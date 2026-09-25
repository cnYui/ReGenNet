"""端到端规范系双人 DCT-Mixer（v2 设计代号 A6 InterMixer），取代"冻结单人 base + 跨人 refiner"。

设计依据（docs/ai/context/20260925-121154-ntu2p-v2-architecture-exploration-design-and-plan.md 的 A6 节）：
- 规范化是最强的归纳偏置（岭回归相机系 0.2280 → 规范系 0.1963，已优于冻结 base 0.2265），
  train 只有 1758 条序列，小而强偏置的模型比加容量更合适；
- 手指 30 个关节主导 mpjpe，但手指相对手腕的局部旋转几乎不变（固定为观测末帧只损失 2.7 mm），
  手可当刚体：手指 = 预测手腕 + 手部整体旋转作用于观测末帧的"手指 - 手腕"偏移；
- 人物顺序是施动/受动，只加 role embedding，不做 A/B 交换。

数据流（全部在规范系内，输出再旋回相机系）：
obs 相机系 -> canonical_frame -> 每人转到"朝向对方"系（B 绕竖直轴转 180°，使"走向对方"对 A/B 同号）
-> 位移历史 (x_t - x_last) 补零到 obs+pred 帧后取前 input_dct_k 个 DCT 系数作 token
-> Mixer 块（系数轴时间混合 / 动作+角色 FiLM 的通道 MLP / 零初始化人物混合）
-> root 低阶 DCT 头 + 局部中阶 DCT 头（刚体手模式下含每只手的轴角）-> IDCT 取未来帧 × saturate ramp
-> 相对观测末帧的残差 delta；pred = obs_last + R^T delta。
输出头零初始化，初始输出与 copy-last 逐位相同，首帧误差恒为 0。

可选 leg_stream（GH 变体）：个人朝向系的腿部专用流替换主干输出头的腿局部行，见 NTU2PLegStream。
"""

from collections import OrderedDict

import torch
from torch import nn

from model.forecasting_ntu2p_residual_xyz import (
    NUM_ACTIONS,
    RAMP_MODES,
    _normalize_action,
    build_residual_ramp,
    count_parameters,
)
from utils.ntu2p_canonical import (
    PELVIS,
    _unit,
    apply_linear,
    canonical_frame,
    facing_direction,
    to_canonical,
)
from utils.ntu2p_gait_losses import LEG_JOINTS
from utils.ntu2p_naturalness import NTU2P_FPS
from utils.ntu_smplx_2p_xyz import (
    NTU_LEFT_WRIST,
    NTU_RIGHT_WRIST,
    NTU_SMPLX_BODY_JOINTS,
    XYZ_COORD_DIM,
    check_ntu_xyz,
    dct_matrix,
)


MODEL_TYPE = "ntu2p_intermixer"
HAND_MODES = ("rigid", "free")
PERSON_FRAMES = ("toward_other", "scene")

# SMPL-X 前 25 个关节为躯干/四肢/头（含 jaw 与双眼），25-39 为左手 15 个指节、40-54 为右手。
NUM_BODY_JOINTS = 25
NUM_FINGERS_PER_HAND = 15
NUM_HANDS = 2
# 左右手腕相邻（20、21），顺序与左手 25-39、右手 40-54 一致；用切片免得每次前向新建索引张量。
WRIST_SLICE = slice(NTU_LEFT_WRIST, NTU_RIGHT_WRIST + 1)
REL_GEOMETRY_DIM = 6
# 规范系 +Y 为上；B 绕它转 180° 即 x、z 取反（真旋转，不改变惯用手）。
_TOWARD_OTHER_SIGNS = ((1.0, 1.0, 1.0), (-1.0, 1.0, -1.0))

CONFIG_KEYS = (
    "obs_len",
    "pred_len",
    "num_actions",
    "hidden",
    "num_blocks",
    "mlp_ratio",
    "cond_dim",
    "root_dct_k",
    "local_dct_k",
    "input_dct_k",
    "dropout",
    "ramp_mode",
    "ramp_saturate_frames",
    "mirror_embed",
    "hand_mode",
    "person_frame",
    "canonical_ab_fallback",
    "kin_proj",
    "leg_stream",
)
# 腿部流：观测末 3 帧朝向之和定个人前向（单帧朝向受扭身与拟合抖动影响）；速度换成 m/s 与位置同量级。
LEG_STREAM_FACING_FRAMES = 3
LEG_STREAM_WIDTH = 512


def rotate_axis_angle(vector, omega):
    """Rodrigues：用轴角 omega 旋转 vector（可广播）。

    不复用 utils.rotation_conversions：它用布尔掩码赋值处理小角度，GPU 上每次前向都会触发 host 同步；
    也不先构造矩阵再 matmul：Ampere 上 TF32 会把 omega=0 时的恒等旋转变成 1e-4 m 级误差，
    破坏"初始输出逐位等于 copy-last"。omega=0 时两个叉积项严格为 0，结果逐位等于 vector。
    """
    # torch 1.7 的 cross 不广播。
    omega, vector = torch.broadcast_tensors(omega, vector)
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-12))
    small = theta2 < 1e-6
    sin_term = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    cos_term = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12))
    cross = torch.cross(omega, vector, dim=-1)
    cross2 = torch.cross(omega, cross, dim=-1)
    return vector + sin_term * cross + cos_term * cross2


def relative_geometry(last_canon):
    """[B,2,55,3] 规范系末帧 -> [B,2,6]：对方 pelvis 在本人朝向系中的 3 维位置、相对 yaw 的 cos/sin、距离。

    全是旋转不变标量，与"场景系/朝向对方系"的选择无关。
    """
    batch_size = int(last_canon.shape[0])
    up = torch.zeros(batch_size * 2, XYZ_COORD_DIM, dtype=last_canon.dtype, device=last_canon.device)
    up[:, 1] = 1.0
    facing = _unit(facing_direction(last_canon.reshape(batch_size * 2, NTU_SMPLX_BODY_JOINTS, 3), up))
    facing = facing.view(batch_size, 2, 3)
    up = up.view(batch_size, 2, 3)
    side = torch.cross(facing, up, dim=-1)
    pelvis = last_canon[:, :, PELVIS]
    offset = pelvis.flip(1) - pelvis
    other_facing = facing.flip(1)
    features = (
        (offset * facing).sum(dim=-1),
        (offset * up).sum(dim=-1),
        (offset * side).sum(dim=-1),
        (facing * other_facing).sum(dim=-1),
        (torch.cross(facing, other_facing, dim=-1) * up).sum(dim=-1),
        offset.norm(dim=-1),
    )
    return torch.stack(features, dim=-1)


class _MixerBlock(nn.Module):
    """系数轴时间混合 -> FiLM 通道 MLP -> 零初始化人物混合，均为 pre-LN 残差。"""

    def __init__(self, num_tokens, hidden, mlp_ratio, cond_dim, dropout):
        super(_MixerBlock, self).__init__()
        self.time_norm = nn.LayerNorm(hidden)
        self.time_mix = nn.Linear(num_tokens, num_tokens)
        self.channel_norm = nn.LayerNorm(hidden)
        # 动作与角色共同调制通道 MLP；零初始化使 FiLM 从恒等开始。
        self.film = nn.Linear(cond_dim, 2 * hidden)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * mlp_ratio, hidden),
        )
        self.person_norm = nn.LayerNorm(hidden)
        # A/B 共享；输入 [h_self; h_other]，零初始化使训练初期先学单人动力学再逐步引入对方信息。
        self.person_mix = nn.Linear(2 * hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.film, self.person_mix):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden, cond):
        # hidden [B,2,K,d]；cond [B,2,cond_dim]（已过 SiLU）。
        mixed = self.time_mix(self.time_norm(hidden).transpose(-1, -2)).transpose(-1, -2)
        hidden = hidden + self.dropout(mixed)
        scale, shift = self.film(cond).unsqueeze(2).chunk(2, dim=-1)
        modulated = self.channel_norm(hidden) * (1.0 + scale) + shift
        hidden = hidden + self.dropout(self.channel_mlp(modulated))
        normed = self.person_norm(hidden)
        return hidden + self.dropout(self.person_mix(torch.cat((normed, normed.flip(1)), dim=-1)))


def rotate_rows(value, rotation):
    """x -> R x，rotation [...,3,3] 与 value [...,3] 按前导维广播；逐元素乘加以避开 TF32。"""
    return (value.unsqueeze(-2) * rotation).sum(dim=-1)


def rotate_rows_transposed(value, rotation):
    """x -> R^T x（rotate_rows 的逆）。"""
    return (value.unsqueeze(-1) * rotation).sum(dim=-2)


def individual_frames(obs_canon, num_frames=LEG_STREAM_FACING_FRAMES):
    """场景规范系（+Y 为上）观测 [B,T,2,55,3] -> 每人个人朝向系旋转 [B,2,3,3]，三行依次为前、左、上。

    前向 = 观测末 num_frames 帧单位朝向之和的水平分量；退化（和向量近 0）时回退规范系 +X。
    """
    batch_size = int(obs_canon.shape[0])
    recent = obs_canon[:, -int(num_frames) :]
    flat = recent.reshape(-1, NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM)
    up = torch.zeros(int(flat.shape[0]), XYZ_COORD_DIM, dtype=obs_canon.dtype, device=obs_canon.device)
    up[:, 1] = 1.0
    # facing_direction 已是水平向量（竖直分量严格为 0），单位向量之和仍水平。
    facing = _unit(facing_direction(flat, up)).view(batch_size, int(num_frames), 2, XYZ_COORD_DIM).sum(dim=1)
    fallback = torch.zeros_like(facing)
    fallback[..., 0] = 1.0
    forward = _unit(torch.where(facing.norm(dim=-1, keepdim=True) >= 1e-6, facing, fallback))
    up = torch.zeros_like(forward)
    up[..., 1] = 1.0
    left = torch.cross(up, forward, dim=-1)
    return torch.stack((forward, left, up), dim=-2)


class NTU2PLegStream(nn.Module):
    """GH：腿与共享主干解耦的专用流，输出替换主干的腿局部残差（A/B 共享权重）。

    诊断：共享主干在朝向对方系把 55 个关节混进 256 通道且被手指梯度主导，腿只占约 3%；只看腿部运动学的 MLP
    在已在走的人上能延续步态相位（f1–20 相关 0.87–0.92），A6 只有 0.42。因此：
    - 输入在个人朝向系中（步态相位只与"前后"有关，与人在场景中朝哪无关）：观测腿局部位置与速度、pelvis 轨迹；
    - 以 detach 的主干 root DCT 预测为条件，使步伐时机与 root 速度耦合（免训练拼接 MLP 腿时滑行帧比 0.145 → 0.335）；
    - 另取 detach 的主干上下文与动作/角色条件，梯度不回主干，主干只学 root 与非腿部分；
    - 输出层零初始化，初始输出与未开启时相同；不用 dropout，不改变训练的随机流。
    """

    def __init__(self, obs_len, root_dct_k, out_dct_k, hidden, cond_dim, width=LEG_STREAM_WIDTH):
        super(NTU2PLegStream, self).__init__()
        num_legs = len(LEG_JOINTS)
        self.out_dct_k = int(out_dct_k)
        leg_dim = (2 * int(obs_len) - 1) * num_legs * XYZ_COORD_DIM
        root_dim = (int(obs_len) + int(root_dct_k)) * XYZ_COORD_DIM
        self.leg_in = nn.Linear(leg_dim, width)
        self.root_in = nn.Linear(root_dim, width, bias=False)
        self.ctx_in = nn.Linear(int(hidden), width, bias=False)
        self.cond_in = nn.Linear(int(cond_dim), width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, self.out_dct_k * num_legs * XYZ_COORD_DIM),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, leg_features, root_features, ctx, cond):
        """各输入 [B,2,D] -> 个人系腿局部残差的 DCT 系数 [B,2,K,8,3]。"""
        hidden = self.leg_in(leg_features) + self.root_in(root_features) + self.ctx_in(ctx) + self.cond_in(cond)
        coeffs = self.mlp(self.norm(hidden))
        return coeffs.view(coeffs.shape[:2] + (self.out_dct_k, len(LEG_JOINTS), XYZ_COORD_DIM))


class NTU2PInterMixer(nn.Module):
    """端到端双人 DCT-Mixer：规范系输入、残差 DCT 输出，零初始化时严格等于 copy-last。"""

    def __init__(
        self,
        obs_len=10,
        pred_len=50,
        num_actions=NUM_ACTIONS,
        hidden=256,
        num_blocks=6,
        mlp_ratio=2,
        cond_dim=64,
        root_dct_k=8,
        local_dct_k=20,
        input_dct_k=30,
        dropout=0.1,
        ramp_mode="saturate",
        ramp_saturate_frames=5,
        mirror_embed=False,
        hand_mode="rigid",
        person_frame="toward_other",
        canonical_ab_fallback="a_facing",
        kin_proj=False,
        leg_stream=False,
    ):
        super(NTU2PInterMixer, self).__init__()
        self.model_type = MODEL_TYPE
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.window_len = self.obs_len + self.pred_len
        self.num_actions = int(num_actions)
        self.hidden = int(hidden)
        self.num_blocks = int(num_blocks)
        self.mlp_ratio = int(mlp_ratio)
        self.cond_dim = int(cond_dim)
        self.root_dct_k = int(root_dct_k)
        self.local_dct_k = int(local_dct_k)
        self.input_dct_k = int(input_dct_k)
        self.dropout = float(dropout)
        self.ramp_mode = str(ramp_mode)
        self.ramp_saturate_frames = int(ramp_saturate_frames)
        self.mirror_embed = bool(mirror_embed)
        self.hand_mode = str(hand_mode)
        self.person_frame = str(person_frame)
        # 旧 checkpoint 的 config 无此键，加载时按默认 a_facing，行为不变。
        self.canonical_ab_fallback = str(canonical_ab_fallback)
        self.kin_proj = bool(kin_proj)
        # 旧 checkpoint 的 config 无此键，加载时按默认 False，结构与行为不变。
        self.leg_stream = bool(leg_stream)
        if self.ramp_mode not in RAMP_MODES:
            raise ValueError("ramp_mode 必须是 {}，当前为 {}".format(RAMP_MODES, self.ramp_mode))
        if self.hand_mode not in HAND_MODES:
            raise ValueError("hand_mode 必须是 {}，当前为 {}".format(HAND_MODES, self.hand_mode))
        if self.person_frame not in PERSON_FRAMES:
            raise ValueError("person_frame 必须是 {}，当前为 {}".format(PERSON_FRAMES, self.person_frame))
        if not 1 <= self.input_dct_k <= self.window_len:
            raise ValueError("input_dct_k 必须在 [1,{}] 内".format(self.window_len))
        # 输出系数直接取自同序号的输入系数 token，因此输出阶数不能超过 token 数。
        for name, value in (("root_dct_k", self.root_dct_k), ("local_dct_k", self.local_dct_k)):
            if not 1 <= value <= self.input_dct_k:
                raise ValueError("{} 必须在 [1,input_dct_k={}] 内".format(name, self.input_dct_k))

        # 窗口长度的 DCT 基：输入与输出的第 k 个 token 对应同一频率（LTD 式补零，未来段位移为 0）。
        window_dct = dct_matrix(self.window_len)
        self.register_buffer("input_dct", window_dct[: self.input_dct_k, : self.obs_len].clone(), persistent=False)
        self.register_buffer("root_idct", window_dct[: self.root_dct_k, self.obs_len :].clone(), persistent=False)
        self.register_buffer("local_idct", window_dct[: self.local_dct_k, self.obs_len :].clone(), persistent=False)
        ramp = build_residual_ramp(self.pred_len, self.ramp_mode, self.ramp_saturate_frames)
        self.register_buffer("ramp", ramp.view(1, self.pred_len, 1, 1), persistent=False)
        signs = _TOWARD_OTHER_SIGNS if self.person_frame == "toward_other" else ((1.0,) * 3,) * 2
        self.register_buffer("person_signs", torch.tensor(signs).view(2, 1, XYZ_COORD_DIM), persistent=False)

        person_dim = NTU_SMPLX_BODY_JOINTS * XYZ_COORD_DIM
        self.disp_proj = nn.Linear(person_dim, self.hidden)
        self.pose_proj = nn.Linear(person_dim, self.hidden)
        self.geometry_proj = nn.Linear(REL_GEOMETRY_DIM, self.hidden)
        self.cond_proj = nn.Linear(self.cond_dim, self.hidden)
        self.coeff_pos = nn.Parameter(torch.randn(1, 1, self.input_dct_k, self.hidden) * 0.02)
        self.action_embed = nn.Embedding(self.num_actions, self.cond_dim)
        self.role_embed = nn.Embedding(2, self.cond_dim)
        # 默认不创建，保证未开镜像的模型 state_dict 不含多余参数；推理时镜像指示恒为 0。
        self.mirror_embedding = nn.Embedding(2, self.cond_dim) if self.mirror_embed else None
        self.cond_act = nn.SiLU()
        if self.kin_proj:
            from utils.ntu2p_kinematic_projection import SkeletonProjector

            # 只有 buffer、不消耗随机数：开关不改变其余参数的初始化，与未投影的同 seed run 配对。
            self.skeleton_projector = SkeletonProjector()

        self.blocks = nn.ModuleList(
            [
                _MixerBlock(self.input_dct_k, self.hidden, self.mlp_ratio, self.cond_dim, self.dropout)
                for _ in range(self.num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(self.hidden)
        self.root_head = nn.Linear(self.hidden, XYZ_COORD_DIM)
        # rigid：躯干 24 个非 pelvis 关节的局部位移 + 左右手各一个轴角；free：54 个关节全部自由。
        if self.hand_mode == "rigid":
            self.num_local_joints = NUM_BODY_JOINTS - 1
            local_channels = (self.num_local_joints + NUM_HANDS) * XYZ_COORD_DIM
        else:
            self.num_local_joints = NTU_SMPLX_BODY_JOINTS - 1
            local_channels = self.num_local_joints * XYZ_COORD_DIM
        self.local_head = nn.Linear(self.hidden, local_channels)
        for layer in (self.root_head, self.local_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.leg_stream:
            # 腿关节 j 在 _decode 的 joints 张量（不含 pelvis）中位于第 j-1 行。
            self.register_buffer("leg_rows", torch.tensor([joint - 1 for joint in LEG_JOINTS], dtype=torch.long), persistent=False)
            # 在 fork 出的 CPU 随机流里初始化：全局随机流不前进，其余参数初始化、dropout 与同 seed 参照 run 逐一对应。
            with torch.random.fork_rng(devices=[]):
                self.leg_net = NTU2PLegStream(self.obs_len, self.root_dct_k, self.local_dct_k, self.hidden, self.cond_dim)

    def config(self):
        config = OrderedDict([("model_type", self.model_type)])
        for key in CONFIG_KEYS:
            config[key] = getattr(self, key)
        config["architecture"] = "canonical_two_person_dct_mixer_end_to_end"
        return config

    def _to_person_frame(self, value):
        """[B,...,2,J,3] 场景规范系 <-> 朝向对方系（对合，正反变换相同）。"""
        return value * self.person_signs.to(dtype=value.dtype)

    def _condition(self, action, context, batch_size, device):
        cond = self.action_embed(action).unsqueeze(1) + self.role_embed.weight.unsqueeze(0)
        mirror = None if context is None else context.get("mirror")
        # mirror_embed=False 时忽略指示：这正是"镜像增广但不告知模型"的消融口径。
        if self.mirror_embedding is not None:
            if mirror is None:
                mirror = torch.zeros(batch_size, dtype=torch.long, device=device)
            cond = cond + self.mirror_embedding(torch.as_tensor(mirror, device=device).long()).unsqueeze(1)
        return cond

    def _embed(self, obs_person, last_person, geometry, cond):
        batch_size = int(obs_person.shape[0])
        # 位移历史只有前 obs_len 帧非零，补零段对 DCT 无贡献，直接用截断的基。
        displacement = (obs_person - last_person.unsqueeze(1)).reshape(batch_size, self.obs_len, 2, -1)
        coeffs = torch.einsum("kt,btpc->bpkc", self.input_dct.to(dtype=displacement.dtype), displacement)
        pose = (last_person - last_person[:, :, PELVIS : PELVIS + 1]).reshape(batch_size, 2, -1)
        static = self.pose_proj(pose) + self.geometry_proj(geometry) + self.cond_proj(cond)
        return self.disp_proj(coeffs) + static.unsqueeze(2) + self.coeff_pos

    def _leg_stream(self, hidden, cond, obs_canon):
        """腿部流 -> 朝向对方系下腿关节相对 pelvis 的局部残差 [B,T,2,8,3]（与 _decode 中 joints 的腿行同义）。"""
        batch_size = int(hidden.shape[0])
        normed = self.out_norm(hidden).detach()
        ctx = normed.mean(dim=2)
        # root 头在朝向对方系；乘符号回到场景规范系（对合），再转到个人系。
        root_coeffs = self._to_person_frame(self.root_head(normed[:, :, : self.root_dct_k]).detach())
        rotation = individual_frames(obs_canon)  # [B,2,3,3]
        per_frame = rotation.unsqueeze(1).unsqueeze(3)  # [B,1,2,1,3,3]
        legs = obs_canon[:, :, :, list(LEG_JOINTS)] - obs_canon[:, :, :, PELVIS : PELVIS + 1]
        legs = rotate_rows(legs, per_frame)  # [B,T,2,8,3]
        velocity = (legs[:, 1:] - legs[:, :-1]) * NTU2P_FPS
        pelvis = obs_canon[:, :, :, PELVIS] - obs_canon[:, -1:, :, PELVIS]
        pelvis = rotate_rows(pelvis, rotation.unsqueeze(1))  # [B,T,2,3]
        root_future = rotate_rows(root_coeffs, rotation.unsqueeze(2))  # [B,2,K,3]

        def per_person(value):
            return value.transpose(1, 2).reshape(batch_size, 2, -1)

        leg_features = torch.cat((per_person(legs), per_person(velocity)), dim=-1)
        root_features = torch.cat((per_person(pelvis), root_future.reshape(batch_size, 2, -1)), dim=-1)
        coeffs = self.leg_net(leg_features, root_features, ctx, cond.detach())  # [B,2,K,8,3]
        num_legs = len(LEG_JOINTS)
        flat = coeffs.reshape(batch_size, 2, self.local_dct_k, num_legs * XYZ_COORD_DIM)
        local = torch.einsum("kt,bpkc->btpc", self.local_idct.to(dtype=hidden.dtype), flat) * self.ramp.to(dtype=hidden.dtype)
        local = local.reshape(batch_size, self.pred_len, 2, num_legs, XYZ_COORD_DIM)
        scene = rotate_rows_transposed(local, rotation.unsqueeze(1).unsqueeze(3))
        return self._to_person_frame(scene)

    def _decode(self, hidden, last_person, leg_local=None):
        """token -> 朝向对方系下相对观测末帧的残差 [B,T,2,55,3]；leg_local 给出时替换腿关节的局部残差。"""
        batch_size = int(hidden.shape[0])
        hidden = self.out_norm(hidden)
        root_coeffs = self.root_head(hidden[:, :, : self.root_dct_k])
        local_coeffs = self.local_head(hidden[:, :, : self.local_dct_k])
        ramp = self.ramp.to(dtype=hidden.dtype)
        root = torch.einsum("kt,bpkc->btpc", self.root_idct.to(dtype=hidden.dtype), root_coeffs) * ramp
        local = torch.einsum("kt,bpkc->btpc", self.local_idct.to(dtype=hidden.dtype), local_coeffs) * ramp
        joint_dims = self.num_local_joints * XYZ_COORD_DIM
        joints = local[..., :joint_dims].reshape(batch_size, self.pred_len, 2, self.num_local_joints, XYZ_COORD_DIM)
        if leg_local is not None:
            # 替换而非叠加：主干腿行的梯度为 0，腿局部类损失只训练腿部流；腿的绝对位置类损失仍经 root 回到主干。
            joints = joints.index_copy(3, self.leg_rows, leg_local)
        # pelvis 的局部位移恒为 0，只随 root 平移。
        delta = torch.cat((torch.zeros_like(joints[:, :, :, :1]), joints), dim=3) + root.unsqueeze(3)
        if self.hand_mode == "free":
            return delta
        # 刚体手：手指 = 预测手腕 + R(omega) · 观测末帧"手指 - 手腕"偏移；omega 已乘 ramp，首帧 R = I。
        omega = local[..., joint_dims:].reshape(batch_size, self.pred_len, 2, NUM_HANDS, 1, XYZ_COORD_DIM)
        fingers = last_person[:, :, NUM_BODY_JOINTS:].reshape(batch_size, 2, NUM_HANDS, NUM_FINGERS_PER_HAND, 3)
        offsets = (fingers - last_person[:, :, WRIST_SLICE].unsqueeze(3)).unsqueeze(1)
        wrist_delta = delta[:, :, :, WRIST_SLICE].unsqueeze(4)
        finger_delta = wrist_delta + (rotate_axis_angle(offsets, omega) - offsets)
        finger_delta = finger_delta.reshape(batch_size, self.pred_len, 2, NUM_HANDS * NUM_FINGERS_PER_HAND, 3)
        return torch.cat((delta, finger_delta), dim=3)

    def forward(self, obs_xyz, action, context=None, return_details=False):
        check_ntu_xyz("obs_xyz", obs_xyz, seq_len=self.obs_len, num_persons=2)
        batch_size = int(obs_xyz.shape[0])
        action = _normalize_action(action, batch_size, self.num_actions, obs_xyz.device)
        frame = canonical_frame(obs_xyz, ab_fallback=self.canonical_ab_fallback)
        obs_canon = to_canonical(obs_xyz, frame)
        last_canon = obs_canon[:, -1]
        obs_person = self._to_person_frame(obs_canon)
        last_person = obs_person[:, -1]

        cond = self._condition(action, context, batch_size, obs_xyz.device)
        hidden = self._embed(obs_person, last_person, relative_geometry(last_canon), cond)
        cond = self.cond_act(cond)
        for block in self.blocks:
            hidden = block(hidden, cond)
        leg_local = self._leg_stream(hidden, cond, obs_canon) if self.leg_stream else None
        delta = self._to_person_frame(self._decode(hidden, last_person, leg_local))

        # 只把残差旋回相机系再加相机系末帧：delta=0 时输出逐位等于 copy-last，首帧误差严格为 0。
        pred = obs_xyz[:, -1:] + apply_linear(delta, frame["rotation"].transpose(-1, -2))
        pred_free = None
        if self.kin_proj:
            # 刚体手已由 hand_mode 保证；投影再固定身体骨长（骨长取观测末帧），消除前臂/小腿伸缩。
            pred_free = pred
            pred = self.skeleton_projector(pred_free, obs_xyz[:, -1])
        if not return_details:
            return pred
        return OrderedDict(
            [
                ("pred", pred),
                ("delta", delta),
                ("frame", frame),
                ("pred_canon", last_canon.unsqueeze(1) + delta),
                ("pred_free", pred_free),
            ]
        )


def create_ntu2p_intermixer_from_config(config):
    config = dict(config or {})
    return NTU2PInterMixer(**{key: config[key] for key in CONFIG_KEYS if key in config})


def load_ntu2p_intermixer_checkpoint(path, device):
    """读取 {model_state_dict, model_type, model_config} 格式的 checkpoint（与旧 refiner 同构）。"""
    state = torch.load(path, map_location=device)
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError("checkpoint model_type 必须是 {}，当前为 {}".format(MODEL_TYPE, state.get("model_type")))
    model = create_ntu2p_intermixer_from_config(state.get("model_config", {}))
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


__all__ = [
    "CONFIG_KEYS",
    "MODEL_TYPE",
    "NTU2PInterMixer",
    "NTU2PLegStream",
    "count_parameters",
    "create_ntu2p_intermixer_from_config",
    "individual_frames",
    "load_ntu2p_intermixer_checkpoint",
    "relative_geometry",
    "rotate_axis_angle",
]
