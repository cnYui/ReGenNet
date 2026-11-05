"""NTU2P JiT 流匹配双人预测：ramp-DCT 残差系数上的 x 预测流，两类去噪器（J1–J4）、条件化腿流与多种读出。

设计：docs/ai/context/20261018-201532-ntu2p-jit-flow-rl-design-and-plan.md 第 3 节。

状态 x：未来 50 帧相对观测末帧残差（场景规范系 + 朝向对方人物系，与 InterMixer 相同）在 ramp-DCT 基上的系数，
root 8 阶 ×3，局部（关节 − pelvis）20 阶 × 28 关节 ×3；28 关节 = 身体 1–24 + 每手 index1/pinky1，正是 FD2 主损失的
hand2 子集。按 root/局部两组标量标准差归一化，保留频谱自然衰减。

流（JiT，t=1 为干净数据）：z_t = t·x + (1−t)·ε；网络直接输出 x̂，v̂ = (x̂ − z_t)/max(1−t, 0.05)。
确定性端点 = t=0、z=0 的一次（自条件时两次）前向，训练时施加 FD2 的完整损失配方，是可部署的点预测读出。

解码：IDCT × ramp（首帧残差严格 0）→ 加回观测末帧 → 其余 26 个手指随本手手腕平移 → SkeletonProjector
（骨长恒定、刚体手；手部 Procrustes 只给 index1/pinky1 权重，即由两个指根与手腕决定手部旋转）。

去噪器：
- mixer（J1/J2）：FD2 的 DCT-Mixer 主干，噪声系数作为同频 token 的输入通道，t 进 FiLM；contact_graph 时
  把自条件估计解码出的双向手—躯干距离/接近速度/因果接触记忆曲线取窗口 DCT 注入 token。
- stg（J3/J4）：JiT 式 Transformer（adaLN-Zero、RMSNorm、qk-norm、SwiGLU、瓶颈 patch 嵌入），token =
  人物 × 身体部位 × 5 帧时间块；contact_graph 时对"对方手臂块 ↔ 本人躯干块"加零初始化的接触图注意力偏置，
  causal_cross 时本人第 i 块只能看对方第 ≤ i 块。输出时域残差，最小二乘投回 DCT 系数。
腿：两类去噪器都由条件化腿流（FD2 腿流 + 噪声腿系数/自条件输入）给出腿关节系数，替换主干的腿行。
"""

import math
from collections import OrderedDict

import torch
from torch import nn

from model.forecasting_ntu2p_intermixer import (
    LEG_STREAM_WIDTH,
    NTU2PLegStream,
    _MixerBlock,
    _TOWARD_OTHER_SIGNS,
    individual_frames,
    relative_geometry,
    rotate_rows,
    rotate_rows_transposed,
)
from model.forecasting_ntu2p_residual_xyz import NUM_ACTIONS, RAMP_MODES, _normalize_action, build_residual_ramp, count_parameters
from utils.ntu2p_canonical import PELVIS, apply_linear, canonical_frame, to_canonical
from utils.ntu2p_contact_graph import causal_contact_memory, hand_body_distances
from utils.ntu2p_gait_losses import LEG_JOINTS
from utils.ntu2p_kinematic_projection import SkeletonProjector
from utils.ntu2p_naturalness import NTU2P_FPS
from utils.ntu_smplx_2p_xyz import NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM, check_ntu_xyz, dct_matrix


MODEL_TYPE = "ntu2p_jit_flow"
DENOISERS = ("mixer", "stg")
READOUTS = ("endpoint", "sample", "mean", "median", "medoid")
FINGER_ROOTS = (25, 31, 40, 46)
LOCAL_JOINTS = tuple(range(1, 25)) + FINGER_ROOTS
STATE_JOINTS = (PELVIS,) + LOCAL_JOINTS
NUM_LOCAL = len(LOCAL_JOINTS)
NUM_STATE = len(STATE_JOINTS)
LEFT_WRIST, RIGHT_WRIST = 20, 21
# 关节 j 在状态中的行号（STATE_JOINTS 的位置）；其余手指取本手手腕所在行，即随手腕平移。
_STATE_ROW = {joint: row for row, joint in enumerate(STATE_JOINTS)}
SOURCE_ROWS = tuple(
    _STATE_ROW.get(joint, _STATE_ROW[LEFT_WRIST if joint < 40 else RIGHT_WRIST]) for joint in range(NTU_SMPLX_BODY_JOINTS)
)
# 腿关节在局部状态（不含 pelvis）中的行号。
LEG_ROWS = tuple(_STATE_ROW[joint] - 1 for joint in LEG_JOINTS)
# STG 的身体部位（状态行号）：躯干+头、左臂+左手两指根、右臂+右手两指根、左腿、右腿。
STG_PARTS = (
    tuple(_STATE_ROW[j] for j in (0, 3, 6, 9, 12, 13, 14, 15, 22, 23, 24)),
    tuple(_STATE_ROW[j] for j in (16, 18, 20, 25, 31)),
    tuple(_STATE_ROW[j] for j in (17, 19, 21, 40, 46)),
    tuple(_STATE_ROW[j] for j in (1, 4, 7, 10)),
    tuple(_STATE_ROW[j] for j in (2, 5, 8, 11)),
)
TORSO_PART, ARM_PARTS = 0, (1, 2)
# 自条件关系曲线：本人双手→对方躯干、对方双手→本人躯干的距离，各自接近速度与因果接触记忆。
REL_CHANNELS = 12
GRAPH_FEATURES = 3
CONTACT_DISTANCE = 0.25
CONTACT_TEMPERATURE = 0.05
V_DENOM_MIN = 0.05
WEISZFELD_ITERATIONS = 10

CONFIG_KEYS = (
    "denoiser",
    "self_cond",
    "contact_graph",
    "causal_cross",
    "obs_len",
    "pred_len",
    "num_actions",
    "root_k",
    "local_k",
    "root_scale",
    "local_scale",
    "hidden",
    "num_blocks",
    "mlp_ratio",
    "cond_dim",
    "input_dct_k",
    "dropout",
    "stg_width",
    "stg_heads",
    "stg_blocks",
    "patch_len",
    "ramp_mode",
    "ramp_saturate_frames",
    "canonical_ab_fallback",
)


class RampDCTCodec(nn.Module):
    """残差 [B,T,P,29,3] <-> (root 系数 [B,P,Kr,3], 局部系数 [B,P,Kl,28,3])。

    子空间与 InterMixer 的输出相同（窗口 DCT 前 K 个基限制到未来帧再乘 ramp），但系数取该子空间按频率顺序
    Gram–Schmidt（QR）得到的正交基上的坐标：原基在未来段上严重病态（K=20 时条件数 4e4），GT 的最小二乘
    系数会是相互抵消的巨大数值，流无法在这种坐标上加噪去噪。正交基下编码就是投影、能量守恒，
    第 k 个系数仍以第 k 个频率为主。
    """

    def __init__(self, obs_len, pred_len, root_k, local_k, ramp_mode, ramp_saturate_frames):
        super(RampDCTCodec, self).__init__()
        window = dct_matrix(int(obs_len) + int(pred_len), dtype=torch.float64)
        ramp = build_residual_ramp(int(pred_len), ramp_mode, ramp_saturate_frames).reshape(int(pred_len)).double()
        for name, order in (("root", root_k), ("local", local_k)):
            basis = ramp.unsqueeze(1) * window[: int(order), int(obs_len) :].t()  # [T,K]
            q, r = torch.qr(basis)
            q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)  # 与原基同向，系数符号约定固定
            self.register_buffer(name + "_basis", q.float(), persistent=False)
            self.register_buffer(name + "_pinv", q.t().contiguous().float(), persistent=False)

    def decode(self, root, local):
        root_t = torch.einsum("tk,bpkc->btpc", self.root_basis.to(root.dtype), root)
        local_t = torch.einsum("tk,bpkjc->btpjc", self.local_basis.to(local.dtype), local)
        return torch.cat((root_t.unsqueeze(3), local_t + root_t.unsqueeze(3)), dim=3)

    def encode(self, residual):
        root = residual[..., 0, :]
        local = residual[..., 1:, :] - root.unsqueeze(3)
        root_c = torch.einsum("kt,btpc->bpkc", self.root_pinv.to(residual.dtype), root)
        local_c = torch.einsum("kt,btpjc->bpkjc", self.local_pinv.to(residual.dtype), local)
        return root_c, local_c


def timestep_embedding(t, dim):
    """t ∈ [0,1] 的正弦编码（按 1000 步缩放，与常见扩散实现量级一致）。"""
    half = int(dim) // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / float(half))
    args = (t.float() * 1000.0).unsqueeze(-1) * freqs
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


class FlowLegStream(NTU2PLegStream):
    """FD2 腿流 + 噪声腿系数（与自条件估计）输入；输出个人朝向系的腿 x̂ 系数（归一化单位）。"""

    def __init__(self, obs_len, root_k, local_k, ctx_dim, cond_dim, self_cond):
        super(FlowLegStream, self).__init__(obs_len, root_k, local_k, ctx_dim, cond_dim)
        state_dim = int(local_k) * len(LEG_JOINTS) * XYZ_COORD_DIM
        self.noisy_in = nn.Linear(state_dim, LEG_STREAM_WIDTH, bias=False)
        self.sc_in = nn.Linear(state_dim, LEG_STREAM_WIDTH, bias=False) if self_cond else None

    def forward(self, leg_features, root_features, ctx, cond, noisy, sc=None):
        hidden = self.leg_in(leg_features) + self.root_in(root_features) + self.ctx_in(ctx) + self.cond_in(cond)
        hidden = hidden + self.noisy_in(noisy)
        if self.sc_in is not None and sc is not None:
            hidden = hidden + self.sc_in(sc)
        coeffs = self.mlp(self.norm(hidden))
        return coeffs.view(coeffs.shape[:2] + (self.out_dct_k, len(LEG_JOINTS), XYZ_COORD_DIM))


def _state_tokens(root, local, num_tokens):
    """root [B,2,Kr,3]、局部 [B,2,Kl,28,3] -> 同频 token 输入 [B,2,K,87]，超出各自阶数的位置补 0。"""
    batch_size = int(root.shape[0])
    out = root.new_zeros(batch_size, 2, int(num_tokens), XYZ_COORD_DIM * (1 + NUM_LOCAL))
    out[:, :, : root.shape[2], :XYZ_COORD_DIM] = root
    out[:, :, : local.shape[2], XYZ_COORD_DIM:] = local.reshape(batch_size, 2, local.shape[2], -1)
    return out


class MixerDenoiser(nn.Module):
    """J1/J2：FD2 的系数轴 DCT-Mixer 主干做 x 预测去噪。"""

    def __init__(self, obs_len, pred_len, root_k, local_k, input_dct_k, hidden, num_blocks, mlp_ratio, cond_dim, dropout,
                 self_cond, contact_graph):
        super(MixerDenoiser, self).__init__()
        self.obs_len, self.root_k, self.local_k = int(obs_len), int(root_k), int(local_k)
        self.input_dct_k = int(input_dct_k)
        window = dct_matrix(int(obs_len) + int(pred_len))
        self.register_buffer("input_dct", window[: self.input_dct_k, : self.obs_len].clone(), persistent=False)
        self.register_buffer("rel_dct", window[: self.input_dct_k].clone(), persistent=False)
        person_dim = NTU_SMPLX_BODY_JOINTS * XYZ_COORD_DIM
        state_dim = XYZ_COORD_DIM * (1 + NUM_LOCAL)
        self.disp_proj = nn.Linear(person_dim, hidden)
        self.pose_proj = nn.Linear(person_dim, hidden)
        self.geometry_proj = nn.Linear(6, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.coeff_pos = nn.Parameter(torch.randn(1, 1, self.input_dct_k, hidden) * 0.02)
        self.noisy_proj = nn.Linear(state_dim, hidden)
        self.sc_proj = nn.Linear(state_dim, hidden) if self_cond else None
        self.rel_proj = nn.Linear(REL_CHANNELS, hidden) if contact_graph else None
        if self.rel_proj is not None:
            # 零初始化：关系通道从"不改变主干"开始，接触图的作用只能由训练引入。
            nn.init.zeros_(self.rel_proj.weight)
            nn.init.zeros_(self.rel_proj.bias)
        self.blocks = nn.ModuleList([_MixerBlock(self.input_dct_k, hidden, mlp_ratio, cond_dim, dropout) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.root_head = nn.Linear(hidden, XYZ_COORD_DIM)
        self.local_head = nn.Linear(hidden, NUM_LOCAL * XYZ_COORD_DIM)
        for layer in (self.root_head, self.local_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.ctx_dim = int(hidden)

    def forward(self, prep, z_root, z_local, cond, sc=None, rel=None):
        obs_person, last_person = prep["obs_person"], prep["last_person"]
        batch_size = int(obs_person.shape[0])
        displacement = (obs_person - last_person.unsqueeze(1)).reshape(batch_size, self.obs_len, 2, -1)
        coeffs = torch.einsum("kt,btpc->bpkc", self.input_dct.to(displacement.dtype), displacement)
        pose = (last_person - last_person[:, :, PELVIS : PELVIS + 1]).reshape(batch_size, 2, -1)
        static = self.pose_proj(pose) + self.geometry_proj(prep["geometry"]) + self.cond_proj(prep["cond"])
        hidden = self.disp_proj(coeffs) + static.unsqueeze(2) + self.coeff_pos
        hidden = hidden + self.noisy_proj(_state_tokens(z_root, z_local, self.input_dct_k))
        if self.sc_proj is not None and sc is not None:
            hidden = hidden + self.sc_proj(_state_tokens(sc[0], sc[1], self.input_dct_k))
        if self.rel_proj is not None and rel is not None:
            hidden = hidden + self.rel_proj(torch.einsum("kt,btpc->bpkc", self.rel_dct.to(rel.dtype), rel))
        for block in self.blocks:
            hidden = block(hidden, cond)
        normed = self.out_norm(hidden)
        root = self.root_head(normed[:, :, : self.root_k])
        local = self.local_head(normed[:, :, : self.local_k]).view(batch_size, 2, self.local_k, NUM_LOCAL, XYZ_COORD_DIM)
        return root, local, normed.mean(dim=2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        super(RMSNorm, self).__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim))) if affine else None

    def forward(self, value):
        value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return value if self.weight is None else value * self.weight


class _STGBlock(nn.Module):
    """adaLN-Zero 预归一化 Transformer 块：qk-norm 多头注意力（外加偏置/掩码）+ SwiGLU。"""

    def __init__(self, width, heads, cond_dim, mlp_ratio, dropout):
        super(_STGBlock, self).__init__()
        self.heads = int(heads)
        self.head_dim = int(width) // self.heads
        self.norm1 = RMSNorm(width, affine=False)
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(width, width)
        self.norm2 = RMSNorm(width, affine=False)
        inner = int(width * mlp_ratio * 2 / 3)
        self.w1 = nn.Linear(width, inner)
        self.w3 = nn.Linear(width, inner)
        self.w2 = nn.Linear(inner, width)
        self.dropout = nn.Dropout(dropout)
        self.ada = nn.Linear(cond_dim, 6 * width)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, tokens, cond, bias):
        # tokens [B,2,M,w]（每人 M 个 token）；cond [B,2,cond_dim]；bias [B,h,N,N]，N=2M。
        batch_size, persons, per_person, width = tokens.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).unsqueeze(2).chunk(6, dim=-1)
        x = self.norm1(tokens) * (1.0 + scale1) + shift1
        qkv = self.qkv(x.reshape(batch_size, persons * per_person, width))
        qkv = qkv.view(batch_size, persons * per_person, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim) + bias
        attn = self.dropout(torch.softmax(logits, dim=-1))
        out = torch.matmul(attn, value).transpose(1, 2).reshape(batch_size, persons, per_person, width)
        tokens = tokens + gate1 * self.proj(out)
        x = self.norm2(tokens) * (1.0 + scale2) + shift2
        mlp = self.w2(self.dropout(nn.functional.silu(self.w1(x)) * self.w3(x)))
        return tokens + gate2 * mlp


class STGDenoiser(nn.Module):
    """J3/J4：人物 × 身体部位 × 5 帧时间块的时空图 Transformer，输出时域残差后最小二乘投回 DCT 系数。"""

    BOTTLENECK = 128

    def __init__(self, obs_len, pred_len, patch_len, width, heads, num_blocks, cond_dim, dropout, self_cond, contact_graph,
                 causal_cross, codec, mlp_ratio=4):
        super(STGDenoiser, self).__init__()
        self.obs_len, self.pred_len, self.patch_len = int(obs_len), int(pred_len), int(patch_len)
        if self.obs_len % self.patch_len or self.pred_len % self.patch_len:
            raise ValueError("obs_len 与 pred_len 必须能被 patch_len 整除")
        self.num_obs_patches = self.obs_len // self.patch_len
        self.num_patches = (self.obs_len + self.pred_len) // self.patch_len
        self.width, self.heads = int(width), int(heads)
        self.self_cond = bool(self_cond)
        self.contact_graph = bool(contact_graph)
        self.causal_cross = bool(causal_cross)
        self.codec = codec
        channels = 2 if self.self_cond else 1
        self.embed = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.patch_len * len(part) * XYZ_COORD_DIM * channels, self.BOTTLENECK), nn.Linear(self.BOTTLENECK, self.width))
                for part in STG_PARTS
            ]
        )
        num_parts = len(STG_PARTS)
        self.patch_pos = nn.Parameter(torch.randn(self.num_patches, self.width) * 0.02)
        self.part_pos = nn.Parameter(torch.randn(num_parts, self.width) * 0.02)
        self.role_pos = nn.Parameter(torch.randn(2, self.width) * 0.02)
        self.obs_flag = nn.Parameter(torch.randn(2, self.width) * 0.02)
        self.static_proj = nn.Linear(6 + NTU_SMPLX_BODY_JOINTS * XYZ_COORD_DIM, self.width)
        # 部位对与相对时间的可学习偏置（同人/跨人分开），零初始化 = 初始全连接无偏置。
        self.part_bias = nn.Parameter(torch.zeros(self.heads, 2, num_parts, num_parts))
        self.time_bias = nn.Parameter(torch.zeros(self.heads, 2, 2 * self.num_patches - 1))
        if self.contact_graph:
            # [对方手 → 本人躯干] 作为键时，与 [本人手 → 对方躯干] 作为键时各一组；零初始化。
            self.graph_recv = nn.Linear(GRAPH_FEATURES, self.heads)
            self.graph_send = nn.Linear(GRAPH_FEATURES, self.heads)
            for layer in (self.graph_recv, self.graph_send):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.blocks = nn.ModuleList([_STGBlock(self.width, self.heads, cond_dim, mlp_ratio, dropout) for _ in range(num_blocks)])
        self.final_norm = RMSNorm(self.width, affine=False)
        self.final_ada = nn.Linear(cond_dim, 2 * self.width)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        self.out = nn.ModuleList([nn.Linear(self.width, self.patch_len * len(part) * XYZ_COORD_DIM) for part in STG_PARTS])
        for layer in self.out:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.ctx_dim = self.width
        self._register_index()

    def _register_index(self):
        """token 顺序 (人物, 部位, 时间块)；预存成对索引，偏置/掩码每次前向只做一次 gather。"""
        num_parts, num_patches = len(STG_PARTS), self.num_patches
        person = torch.arange(2).view(2, 1, 1).expand(2, num_parts, num_patches).reshape(-1)
        part = torch.arange(num_parts).view(1, num_parts, 1).expand(2, num_parts, num_patches).reshape(-1)
        patch = torch.arange(num_patches).view(1, 1, num_patches).expand(2, num_parts, num_patches).reshape(-1)
        cross = (person.view(-1, 1) != person.view(1, -1)).long()
        self.register_buffer("pair_cross", cross, persistent=False)
        self.register_buffer("pair_part_q", part.view(-1, 1).expand_as(cross).clone(), persistent=False)
        self.register_buffer("pair_part_k", part.view(1, -1).expand_as(cross).clone(), persistent=False)
        self.register_buffer("pair_dt", (patch.view(1, -1) - patch.view(-1, 1) + num_patches - 1), persistent=False)
        future_cross = (cross > 0) & (patch.view(1, -1) > patch.view(-1, 1))
        self.register_buffer("causal_mask", future_cross, persistent=False)

    def _static_bias(self):
        part = self.part_bias[:, self.pair_cross, self.pair_part_q, self.pair_part_k]
        time = self.time_bias[:, self.pair_cross, self.pair_dt]
        bias = part + time
        if self.causal_cross:
            bias = bias.masked_fill(self.causal_mask, -1e4)
        return bias.unsqueeze(0)  # [1,h,N,N]

    def _graph_bias(self, graph):
        """graph [B,2(手的主人 q),2(手 h),P,F] -> [B,h,N,N]：躯干查询看对方手臂键、手臂查询看对方躯干键。"""
        batch_size = int(graph.shape[0])
        num_parts, num_patches = len(STG_PARTS), self.num_patches
        bias = graph.new_zeros(batch_size, self.heads, 2, num_parts, num_patches, 2, num_parts, num_patches)
        recv = self.graph_recv(graph).permute(0, 4, 1, 2, 3)  # [B,h,q,hand,P]
        send = self.graph_send(graph).permute(0, 4, 1, 2, 3)
        for owner in (0, 1):
            other = 1 - owner
            for hand, arm in enumerate(ARM_PARTS):
                # 受动者 other 的躯干块（任意查询时刻）看 owner 手臂第 j 块时，按 owner 的手在 j 时刻离 other 躯干的远近加偏置。
                bias[:, :, other, TORSO_PART, :, owner, arm, :] = recv[:, :, owner, hand].unsqueeze(2)
                # owner 的手臂块看 other 的躯干第 j 块时，同一几何关系的另一方向。
                bias[:, :, owner, arm, :, other, TORSO_PART, :] = send[:, :, owner, hand].unsqueeze(2)
        size = 2 * num_parts * num_patches
        return bias.view(batch_size, self.heads, size, size)

    def _patchify(self, traj):
        """[B,T,2,29,3] -> 每个部位 [B,2,T/patch,patch*n*3]。"""
        batch_size, frames = int(traj.shape[0]), int(traj.shape[1])
        out = []
        for part in STG_PARTS:
            value = traj[:, :, :, list(part)]  # [B,T,2,n,3]
            value = value.view(batch_size, frames // self.patch_len, self.patch_len, 2, len(part), XYZ_COORD_DIM)
            out.append(value.permute(0, 3, 1, 2, 4, 5).reshape(batch_size, 2, frames // self.patch_len, -1))
        return out

    def forward(self, prep, z_root, z_local, cond, sc=None, graph=None, scales=(1.0, 1.0)):
        root_scale, local_scale = scales
        obs_state = (prep["obs_person"] - prep["last_person"].unsqueeze(1))[:, :, :, list(STATE_JOINTS)]
        noisy = self.codec.decode(z_root * root_scale, z_local * local_scale)
        obs_parts = self._patchify(obs_state)
        noisy_parts = self._patchify(noisy)
        if self.self_cond:
            sc_traj = noisy.new_zeros(noisy.shape) if sc is None else self.codec.decode(sc[0] * root_scale, sc[1] * local_scale)
            sc_parts = self._patchify(sc_traj)
        tokens = []
        for index, embed in enumerate(self.embed):
            obs_in, fut_in = obs_parts[index], noisy_parts[index]
            if self.self_cond:
                obs_in = torch.cat((obs_in, obs_in), dim=-1)
                fut_in = torch.cat((fut_in, sc_parts[index]), dim=-1)
            value = embed(torch.cat((obs_in, fut_in), dim=2))  # [B,2,P,w]
            tokens.append(value + self.part_pos[index])
        tokens = torch.stack(tokens, dim=2)  # [B,2,parts,P,w]
        flags = torch.cat((self.obs_flag[:1].expand(self.num_obs_patches, -1), self.obs_flag[1:].expand(self.num_patches - self.num_obs_patches, -1)))
        tokens = tokens + self.patch_pos + flags + self.role_pos.view(1, 2, 1, 1, -1)
        last = prep["last_person"]
        pose = (last - last[:, :, PELVIS : PELVIS + 1]).reshape(last.shape[0], 2, -1)
        tokens = tokens + self.static_proj(torch.cat((prep["geometry"], pose), dim=-1)).view(last.shape[0], 2, 1, 1, -1)
        batch_size = int(tokens.shape[0])
        tokens = tokens.reshape(batch_size, 2, -1, self.width)
        bias = self._static_bias()
        if self.contact_graph and graph is not None:
            bias = bias + self._graph_bias(graph)
        for block in self.blocks:
            tokens = block(tokens, cond, bias)
        shift, scale = self.final_ada(cond).unsqueeze(2).chunk(2, dim=-1)
        normed = self.final_norm(tokens) * (1.0 + scale) + shift
        ctx = normed.mean(dim=2)
        normed = normed.view(batch_size, 2, len(STG_PARTS), self.num_patches, self.width)[:, :, :, self.num_obs_patches :]
        traj = noisy.new_zeros(batch_size, self.pred_len, 2, NUM_STATE, XYZ_COORD_DIM)
        num_future = self.num_patches - self.num_obs_patches
        for index, (part, layer) in enumerate(zip(STG_PARTS, self.out)):
            value = layer(normed[:, :, index]).view(batch_size, 2, num_future, self.patch_len, len(part), XYZ_COORD_DIM)
            value = value.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, self.pred_len, 2, len(part), XYZ_COORD_DIM)
            traj[:, :, :, list(part)] = value
        root, local = self.codec.encode(traj)
        return root / root_scale, local / local_scale, ctx


def _repeat(value, times):
    if torch.is_tensor(value):
        return value.repeat_interleave(int(times), dim=0)
    if isinstance(value, dict):
        return OrderedDict((key, _repeat(item, times)) for key, item in value.items())
    return value


class NTU2PJiTFlow(nn.Module):
    """JiT 流匹配包装：表示、条件、去噪、腿流、采样与读出；forward 走统一 v2 接口（返回相机系 pred）。"""

    def __init__(
        self,
        denoiser="mixer",
        self_cond=False,
        contact_graph=False,
        causal_cross=False,
        obs_len=10,
        pred_len=50,
        num_actions=NUM_ACTIONS,
        root_k=8,
        local_k=20,
        root_scale=1.0,
        local_scale=1.0,
        hidden=256,
        num_blocks=6,
        mlp_ratio=2,
        cond_dim=64,
        input_dct_k=30,
        dropout=0.1,
        stg_width=192,
        stg_heads=4,
        stg_blocks=6,
        patch_len=5,
        ramp_mode="saturate",
        ramp_saturate_frames=5,
        canonical_ab_fallback="camera_x",
    ):
        super(NTU2PJiTFlow, self).__init__()
        self.model_type = MODEL_TYPE
        if denoiser not in DENOISERS:
            raise ValueError("denoiser 必须是 {}".format(DENOISERS))
        if ramp_mode not in RAMP_MODES:
            raise ValueError("ramp_mode 必须是 {}".format(RAMP_MODES))
        if causal_cross and denoiser != "stg":
            raise ValueError("causal_cross 只支持 stg 去噪器")
        if contact_graph and not self_cond:
            # 接触图读的是模型自己上一轮的估计；没有自条件就只能读 copy-last，图没有信息。
            raise ValueError("contact_graph 需要 self_cond")
        self.denoiser_type = str(denoiser)
        self.self_cond, self.contact_graph, self.causal_cross = bool(self_cond), bool(contact_graph), bool(causal_cross)
        self.obs_len, self.pred_len, self.num_actions = int(obs_len), int(pred_len), int(num_actions)
        self.root_k, self.local_k = int(root_k), int(local_k)
        self.root_scale, self.local_scale = float(root_scale), float(local_scale)
        self.hidden, self.num_blocks, self.mlp_ratio = int(hidden), int(num_blocks), int(mlp_ratio)
        self.cond_dim, self.input_dct_k, self.dropout = int(cond_dim), int(input_dct_k), float(dropout)
        self.stg_width, self.stg_heads, self.stg_blocks, self.patch_len = int(stg_width), int(stg_heads), int(stg_blocks), int(patch_len)
        self.ramp_mode, self.ramp_saturate_frames = str(ramp_mode), int(ramp_saturate_frames)
        self.canonical_ab_fallback = str(canonical_ab_fallback)
        self.codec = RampDCTCodec(self.obs_len, self.pred_len, self.root_k, self.local_k, self.ramp_mode, self.ramp_saturate_frames)
        self.register_buffer("person_signs", torch.tensor(_TOWARD_OTHER_SIGNS).view(2, 1, XYZ_COORD_DIM), persistent=False)
        self.register_buffer("source_rows", torch.tensor(SOURCE_ROWS, dtype=torch.long), persistent=False)
        self.register_buffer("leg_rows", torch.tensor(LEG_ROWS, dtype=torch.long), persistent=False)
        self.action_embed = nn.Embedding(self.num_actions, self.cond_dim)
        self.role_embed = nn.Embedding(2, self.cond_dim)
        self.time_mlp = nn.Sequential(nn.Linear(self.cond_dim, self.cond_dim), nn.SiLU(), nn.Linear(self.cond_dim, self.cond_dim))
        self.cond_act = nn.SiLU()
        if self.denoiser_type == "mixer":
            self.net = MixerDenoiser(self.obs_len, self.pred_len, self.root_k, self.local_k, self.input_dct_k, self.hidden,
                                     self.num_blocks, self.mlp_ratio, self.cond_dim, self.dropout, self.self_cond, self.contact_graph)
        else:
            self.net = STGDenoiser(self.obs_len, self.pred_len, self.patch_len, self.stg_width, self.stg_heads, self.stg_blocks,
                                   self.cond_dim, self.dropout, self.self_cond, self.contact_graph, self.causal_cross, self.codec)
        self.leg_net = FlowLegStream(self.obs_len, self.root_k, self.local_k, self.net.ctx_dim, self.cond_dim, self.self_cond)
        weights = torch.ones(NTU_SMPLX_BODY_JOINTS, dtype=torch.float64)
        weights[25:] = 0.0
        weights[list(FINGER_ROOTS)] = 1.0
        # 手部 Procrustes 只看两个指根：其余手指只是随手腕平移的模板，不含旋转信息。
        self.skeleton_projector = SkeletonProjector(child_weights=weights)
        self.readout = "endpoint"
        self.num_samples = 16
        self.nfe_steps = 20
        self.solver = "heun"
        self.reset_sampler(0)

    # ------------------------------------------------------------------ 配置与条件

    def config(self):
        config = OrderedDict([("model_type", self.model_type)])
        for key in CONFIG_KEYS:
            config[key] = getattr(self, "denoiser_type" if key == "denoiser" else key)
        config["architecture"] = "jit_x_prediction_flow_on_ramp_dct_residual_coefficients"
        return config

    def reset_sampler(self, seed):
        """采样噪声来自独立的 CPU generator：不消耗全局 RNG，固定 seed 下样本与 batch 划分以外的因素无关。"""
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.__dict__["_generator"] = generator

    def __getstate__(self):
        # torch.Generator 不能 deepcopy/pickle（EMA 用 deepcopy 复制模型）；副本按 seed 0 重建。
        state = self.__dict__.copy()
        state.pop("_generator", None)
        return state

    def __setstate__(self, state):
        super(NTU2PJiTFlow, self).__setstate__(state)
        self.reset_sampler(0)

    def _noise(self, shape, reference):
        return torch.randn(shape, generator=self.__dict__["_generator"]).to(device=reference.device, dtype=reference.dtype)

    def _to_person(self, value):
        return value * self.person_signs.to(dtype=value.dtype)

    def prepare(self, obs_xyz, action, context=None):
        check_ntu_xyz("obs_xyz", obs_xyz, seq_len=self.obs_len, num_persons=2)
        batch_size = int(obs_xyz.shape[0])
        action = _normalize_action(action, batch_size, self.num_actions, obs_xyz.device)
        frame = canonical_frame(obs_xyz, ab_fallback=self.canonical_ab_fallback)
        obs_canon = to_canonical(obs_xyz, frame)
        obs_person = self._to_person(obs_canon)
        cond = self.action_embed(action).unsqueeze(1) + self.role_embed.weight.unsqueeze(0)
        rotation = individual_frames(obs_canon)
        legs = obs_canon[:, :, :, list(LEG_JOINTS)] - obs_canon[:, :, :, PELVIS : PELVIS + 1]
        legs = rotate_rows(legs, rotation.unsqueeze(1).unsqueeze(3))
        velocity = (legs[:, 1:] - legs[:, :-1]) * NTU2P_FPS
        pelvis = rotate_rows(obs_canon[:, :, :, PELVIS] - obs_canon[:, -1:, :, PELVIS], rotation.unsqueeze(1))

        def per_person(value):
            return value.transpose(1, 2).reshape(batch_size, 2, -1)

        return OrderedDict(
            [
                ("obs_xyz", obs_xyz),
                ("frame", frame),
                ("obs_canon", obs_canon),
                ("obs_person", obs_person),
                ("last_person", obs_person[:, -1]),
                ("geometry", relative_geometry(obs_canon[:, -1])),
                ("cond", cond),
                ("rotation", rotation),
                ("leg_features", torch.cat((per_person(legs), per_person(velocity)), dim=-1)),
                ("pelvis_features", per_person(pelvis)),
            ]
        )

    # ------------------------------------------------------------------ 状态 <-> xyz

    def encode_target(self, prep, target_xyz):
        """GT 未来 -> 归一化系数（最小二乘拟合，训练目标）。"""
        residual = self._to_person(to_canonical(target_xyz, prep["frame"]) - prep["obs_canon"][:, -1:])
        root, local = self.codec.encode(residual[:, :, :, list(STATE_JOINTS)])
        return root / self.root_scale, local / self.local_scale

    def residual55(self, root_n, local_n):
        """归一化系数 -> 人物系 55 关节残差 [B,T,2,55,3]（其余手指随手腕）。"""
        state = self.codec.decode(root_n * self.root_scale, local_n * self.local_scale)
        return state.index_select(3, self.source_rows)

    def decode_xyz(self, prep, root_n, local_n, project=True):
        """返回 (pred, pred_free, delta)：相机系预测（骨架投影后）、投影前、人物系残差。"""
        delta = self.residual55(root_n, local_n)
        residual = apply_linear(self._to_person(delta), prep["frame"]["rotation"].transpose(-1, -2))
        last = prep["obs_xyz"][:, -1]
        free = last.unsqueeze(1) + residual
        pred = self.skeleton_projector(free, last) if project else free
        return pred, free, delta

    def _relations(self, prep, sc):
        """自条件估计 -> 场景规范系完整 60 帧 -> 关系曲线 [B,60,2,12] 与逐块图特征 [B,2,2,P,3]。"""
        batch_size = int(prep["obs_canon"].shape[0])
        if sc is None:
            delta = prep["obs_canon"].new_zeros(batch_size, self.pred_len, 2, NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM)
        else:
            delta = self.residual55(sc[0], sc[1])
        future = prep["obs_canon"][:, -1:] + self._to_person(delta)
        full = torch.cat((prep["obs_canon"], future), dim=1)
        distance = hand_body_distances(full).amin(dim=-1)  # [B,N,2(手的主人),2(手)]
        approach = torch.cat((torch.zeros_like(distance[:, :1]), (distance[:, :-1] - distance[:, 1:]) * NTU2P_FPS), dim=1)
        approach = approach.clamp(-2.0, 2.0) / 2.0
        contact = torch.sigmoid((CONTACT_DISTANCE - distance) / CONTACT_TEMPERATURE)
        memory = causal_contact_memory(contact)
        received = lambda value: value.flip(2)  # noqa: E731 —— 对方双手到本人躯干
        rel = torch.cat((distance, received(distance), approach, received(approach), memory, received(memory)), dim=-1)
        if self.denoiser_type == "mixer":
            return rel, None
        patches = (self.obs_len + self.pred_len) // self.patch_len
        graph = torch.stack((contact, approach, memory), dim=-1)  # [B,N,2,2,3]
        graph = graph.view(batch_size, patches, self.patch_len, 2, 2, GRAPH_FEATURES).mean(dim=2)
        return None, graph.permute(0, 2, 3, 1, 4)  # [B,2,2,P,3]

    # ------------------------------------------------------------------ 去噪

    def denoise(self, prep, z_root, z_local, t, sc=None):
        """x 预测：返回归一化系数 (root, local)；t 为 [B]（0 为纯噪声，1 为干净）。"""
        batch_size = int(z_root.shape[0])
        time = self.time_mlp(timestep_embedding(t.view(batch_size).to(z_root.dtype), self.cond_dim))
        cond = self.cond_act(prep["cond"] + time.unsqueeze(1))
        rel = graph = None
        if self.contact_graph:
            rel, graph = self._relations(prep, sc)
        if self.denoiser_type == "mixer":
            root, local, ctx = self.net(prep, z_root, z_local, cond, sc=sc, rel=rel)
        else:
            root, local, ctx = self.net(prep, z_root, z_local, cond, sc=sc, graph=graph, scales=(self.root_scale, self.local_scale))
        return self._legs(prep, root, local, ctx, cond, z_local, sc)

    def _legs(self, prep, root, local, ctx, cond, z_local, sc):
        """条件化腿流在个人朝向系里给出腿 x̂ 系数，替换主干的腿行；系数与向量同样按旋转变换。"""
        rotation = prep["rotation"]
        per_coeff = rotation.unsqueeze(2)  # [B,2,1,3,3]
        per_joint = rotation.unsqueeze(2).unsqueeze(3)  # [B,2,1,1,3,3]
        batch_size = int(root.shape[0])
        root_individual = rotate_rows(self._to_person_coeff(root.detach() * self.root_scale), per_coeff)

        def to_individual(value):
            legs = value.index_select(3, self.leg_rows)
            return rotate_rows(self._to_person_coeff(legs), per_joint).reshape(batch_size, 2, -1)

        root_features = torch.cat((prep["pelvis_features"], root_individual.reshape(batch_size, 2, -1)), dim=-1)
        leg_sc = None if sc is None else to_individual(sc[1])
        legs = self.leg_net(prep["leg_features"], root_features, ctx.detach(), cond.detach(), to_individual(z_local), leg_sc)
        legs = self._to_person_coeff(rotate_rows_transposed(legs, per_joint))
        return root, local.index_copy(3, self.leg_rows, legs)

    def _to_person_coeff(self, value):
        """人物系 <-> 场景系（对合）：系数张量的人物维在 dim=1。"""
        signs = self.person_signs.to(dtype=value.dtype).view((1, 2) + (1,) * (value.dim() - 3) + (XYZ_COORD_DIM,))
        return value * signs

    def endpoint(self, prep):
        """确定性端点：t=0、z=0；自条件模型再以第一遍估计为条件精炼一次。"""
        batch_size = int(prep["obs_xyz"].shape[0])
        dtype, device = prep["obs_xyz"].dtype, prep["obs_xyz"].device
        z_root = torch.zeros(batch_size, 2, self.root_k, XYZ_COORD_DIM, dtype=dtype, device=device)
        z_local = torch.zeros(batch_size, 2, self.local_k, NUM_LOCAL, XYZ_COORD_DIM, dtype=dtype, device=device)
        t = torch.zeros(batch_size, dtype=dtype, device=device)
        estimate = self.denoise(prep, z_root, z_local, t)
        if self.self_cond:
            estimate = self.denoise(prep, z_root, z_local, t, sc=estimate)
        return estimate

    def sample_ode(self, prep, steps=None, solver=None):
        """JiT 采样：线性 t 网格，Heun（末步 Euler 直接落到 x̂）；自条件取上一次 x̂。"""
        steps = int(steps or self.nfe_steps)
        solver = solver or self.solver
        batch_size = int(prep["obs_xyz"].shape[0])
        reference = prep["obs_xyz"]
        z_root = self._noise((batch_size, 2, self.root_k, XYZ_COORD_DIM), reference)
        z_local = self._noise((batch_size, 2, self.local_k, NUM_LOCAL, XYZ_COORD_DIM), reference)
        grid = torch.linspace(0.0, 1.0, steps + 1, dtype=reference.dtype, device=reference.device)
        sc = None
        for index in range(steps):
            t, t_next = grid[index], grid[index + 1]
            estimate = self.denoise(prep, z_root, z_local, t.expand(batch_size), sc=sc if self.self_cond else None)
            sc = estimate
            if index == steps - 1:
                z_root, z_local = estimate
                break
            denom = (1.0 - t).clamp_min(V_DENOM_MIN)
            v_root, v_local = (estimate[0] - z_root) / denom, (estimate[1] - z_local) / denom
            dt = t_next - t
            euler_root, euler_local = z_root + dt * v_root, z_local + dt * v_local
            if solver == "euler":
                z_root, z_local = euler_root, euler_local
                continue
            second = self.denoise(prep, euler_root, euler_local, t_next.expand(batch_size), sc=sc if self.self_cond else None)
            sc = second
            denom_next = (1.0 - t_next).clamp_min(V_DENOM_MIN)
            v2_root, v2_local = (second[0] - euler_root) / denom_next, (second[1] - euler_local) / denom_next
            z_root = z_root + dt * 0.5 * (v_root + v2_root)
            z_local = z_local + dt * 0.5 * (v_local + v2_local)
        return z_root, z_local

    # ------------------------------------------------------------------ 读出

    def samples_xyz(self, prep, num_samples):
        """K 个 ODE 样本 -> [K,B,T,2,55,3]（骨架投影后）。"""
        batch_size = int(prep["obs_xyz"].shape[0])
        repeated = _repeat(prep, num_samples)
        root, local = self.sample_ode(repeated)
        pred, _, _ = self.decode_xyz(repeated, root, local)
        return pred.view((batch_size, int(num_samples)) + tuple(pred.shape[1:])).transpose(0, 1)

    def aggregate(self, samples, last, mode):
        if mode == "mean":
            center = samples.mean(dim=0)
        elif mode == "median":
            center = samples.mean(dim=0)
            for _ in range(WEISZFELD_ITERATIONS):
                weight = 1.0 / (samples - center.unsqueeze(0)).norm(dim=-1, keepdim=True).clamp_min(1e-6)
                center = (weight * samples).sum(dim=0) / weight.sum(dim=0)
        elif mode == "medoid":
            flat = samples.transpose(0, 1)  # [B,K,T,2,55,3]
            # 逐个样本算到其余样本的平均关节距离，避免一次性展开 K×K 的全部差值。
            total = torch.stack(
                [(flat - flat[:, k : k + 1]).norm(dim=-1).mean(dim=(2, 3, 4)).sum(dim=1) for k in range(int(flat.shape[1]))], dim=1
            )
            choice = total.argmin(dim=1)  # [B]
            index = choice.view(-1, 1, *([1] * (flat.dim() - 2))).expand((flat.shape[0], 1) + tuple(flat.shape[2:]))
            return flat.gather(1, index).squeeze(1)
        else:
            raise ValueError("未知聚合方式 {}".format(mode))
        return self.skeleton_projector(center, last)

    def forward(self, obs_xyz, action, context=None, return_details=False):
        if self.readout not in READOUTS:
            raise ValueError("readout 必须是 {}".format(READOUTS))
        prep = self.prepare(obs_xyz, action, context)
        endpoint_root, endpoint_local = self.endpoint(prep)
        endpoint_pred, endpoint_free, delta = self.decode_xyz(prep, endpoint_root, endpoint_local)
        pred = endpoint_pred
        if self.readout == "sample":
            pred = self.samples_xyz(prep, 1)[0]
        elif self.readout != "endpoint":
            pred = self.aggregate(self.samples_xyz(prep, self.num_samples), obs_xyz[:, -1], self.readout)
        check_ntu_xyz("pred_xyz", pred, seq_len=self.pred_len, num_persons=2)
        if not return_details:
            return pred
        details = OrderedDict([("pred", pred), ("delta", delta), ("frame", prep["frame"]), ("pred_free", endpoint_free)])
        if self.readout != "endpoint":
            details["endpoint"] = endpoint_pred
        return details


def create_ntu2p_jitflow_from_config(config):
    config = dict(config or {})
    return NTU2PJiTFlow(**{key: config[key] for key in CONFIG_KEYS if key in config})


def load_ntu2p_jitflow_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError("checkpoint model_type 必须是 {}，当前为 {}".format(MODEL_TYPE, state.get("model_type")))
    model = create_ntu2p_jitflow_from_config(state.get("model_config", {}))
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


__all__ = [
    "CONFIG_KEYS",
    "LOCAL_JOINTS",
    "MODEL_TYPE",
    "NTU2PJiTFlow",
    "READOUTS",
    "RampDCTCodec",
    "STATE_JOINTS",
    "V_DENOM_MIN",
    "count_parameters",
    "create_ntu2p_jitflow_from_config",
    "load_ntu2p_jitflow_checkpoint",
]
