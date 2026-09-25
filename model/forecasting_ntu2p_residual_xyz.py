"""继承独立单人 xyz baseline 的双人跨人残差预测器。"""

import math
from collections import OrderedDict

import torch
from torch import nn

from model.forecasting_ntu_xyz import (
    NTULabelXYZTransformer,
    create_ntu_label_xyz_model_from_config,
)
from model.two_person_transformer import TwoPersonForecastingDecoder, TwoPersonInteractionEncoder
from utils.ntu_smplx_2p_xyz import (
    NTU_SMPLX_BODY_JOINTS,
    XYZ_COORD_DIM,
    check_ntu_xyz,
    dct_matrix,
)


MODEL_TYPE = "ntu2p_residual_refiner_xyz"
NUM_ACTIONS = 26


RAMP_MODES = ("linear", "saturate")
FUTURE_POS_MODES = ("learned_zero", "sinusoidal")
ROOT_HEAD_MODES = ("none", "dct")


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_residual_ramp(pred_len, mode="linear", saturate_frames=5):
    """首帧恒为 0 保证连续；linear 平均只放行 50% 残差，saturate 在 saturate_frames 帧后全额放行。"""
    pred_len = int(pred_len)
    if mode == "linear":
        ramp = torch.linspace(0.0, 1.0, pred_len)
    elif mode == "saturate":
        ramp = (torch.arange(pred_len, dtype=torch.float32) / float(saturate_frames)).clamp(max=1.0)
    else:
        raise ValueError("ramp_mode 必须是 {}，当前为 {}".format(RAMP_MODES, mode))
    return ramp.view(1, pred_len, 1, 1, 1)


def build_sinusoidal_position(seq_len, dim):
    """固定多频正弦编码，让解码器从初始化起就能区分未来各帧、表达周期性输出。"""
    position = torch.arange(int(seq_len), dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, int(dim), 2, dtype=torch.float32) * (-math.log(10000.0) / float(dim)))
    encoding = torch.zeros(int(seq_len), int(dim))
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding.unsqueeze(1)


def build_anchored_dct_basis(pred_len, num_coeffs):
    """phi_k(t) - phi_k(0)，k=1..K：首帧恰为 0 保证与观测末帧连续，低频截断保证 root 轨迹平滑。"""
    phi = dct_matrix(int(pred_len))
    return (phi - phi[:, :1])[1 : int(num_coeffs) + 1]


def _normalize_action(action, batch_size, num_actions, device):
    if action is None:
        raise ValueError("action 不能为空")
    if not torch.is_tensor(action):
        action = torch.as_tensor(action)
    if action.dim() == 2 and action.shape[1] == 1:
        action = action[:, 0]
    if action.dim() != 1 or int(action.shape[0]) != int(batch_size):
        raise ValueError("action 必须是 [B] 或 [B,1]，且 batch={}，当前为 {}".format(batch_size, tuple(action.shape)))
    action = action.to(device=device, dtype=torch.long)
    if int(action.min().item()) < 0 or int(action.max().item()) >= int(num_actions):
        raise ValueError("action 必须在 [0,{}] 内".format(int(num_actions) - 1))
    return action


class NTU2PResidualRefinerXYZ(nn.Module):
    """以冻结单人预测为锚点，仅学习双人跨人 residual。"""

    def __init__(
        self,
        base_model,
        obs_len=10,
        pred_len=50,
        num_actions=NUM_ACTIONS,
        latent_dim=256,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        alpha=1.0,
        freeze_base=True,
        ramp_mode="linear",
        ramp_saturate_frames=5,
        future_pos_mode="learned_zero",
        root_head_mode="none",
        root_dct_k=5,
    ):
        super(NTU2PResidualRefinerXYZ, self).__init__()
        if not isinstance(base_model, NTULabelXYZTransformer):
            raise ValueError("base_model 必须是 NTULabelXYZTransformer")
        if int(base_model.num_persons) != 1:
            raise ValueError("base_model 必须是 num_persons=1")
        if int(base_model.obs_len) != int(obs_len) or int(base_model.pred_len) != int(pred_len):
            raise ValueError("base_model 的 obs/pred 长度必须与 residual refiner 一致")

        self.model_type = MODEL_TYPE
        self.base_model = base_model
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.num_actions = int(num_actions)
        self.num_joints = NTU_SMPLX_BODY_JOINTS
        self.coord_dim = XYZ_COORD_DIM
        self.person_dim = self.num_joints * self.coord_dim
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.freeze_base = bool(freeze_base)
        self.ramp_mode = str(ramp_mode)
        self.ramp_saturate_frames = int(ramp_saturate_frames)
        self.future_pos_mode = str(future_pos_mode)
        self.root_head_mode = str(root_head_mode)
        self.root_dct_k = int(root_dct_k)
        if self.root_head_mode not in ROOT_HEAD_MODES:
            raise ValueError("root_head_mode 必须是 {}，当前为 {}".format(ROOT_HEAD_MODES, self.root_head_mode))
        if self.root_head_mode == "dct" and not 1 <= self.root_dct_k < self.pred_len:
            raise ValueError("root_dct_k 必须在 [1,{}) 内".format(self.pred_len))
        if self.ramp_mode not in RAMP_MODES:
            raise ValueError("ramp_mode 必须是 {}，当前为 {}".format(RAMP_MODES, self.ramp_mode))
        if self.future_pos_mode not in FUTURE_POS_MODES:
            raise ValueError("future_pos_mode 必须是 {}，当前为 {}".format(FUTURE_POS_MODES, self.future_pos_mode))
        if self.ramp_mode == "saturate" and self.ramp_saturate_frames < 1:
            raise ValueError("ramp_saturate_frames 必须 >= 1")
        # 非持久 buffer：不进入 state_dict，旧 checkpoint 可原样加载。
        self.register_buffer(
            "ramp",
            build_residual_ramp(self.pred_len, self.ramp_mode, self.ramp_saturate_frames),
            persistent=False,
        )
        self.register_buffer(
            "future_sin_pos",
            build_sinusoidal_position(self.pred_len, self.latent_dim),
            persistent=False,
        )

        self.obs_input_proj = nn.Linear(self.person_dim, self.latent_dim)
        self.future_input_proj = nn.Linear(self.person_dim, self.latent_dim)
        self.obs_pos = nn.Parameter(torch.zeros(1, self.obs_len, self.latent_dim))
        self.future_pos = nn.Parameter(torch.zeros(1, self.pred_len, self.latent_dim))
        self.action_embed = nn.Embedding(self.num_actions, self.latent_dim)
        self.action_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.obs_summary_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.base_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

        self.obs_encoder = TwoPersonInteractionEncoder(
            num_layers=self.encoder_layers,
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
        )
        self.future_decoder = TwoPersonForecastingDecoder(
            num_layers=self.decoder_layers,
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
        )
        self.memory_norm = nn.LayerNorm(self.latent_dim)
        self.memory_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.future_norm = nn.LayerNorm(self.latent_dim)
        self.delta_norm = nn.LayerNorm(self.latent_dim)
        self.delta_proj = nn.Linear(self.latent_dim, self.person_dim)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)

        # alpha=1 保留零初始化 delta head 的梯度；alpha=0 仍用于严格等价性测试。
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

        # none 时不创建任何模块，训练与历史 run 逐位等价；dct 时在 fork 出的 CPU 随机数流里初始化，
        # 使同 seed 下数据打乱顺序与对照一致（配对比较），零初始化输出层保证初始输出也一致。
        if self.root_head_mode == "dct":
            self.register_buffer(
                "root_dct_basis",
                build_anchored_dct_basis(self.pred_len, self.root_dct_k),
                persistent=False,
            )
            with torch.random.fork_rng(devices=[]):
                # 输入：本人观测 root 相对末帧的轨迹 + 对方相对本人的观测 root 轨迹，各 obs_len×3。
                self.root_kin_proj = nn.Linear(2 * self.obs_len * self.coord_dim, self.latent_dim)
                self.root_norm = nn.LayerNorm(self.latent_dim)
                self.root_hidden = nn.Linear(self.latent_dim, self.latent_dim)
                self.root_out = nn.Linear(self.latent_dim, self.root_dct_k * self.coord_dim)
            nn.init.zeros_(self.root_out.weight)
            nn.init.zeros_(self.root_out.bias)

        if self.freeze_base:
            self.set_base_trainable(False)

    def set_base_trainable(self, trainable):
        self.freeze_base = not bool(trainable)
        for param in self.base_model.parameters():
            param.requires_grad = bool(trainable)
        if not trainable:
            self.base_model.eval()

    def train(self, mode=True):
        super(NTU2PResidualRefinerXYZ, self).train(mode)
        if self.freeze_base:
            self.base_model.eval()
        return self

    def config(self):
        return OrderedDict(
            [
                ("model_type", self.model_type),
                ("obs_len", self.obs_len),
                ("pred_len", self.pred_len),
                ("num_actions", self.num_actions),
                ("latent_dim", self.latent_dim),
                ("num_heads", self.num_heads),
                ("encoder_layers", self.encoder_layers),
                ("decoder_layers", self.decoder_layers),
                ("dim_feedforward", self.dim_feedforward),
                ("dropout", self.dropout),
                ("base_model_config", self.base_model.config()),
                ("ramp_mode", self.ramp_mode),
                ("ramp_saturate_frames", self.ramp_saturate_frames),
                ("future_pos_mode", self.future_pos_mode),
                ("root_head_mode", self.root_head_mode),
                ("root_dct_k", self.root_dct_k),
                ("architecture", "frozen_single_person_base_plus_cross_person_residual"),
                ("base_frozen_by_default", True),
            ]
        )

    def _base_forward(self, obs_xyz, action):
        batch_size = int(obs_xyz.shape[0])
        obs_single = torch.cat((obs_xyz[:, :, 0:1], obs_xyz[:, :, 1:2]), dim=0)
        action_single = torch.cat((action, action), dim=0)
        if self.freeze_base:
            with torch.no_grad():
                pred_single = self.base_model(obs_single, action_single)
        else:
            pred_single = self.base_model(obs_single, action_single)
        if tuple(pred_single.shape) != (batch_size * 2, self.pred_len, 1, self.num_joints, self.coord_dim):
            raise ValueError("base_model 输出 shape 错误: {}".format(tuple(pred_single.shape)))
        return torch.cat((pred_single[:batch_size], pred_single[batch_size:]), dim=2)

    def _person_tokens(self, value, projection, position, token_type=None):
        batch_size, seq_len, num_persons, num_joints, coord_dim = value.shape
        if int(num_persons) != 2 or int(num_joints) != self.num_joints or int(coord_dim) != self.coord_dim:
            raise ValueError("双人 xyz 输入 shape 错误: {}".format(tuple(value.shape)))
        person_a = value[:, :, 0].reshape(batch_size, seq_len, self.person_dim)
        person_b = value[:, :, 1].reshape(batch_size, seq_len, self.person_dim)
        person_a = projection(person_a).transpose(0, 1).contiguous()
        person_b = projection(person_b).transpose(0, 1).contiguous()
        position = position[:, :seq_len].transpose(0, 1)
        person_a = person_a + position
        person_b = person_b + position
        if token_type is not None:
            person_a = person_a + token_type
            person_b = person_b + token_type
        return person_a, person_b

    def _tokens_to_xyz(self, tokens, batch_size):
        seq_len = int(tokens.shape[0])
        output = self.delta_proj(self.delta_norm(tokens)).transpose(0, 1).contiguous()
        return output.reshape(batch_size, seq_len, 1, self.num_joints, self.coord_dim)

    def _root_offset(self, obs_xyz, decoded_a, decoded_b):
        """每人一组锚定 DCT 系数 -> 整体平移轨迹 [B,T,2,3]；A/B 共享参数，输入按本人视角构造。"""
        root = obs_xyz[:, :, :, 0]
        batch_size = int(obs_xyz.shape[0])
        offsets = []
        for person, other, decoded in ((0, 1, decoded_a), (1, 0, decoded_b)):
            own = root[:, :, person]
            kinematics = torch.cat((own - own[:, -1:], root[:, :, other] - own), dim=1).reshape(batch_size, -1)
            hidden = self.root_kin_proj(kinematics) + decoded.mean(dim=0)
            hidden = torch.nn.functional.gelu(self.root_hidden(self.root_norm(hidden)))
            coeffs = self.root_out(hidden).reshape(batch_size, self.root_dct_k, self.coord_dim)
            offsets.append(torch.einsum("kt,bkc->btc", self.root_dct_basis.to(dtype=coeffs.dtype), coeffs))
        return torch.stack(offsets, dim=2)

    def forward(self, obs_xyz, action, return_details=False):
        check_ntu_xyz("obs_xyz", obs_xyz, seq_len=self.obs_len, num_persons=2)
        batch_size = int(obs_xyz.shape[0])
        action = _normalize_action(action, batch_size, self.num_actions, obs_xyz.device)
        base_xyz = self._base_forward(obs_xyz, action)

        obs_a, obs_b = self._person_tokens(obs_xyz, self.obs_input_proj, self.obs_pos)
        obs_a, obs_b = self.obs_encoder(obs_a, obs_b)
        action_token = self.action_embed(action).unsqueeze(0) + self.action_type
        obs_a_summary = obs_a.mean(dim=0, keepdim=True) + self.obs_summary_type
        obs_b_summary = obs_b.mean(dim=0, keepdim=True) + self.obs_summary_type
        memory = torch.cat((action_token, obs_a_summary, obs_b_summary, obs_a, obs_b), dim=0)
        memory = self.memory_proj(self.memory_norm(memory))

        future_a, future_b = self._person_tokens(
            base_xyz,
            self.future_input_proj,
            self.future_pos,
            token_type=self.base_type,
        )
        if self.future_pos_mode == "sinusoidal":
            future_a = future_a + self.future_sin_pos
            future_b = future_b + self.future_sin_pos
        future_a = self.future_norm(future_a)
        future_b = self.future_norm(future_b)
        decoded_a, decoded_b = self.future_decoder(future_a, future_b, memory=memory)
        delta_a = self._tokens_to_xyz(decoded_a, batch_size)
        delta_b = self._tokens_to_xyz(decoded_b, batch_size)
        delta = torch.cat((delta_a, delta_b), dim=2)
        delta = delta * self.ramp.to(dtype=delta.dtype)
        pred_xyz = base_xyz + self.alpha * delta
        if self.root_head_mode == "dct":
            pred_xyz = pred_xyz + self._root_offset(obs_xyz, decoded_a, decoded_b).unsqueeze(3)
        check_ntu_xyz("pred_xyz", pred_xyz, seq_len=self.pred_len, num_persons=2)
        if return_details:
            return pred_xyz, base_xyz, delta
        return pred_xyz


def load_base_model_from_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if "model_state_dict" not in state or "model_config" not in state:
        raise ValueError("baseline checkpoint 缺少 model_state_dict/model_config")
    if state.get("representation") not in (None, "independent_single_person_xyz"):
        raise ValueError("baseline checkpoint representation 不正确")
    base_model = create_ntu_label_xyz_model_from_config(state["model_config"])
    base_model.load_state_dict(state["model_state_dict"])
    base_model.to(device)
    base_model.eval()
    return base_model, state


def create_ntu2p_residual_refiner_from_checkpoint(path, device, config=None):
    base_model, base_state = load_base_model_from_checkpoint(path, device)
    config = dict(config or {})
    model = NTU2PResidualRefinerXYZ(base_model=base_model, **config)
    model.to(device)
    return model, base_state


def load_ntu2p_residual_refiner_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError("checkpoint model_type 必须是 {}".format(MODEL_TYPE))
    model_config = state.get("model_config", {})
    base_config = model_config.get("base_model_config")
    if not base_config:
        raise ValueError("residual checkpoint 缺少 base_model_config")
    base_model = create_ntu_label_xyz_model_from_config(base_config)
    model = NTU2PResidualRefinerXYZ(
        base_model=base_model,
        obs_len=model_config.get("obs_len", 10),
        pred_len=model_config.get("pred_len", 50),
        num_actions=model_config.get("num_actions", NUM_ACTIONS),
        latent_dim=model_config.get("latent_dim", 256),
        num_heads=model_config.get("num_heads", 4),
        encoder_layers=model_config.get("encoder_layers", 2),
        decoder_layers=model_config.get("decoder_layers", 2),
        dim_feedforward=model_config.get("dim_feedforward", 1024),
        dropout=model_config.get("dropout", 0.1),
        alpha=1.0,
        freeze_base=True,
        ramp_mode=model_config.get("ramp_mode", "linear"),
        ramp_saturate_frames=model_config.get("ramp_saturate_frames", 5),
        future_pos_mode=model_config.get("future_pos_mode", "learned_zero"),
        root_head_mode=model_config.get("root_head_mode", "none"),
        root_dct_k=model_config.get("root_dct_k", 5),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state
