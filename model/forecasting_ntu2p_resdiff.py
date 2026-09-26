"""Track B 生成器：v2 底座残差系数的条件扩散去噪网络（x0 预测）、扩散构建、采样与 checkpoint 读写。

设计：docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md 第 3.6–3.8 节。
- 47 个 token（全局 1、几何 2、观测 20、系数 24），参照 CMDM trans_enc 的全连接注意，两人全部 token 互相可见，
  以联合表达 A/B 残差的耦合（终帧 root 残差 A/B 相关 0.23–0.28）；
- 输出层零初始化：初始 x̂0 ≡ 0，即输出等于 v2 底座；
- root_known：输入 x_t 与输出 x̂0 的 root 通道都替换为给定值，训练时以 p=0.25 给干净 root，模型学到 p(局部 | root)，
  mode R（root_value=0）因此是训练过的条件，而不是替换式 inpainting 的近似；
- 采样用 space_timesteps(1000,[N]) 含 t=999，从 ᾱ≈2.4e-9 的纯噪声起步；不用 "ddimN"：它的最大步是 t=900
  （ᾱ=0.024，x_900 仍含 0.154·x0），从纯噪声起步与训练分布不对齐，这正是历史 rot6d 扩散失败的原因之一。
- 主采样 NFE=50：DDIM（eta=0）的离散化误差会系统性收缩样本离散度，解析最优去噪器下条件 std=0.3 时
  NFE10 只有真 std 的 0.78、NFE50 为 0.955（std=1.0 时 0.87 / 0.975），τ=1 在 NFE10 下并不是校准采样；
  修订见 docs/ai/context/20260926-172000-ntu2p-trackb-review-fixes-and-nfe-preregistration-amendment.md。
"""

import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch import nn

import diffusion.gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.cmdm import PositionalEncoding, TimestepEmbedder


MODEL_TYPE = "ntu2p_resdiff"
DIFFUSION_STEPS = 1000
NOISE_SCHEDULE = "cosine"
MODES = ("F", "R")
CONFIG_KEYS = ("num_coeffs", "channels", "obs_len", "num_actions", "latent_dim", "num_layers", "num_heads", "ff_size", "dropout")
REL_GEOMETRY_DIM = 6
ROOT_CHANNELS = 3
NUM_PERSONS = 2
# 预登记主采样步数：离散度收缩随 NFE 单调减小，50 步时解析 oracle 的样本 std 已 ≥ 真 std 的 0.95。
DEFAULT_NFE = 50


class NTU2PResDiffDenoiser(nn.Module):
    """x_t（归一化系数）+ 条件 -> x̂0（归一化系数），[B,2,K,C]。"""

    def __init__(
        self,
        stats=None,
        num_coeffs=12,
        channels=66,
        obs_len=10,
        num_actions=26,
        latent_dim=256,
        num_layers=6,
        num_heads=4,
        ff_size=1024,
        dropout=0.1,
    ):
        super(NTU2PResDiffDenoiser, self).__init__()
        self.model_type = MODEL_TYPE
        self.num_coeffs = int(num_coeffs)
        self.channels = int(channels)
        self.obs_len = int(obs_len)
        self.num_actions = int(num_actions)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ff_size = int(ff_size)
        self.dropout = float(dropout)
        stats = stats or {}
        shapes = OrderedDict(
            [("sigma_target", (self.num_coeffs, self.channels)), ("sigma_draft", (self.num_coeffs, self.channels)), ("feat_std", (self.channels,))]
        )
        for name, shape in shapes.items():
            value = stats.get(name)
            value = torch.ones(shape) if value is None else torch.as_tensor(value, dtype=torch.float32).reshape(shape).clone()
            # 持久 buffer：统计量随 checkpoint 保存，评估时不依赖残差库文件也能反归一化。
            self.register_buffer(name, value)

        d = self.latent_dim
        self.coeff_in = nn.Linear(2 * self.channels, d)
        self.obs_in = nn.Linear(self.channels, d)
        self.geom_in = nn.Linear(REL_GEOMETRY_DIM, d)
        self.coeff_pos = nn.Parameter(torch.randn(self.num_coeffs, d) * 0.02)
        self.frame_pos = nn.Parameter(torch.randn(self.obs_len, d) * 0.02)
        self.role = nn.Embedding(NUM_PERSONS, d)
        self.root_known_embed = nn.Parameter(torch.randn(d) * 0.02)
        self.action_embed = nn.Embedding(self.num_actions, d)
        self.time_embed = TimestepEmbedder(d, PositionalEncoding(d, self.dropout))
        layer = nn.TransformerEncoderLayer(d, self.num_heads, self.ff_size, self.dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, self.num_layers)
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def config(self):
        config = OrderedDict([("model_type", self.model_type)])
        for key in CONFIG_KEYS:
            config[key] = getattr(self, key)
        config["architecture"] = "trans_enc_47_tokens_x0_pred_on_v2_residual_ramp_dct"
        return config

    def normalize_features(self, raw):
        return OrderedDict(
            [
                ("draft", raw["draft"] / self.sigma_draft),
                ("obs_feats", raw["obs_feats"] / self.feat_std),
                ("rel_geom", raw["rel_geom"]),
            ]
        )

    @staticmethod
    def _replace_root(value, root_known, root_value):
        flag = root_known.view(-1, 1, 1, 1)
        root = torch.where(flag, root_value.to(value.dtype), value[..., :ROOT_CHANNELS])
        return torch.cat((root, value[..., ROOT_CHANNELS:]), dim=-1)

    def forward(self, x, timesteps, y):
        batch_size = int(x.shape[0])
        root_known = y["root_known"].to(device=x.device, dtype=torch.bool)
        root_value = y["root_value"]
        x = self._replace_root(x, root_known, root_value)
        role = self.role.weight  # [2,d]

        global_token = self.time_embed(timesteps.long()) + self.action_embed(y["action"].long()).unsqueeze(0)  # [1,B,d]
        geom = self.geom_in(y["rel_geom"]) + role.unsqueeze(0)  # [B,2,d]
        obs = self.obs_in(y["obs_feats"]) + self.frame_pos.unsqueeze(0).unsqueeze(0) + role.view(1, NUM_PERSONS, 1, -1)  # [B,2,10,d]
        coeff = self.coeff_in(torch.cat((x, y["draft"]), dim=-1))
        coeff = coeff + self.coeff_pos.unsqueeze(0).unsqueeze(0) + role.view(1, NUM_PERSONS, 1, -1)
        coeff = coeff + root_known.to(coeff.dtype).view(-1, 1, 1, 1) * self.root_known_embed  # [B,2,K,d]

        tokens = torch.cat(
            (
                global_token,
                geom.transpose(0, 1),
                obs.reshape(batch_size, -1, self.latent_dim).transpose(0, 1),
                coeff.reshape(batch_size, -1, self.latent_dim).transpose(0, 1),
            ),
            dim=0,
        )  # [47,B,d]
        hidden = self.encoder(tokens)
        coeff_hidden = hidden[-NUM_PERSONS * self.num_coeffs :].transpose(0, 1).reshape(batch_size, NUM_PERSONS, self.num_coeffs, self.latent_dim)
        output = self.out(self.out_norm(coeff_hidden))
        return self._replace_root(output, root_known, root_value)


def make_condition(model, feats_raw, action, mode, root_value=None, root_known=None):
    """条件字典 y（归一化）。mode F：root 全部待采样；mode R：root_known=1、root_value=0，即 root 逐位等于 v2。

    训练时由调用方传入 root_known 掩码与干净 root（归一化空间）。
    """
    if mode not in MODES:
        raise ValueError("mode 必须是 {}，当前为 {}".format(MODES, mode))
    y = model.normalize_features(feats_raw)
    batch_size = int(y["draft"].shape[0])
    device = y["draft"].device
    if root_known is None:
        root_known = torch.full((batch_size,), mode == "R", dtype=torch.bool, device=device)
    if root_value is None:
        root_value = torch.zeros(batch_size, NUM_PERSONS, model.num_coeffs, ROOT_CHANNELS, dtype=y["draft"].dtype, device=device)
    y["action"] = action.to(device).long()
    y["root_known"] = root_known
    y["root_value"] = root_value
    return y


def repeat_condition(y, repeats):
    """条件的每个样本重复 repeats 次（与 [B,K'] 展平成 B×K' 的噪声对齐）。"""
    return OrderedDict((key, value.repeat_interleave(int(repeats), dim=0)) for key, value in y.items())


def _diffusion_kwargs():
    return dict(
        betas=gd.get_named_beta_schedule(NOISE_SCHEDULE, DIFFUSION_STEPS),
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )


def build_training_diffusion():
    return gd.GaussianDiffusion(**_diffusion_kwargs())


def build_sampling_diffusion(num_steps=DEFAULT_NFE):
    """含 t=0 与 t=999 的等距子序列；不用 "ddimN"（最大步 900，与训练的纯噪声端不对齐）。"""
    steps = space_timesteps(DIFFUSION_STEPS, [int(num_steps)])
    if 0 not in steps or DIFFUSION_STEPS - 1 not in steps:
        raise AssertionError("采样步必须同时包含 t=0 与 t={}".format(DIFFUSION_STEPS - 1))
    return SpacedDiffusion(use_timesteps=steps, **_diffusion_kwargs())


def sample_x0(model, sampler, y, noise, tau=1.0):
    """DDIM（eta=0）确定性 ODE；温度只缩放初始噪声。clip_denoised 必须关闭：x0 是单位方差量，截到 ±1 会压扁分布。

    步数少时离散化误差让样本离散度偏小（eta=1 更差），所以 τ=1 只在足够的 NFE 下才近似校准，见 DEFAULT_NFE。
    """
    with torch.no_grad():
        return sampler.ddim_sample_loop(
            model,
            tuple(noise.shape),
            noise=float(tau) * noise,
            clip_denoised=False,
            model_kwargs={"y": y},
            device=noise.device,
            eta=0.0,
        )


def one_step_mean(model, y, noise, tau=1.0):
    """单步条件均值：x̂0(x_T, t=999) 对 K' 个噪声取平均；noise [B,K',2,K,C]，y 为 B 个样本的条件。

    是生成器对条件均值的估计，没有 mean-of-K 的 var/K 项，作点估计判据。
    """
    batch_size, repeats = int(noise.shape[0]), int(noise.shape[1])
    flat = float(tau) * noise.reshape((batch_size * repeats,) + tuple(noise.shape[2:]))
    timesteps = torch.full((batch_size * repeats,), DIFFUSION_STEPS - 1, dtype=torch.long, device=noise.device)
    with torch.no_grad():
        x0 = model(flat, timesteps, repeat_condition(y, repeats))
    return x0.reshape(noise.shape).mean(dim=1)


def to_meters(model, x0):
    return x0 * model.sigma_target


def from_meters(model, coeffs):
    return coeffs / model.sigma_target


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def create_ntu2p_resdiff_from_config(config, stats=None):
    config = dict(config or {})
    return NTU2PResDiffDenoiser(stats=stats, **{key: config[key] for key in CONFIG_KEYS if key in config})


def save_ntu2p_resdiff_checkpoint(path, model, extra):
    """extra 至少含 codec_config、bank_path、bank_config_sha256、protocol、arm、step、seed、device。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = OrderedDict(
        [
            ("model_state_dict", model.state_dict()),
            ("model_type", MODEL_TYPE),
            ("model_config", model.config()),
            ("num_params", int(sum(param.numel() for param in model.parameters()))),
            ("created_at", _utc_now()),
        ]
    )
    for key, value in extra.items():
        state.setdefault(key, value)
    torch.save(state, path)
    return path


def load_ntu2p_resdiff_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError("checkpoint model_type 必须是 {}，当前为 {}".format(MODEL_TYPE, state.get("model_type")))
    model = create_ntu2p_resdiff_from_config(state.get("model_config", {}))
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


__all__ = [
    "DEFAULT_NFE",
    "DIFFUSION_STEPS",
    "MODEL_TYPE",
    "MODES",
    "NTU2PResDiffDenoiser",
    "build_sampling_diffusion",
    "build_training_diffusion",
    "count_parameters",
    "from_meters",
    "load_ntu2p_resdiff_checkpoint",
    "make_condition",
    "one_step_mean",
    "repeat_condition",
    "sample_x0",
    "save_ntu2p_resdiff_checkpoint",
    "to_meters",
]
