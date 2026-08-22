from copy import deepcopy

import torch
import torch.nn as nn

from model.cmdm import (
    EmbedAction,
    InputProcess,
    OutputProcess,
    PositionalEncoding,
    TimestepEmbedder,
)
from utils.ntu_2p_rot6d import (
    NTU_2P_NUM_JOINTS_WITH_TRANS,
    NTU_2P_PERSON_ORDER,
    NTU_2P_REPRESENTATION,
    NTU_2P_ROT6D_FEATS,
)


MODEL_TYPE = "ntu2p_forecasting_diffusion_decoder"
NUM_ACTIONS = 26
DIFFUSION_STEPS = 1000


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _ensure_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError("{} 存在 NaN 或 Inf".format(name))


def _shape_text(tensor):
    return tuple(tensor.shape)


def _normalize_action(action, batch_size, num_actions, device):
    if action is None:
        raise ValueError("y['action'] 不能为空")
    if not torch.is_tensor(action):
        action = torch.as_tensor(action)
    if action.dim() == 1:
        action = action.view(-1, 1)
    elif action.dim() != 2 or action.shape[1] != 1:
        raise ValueError("action 必须是 [B] 或 [B,1]，当前为 {}".format(_shape_text(action)))
    if action.shape[0] != batch_size:
        raise ValueError("action batch 必须是 {}，当前为 {}".format(batch_size, action.shape[0]))

    action = action.to(device=device, dtype=torch.long)
    if int(action.min().item()) < 0 or int(action.max().item()) >= int(num_actions):
        raise ValueError("action 必须在 [0,{}] 内".format(int(num_actions) - 1))
    return action


def _normalize_timesteps(timesteps, batch_size, device, max_timesteps):
    if timesteps is None:
        raise ValueError("timesteps 不能为空")
    if not torch.is_tensor(timesteps):
        timesteps = torch.as_tensor(timesteps)
    if timesteps.dim() != 1:
        raise ValueError("timesteps 必须是 [B]，当前为 {}".format(_shape_text(timesteps)))
    if timesteps.shape[0] != batch_size:
        raise ValueError("timesteps batch 必须是 {}，当前为 {}".format(batch_size, timesteps.shape[0]))
    if timesteps.dtype.is_floating_point:
        raise ValueError("timesteps 必须是整数 index；当前模型要求 diffusion rescale_timesteps=False")

    timesteps = timesteps.to(device=device, dtype=torch.long)
    if int(timesteps.min().item()) < 0 or int(timesteps.max().item()) >= int(max_timesteps):
        raise ValueError("timesteps 超出位置编码范围 [0,{})".format(int(max_timesteps)))
    return timesteps


class NTU2PForecastingDiffusionDecoder(nn.Module):
    def __init__(
        self,
        model_type=MODEL_TYPE,
        njoints=NTU_2P_NUM_JOINTS_WITH_TRANS,
        nfeats=NTU_2P_ROT6D_FEATS,
        num_actions=NUM_ACTIONS,
        obs_len=10,
        pred_len=50,
        window_len=None,
        latent_dim=256,
        obs_encoder_layers=2,
        decoder_layers=4,
        num_heads=4,
        ff_size=1024,
        dropout=0.1,
        activation="gelu",
        cond_mask_prob=0.1,
        causal_future_mask=False,
        data_rep="rot6d",
        body_model="smplx",
        dataset="ntu120_2p",
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=None,
        representation=NTU_2P_REPRESENTATION,
        person_order=NTU_2P_PERSON_ORDER,
        init_rot2xyz=False,
        **kwargs
    ):
        super(NTU2PForecastingDiffusionDecoder, self).__init__()

        if model_type != MODEL_TYPE:
            raise ValueError("model_type 必须是 {}".format(MODEL_TYPE))
        if int(njoints) != NTU_2P_NUM_JOINTS_WITH_TRANS:
            raise ValueError("njoints 必须是 {}".format(NTU_2P_NUM_JOINTS_WITH_TRANS))
        if int(nfeats) != NTU_2P_ROT6D_FEATS:
            raise ValueError("nfeats 必须是 {}".format(NTU_2P_ROT6D_FEATS))

        self.model_type = model_type
        self.njoints = int(njoints)
        self.nfeats = int(nfeats)
        self.input_feats = self.njoints * self.nfeats
        self.num_actions = int(num_actions)
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.window_len = int(window_len) if window_len is not None else self.obs_len + self.pred_len
        if self.obs_len + self.pred_len != self.window_len:
            raise ValueError("obs_len + pred_len 必须等于 window_len")

        self.latent_dim = int(latent_dim)
        self.obs_encoder_layers = int(obs_encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.num_heads = int(num_heads)
        self.ff_size = int(ff_size)
        self.dropout = float(dropout)
        self.activation = activation
        self.cond_mask_prob = float(cond_mask_prob)
        self.causal_future_mask = bool(causal_future_mask)
        self.data_rep = data_rep
        self.cond_mode = "action"
        self.body_model = body_model
        self.dataset = dataset
        self.translation = bool(translation)
        self.pose_rep = pose_rep
        self.glob = bool(glob)
        self.glob_rot = glob_rot
        self.representation = representation
        self.person_order = person_order
        self.init_rot2xyz = bool(init_rot2xyz)

        self.obs_input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        self.future_input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)
        self.output_process = OutputProcess(
            self.data_rep,
            self.input_feats,
            self.latent_dim,
            self.njoints,
            self.nfeats,
        )
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_action = EmbedAction(self.num_actions, self.latent_dim)

        obs_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.obs_encoder = nn.TransformerEncoder(
            obs_encoder_layer,
            num_layers=self.obs_encoder_layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransDecoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.decoder_layers,
        )

        self.time_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.action_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.obs_summary_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.obs_frame_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.future_type = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

        self.memory_norm = nn.LayerNorm(self.latent_dim)
        self.memory_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.future_norm = nn.LayerNorm(self.latent_dim)

        self.rot2xyz = None
        if self.init_rot2xyz:
            from model.rotation2xyz import Rotation2xyz_x

            self.rot2xyz = Rotation2xyz_x(device="cpu", dataset=self.dataset)

    def parameters_wo_clip(self):
        return list(self.parameters())

    def config(self):
        return {
            "model_type": self.model_type,
            "njoints": self.njoints,
            "nfeats": self.nfeats,
            "num_actions": self.num_actions,
            "obs_len": self.obs_len,
            "pred_len": self.pred_len,
            "window_len": self.window_len,
            "latent_dim": self.latent_dim,
            "obs_encoder_layers": self.obs_encoder_layers,
            "decoder_layers": self.decoder_layers,
            "num_heads": self.num_heads,
            "ff_size": self.ff_size,
            "dropout": self.dropout,
            "activation": self.activation,
            "cond_mask_prob": self.cond_mask_prob,
            "causal_future_mask": self.causal_future_mask,
            "data_rep": self.data_rep,
            "body_model": self.body_model,
            "dataset": self.dataset,
            "translation": self.translation,
            "pose_rep": self.pose_rep,
            "glob": self.glob,
            "glob_rot": self.glob_rot,
            "representation": self.representation,
            "person_order": self.person_order,
            "num_person": 2,
            "init_rot2xyz": self.init_rot2xyz,
        }

    def add_window_pos(self, tokens, start):
        end = int(start) + int(tokens.shape[0])
        if end > self.sequence_pos_encoder.pe.shape[0]:
            raise ValueError("位置编码长度不足，end={} max={}".format(end, self.sequence_pos_encoder.pe.shape[0]))
        pe = self.sequence_pos_encoder.pe[int(start):end]
        return self.sequence_pos_encoder.dropout(tokens + pe)

    def mask_action(self, action_emb, force_mask=False):
        if force_mask:
            return torch.zeros_like(action_emb)
        if self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(action_emb.shape[0], device=action_emb.device) * self.cond_mask_prob
            ).view(-1, 1)
            return action_emb * (1.0 - mask)
        return action_emb

    def _force_uncond(self, y):
        force_uncond = y.get("uncond", False)
        if torch.is_tensor(force_uncond):
            if force_uncond.numel() != 1:
                raise ValueError("y['uncond'] 必须是 bool 或单元素 tensor")
            return bool(force_uncond.item())
        return bool(force_uncond)

    def _check_future(self, x_t):
        if not torch.is_tensor(x_t):
            raise ValueError("x_t 必须是 tensor")
        if x_t.dim() != 4:
            raise ValueError(
                "x_t 必须是 [B,{},{},{}]，当前维度数为 {}".format(
                    self.njoints,
                    self.nfeats,
                    self.pred_len,
                    x_t.dim(),
                )
            )
        expected_tail = (self.njoints, self.nfeats, self.pred_len)
        if tuple(x_t.shape[1:]) != expected_tail:
            raise ValueError("x_t 后三维必须是 {}，当前为 {}".format(expected_tail, _shape_text(x_t)[1:]))
        _ensure_finite("x_t", x_t)

    def _check_obs(self, obs_motion, batch_size, device):
        if obs_motion is None:
            raise ValueError("y['obs_motion'] 不能为空")
        if not torch.is_tensor(obs_motion):
            raise ValueError("y['obs_motion'] 必须是 tensor")
        if obs_motion.dim() != 4:
            raise ValueError(
                "obs_motion 必须是 [B,{},{},{}]，当前维度数为 {}".format(
                    self.njoints,
                    self.nfeats,
                    self.obs_len,
                    obs_motion.dim(),
                )
            )
        expected = (batch_size, self.njoints, self.nfeats, self.obs_len)
        if tuple(obs_motion.shape) != expected:
            raise ValueError("obs_motion 必须是 {}，当前为 {}".format(expected, _shape_text(obs_motion)))
        if obs_motion.device != device:
            raise ValueError("obs_motion device 必须与 x_t 一致")
        _ensure_finite("obs_motion", obs_motion)
        return obs_motion

    def _future_mask(self, device):
        if not self.causal_future_mask:
            return None
        mask = torch.full((self.pred_len, self.pred_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, x_t, timesteps, y=None):
        if y is None or not isinstance(y, dict):
            raise ValueError("NTU2PForecastingDiffusionDecoder.forward 需要 dict 类型的 y")
        self._check_future(x_t)

        batch_size = x_t.shape[0]
        device = x_t.device
        obs_motion = self._check_obs(y.get("obs_motion"), batch_size, device)
        action = _normalize_action(y.get("action"), batch_size, self.num_actions, device)
        timesteps = _normalize_timesteps(
            timesteps,
            batch_size,
            device,
            self.sequence_pos_encoder.pe.shape[0],
        )
        force_uncond = self._force_uncond(y)

        obs_tokens = self.obs_input_process(obs_motion)
        obs_tokens = self.add_window_pos(obs_tokens, 0)
        obs_tokens = obs_tokens + self.obs_frame_type
        obs_tokens = self.obs_encoder(obs_tokens)

        obs_summary = obs_tokens.mean(dim=0, keepdim=True)
        obs_summary = obs_summary + self.obs_summary_type

        time_token = self.embed_timestep(timesteps) + self.time_type
        action_emb = self.embed_action(action)
        action_emb = self.mask_action(action_emb, force_mask=force_uncond)
        action_token = action_emb.unsqueeze(0) + self.action_type

        memory = torch.cat([time_token, action_token, obs_summary, obs_tokens], dim=0)
        memory = self.memory_proj(self.memory_norm(memory))

        future_tokens = self.future_input_process(x_t)
        future_tokens = self.add_window_pos(future_tokens, self.obs_len)
        future_tokens = self.future_norm(future_tokens + self.future_type)

        decoded = self.seqTransDecoder(
            tgt=future_tokens,
            memory=memory,
            tgt_mask=self._future_mask(device),
            memory_mask=None,
        )
        output = self.output_process(decoded)
        if tuple(output.shape) != tuple(x_t.shape):
            raise ValueError("输出 shape 必须等于 x_t，当前 {} vs {}".format(_shape_text(output), _shape_text(x_t)))
        _ensure_finite("output", output)
        return output

    def _apply(self, fn):
        super(NTU2PForecastingDiffusionDecoder, self)._apply(fn)
        if self.rot2xyz is not None:
            self.rot2xyz.smpl_model._apply(fn)
        return self

    def train(self, *args, **kwargs):
        super(NTU2PForecastingDiffusionDecoder, self).train(*args, **kwargs)
        if self.rot2xyz is not None:
            self.rot2xyz.smpl_model.train(*args, **kwargs)
        return self


class NTU2PClassifierFreeSampleModel(nn.Module):
    def __init__(self, model, guidance_scale=1.0):
        super(NTU2PClassifierFreeSampleModel, self).__init__()
        if getattr(model, "cond_mask_prob", 0.0) <= 0.0:
            raise ValueError("cond_mask_prob 必须大于 0，才能使用 classifier-free guidance")
        self.model = model
        self.guidance_scale = float(guidance_scale)
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode

    def _scale_tensor(self, scale, x):
        batch_size = x.shape[0]
        if scale is None:
            scale = self.guidance_scale
        if torch.is_tensor(scale):
            scale = scale.to(device=x.device, dtype=x.dtype)
            if scale.dim() == 0:
                scale = scale.view(1).repeat(batch_size)
            elif scale.dim() == 1 and scale.shape[0] == 1:
                scale = scale.repeat(batch_size)
            elif scale.dim() != 1 or scale.shape[0] != batch_size:
                raise ValueError("CFG scale 必须是标量或 [B]，当前为 {}".format(_shape_text(scale)))
        else:
            scale = torch.ones(batch_size, device=x.device, dtype=x.dtype) * float(scale)
        return scale.view(-1, 1, 1, 1)

    def forward(self, x, timesteps, y=None):
        if y is None or not isinstance(y, dict):
            raise ValueError("NTU2PClassifierFreeSampleModel.forward 需要 dict 类型的 y")

        y_cond = deepcopy(y)
        y_uncond = deepcopy(y)
        y_uncond["uncond"] = True

        out_cond = self.model(x, timesteps, y_cond)
        out_uncond = self.model(x, timesteps, y_uncond)
        scale = self._scale_tensor(y.get("scale", None), x)
        return out_uncond + scale * (out_cond - out_uncond)


def create_ntu_2p_diffusion_model_from_config(config):
    return NTU2PForecastingDiffusionDecoder(
        model_type=config.get("model_type", MODEL_TYPE),
        njoints=config.get("njoints", NTU_2P_NUM_JOINTS_WITH_TRANS),
        nfeats=config.get("nfeats", NTU_2P_ROT6D_FEATS),
        num_actions=config.get("num_actions", NUM_ACTIONS),
        obs_len=config.get("obs_len", 10),
        pred_len=config.get("pred_len", 50),
        window_len=config.get("window_len"),
        latent_dim=config.get("latent_dim", 256),
        obs_encoder_layers=config.get("obs_encoder_layers", 2),
        decoder_layers=config.get("decoder_layers", 4),
        num_heads=config.get("num_heads", 4),
        ff_size=config.get("ff_size", 1024),
        dropout=config.get("dropout", 0.1),
        activation=config.get("activation", "gelu"),
        cond_mask_prob=config.get("cond_mask_prob", 0.1),
        causal_future_mask=config.get("causal_future_mask", False),
        data_rep=config.get("data_rep", "rot6d"),
        body_model=config.get("body_model", "smplx"),
        dataset=config.get("dataset", "ntu120_2p"),
        translation=config.get("translation", True),
        pose_rep=config.get("pose_rep", "rot6d"),
        glob=config.get("glob", True),
        glob_rot=config.get("glob_rot"),
        representation=config.get("representation", NTU_2P_REPRESENTATION),
        person_order=config.get("person_order", NTU_2P_PERSON_ORDER),
        init_rot2xyz=config.get("init_rot2xyz", False),
    )
