"""接触事件图与双腿相位条件化的 DCT 纠正器，支持 ReGenNet 系数扩散。

设计与公平性：docs/ai/context/20261014-112117-ntu2p-contact-graph-diffusion-design-and-plan.md
及 20261014-111722 的预登记修订。确定性端点与 DDIM 样本必须分别报告。
"""

import copy

import torch
from torch import nn

from model.cmdm import PositionalEncoding, TimestepEmbedder
from model.forecasting_ntu2p_intermixer import create_ntu2p_intermixer_from_config, rotate_rows_transposed
from model.forecasting_ntu2p_resdiff import build_training_diffusion, build_sampling_diffusion
from utils.ntu2p_canonical import to_canonical
from utils.ntu2p_gait_losses import LEG_JOINTS
from utils.ntu2p_naturalness import SMPLX_PARENTS, NTU2P_FPS
from utils.ntu2p_residual_codec import ResidualCodec, codec_frames, obs_features, to_person


MODEL_TYPE = "ntu2p_contact_graph_diffusion"
CONFIG_KEYS = ("graph", "phase", "diffusion", "hidden", "num_blocks", "num_coeffs", "smooth_frames", "dropout", "fine_tune_prior")
NUM_BODY, CHANNELS, PRED_LEN = 22, 66, 50


def _bone_graph():
    adjacency = torch.eye(NUM_BODY)
    for child in range(1, NUM_BODY):
        parent = SMPLX_PARENTS[child]
        adjacency[child, parent] = adjacency[parent, child] = 1.0
    return adjacency / adjacency.sum(-1, keepdim=True)


class _EventGraphBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout):
        super(_EventGraphBlock, self).__init__()
        self.norm = nn.LayerNorm(hidden)
        self.spatial = nn.Linear(3 * hidden, hidden)
        self.temporal = nn.Conv1d(hidden, hidden, 5, padding=2 * dilation, dilation=dilation, groups=hidden)
        self.ff = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.drop = nn.Dropout(dropout)
        self.register_buffer("bones", _bone_graph(), persistent=False)

    def forward(self, hidden, adjacency, use_graph):
        batch, time, persons, joints, width = hidden.shape
        normed = self.norm(hidden)
        if use_graph:
            intra = torch.einsum("ij,btpjh->btpih", self.bones, normed)
            contact = torch.einsum("btpij,btpjh->btpih", adjacency, normed.flip(2))
        else:
            intra = normed.mean(3, keepdim=True).expand_as(normed)
            contact = torch.zeros_like(normed)
        # 一般跨人上下文允许避让与漏接触修复；接触图只承载额外局部关系。
        other = normed.flip(2).mean(3, keepdim=True).expand_as(normed)
        hidden = hidden + self.drop(self.spatial(torch.cat((intra, other, contact), dim=-1)))
        temporal = normed.permute(0, 2, 3, 4, 1).reshape(batch * persons * joints, width, time)
        temporal = self.temporal(temporal).reshape(batch, persons, joints, width, time).permute(0, 4, 1, 2, 3)
        hidden = hidden + self.drop(temporal)
        return hidden + self.drop(self.ff(hidden))


class NTU2PContactGraphDiffusion(nn.Module):
    """先验与纠正器组合；端到端模式另存冻结参照，forward 为明确监督的确定性预测。"""

    def __init__(self, prior, graph=True, phase=True, diffusion=True, hidden=96, num_blocks=3,
                 num_coeffs=16, smooth_frames=8, dropout=0.1, fine_tune_prior=False):
        super(NTU2PContactGraphDiffusion, self).__init__()
        if prior.model_type != "ntu2p_intermixer" or not prior.kin_proj:
            raise ValueError("先验必须为开启骨架投影的 InterMixer")
        self.model_type = MODEL_TYPE
        self.obs_len, self.pred_len, self.num_actions = prior.obs_len, prior.pred_len, prior.num_actions
        for name, value in dict(graph=graph, phase=phase, diffusion=diffusion, hidden=hidden,
                                num_blocks=num_blocks, num_coeffs=num_coeffs,
                                smooth_frames=smooth_frames, dropout=dropout,
                                fine_tune_prior=fine_tune_prior).items():
            setattr(self, name, value)
        if self.obs_len != 10 or self.pred_len != 50 or not 1 <= int(num_coeffs) < 50:
            raise ValueError("图扩散固定 obs10/future50，DCT阶数必须在 [1,49]")
        self.prior = prior.eval()
        # deepcopy 不消耗随机数；原 FD2 独立保存，避免微调后的预测被冒充评估对照。
        self.reference = copy.deepcopy(prior).eval() if fine_tune_prior else None
        for param in self.prior.parameters():
            param.requires_grad_(bool(fine_tune_prior))
        if self.reference is not None:
            for param in self.reference.parameters():
                param.requires_grad_(False)
        self.codec = ResidualCodec(num_coeffs=int(num_coeffs), ramp_mode="smoothstep", saturate_frames=int(smooth_frames))
        self.register_buffer("sigma", torch.ones(2, int(num_coeffs), CHANNELS))
        self.obs_embed = nn.Sequential(nn.Linear(57, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.input_embed = nn.Linear(9, hidden)
        self.event_embed = nn.Linear(6, hidden)
        self.joint_embed = nn.Embedding(NUM_BODY, hidden)
        self.role_embed = nn.Embedding(2, hidden)
        self.action_embed = nn.Embedding(self.num_actions, hidden)
        self.time_embed = TimestepEmbedder(hidden, PositionalEncoding(hidden, 0.0))
        self.frame_embed = nn.Parameter(torch.randn(1, PRED_LEN, 1, 1, hidden) * 0.02)
        self.blocks = nn.ModuleList([_EventGraphBlock(hidden, 2 ** index, dropout) for index in range(int(num_blocks))])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        # 所有候选都创建同一参数集合，保证初始化、容量和消融口径一致。
        leg_input = len(LEG_JOINTS) * 57 + PRED_LEN * 3
        self.leg_net = nn.Sequential(nn.Linear(leg_input, 256), nn.GELU(), nn.Linear(256, int(num_coeffs) * len(LEG_JOINTS) * 3))
        for layer in (self.out[-1], self.leg_net[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super(NTU2PContactGraphDiffusion, self).train(mode)
        if not self.fine_tune_prior:
            self.prior.eval()
        if self.reference is not None:
            self.reference.eval()
        return self

    def config(self):
        return dict(model_type=MODEL_TYPE, prior_config=self.prior.config(),
                    **{key: getattr(self, key) for key in CONFIG_KEYS})

    def prepare(self, obs, action):
        # 冻结模式保留旧数值与随机流；联合模式允许点轨迹及几何监督回到 InterMixer。
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.fine_tune_prior):
            draft = self.prior(obs, action)
            frames = codec_frames(obs)
            feats = obs_features(obs, frames).reshape(obs.shape[0], 2, self.obs_len, NUM_BODY, 3).permute(0, 1, 3, 2, 4)
            history = torch.cat((feats.flatten(-2), ((feats[..., 1:, :] - feats[..., :-1, :]) * NTU2P_FPS).flatten(-2)), -1)
            draft_body = to_person(draft - obs[:, -1:, :, :1], frames)
            root = draft_body[..., :1, :]
            draft_local = torch.cat((root, draft_body[..., 1:NUM_BODY, :] - root), -2)
            velocity = torch.cat((torch.zeros_like(draft_local[:, :1]), draft_local[:, 1:] - draft_local[:, :-1]), 1) * NTU2P_FPS
        prepared = dict(obs=obs, action=action.long(), draft=draft, frames=frames, history=history,
                        draft_local=draft_local, velocity=velocity, draft_scene=to_canonical(draft, frames["scene"]))
        if self.reference is not None:
            with torch.no_grad():
                prepared["reference"] = self.reference(obs, action)
        return prepared

    def predict_coeffs(self, x, timesteps, y):
        from utils.ntu2p_contact_graph import contact_graph_features

        batch = int(x.shape[0])
        state = self.codec.decode_channels(x).reshape(batch, PRED_LEN, 2, NUM_BODY, 3)
        hidden = self.input_embed(torch.cat((state, y["draft_local"], y["velocity"]), -1))
        hidden = hidden + self.obs_embed(y["history"]).unsqueeze(1)
        hidden = hidden + self.action_embed(y["action"])[:, None, None, None]
        hidden = hidden + self.role_embed.weight[None, None, :, None] + self.joint_embed.weight[None, None, None]
        hidden = hidden + self.time_embed(timesteps.long()).squeeze(0)[:, None, None, None] + self.frame_embed
        coeffs_m = x * self.sigma
        body_state = self.codec.channels_to_residual(self.codec.decode_channels(coeffs_m))
        # 高噪声阶段的几何仅信先验，低噪声阶段重新估计接触图，避免噪声伪造接触。
        trust = torch.cos(((timesteps.float() / 1000.0 + 0.008) / 1.008) * (torch.pi if hasattr(torch, "pi") else 3.141592653589793) / 2).pow(2)
        scene_state = rotate_rows_transposed(body_state, y["frames"]["person"][:, None, :, None])
        geometry = y["draft_scene"] + trust[:, None, None, None, None] * scene_state
        graph = contact_graph_features(geometry)
        adjacency = graph["adjacency"]
        # 每人两个手节点保留接触/事件记忆；躯干节点聚合对方两只手，保留关系方向。
        contact, memory = graph["contact"], graph["memory"]
        hand = torch.stack((contact.amax(-1), memory.amax(-1)), -1)
        torso = torch.stack((contact.flip(2).amax(-2).amax(-1), memory.flip(2).amax(-2).amax(-1)), -1)
        events = torch.zeros(batch, PRED_LEN, 2, NUM_BODY, 6, dtype=x.dtype, device=x.device)
        events[:, :, :, 20:22, :2] = hand
        events[:, :, :, :20, :2] = torso.unsqueeze(-2)
        events[..., 2:5] = geometry[..., :NUM_BODY, :] - geometry[..., :1, :]
        events[..., 5] = (contact.amax(-1).amax(-1))[:, :, :, None]
        if self.graph:
            hidden = hidden + self.event_embed(events)
        for block in self.blocks:
            hidden = block(hidden, adjacency, self.graph)
        channels = self.out(hidden).flatten(-2)
        coeffs = self.codec.encode_channels(channels)
        if self.phase:
            leg_history = y["history"][:, :, list(LEG_JOINTS)].flatten(-2)
            root_plan = y["draft_local"][:, :, :, 0].transpose(1, 2).flatten(-2)
            legs = self.leg_net(torch.cat((leg_history, root_plan), -1)).reshape(batch, 2, int(self.num_coeffs), len(LEG_JOINTS), 3)
            correction = torch.zeros(batch, 2, int(self.num_coeffs), NUM_BODY, 3, dtype=x.dtype, device=x.device)
            correction[:, :, :, list(LEG_JOINTS)] = legs
            coeffs = coeffs + correction.flatten(-2)
        return coeffs

    def apply_coeffs(self, coeffs, y):
        meters = coeffs * self.sigma
        delta = self.codec.residual_camera(meters, y["frames"])
        free = y["draft"] + delta
        pred = self.prior.skeleton_projector(free, y["obs"][:, -1])
        # 投影的舍入误差无需影响严格连续的首帧。
        pred = torch.cat((y["obs"][:, -1:], pred[:, 1:]), 1)
        reference = y["reference"] if self.fine_tune_prior else y["draft"]
        return dict(pred=pred, pred_free=free, delta=pred - y["obs"][:, -1:], prior=reference, coeffs=coeffs)

    def point_details(self, obs, action, prepared=None):
        y = self.prepare(obs, action) if prepared is None else prepared
        x = torch.zeros(int(obs.shape[0]), 2, int(self.num_coeffs), CHANNELS, dtype=obs.dtype, device=obs.device)
        time = torch.full((int(obs.shape[0]),), 999, dtype=torch.long, device=obs.device)
        return self.apply_coeffs(self.predict_coeffs(x, time, y), y)

    def forward(self, obs, action, context=None, return_details=False):
        details = self.point_details(obs, action)
        return details if return_details else details["pred"]

    def sample_details(self, obs, action, noise, num_steps=50):
        if not self.diffusion:
            raise ValueError("未训练扩散的候选不支持 DDIM")
        y = self.prepare(obs, action)
        sampler = build_sampling_diffusion(num_steps)
        # 重用 ReGenNet 的采样器，明确适配系数去噪接口，避免把点预测当扩散 forward。
        def denoise(state, timesteps, **kwargs):
            return self.predict_coeffs(state, timesteps, y)
        with torch.no_grad():
            coeffs = sampler.ddim_sample_loop(denoise, tuple(noise.shape), noise=noise,
                                              clip_denoised=False, device=obs.device, eta=0.0,
                                              model_kwargs={"y": y})
        return self.apply_coeffs(coeffs, y)


def load_ntu2p_graphdiff_checkpoint(path, device):
    state = torch.load(path, map_location="cpu")
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError("checkpoint 不是接触图扩散模型")
    config = state["model_config"]
    prior = create_ntu2p_intermixer_from_config(config["prior_config"])
    model = NTU2PContactGraphDiffusion(prior, **{key: config[key] for key in CONFIG_KEYS if key in config})
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval(), state
