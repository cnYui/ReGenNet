"""NTU2P 第二代（v2）模型：规范系 residual refiner（A1/A2），可挂检索锚点（A3）、骨架投影（A4）与镜像指示（A7）。

设计：docs/ai/context/20260925-121154-ntu2p-v2-architecture-exploration-design-and-plan.md 第 3 节。

v2 统一接口（所有 v2 架构共用，训练/评估脚本只依赖它）：
    forward(obs_xyz, action, context=None, return_details=False)
    obs_xyz 为相机系 [B,10,2,55,3]，返回相机系 pred [B,50,2,55,3]；规范化在模型内部完成。
    return_details=True 返回 dict：pred、delta（delta_reg 用，可为 0 张量）、frame（规范化参数或 None）、
    base（可选）、pred_free（骨架投影前的输出，未投影时为 None）、anchor（检索锚点，仅 A3）。
    context 为 dict，可含 performer [B] long（检索排除同受试者）、seq_index [B] long（训练时排除同序列）、
    mirror [B] bool（镜像指示）。
旧 refiner 的 forward 返回 tuple，用 `forward_details` 统一成上面的 dict。
"""

import inspect
from collections import OrderedDict

import torch
from torch import nn

from model.forecasting_ntu2p_residual_xyz import (
    MODEL_TYPE as REFINER_MODEL_TYPE,
    NTU2PResidualRefinerXYZ,
    _normalize_action,
    load_ntu2p_residual_refiner_checkpoint,
)
from model.forecasting_ntu_xyz import create_ntu_label_xyz_model_from_config
from utils.ntu2p_canonical import apply_linear, canonical_frame, to_canonical
from utils.ntu_smplx_2p_xyz import check_ntu_xyz


CANON_MODEL_TYPE = "ntu2p_canon_refiner_xyz"
NUM_ROLES = 2

# 父类构造参数；保存进 config，加载时原样传回，保证重建的结构与训练时一致。
REFINER_INIT_KEYS = (
    "obs_len",
    "pred_len",
    "num_actions",
    "latent_dim",
    "num_heads",
    "encoder_layers",
    "decoder_layers",
    "dim_feedforward",
    "dropout",
    "ramp_mode",
    "ramp_saturate_frames",
    "future_pos_mode",
    "root_head_mode",
    "root_dct_k",
)
CANON_SWITCH_KEYS = ("canonical", "disp_channel", "role_embed", "mirror_embed", "retrieval_k", "kin_proj", "canonical_ab_fallback")


def accepted_kwargs(target, candidates):
    """只保留 target（类或函数）构造签名接受的参数：并行开发的模块签名未冻结时，避免多传参数直接报错。"""
    parameters = inspect.signature(target).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(candidates)
    return {key: value for key, value in candidates.items() if key in parameters}


def rotate_to_camera(value, frame):
    """规范系下的位移/残差（不含平移）旋回相机系；只转向量不加平移，首帧零残差旋回后仍严格为 0。"""
    return apply_linear(value, frame["rotation"].transpose(-1, -2))


def independent_base_forward(base_model, obs_xyz, action):
    """冻结独立单人 base 对 A/B 各跑一次；评估时各架构共用它算 base 指标，gate 口径一致。"""
    batch_size = int(obs_xyz.shape[0])
    action = _normalize_action(action, batch_size, base_model.num_actions, obs_xyz.device)
    obs_single = torch.cat((obs_xyz[:, :, 0:1], obs_xyz[:, :, 1:2]), dim=0)
    pred_single = base_model(obs_single, torch.cat((action, action), dim=0))
    return torch.cat((pred_single[:batch_size], pred_single[batch_size:]), dim=2)


class NTU2PCanonRefiner(NTU2PResidualRefinerXYZ):
    """冻结单人 base（相机系）+ 规范系跨人 residual；全部开关关闭时前向与父类逐位相同。

    新模块全部在 fork 出的 CPU 随机数流里初始化，且输出层零初始化：同 seed 下与对照（A0）的
    父类参数初始化、dropout 随机流逐一对应，开关只改变结构，便于配对比较。
    """

    def __init__(
        self,
        base_model,
        canonical=False,
        disp_channel=False,
        role_embed=False,
        mirror_embed=False,
        retrieval_k=0,
        kin_proj=False,
        retrieval_anchor_kwargs=None,
        canonical_ab_fallback="a_facing",
        **kwargs
    ):
        super(NTU2PCanonRefiner, self).__init__(base_model, **kwargs)
        self.model_type = CANON_MODEL_TYPE
        self.canonical = bool(canonical)
        self.disp_channel = bool(disp_channel)
        self.role_embed = bool(role_embed)
        self.mirror_embed = bool(mirror_embed)
        self.retrieval_k = int(retrieval_k)
        self.kin_proj = bool(kin_proj)
        # 旧 checkpoint 的 config 无此键，加载时按默认 a_facing，行为不变。
        self.canonical_ab_fallback = str(canonical_ab_fallback)
        self.retrieval_anchor_kwargs = dict(retrieval_anchor_kwargs or {})
        if self.retrieval_k < 0:
            raise ValueError("retrieval_k 必须 >= 0")
        if self.retrieval_k > 0 and not self.canonical:
            # 检索库的键与邻居位移都在场景规范系中，锚点只能在同一坐标系里与 base 融合。
            raise ValueError("retrieval 需要同时开启 canonical")
        # 检索库是数据（含整份 train 缓存引用），不进 state_dict，也不注册成子模块。
        self.__dict__["_retrieval_bank"] = None

        with torch.random.fork_rng(devices=[]):
            if self.disp_channel:
                self.disp_proj = nn.Linear(self.person_dim, self.latent_dim)
            if self.role_embed:
                self.role_embedding = nn.Embedding(NUM_ROLES, self.latent_dim)
            if self.mirror_embed:
                self.mirror_embedding = nn.Embedding(2, self.latent_dim)
            if self.retrieval_k > 0:
                from model.ntu2p_retrieval_anchor import RetrievalAnchor

                candidates = OrderedDict(
                    [
                        ("latent_dim", self.latent_dim),
                        ("d_model", self.latent_dim),
                        ("k", self.retrieval_k),
                        ("num_neighbors", self.retrieval_k),
                        ("obs_len", self.obs_len),
                        ("pred_len", self.pred_len),
                    ]
                )
                anchor_kwargs = accepted_kwargs(RetrievalAnchor, candidates)
                # 显式给出的参数不过滤：拼错或不支持时应直接报错，而不是被静默丢弃。
                anchor_kwargs.update(self.retrieval_anchor_kwargs)
                self.retrieval_anchor = RetrievalAnchor(**anchor_kwargs)
            if self.kin_proj:
                from utils.ntu2p_kinematic_projection import SkeletonProjector

                self.skeleton_projector = SkeletonProjector()
        # 零初始化：输入（位移、角色、镜像标记）本身非零，delta head 首步更新后上游梯度即非零，
        # 权重梯度 = 上游梯度 × 输入，不会出现 Stage 2 那种"两个因子同时为零"的鞍点；
        # 同时保证开关开启时初始输出与关闭时逐位一致。小随机初始化没有额外收益，反而破坏这一点。
        if self.disp_channel:
            nn.init.zeros_(self.disp_proj.weight)
            nn.init.zeros_(self.disp_proj.bias)
        if self.role_embed:
            nn.init.zeros_(self.role_embedding.weight)
        if self.mirror_embed:
            nn.init.zeros_(self.mirror_embedding.weight)

    def config(self):
        config = super(NTU2PCanonRefiner, self).config()
        config["architecture"] = "frozen_single_person_base_plus_canonical_cross_person_residual"
        for key in CANON_SWITCH_KEYS:
            config[key] = getattr(self, key)
        config["retrieval_anchor_kwargs"] = dict(self.retrieval_anchor_kwargs)
        config["canonical_frame"] = "utils.ntu2p_canonical.canonical_frame defaults (up=obs, yaw=ab_line, origin=pelvis)"
        return config

    def set_retrieval_bank(self, bank):
        if self.retrieval_k <= 0:
            raise ValueError("模型未开启 retrieval（retrieval_k=0），不需要检索库")
        if dict(getattr(bank, "canonical_kwargs", {})):
            # 模型用 canonical_frame 默认参数规范化 obs 与 base，检索库必须同一规范系，否则锚点坐标错位。
            raise ValueError("检索库 canonical_kwargs={} 与模型的默认规范系不一致".format(bank.canonical_kwargs))
        self.__dict__["_retrieval_bank"] = bank

    @property
    def retrieval_bank(self):
        return self.__dict__.get("_retrieval_bank")

    def _embed_extras(self, token_a, token_b, context, batch_size):
        """角色与镜像指示加到 A/B 的 token 上（token 为 [T,B,d]）。"""
        if self.role_embed:
            token_a = token_a + self.role_embedding.weight[0]
            token_b = token_b + self.role_embedding.weight[1]
        if self.mirror_embed:
            mirror = context.get("mirror")
            if mirror is None:
                # 测试与未增广训练时没有镜像，指示恒为 0。
                mirror = torch.zeros(batch_size, dtype=torch.long, device=token_a.device)
            mirror = self.mirror_embedding(mirror.to(device=token_a.device, dtype=torch.long)).unsqueeze(0)
            token_a = token_a + mirror
            token_b = token_b + mirror
        return token_a, token_b

    def _disp_tokens(self, obs_in, batch_size):
        disp = obs_in - obs_in[:, -1:]
        tokens = []
        for person in (0, 1):
            value = disp[:, :, person].reshape(batch_size, self.obs_len, self.person_dim)
            tokens.append(self.disp_proj(value).transpose(0, 1))
        return tokens

    def _retrieval(self, obs_canon, base_canon, action, context):
        bank = self.retrieval_bank
        if bank is None:
            raise RuntimeError("retrieval 模型需要先 set_retrieval_bank(bank)")
        performer = context.get("performer")
        if performer is None:
            raise ValueError("retrieval 需要 context['performer'] 以排除同受试者")
        with torch.no_grad():
            neighbors = bank.query(
                obs_canon,
                action,
                performer=performer,
                exclude_seq=context.get("seq_index"),
                k=self.retrieval_k,
            )
        device = obs_canon.device
        disp = neighbors["disp"].to(device=device, dtype=obs_canon.dtype)
        dist = neighbors["dist"].to(device=device, dtype=obs_canon.dtype)
        tier = neighbors.get("tier")
        if tier is not None:
            tier = tier.to(device)
        # tier 标出回退邻居（异动作/同受试者），让锚点能区分首选与回退候选。
        output = self.retrieval_anchor(obs_canon, base_canon, disp, dist, self.ramp.to(dtype=obs_canon.dtype), tier=tier)
        return output["anchor"], output["tokens"]

    def forward(self, obs_xyz, action, context=None, return_details=False):
        check_ntu_xyz("obs_xyz", obs_xyz, seq_len=self.obs_len, num_persons=2)
        batch_size = int(obs_xyz.shape[0])
        action = _normalize_action(action, batch_size, self.num_actions, obs_xyz.device)
        context = context or {}
        # base 在相机系上训练，必须在相机系运行；只有 refiner 的输入与残差进入规范系。
        base_xyz = self._base_forward(obs_xyz, action)

        frame = None
        obs_in, base_in = obs_xyz, base_xyz
        if self.canonical:
            frame = canonical_frame(obs_xyz, ab_fallback=self.canonical_ab_fallback)
            obs_in = to_canonical(obs_xyz, frame)
            base_in = to_canonical(base_xyz, frame)
        anchor_in, exemplar_tokens = base_in, None
        if self.retrieval_k > 0:
            anchor_in, exemplar_tokens = self._retrieval(obs_in, base_in, action, context)

        obs_a, obs_b = self._person_tokens(obs_in, self.obs_input_proj, self.obs_pos)
        if self.disp_channel:
            disp_a, disp_b = self._disp_tokens(obs_in, batch_size)
            obs_a = obs_a + disp_a
            obs_b = obs_b + disp_b
        obs_a, obs_b = self._embed_extras(obs_a, obs_b, context, batch_size)
        obs_a, obs_b = self.obs_encoder(obs_a, obs_b)
        action_token = self.action_embed(action).unsqueeze(0) + self.action_type
        obs_a_summary = obs_a.mean(dim=0, keepdim=True) + self.obs_summary_type
        obs_b_summary = obs_b.mean(dim=0, keepdim=True) + self.obs_summary_type
        parts = [action_token, obs_a_summary, obs_b_summary, obs_a, obs_b]
        if exemplar_tokens is not None:
            # 与其它 memory token 一起过 LayerNorm，使 exemplar token 的尺度与观测 token 对齐。
            parts.append(exemplar_tokens)
        memory = torch.cat(parts, dim=0)
        memory = self.memory_proj(self.memory_norm(memory))

        # future token 取残差锚点本身（A3 时为检索锚点），解码器直接看到它要修正的轨迹。
        future_a, future_b = self._person_tokens(anchor_in, self.future_input_proj, self.future_pos, token_type=self.base_type)
        if self.future_pos_mode == "sinusoidal":
            future_a = future_a + self.future_sin_pos
            future_b = future_b + self.future_sin_pos
        future_a, future_b = self._embed_extras(future_a, future_b, context, batch_size)
        future_a = self.future_norm(future_a)
        future_b = self.future_norm(future_b)
        decoded_a, decoded_b = self.future_decoder(future_a, future_b, memory=memory)
        delta_a = self._tokens_to_xyz(decoded_a, batch_size)
        delta_b = self._tokens_to_xyz(decoded_b, batch_size)
        delta = torch.cat((delta_a, delta_b), dim=2)
        delta = delta * self.ramp.to(dtype=delta.dtype)

        anchor_xyz = None
        if self.canonical:
            residual = self.alpha * delta
            if exemplar_tokens is not None:
                anchor_offset = anchor_in - base_in
                residual = anchor_offset + residual
                anchor_xyz = base_xyz + rotate_to_camera(anchor_offset, frame)
            if self.root_head_mode == "dct":
                residual = residual + self._root_offset(obs_in, decoded_a, decoded_b).unsqueeze(3)
            pred_xyz = base_xyz + rotate_to_camera(residual, frame)
        else:
            # 与父类 forward 的运算顺序完全相同，保证开关关闭时逐位等价。
            pred_xyz = base_xyz + self.alpha * delta
            if self.root_head_mode == "dct":
                pred_xyz = pred_xyz + self._root_offset(obs_xyz, decoded_a, decoded_b).unsqueeze(3)

        pred_free = None
        if self.kin_proj:
            pred_free = pred_xyz
            pred_xyz = self.skeleton_projector(pred_free, obs_xyz[:, -1])
        check_ntu_xyz("pred_xyz", pred_xyz, seq_len=self.pred_len, num_persons=2)
        if not return_details:
            return pred_xyz
        details = OrderedDict(
            [
                ("pred", pred_xyz),
                ("delta", delta),
                ("frame", frame),
                ("base", base_xyz),
                ("pred_free", pred_free),
            ]
        )
        if anchor_xyz is not None:
            details["anchor"] = anchor_xyz
        return details


def forward_details(model, obs_xyz, action, context=None):
    """统一取 details dict：旧 refiner（A0）返回 (pred, base, delta)，v2 模型原生返回 dict。"""
    if getattr(model, "model_type", None) == REFINER_MODEL_TYPE:
        pred, base, delta = model(obs_xyz, action, return_details=True)
        return OrderedDict([("pred", pred), ("delta", delta), ("frame", None), ("base", base), ("pred_free", None)])
    return model(obs_xyz, action, context=context, return_details=True)


def canon_refiner_kwargs_from_config(config):
    kwargs = {key: config[key] for key in REFINER_INIT_KEYS + CANON_SWITCH_KEYS if key in config}
    kwargs["retrieval_anchor_kwargs"] = dict(config.get("retrieval_anchor_kwargs") or {})
    return kwargs


def load_ntu2p_canon_refiner_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if state.get("model_type") != CANON_MODEL_TYPE:
        raise ValueError("checkpoint model_type 必须是 {}".format(CANON_MODEL_TYPE))
    config = state.get("model_config", {})
    if not config.get("base_model_config"):
        raise ValueError("checkpoint 缺少 base_model_config")
    base_model = create_ntu_label_xyz_model_from_config(config["base_model_config"])
    # 推理时 base 一律冻结；解冻训练（A2）得到的 base 权重随 state_dict 一起加载。
    model = NTU2PCanonRefiner(base_model=base_model, alpha=1.0, freeze_base=True, **canon_refiner_kwargs_from_config(config))
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


def load_ntu2p_model_checkpoint(path, device):
    """按 checkpoint 的 model_type 分派：旧 refiner、规范系 refiner、InterMixer（A6）。"""
    model_type = torch.load(path, map_location="cpu").get("model_type")
    if model_type == REFINER_MODEL_TYPE:
        return load_ntu2p_residual_refiner_checkpoint(path, device)
    if model_type == CANON_MODEL_TYPE:
        return load_ntu2p_canon_refiner_checkpoint(path, device)
    from model.forecasting_ntu2p_intermixer import load_ntu2p_intermixer_checkpoint

    return load_ntu2p_intermixer_checkpoint(path, device)
