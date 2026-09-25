"""NTU 双人检索锚点（设计代号 A3）的检索库：train 全部窗口在各自场景规范系下的观测键与索引。

设计取舍（背景见 docs/ai/context/20260925-121154-ntu2p-v2-architecture-exploration-design-and-plan.md 1.4 / A3）：
- 只由 train 构建。未来位移不单独存储：查询时按 (seq, start) 从常驻缓存 gather，再用存下的规范系旋转变换，
  库文件只有几 MB，且与缓存逐位一致；缓存可常驻 GPU，与训练采样器共用同一份帧。
- 规范系沿用 `utils/ntu2p_canonical.canonical_frame`（原点 = 观测末帧双人 pelvis 中点，+X = A->B，+Y = 上），
  使检索对相机位姿、双人左右排布不变，只剩双人距离一个布局自由度。
- 键 = 规范系观测 10 帧 × 22 个身体关节坐标，除以 sqrt(位置维数) 使距离成为逐坐标 RMS（单位 m），
  τ 初始化与不同 K 之间可直接比较。不用手指 30 关节：R2 分析显示手指误差几乎全来自手腕位置，
  手指自身构型噪声大且维数多，会稀释身体姿态与运动的权重。不做逐维 z-score：低方差维（如站定者的
  pelvis 竖直分量）会被放大成噪声。速度项默认关闭，其权重由 train 内留受试者检索实验决定（见构建脚本）。
- 受试者重叠：val 受试者全部出现在 train，同人以 R001/R002 重复同一动作，不排除会高估检索效果；
  因此查询默认排除同受试者，训练时再排除同序列；test 天然不重叠，对所有 split 用同一规则即可。
- 候选不足 k（稀有动作，如 train 中 A018 只有 1 名受试者、6 个窗口）时按层级回退，同序列永远排除。
  默认 `fallback="other_action"`：先同动作异受试者，不足时补异动作异受试者，最后才用同受试者。
  这样训练 / val 的回退形态与 test 一致（test 永远没有同受试者候选）；`"same_performer"` 则先放宽受试者。
  每个邻居返回 `tier` 代码（bit1 = 异动作，bit0 = 同受试者，0 = 首选），下游可以感知回退。
- 默认每条 train 序列至多取 1 个窗口（`one_per_sequence=True`，取该序列内距离最小的起点）：步长 1 的相邻窗口
  几乎相同，不去重时 top-16 实际只来自 3-4 条序列。train 内留受试者检索（1758 个中心窗口）的 mpjpe：
  去重 K=8/16/40 为 0.1835/0.1820/0.1858，不去重为 0.2090/0.1988/0.1893；速度项（权重 3/10）无收益。
"""

import contextlib
import math
import re
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu_2p_diffusion import DEFAULT_OBS_LEN, DEFAULT_PRED_LEN
from utils.ntu2p_canonical import canonical_frame, canonicalize


BANK_FORMAT_VERSION = 1
DEFAULT_BANK_DIR = "results/forecasting/ntu120_label/ntu2p_retrieval_bank"
DEFAULT_KEY_JOINTS = 22
DEFAULT_K = 16
FALLBACK_MODES = ("other_action", "same_performer")
TIER_OTHER_ACTION = 2
TIER_SAME_PERFORMER = 1
NUM_TIERS = 4
_PERFORMER_PATTERN = re.compile(r"P(\d{3})")
# 分块计算规范系与键，控制构建时的峰值内存（每块观测约 5 MB / 100 窗口）。
_BUILD_CHUNK = 4096


def performer_ids(sample_ids):
    """NTU sample_id（如 S001C001P002R002A003）中的受试者编号 -> LongTensor [N]。"""
    values = []
    for sample_id in sample_ids:
        match = _PERFORMER_PATTERN.search(str(sample_id))
        if match is None:
            raise ValueError("sample_id {} 中找不到受试者字段 P\\d{{3}}".format(sample_id))
        values.append(int(match.group(1)))
    return torch.as_tensor(values, dtype=torch.long)


def retrieval_key(obs_canon, key_joints=DEFAULT_KEY_JOINTS, velocity_weight=0.0):
    """规范系观测 [B,T,2,55,3] -> 检索键 [B,D]；位置部分的欧氏距离除以 sqrt(D_pos) 即逐坐标 RMS。"""
    body = obs_canon[:, :, :, : int(key_joints)]
    batch_size = int(body.shape[0])
    scale = 1.0 / math.sqrt(float(body[0].numel()))
    parts = [body.reshape(batch_size, -1)]
    if float(velocity_weight) > 0.0:
        velocity = body[:, 1:] - body[:, :-1]
        parts.append(float(velocity_weight) * velocity.reshape(batch_size, -1))
    return torch.cat(parts, dim=1) * scale


@contextlib.contextmanager
def full_precision_matmul():
    """距离用 |q|²+|k|²-2q·k 计算，TF32 的 10 位尾数会让近邻排序出错；在检索期间临时关闭。"""
    matmul = getattr(torch.backends.cuda, "matmul", None)
    previous = getattr(matmul, "allow_tf32", None)
    if previous is None:
        yield
        return
    matmul.allow_tf32 = False
    try:
        yield
    finally:
        matmul.allow_tf32 = previous


def rotate(value, rotation):
    """逐样本 x -> R x（value [M,...,3]，rotation [M,3,3]）；与 apply_linear 等价，但按列展开，
    不生成 [...,3,3] 中间张量（检索 gather 的 K×50 帧位移在 CPU 上快约 3 倍），同样避开 TF32。"""
    columns = rotation.view((int(rotation.shape[0]),) + (1,) * (value.dim() - 2) + (3, 3))
    return value[..., 0:1] * columns[..., 0] + value[..., 1:2] * columns[..., 1] + value[..., 2:3] * columns[..., 2]


def enumerate_windows(lengths, window_len, stride=1):
    """每条序列起点 0, stride, ..., length-window_len -> (seq_index [N], start [N])。"""
    seq_index, start = [], []
    for index, length in enumerate(lengths.tolist()):
        count = int(length) - int(window_len)
        if count < 0:
            continue
        starts = torch.arange(0, count + 1, int(stride), dtype=torch.long)
        seq_index.append(torch.full_like(starts, index))
        start.append(starts)
    return torch.cat(seq_index), torch.cat(start)


class NTU2PRetrievalBank(object):
    """train 窗口检索库。键与规范系参数常驻 `device`；未来位移从 `train_cache` 实时 gather。"""

    def __init__(self, tables, config, train_cache, device="cpu"):
        self.config = OrderedDict(config)
        self.device = torch.device(device)
        self.train_cache = train_cache
        self.obs_len = int(config["obs_len"])
        self.pred_len = int(config["pred_len"])
        self.key_joints = int(config["key_joints"])
        self.velocity_weight = float(config["velocity_weight"])
        self.canonical_kwargs = dict(config["canonical_kwargs"])
        self.seq_index = tables["seq_index"].to(self.device, dtype=torch.long)
        self.start = tables["start"].to(self.device, dtype=torch.long)
        self.action = tables["action"].to(self.device, dtype=torch.long)
        self.performer = tables["performer"].to(self.device, dtype=torch.long)
        self.rotation = tables["rotation"].to(self.device, dtype=torch.float32)
        self.translation = tables["translation"].to(self.device, dtype=torch.float32)
        self.up = tables["up"].to(self.device, dtype=torch.float32)
        # 每条 train 序列的受试者：训练查询用 seq_index 直接取 performer 与 exclude_seq。
        self.seq_performers = tables["seq_performers"].to(self.device, dtype=torch.long)
        self.seq_actions = train_cache.actions.to(self.device, dtype=torch.long)
        self._cache_offsets = train_cache.offsets
        self._build_sequence_layout()
        self._set_keys(self._compute_keys())

    # ---- 构建 / 保存 / 加载 ----

    @classmethod
    def build(
        cls,
        train_cache,
        obs_len=DEFAULT_OBS_LEN,
        pred_len=DEFAULT_PRED_LEN,
        stride=1,
        key_joints=DEFAULT_KEY_JOINTS,
        velocity_weight=0.0,
        canonical_kwargs=None,
        device=None,
    ):
        if train_cache.split != "train":
            raise ValueError("检索库只能由 train 构建，当前 split={}".format(train_cache.split))
        device = train_cache.device if device is None else torch.device(device)
        canonical_kwargs = dict(canonical_kwargs or {})
        window_len = int(obs_len) + int(pred_len)
        seq_index, start = enumerate_windows(train_cache.lengths_cpu, window_len, stride)
        rotation, translation, up = [], [], []
        for begin in range(0, int(seq_index.numel()), _BUILD_CHUNK):
            end = min(begin + _BUILD_CHUNK, int(seq_index.numel()))
            obs = train_cache.gather_windows(seq_index[begin:end], start[begin:end], int(obs_len))
            frame = canonical_frame(obs, **canonical_kwargs)
            rotation.append(frame["rotation"].cpu())
            translation.append(frame["translation"].cpu())
            up.append(frame["up"].cpu())
        seq_performers = performer_ids(train_cache.sample_ids)
        tables = OrderedDict(
            [
                ("seq_index", seq_index),
                ("start", start),
                ("action", train_cache.actions.cpu()[seq_index]),
                ("performer", seq_performers[seq_index]),
                ("rotation", torch.cat(rotation)),
                ("translation", torch.cat(translation)),
                ("up", torch.cat(up)),
                ("seq_performers", seq_performers),
            ]
        )
        config = OrderedDict(
            [
                ("format_version", BANK_FORMAT_VERSION),
                ("obs_len", int(obs_len)),
                ("pred_len", int(pred_len)),
                ("stride", int(stride)),
                ("key_joints", int(key_joints)),
                ("velocity_weight", float(velocity_weight)),
                ("canonical_kwargs", canonical_kwargs),
                ("manifest_hash", train_cache.manifest_hash),
                ("num_train_sequences", len(train_cache)),
                ("num_windows", int(seq_index.numel())),
                ("num_performers", int(torch.unique(seq_performers).numel())),
            ]
        )
        return cls(tables, config, train_cache, device=device)

    def state_dict(self):
        return OrderedDict(
            [
                ("config", OrderedDict(self.config)),
                ("seq_index", self.seq_index.cpu().int()),
                ("start", self.start.cpu().int()),
                ("action", self.action.cpu().int()),
                ("performer", self.performer.cpu().int()),
                ("rotation", self.rotation.cpu()),
                ("translation", self.translation.cpu()),
                ("up", self.up.cpu()),
                ("seq_performers", self.seq_performers.cpu().int()),
                # 键由规范系参数与缓存重算，只存校验和：加载时发现缓存或代码漂移。
                ("key_checksum", self.key_checksum()),
            ]
        )

    def save(self, path):
        torch.save(self.state_dict(), str(path))

    @classmethod
    def load(cls, path, train_cache, device="cpu", checksum_tol=1e-4):
        payload = torch.load(str(path), map_location="cpu")
        config = payload["config"]
        if int(config.get("format_version", -1)) != BANK_FORMAT_VERSION:
            raise ValueError("检索库 format_version={} 不受支持".format(config.get("format_version")))
        if config.get("manifest_hash") != train_cache.manifest_hash:
            raise ValueError("检索库 manifest_hash 与 train 缓存不一致，需重建检索库")
        if int(config["num_train_sequences"]) != len(train_cache):
            raise ValueError("检索库序列数与 train 缓存不一致")
        tables = {key: payload[key] for key in (
            "seq_index", "start", "action", "performer", "rotation", "translation", "up", "seq_performers")}
        bank = cls(tables, config, train_cache, device=device)
        stored = payload["key_checksum"]
        current = bank.key_checksum()
        for name in stored:
            if abs(float(stored[name]) - float(current[name])) > checksum_tol * max(1.0, abs(float(stored[name]))):
                raise ValueError("检索键校验和 {} 不一致：存储 {}，重算 {}".format(name, stored[name], current[name]))
        return bank

    # ---- 键 ----

    def canonicalize(self, obs_xyz, target_xyz=None):
        """与建库同一套规范化参数；返回 (obs_canon, target_canon, frame)。"""
        return canonicalize(obs_xyz, target_xyz, **self.canonical_kwargs)

    def key(self, obs_canon):
        return retrieval_key(obs_canon, self.key_joints, self.velocity_weight)

    def gather_canonical_obs(self, index):
        """库窗口 index [M] -> 各自规范系下的观测 [M,obs_len,2,55,3]。"""
        index = index.to(self.device)
        obs = self.train_cache.gather_windows(self.seq_index[index], self.start[index], self.obs_len).to(self.device)
        return rotate(obs - self.translation[index].view(-1, 1, 1, 1, 3), self.rotation[index])

    def _compute_keys(self):
        keys = []
        total = int(self.seq_index.numel())
        for begin in range(0, total, _BUILD_CHUNK):
            index = torch.arange(begin, min(begin + _BUILD_CHUNK, total), device=self.device)
            keys.append(self.key(self.gather_canonical_obs(index)))
        return torch.cat(keys)

    def _build_sequence_layout(self):
        """[S,W] 每条序列的库窗口下标，空位指向哑元 N：去重检索用一次 gather + min 完成，无逐样本循环。"""
        num_seq = int(self.seq_performers.numel())
        total = int(self.seq_index.numel())
        counts = torch.bincount(self.seq_index, minlength=num_seq)
        first = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
        position = torch.arange(total, device=self.device) - first[self.seq_index]
        layout = torch.full((num_seq, int(counts.max().item())), total, dtype=torch.long, device=self.device)
        layout[self.seq_index, position] = torch.arange(total, device=self.device)
        self.sequence_layout = layout

    def _set_keys(self, keys):
        # 中心化后再做 |q|²+|k|²-2q·k，减小大数相消的舍入误差。
        self.key_mean = keys.mean(dim=0)
        self.keys_centered = (keys - self.key_mean).contiguous()
        self.keys_sq_norm = (self.keys_centered * self.keys_centered).sum(dim=1)

    def key_checksum(self):
        keys = self.keys_centered + self.key_mean
        return OrderedDict([("sum", float(keys.double().sum().item())), ("abs_sum", float(keys.double().abs().sum().item()))])

    def __len__(self):
        return int(self.seq_index.numel())

    # ---- 查询 ----

    def squared_distance(self, obs_canon):
        """[B,obs_len,2,55,3] -> 与全部库窗口的均方距离 [B,N]（单位 m²）。"""
        query = self.key(obs_canon.to(self.device)) - self.key_mean
        with full_precision_matmul():
            cross = torch.matmul(query, self.keys_centered.t())
        return ((query * query).sum(dim=1, keepdim=True) + self.keys_sq_norm.unsqueeze(0) - 2.0 * cross).clamp_min(0.0)

    def gather_displacement(self, index):
        """邻居 index [B,K] -> 未来相对观测末帧的位移 [B,K,pred_len,2,55,3]，处在邻居自己的规范系。"""
        batch_size, k = int(index.shape[0]), int(index.shape[1])
        flat = index.reshape(-1).to(self.device)
        cache_device = self.train_cache.device
        first = self._cache_offsets[self.seq_index[flat].to(cache_device)] + self.start[flat].to(cache_device)
        frames = first.unsqueeze(1) + (self.obs_len - 1) + torch.arange(self.pred_len + 1, device=cache_device)
        window = self.train_cache.xyz[frames].to(self.device)
        # 平移在差分中抵消，只需旋转。
        disp = rotate(window[:, 1:] - window[:, :1], self.rotation[flat])
        return disp.view(batch_size, k, self.pred_len, *disp.shape[2:])

    def rank(
        self, d2, action, performer=None, exclude_seq=None, k=DEFAULT_K, fallback="other_action", one_per_sequence=True
    ):
        """按 (层级, 距离) 取 top-k；返回 index [B,K]、均方距离 [B,K]、tier [B,K]。

        动作、受试者与排除规则都是序列级属性：去重时先在每条序列内取最近窗口，再在 [B,S] 上排序，
        层级与排除只需按序列计算。
        """
        if fallback not in FALLBACK_MODES:
            raise ValueError("fallback 必须是 {}，当前为 {}".format(FALLBACK_MODES, fallback))
        if one_per_sequence:
            padded = torch.cat((d2, d2.new_full((d2.shape[0], 1), float("inf"))), dim=1)
            item_d2, choice = padded[:, self.sequence_layout].min(dim=2)
            item_seq = torch.arange(int(self.seq_actions.numel()), device=self.device)
            item_action, item_performer = self.seq_actions, self.seq_performers
        else:
            item_d2, item_seq, item_action, item_performer = d2, self.seq_index, self.action, self.performer
        action = torch.as_tensor(action).to(self.device, dtype=torch.long).view(-1)
        other_action = item_action.unsqueeze(0) != action.unsqueeze(1)
        if performer is None:
            same_performer = torch.zeros_like(other_action)
        else:
            performer = torch.as_tensor(performer).to(self.device, dtype=torch.long).view(-1)
            same_performer = item_performer.unsqueeze(0) == performer.unsqueeze(1)
        if fallback == "other_action":
            rank_cost = other_action.double() + 2.0 * same_performer.double()
        else:
            rank_cost = 2.0 * other_action.double() + same_performer.double()
        # 层级代价乘以本 batch 最大距离 + 1 使层级严格优先；用 float64 相加，否则 float32 在偏移后
        # 只剩约 1e-7 的分辨率，回退层内的距离排序会出现 1e-6 m 量级的错位。
        wide = item_d2.detach().double()
        score = wide + rank_cost * (wide.max() + 1.0)
        if exclude_seq is not None:
            exclude_seq = torch.as_tensor(exclude_seq).to(self.device, dtype=torch.long).view(-1)
            score = score.masked_fill(item_seq.unsqueeze(0) == exclude_seq.unsqueeze(1), float("inf"))
        _, item = torch.topk(score, int(k), dim=1, largest=False, sorted=True)
        if not bool(torch.isfinite(score.gather(1, item)).all()):
            raise ValueError("排除同序列后库中候选不足 k={}".format(int(k)))
        tier = other_action.long() * TIER_OTHER_ACTION + same_performer.long() * TIER_SAME_PERFORMER
        index = self.sequence_layout[item, choice.gather(1, item)] if one_per_sequence else item
        return index, item_d2.gather(1, item), tier.gather(1, item)

    def query(
        self, obs_canon, action, performer=None, exclude_seq=None, k=DEFAULT_K, fallback="other_action", one_per_sequence=True
    ):
        """同动作内检索 top-k。

        obs_canon [B,obs_len,2,55,3]（已用 `canonicalize` 规范化）；action [B]；
        performer [B] 或 None（None = 不排除同受试者，仅用于量化受试者重叠带来的偏差）；
        exclude_seq [B] train 序列号或 None（训练时传查询自身的序列号；-1 表示不排除）。
        返回 disp [B,K,pred_len,2,55,3]（邻居未来 - 邻居观测末帧，邻居自己的规范系）、
        dist [B,K]（逐坐标 RMS，单位 m）、index [B,K]、tier [B,K]（bit1 = 异动作，bit0 = 同受试者）、
        fallback [B]（是否用到非首选候选）。one_per_sequence=True 时每条 train 序列至多 1 个邻居。
        """
        index, d2, tier = self.rank(
            self.squared_distance(obs_canon), action, performer, exclude_seq, k, fallback, one_per_sequence
        )
        return OrderedDict(
            [
                ("disp", self.gather_displacement(index)),
                ("dist", d2.sqrt()),
                ("index", index),
                ("tier", tier),
                ("fallback", (tier != 0).any(dim=1)),
            ]
        )

    def query_train(self, obs_canon, action, seq_index, k=DEFAULT_K, fallback="other_action", one_per_sequence=True):
        """训练查询：排除同受试者与同序列（seq_index 为 train 序列号 [B]）。"""
        seq_index = seq_index.to(self.device, dtype=torch.long)
        return self.query(
            obs_canon, action, self.seq_performers[seq_index], seq_index, k, fallback, one_per_sequence
        )

    def median_neighbor_distance(self, k=DEFAULT_K, batch_size=256, fallback="other_action"):
        """train 每条序列中心窗口按训练规则查询 top-k 的距离中位数，用作 RetrievalAnchor 的 τ 初始值。"""
        lengths = self.train_cache.lengths_cpu
        start = (lengths - (self.obs_len + self.pred_len)) // 2
        actions = self.train_cache.actions.to(self.device)
        dists = []
        for begin in range(0, int(lengths.numel()), int(batch_size)):
            seq = torch.arange(begin, min(begin + int(batch_size), int(lengths.numel())), dtype=torch.long)
            obs = self.train_cache.gather_windows(seq, start[seq], self.obs_len).to(self.device)
            obs_canon, _, _ = self.canonicalize(obs)
            seq = seq.to(self.device)
            _, d2, _ = self.rank(self.squared_distance(obs_canon), actions[seq], self.seq_performers[seq], seq, k, fallback)
            dists.append(d2.sqrt().cpu())
        return float(torch.cat(dists).median().item())
