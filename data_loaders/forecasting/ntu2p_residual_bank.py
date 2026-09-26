"""Track B 残差库：协议 train 全部 stride-1 窗口的 OOF（或 in-sample）残差系数、条件特征与统计量。

由 `scripts/build_ntu2p_oof_residual_bank.py` 生成；生成器训练从干净窗口采样，评估的平凡 bootstrap 基线从干净窗口
按 (动作, v2_walk_A, v2_walk_B) 分层抽取整场景 target 系数。系数一律以米为单位存盘，归一化只在模型里做。
设计：docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md 第 4.3–4.4、5.2 节。
"""

import hashlib
import json
import os
from collections import OrderedDict

import torch

from utils.ntu2p_residual_codec import STD_FLOOR_FRAC


BANK_FORMAT_VERSION = 1
TENSOR_FIELDS = (
    "seq_index",
    "start",
    "fold",
    "performer",
    "action",
    "clean",
    "glitch",
    "target",
    "draft",
    "obs_feats",
    "rel_geom",
    "v2_walk",
    "base_mpjpe",
)
# bootstrap 分层回退的层级编号：0=(动作, walkA, walkB)，1=动作，2=全体。
STRATA_LEVELS = ("action_walk", "action", "all")


def bank_config_sha256(config):
    """残差库 config 的指纹，写进生成器 checkpoint，用于核对评估时用的是同一个库。"""
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rms(value, dims):
    return value.double().pow(2).mean(dim=dims).sqrt()


def _floor(sigma, floor_frac):
    # 下限防止极小方差的通道（如 DC 以外近乎恒 0 的系数）在归一化后被放大成噪声。
    floor = float(floor_frac) * float(sigma.median().item())
    return sigma.clamp_min(floor).float()


def compute_bank_stats(target, draft, obs_feats, clean, floor_frac=STD_FLOOR_FRAC):
    """只在干净窗口上计算、两人合并、不减均值（RMS）：残差均值接近 0，减均值反而让 DC 通道的尺度随抽样漂移。"""
    if int(clean.sum().item()) <= 0:
        raise ValueError("残差库没有干净窗口，无法估计统计量")
    target, draft, obs_feats = target[clean], draft[clean], obs_feats[clean]
    return OrderedDict(
        [
            ("sigma_target", _floor(_rms(target, (0, 1)), floor_frac)),
            ("sigma_draft", _floor(_rms(draft, (0, 1)), floor_frac)),
            ("feat_std", _floor(_rms(obs_feats, (0, 1, 2)), floor_frac)),
        ]
    )


class NTU2PResidualBankSampler(object):
    """按序列均匀的干净窗口采样器：序列逐 epoch 随机排列，序列内在干净窗口上均匀抽取。

    与 v2 训练采样器同一分布（每条序列出现频率相同），长序列不会因窗口多而被过采样；
    全部随机数来自独立的 CPU Generator，不消耗全局 RNG，模型初始化与 dropout 的随机流与数据采样解耦。
    """

    def __init__(self, bank, batch_size, seed=0):
        clean_ids = torch.nonzero(bank.clean.cpu(), as_tuple=False).view(-1)
        if int(clean_ids.numel()) == 0:
            raise ValueError("残差库没有干净窗口")
        seq = bank.seq_index.cpu()[clean_ids]
        # 库按序列、起点顺序生成，同一序列的干净窗口在 clean_ids 中连续。
        order = torch.argsort(seq * (int(bank.start.max().item()) + 1) + bank.start.cpu()[clean_ids])
        self.clean_ids = clean_ids[order]
        _, counts = torch.unique_consecutive(seq[order], return_counts=True)
        self.clean_counts = counts.long()
        self.clean_offsets = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(self.clean_counts, 0)[:-1]))
        self.batch_size = int(batch_size)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        self._queue = torch.zeros(0, dtype=torch.long)
        self.epoch = 0

    @property
    def num_sequences(self):
        return int(self.clean_counts.numel())

    def _next_sequences(self):
        while int(self._queue.numel()) < self.batch_size:
            self._queue = torch.cat((self._queue, torch.randperm(self.num_sequences, generator=self.generator)))
            self.epoch += 1
        chosen = self._queue[: self.batch_size]
        self._queue = self._queue[self.batch_size :]
        return chosen

    def sample(self):
        seq = self._next_sequences()
        counts = self.clean_counts[seq]
        uniform = torch.rand(self.batch_size, generator=self.generator)
        within = torch.min((uniform * counts.double()).long(), counts - 1)
        return self.clean_ids[self.clean_offsets[seq] + within]

    def state_dict(self):
        return OrderedDict([("generator", self.generator.get_state()), ("queue", self._queue.clone()), ("epoch", int(self.epoch))])

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])
        self._queue = state["queue"].clone()
        self.epoch = int(state["epoch"])


class BootstrapPool(object):
    """平凡随机基线的分层抽样：先按 (动作, v2_walk_A, v2_walk_B)，少于 min_count 个干净窗口时回退到动作，再回退到全体。"""

    def __init__(self, bank, min_count=30):
        clean = bank.clean.cpu()
        ids = torch.nonzero(clean, as_tuple=False).view(-1)
        action = bank.action.cpu()[ids]
        walk = bank.v2_walk.cpu()[ids].long()
        self.min_count = int(min_count)
        self.all_ids = ids
        self.by_action = OrderedDict()
        self.by_stratum = OrderedDict()
        for pos in range(int(ids.numel())):
            key = (int(action[pos]), int(walk[pos, 0]), int(walk[pos, 1]))
            self.by_stratum.setdefault(key, []).append(int(ids[pos]))
            self.by_action.setdefault(key[0], []).append(int(ids[pos]))
        self.by_stratum = OrderedDict((k, torch.tensor(v, dtype=torch.long)) for k, v in self.by_stratum.items())
        self.by_action = OrderedDict((k, torch.tensor(v, dtype=torch.long)) for k, v in self.by_action.items())

    def candidates(self, action, walk_a, walk_b):
        """返回 (候选 id, 层级编号)。"""
        stratum = self.by_stratum.get((int(action), int(walk_a), int(walk_b)))
        if stratum is not None and int(stratum.numel()) >= self.min_count:
            return stratum, 0
        by_action = self.by_action.get(int(action))
        if by_action is not None and int(by_action.numel()) >= self.min_count:
            return by_action, 1
        return self.all_ids, 2

    def draw(self, action, walk, uniform):
        """action [B]、walk bool [B,2]、uniform [B,K]（调用方给出，保证所有变体共用同一份随机数）-> (ids [B,K], levels [B])。"""
        action, walk, uniform = action.cpu(), walk.cpu(), uniform.cpu().double()
        ids = torch.empty(uniform.shape, dtype=torch.long)
        levels = torch.empty(uniform.shape[0], dtype=torch.long)
        for row in range(int(uniform.shape[0])):
            cand, level = self.candidates(action[row], walk[row, 0], walk[row, 1])
            count = int(cand.numel())
            pick = torch.clamp((uniform[row] * count).long(), max=count - 1)
            ids[row] = cand[pick]
            levels[row] = level
        return ids, levels


class NTU2PResidualBank(object):
    """字段见 TENSOR_FIELDS；系数为米，窗口按 (seq_index, start) 顺序。"""

    def __init__(self, fields, stats, config, sample_ids, device="cpu"):
        missing = [name for name in TENSOR_FIELDS if name not in fields]
        if missing:
            raise ValueError("残差库缺少字段 {}".format(missing))
        if int(config.get("format_version", -1)) != BANK_FORMAT_VERSION:
            raise ValueError("残差库 format_version={} 不受支持".format(config.get("format_version")))
        self.device = torch.device(device)
        for name in TENSOR_FIELDS:
            setattr(self, name, fields[name].to(self.device))
        self.stats = OrderedDict((key, value.to(self.device)) for key, value in stats.items())
        self.config = dict(config)
        self.sample_ids = list(sample_ids)
        count = int(self.target.shape[0])
        for name in TENSOR_FIELDS:
            if int(getattr(self, name).shape[0]) != count:
                raise ValueError("残差库字段 {} 的窗口数与 target 不一致".format(name))

    def __len__(self):
        return int(self.target.shape[0])

    @property
    def num_coeffs(self):
        return int(self.target.shape[2])

    @property
    def channels(self):
        return int(self.target.shape[3])

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu")
        return cls(payload["fields"], payload["stats"], payload["config"], payload["sample_ids"], device=device)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = OrderedDict(
            [
                ("fields", OrderedDict((name, getattr(self, name).cpu()) for name in TENSOR_FIELDS)),
                ("stats", OrderedDict((key, value.cpu()) for key, value in self.stats.items())),
                ("config", self.config),
                ("sample_ids", self.sample_ids),
            ]
        )
        torch.save(payload, path)

    def config_sha256(self):
        return bank_config_sha256(self.config)

    def batch(self, ids):
        ids = ids.to(self.device)
        return OrderedDict(
            [
                ("target", self.target[ids]),
                ("draft", self.draft[ids]),
                ("obs_feats", self.obs_feats[ids]),
                ("rel_geom", self.rel_geom[ids]),
                ("action", self.action[ids]),
            ]
        )

    def make_sampler(self, batch_size, seed):
        return NTU2PResidualBankSampler(self, batch_size, seed)

    def bootstrap_pool(self, min_count=30):
        return BootstrapPool(self, min_count=min_count)

    def summary(self):
        return OrderedDict(
            [
                ("num_windows", len(self)),
                ("num_clean", int(self.clean.sum().item())),
                ("glitch_fraction", float(self.glitch.float().mean().item())),
                ("num_sequences", int(torch.unique(self.seq_index).numel())),
                ("base_mpjpe_mean", float(self.base_mpjpe.double().mean().item())),
                ("base_mpjpe_clean_mean", float(self.base_mpjpe[self.clean].double().mean().item())),
            ]
        )


__all__ = [
    "BANK_FORMAT_VERSION",
    "BootstrapPool",
    "NTU2PResidualBank",
    "NTU2PResidualBankSampler",
    "STRATA_LEVELS",
    "TENSOR_FIELDS",
    "bank_config_sha256",
    "compute_bank_stats",
]
