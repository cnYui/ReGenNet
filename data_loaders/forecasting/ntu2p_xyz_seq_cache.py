"""NTU 双人完整序列 xyz 缓存的加载、GPU 常驻随机截窗采样与 val/test 中心窗口复现。

缓存由 `scripts/build_ntu2p_xyz_seq_cache.py` 生成。训练时整份 train 帧常驻 GPU（约 178MB），
每步只做一次索引 gather，省掉逐 batch 的 SMPL-X FK（含 10475 顶点 LBS）与 h5 读取。

用法：
    cache = NTU2PXYZSeqCache.load(DEFAULT_CACHE_DIR, "train", manifest_path=manifest, device="cuda:0")
    sampler = NTU2PXYZTrainSampler(cache, batch_size=32, seed=0)
    batch = sampler.sample()  # obs_xyz [B,10,2,55,3], target_xyz [B,50,2,55,3], action [B]

    val = NTU2PXYZSeqCache.load(DEFAULT_CACHE_DIR, "val", manifest_path=manifest, device="cuda:0")
    for batch in iter_eval_batches(eval_windows(val), batch_size=16):
        ...
"""

import os
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu_2p_diffusion import (
    DEFAULT_OBS_LEN,
    DEFAULT_PRED_LEN,
    load_ntu_2p_diffusion_manifest,
)


CACHE_FORMAT_VERSION = 1
CACHE_REPRESENTATION = "two_person_rot6d_fk_xyz_full_sequence"
DEFAULT_CACHE_DIR = "results/forecasting/ntu120_label/ntu2p_xyz_seq_cache"
# 增广用独立随机流，保证开关增广时序列与起点的抽样顺序不变（可配对比较）。
AUGMENT_SEED_OFFSET = 7919


def cache_file_path(cache_dir, split):
    return os.path.join(str(cache_dir), "{}_xyz_seq.pt".format(split))


def _check_against_manifest(payload, manifest, split):
    entries = manifest["splits"][split]
    if payload["config"].get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError(
            "缓存 manifest_hash={} 与 manifest {} 不一致，需重新构建缓存".format(
                payload["config"].get("manifest_hash"), manifest.get("manifest_hash")
            )
        )
    expected_ids = [str(item["sample_id"]) for item in entries]
    if list(payload["sample_ids"]) != expected_ids:
        raise ValueError("缓存 sample_id 顺序与 manifest split={} 不一致".format(split))
    expected_lengths = torch.as_tensor([int(item["length"]) for item in entries], dtype=torch.long)
    expected_actions = torch.as_tensor([int(item["action"]) for item in entries], dtype=torch.long)
    if not torch.equal(payload["lengths"].cpu(), expected_lengths):
        raise ValueError("缓存 length 与 manifest split={} 不一致".format(split))
    if not torch.equal(payload["actions"].cpu(), expected_actions):
        raise ValueError("缓存 action 与 manifest split={} 不一致".format(split))


class NTU2PXYZSeqCache(object):
    """一个 split 的全部序列：xyz [F,2,55,3] 扁平帧 + offsets/lengths/actions，均在同一 device。"""

    def __init__(self, payload, device="cpu"):
        config = payload["config"]
        if int(config.get("format_version", -1)) != CACHE_FORMAT_VERSION:
            raise ValueError("缓存 format_version={} 不受支持".format(config.get("format_version")))
        if config.get("representation") != CACHE_REPRESENTATION:
            raise ValueError("缓存 representation={} 不正确".format(config.get("representation")))
        self.device = torch.device(device)
        self.config = config
        self.split = str(config["split"])
        self.manifest_hash = config.get("manifest_hash")
        self.sample_ids = list(payload["sample_ids"])
        self.action_codes = list(payload["action_codes"])
        self.xyz = payload["xyz"].to(self.device, dtype=torch.float32).contiguous()
        self.offsets = payload["offsets"].to(self.device, dtype=torch.long)
        self.lengths = payload["lengths"].to(self.device, dtype=torch.long)
        self.actions = payload["actions"].to(self.device, dtype=torch.long)
        # CPU 副本用于在 CPU generator 上计算起点，避免每步 device->host 同步。
        self.offsets_cpu = payload["offsets"].cpu().long()
        self.lengths_cpu = payload["lengths"].cpu().long()
        if int(self.xyz.shape[0]) != int(self.lengths_cpu.sum().item()):
            raise ValueError("缓存帧数与 lengths 之和不一致")
        if tuple(self.xyz.shape[1:]) != (2, 55, 3):
            raise ValueError("缓存 xyz 必须是 [F,2,55,3]，当前为 {}".format(tuple(self.xyz.shape)))

    @classmethod
    def load(cls, cache_dir, split, manifest_path=None, device="cpu"):
        payload = torch.load(cache_file_path(cache_dir, split), map_location="cpu")
        if str(payload["config"]["split"]) != split:
            raise ValueError("缓存文件 split={} 与请求 {} 不一致".format(payload["config"]["split"], split))
        if manifest_path is not None:
            _check_against_manifest(payload, load_ntu_2p_diffusion_manifest(manifest_path), split)
        return cls(payload, device=device)

    def __len__(self):
        return len(self.sample_ids)

    def sequence(self, index):
        start = int(self.offsets_cpu[index].item())
        return self.xyz[start : start + int(self.lengths_cpu[index].item())]

    def gather_windows(self, seq_index, start, window_len):
        """seq_index/start 为 [B] long（任意 device），返回 [B,window_len,2,55,3]。"""
        first = self.offsets[seq_index.to(self.device)] + start.to(self.device)
        frames = first.unsqueeze(1) + torch.arange(int(window_len), device=self.device).unsqueeze(0)
        return self.xyz[frames]


def _split_window(window, obs_len):
    return window[:, :obs_len].contiguous(), window[:, obs_len:].contiguous()


class NTU2PXYZTrainSampler(object):
    """GPU 常驻随机截窗采样器。

    序列按"逐 epoch 随机排列、跨 epoch 首尾拼接"抽取：每条序列出现频率与 DataLoader(shuffle=True)
    相同，但每步都是满 batch；起点在 [0, length-window_len] 上均匀。全部随机数来自独立的 CPU
    `torch.Generator`（torch 1.7.1 的 CUDA generator 调 randperm 会段错误），不消耗全局 RNG，
    因此模型初始化与 dropout 的随机流和数据采样解耦。
    """

    def __init__(self, cache, batch_size, seed=0, obs_len=DEFAULT_OBS_LEN, pred_len=DEFAULT_PRED_LEN, augment=None):
        self.cache = cache
        self.batch_size = int(batch_size)
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.window_len = self.obs_len + self.pred_len
        self.augment = augment
        self.seed = int(seed)
        if int(cache.lengths_cpu.min().item()) < self.window_len:
            raise ValueError("缓存中存在短于 window_len={} 的序列".format(self.window_len))
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.augment_generator = torch.Generator()
        self.augment_generator.manual_seed(self.seed + AUGMENT_SEED_OFFSET)
        self._queue = torch.zeros(0, dtype=torch.long)
        self.epoch = 0

    def _next_sequences(self):
        while int(self._queue.numel()) < self.batch_size:
            perm = torch.randperm(len(self.cache), generator=self.generator)
            self._queue = torch.cat((self._queue, perm))
            self.epoch += 1
        seq_index = self._queue[: self.batch_size]
        self._queue = self._queue[self.batch_size :]
        return seq_index

    def sample(self):
        seq_index = self._next_sequences()
        max_start = self.cache.lengths_cpu[seq_index] - self.window_len
        uniform = torch.rand(self.batch_size, generator=self.generator)
        start = torch.min((uniform * (max_start + 1).float()).long(), max_start)
        # 打包成一次 host->device 拷贝；non_blocking 避免与上一步 GPU 计算串行。
        packed = torch.stack((seq_index, start)).to(self.cache.device, non_blocking=True)
        window = self.cache.gather_windows(packed[0], packed[1], self.window_len)
        obs_xyz, target_xyz = _split_window(window, self.obs_len)
        if self.augment is not None:
            obs_xyz, target_xyz = self.augment(obs_xyz, target_xyz, self.augment_generator)
        return OrderedDict(
            [
                ("obs_xyz", obs_xyz),
                ("target_xyz", target_xyz),
                ("action", self.cache.actions[packed[0]]),
                ("seq_index", seq_index),
                ("start", start),
            ]
        )

    def __iter__(self):
        while True:
            yield self.sample()

    def state_dict(self):
        return OrderedDict(
            [
                ("generator", self.generator.get_state()),
                ("augment_generator", self.augment_generator.get_state()),
                ("queue", self._queue.clone()),
                ("epoch", int(self.epoch)),
            ]
        )

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])
        self.augment_generator.set_state(state["augment_generator"])
        self._queue = state["queue"].clone()
        self.epoch = int(state["epoch"])


def center_starts(lengths, window_len):
    """与 NTU2PDiffusionForecastDataset 的 val/test 取窗一致：start = (length - window_len) // 2。"""
    return (lengths - int(window_len)) // 2


def eval_windows(cache, obs_len=DEFAULT_OBS_LEN, pred_len=DEFAULT_PRED_LEN):
    """按 manifest 顺序返回全部中心窗口；样本顺序与原数据集 shuffle=False 的 DataLoader 相同。"""
    window_len = int(obs_len) + int(pred_len)
    if int(cache.lengths_cpu.min().item()) < window_len:
        raise ValueError("缓存中存在短于 window_len={} 的序列".format(window_len))
    seq_index = torch.arange(len(cache), dtype=torch.long)
    start = center_starts(cache.lengths_cpu, window_len)
    obs_xyz, target_xyz = _split_window(cache.gather_windows(seq_index, start, window_len), int(obs_len))
    meta = [
        OrderedDict(
            [
                ("sample_id", cache.sample_ids[index]),
                ("start", int(start[index].item())),
                ("length", int(cache.lengths_cpu[index].item())),
                ("action", int(cache.actions[index].item())),
                ("action_code", cache.action_codes[index]),
                ("split", cache.split),
                ("manifest_hash", cache.manifest_hash),
            ]
        )
        for index in range(len(cache))
    ]
    return OrderedDict(
        [
            ("obs_xyz", obs_xyz),
            ("target_xyz", target_xyz),
            ("action", cache.actions.clone()),
            ("start", start),
            ("meta", meta),
        ]
    )


def iter_eval_batches(windows, batch_size):
    """按顺序切 batch，drop_last=False，与原评估 DataLoader 的分批一致。"""
    total = int(windows["obs_xyz"].shape[0])
    for begin in range(0, total, int(batch_size)):
        end = min(begin + int(batch_size), total)
        yield OrderedDict(
            [
                ("obs_xyz", windows["obs_xyz"][begin:end]),
                ("target_xyz", windows["target_xyz"][begin:end]),
                ("action", windows["action"][begin:end]),
                ("meta", windows["meta"][begin:end]),
            ]
        )
