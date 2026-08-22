import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import hashlib
import h5py
import torch
from torch.utils.data import Dataset, get_worker_info

from data_loaders.forecasting.ntu_label import (
    NUM_ACTIONS,
    NTU_FEATS,
    NTU_JOINTS,
    parse_ntu_action_label,
    scan_ntu_label_forecasting_entries,
    summarize_entries,
)
from utils.ntu_2p_rot6d import (
    NTU_2P_ROT6D_FEATS,
    check_ntu_2p_rot6d,
    check_raw_ntu_2p_motion,
    raw_ntu_2p_to_rot6d,
)


DEFAULT_WINDOW_LEN = 60
DEFAULT_OBS_LEN = 10
DEFAULT_PRED_LEN = 50
DEFAULT_VAL_RATIO = 0.1


def _utc_free_payload(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_payload_hash(payload):
    payload = dict(payload)
    payload.pop("manifest_hash", None)
    return hashlib.sha256(_utc_free_payload(payload)).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")


def load_ntu_2p_diffusion_manifest(path, validate_hash=True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r") as f:
        payload = json.load(f)
    if validate_hash and payload.get("manifest_hash") != manifest_payload_hash(payload):
        raise ValueError("manifest_hash 校验失败: {}".format(path))
    return payload


def _validate_lengths(window_len, obs_len, pred_len):
    if int(obs_len) + int(pred_len) != int(window_len):
        raise ValueError("obs_len + pred_len 必须等于 window_len")


def scan_ntu_2p_diffusion_entries(h5_path, window_len=DEFAULT_WINDOW_LEN, strict=True):
    return scan_ntu_label_forecasting_entries(
        h5_path,
        window_len=window_len,
        num_joints=NTU_JOINTS,
        num_feats=NTU_FEATS,
        strict=strict,
    )


def _group_entries_by_action(entries, num_actions):
    groups = OrderedDict((idx, []) for idx in range(int(num_actions)))
    for entry in entries:
        action = int(entry["action"])
        if action < 0 or action >= int(num_actions):
            raise ValueError("action 必须在 [0,{}] 内，当前为 {}".format(int(num_actions) - 1, action))
        groups[action].append(dict(entry))
    return groups


def _split_group(entries, val_ratio, rng):
    entries = [dict(item) for item in entries]
    rng.shuffle(entries)
    count = len(entries)
    if count <= 1:
        return entries, []
    val_count = int(round(float(count) * float(val_ratio)))
    val_count = max(1, val_count)
    val_count = min(count - 1, val_count)
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]
    return train_entries, val_entries


def stratified_train_val_split(entries, val_ratio=DEFAULT_VAL_RATIO, seed=0, num_actions=NUM_ACTIONS):
    rng = random.Random(int(seed))
    groups = _group_entries_by_action(entries, num_actions)
    train_entries = []
    val_entries = []
    for action, group in groups.items():
        group_train, group_val = _split_group(group, val_ratio, rng)
        train_entries.extend(group_train)
        val_entries.extend(group_val)
    train_entries = sorted(train_entries, key=lambda item: item["sample_id"])
    val_entries = sorted(val_entries, key=lambda item: item["sample_id"])
    return train_entries, val_entries


def _entry_for_manifest(entry, split):
    label_info = parse_ntu_action_label(entry["sample_id"])
    action = int(entry.get("action", label_info["action"]))
    if action != int(label_info["action"]):
        raise ValueError("manifest action 与 sample_id 不一致: {}".format(entry["sample_id"]))
    return OrderedDict(
        [
            ("sample_id", str(entry["sample_id"])),
            ("length", int(entry["length"])),
            ("action", action),
            ("action_code", str(entry.get("action_code", label_info["action_code"]))),
            ("action_name", str(entry.get("action_name", entry.get("action_code", label_info["action_code"])))),
            ("split", split),
        ]
    )


def _split_summary(entries, num_actions):
    return summarize_entries({"entries": entries, "raw_count": len(entries)}, num_actions=num_actions)


def assert_manifest_no_sample_id_leak(manifest):
    split_ids = {
        split: {str(item["sample_id"]) for item in manifest["splits"].get(split, [])}
        for split in ("train", "val", "test")
    }
    overlaps = OrderedDict()
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = sorted(split_ids[left].intersection(split_ids[right]))
        overlaps["{}_{}".format(left, right)] = common
    leaked = [key for key, value in overlaps.items() if value]
    if leaked:
        raise ValueError("manifest 存在 sample_id 泄漏: {}".format(overlaps))
    return overlaps


def prepare_ntu_2p_diffusion_manifest(
    train_h5_path,
    test_h5_path,
    manifest_path,
    window_len=DEFAULT_WINDOW_LEN,
    obs_len=DEFAULT_OBS_LEN,
    pred_len=DEFAULT_PRED_LEN,
    val_ratio=DEFAULT_VAL_RATIO,
    seed=0,
    num_actions=NUM_ACTIONS,
    strict=True,
):
    _validate_lengths(window_len, obs_len, pred_len)
    train_h5_path = Path(train_h5_path)
    test_h5_path = Path(test_h5_path)
    if not train_h5_path.exists():
        raise FileNotFoundError(str(train_h5_path))
    if not test_h5_path.exists():
        raise FileNotFoundError(str(test_h5_path))

    train_scan = scan_ntu_2p_diffusion_entries(train_h5_path, window_len=window_len, strict=strict)
    test_scan = scan_ntu_2p_diffusion_entries(test_h5_path, window_len=window_len, strict=strict)
    train_entries, val_entries = stratified_train_val_split(
        train_scan["entries"],
        val_ratio=val_ratio,
        seed=seed,
        num_actions=num_actions,
    )
    test_entries = sorted(test_scan["entries"], key=lambda item: item["sample_id"])

    manifest = OrderedDict()
    manifest["protocol"] = OrderedDict(
        [
            ("dataset", "ntu120_2p_smplx"),
            ("representation", "two_person_rot6d"),
            ("window_len", int(window_len)),
            ("obs_len", int(obs_len)),
            ("pred_len", int(pred_len)),
            ("num_actions", int(num_actions)),
            ("person_order", "person_a_then_person_b_assumed"),
        ]
    )
    manifest["source_paths"] = OrderedDict(
        [
            ("train_h5_path", str(train_h5_path)),
            ("test_h5_path", str(test_h5_path)),
        ]
    )
    manifest["split_config"] = OrderedDict(
        [
            ("seed", int(seed)),
            ("val_ratio", float(val_ratio)),
            ("split_unit", "sample_id"),
            ("train_val_source", "xsub.train"),
            ("test_source", "xsub.test"),
        ]
    )
    manifest["source_scan"] = OrderedDict(
        [
            ("train", summarize_entries(train_scan, num_actions=num_actions)),
            ("test", summarize_entries(test_scan, num_actions=num_actions)),
        ]
    )
    manifest["splits"] = OrderedDict(
        [
            ("train", [_entry_for_manifest(item, "train") for item in train_entries]),
            ("val", [_entry_for_manifest(item, "val") for item in val_entries]),
            ("test", [_entry_for_manifest(item, "test") for item in test_entries]),
        ]
    )
    manifest["split_summary"] = OrderedDict(
        [
            ("train", _split_summary(manifest["splits"]["train"], num_actions)),
            ("val", _split_summary(manifest["splits"]["val"], num_actions)),
            ("test", _split_summary(manifest["splits"]["test"], num_actions)),
        ]
    )
    manifest["sample_id_overlaps"] = assert_manifest_no_sample_id_leak(manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_hash"] = manifest_payload_hash(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def ensure_ntu_2p_diffusion_manifest(
    train_h5_path,
    test_h5_path,
    manifest_path,
    window_len=DEFAULT_WINDOW_LEN,
    obs_len=DEFAULT_OBS_LEN,
    pred_len=DEFAULT_PRED_LEN,
    val_ratio=DEFAULT_VAL_RATIO,
    seed=0,
    num_actions=NUM_ACTIONS,
    overwrite=False,
):
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not bool(overwrite):
        return load_ntu_2p_diffusion_manifest(manifest_path)
    return prepare_ntu_2p_diffusion_manifest(
        train_h5_path=train_h5_path,
        test_h5_path=test_h5_path,
        manifest_path=manifest_path,
        window_len=window_len,
        obs_len=obs_len,
        pred_len=pred_len,
        val_ratio=val_ratio,
        seed=seed,
        num_actions=num_actions,
    )


class NTU2PDiffusionForecastDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split,
        train_h5_path=None,
        test_h5_path=None,
        window_len=DEFAULT_WINDOW_LEN,
        obs_len=DEFAULT_OBS_LEN,
        pred_len=DEFAULT_PRED_LEN,
        max_samples=-1,
        seed=0,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("split 必须是 train/val/test，当前为 {}".format(split))
        _validate_lengths(window_len, obs_len, pred_len)

        self.manifest_path = Path(manifest_path)
        self.manifest = load_ntu_2p_diffusion_manifest(self.manifest_path)
        protocol = self.manifest.get("protocol", {})
        for key, expected in (("window_len", window_len), ("obs_len", obs_len), ("pred_len", pred_len)):
            if int(protocol.get(key)) != int(expected):
                raise ValueError("manifest {}={} 与参数 {} 不一致".format(key, protocol.get(key), expected))

        self.split = split
        self.window_len = int(window_len)
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.max_samples = int(max_samples) if max_samples is not None else -1
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._h5_handle = None
        self.manifest_hash = self.manifest.get("manifest_hash")

        source_paths = self.manifest.get("source_paths", {})
        if split in ("train", "val"):
            self.h5_path = Path(train_h5_path or source_paths.get("train_h5_path"))
        else:
            self.h5_path = Path(test_h5_path or source_paths.get("test_h5_path"))
        if not self.h5_path.exists():
            raise FileNotFoundError(str(self.h5_path))

        self.entries = [dict(item) for item in self.manifest["splits"].get(split, [])]
        if self.max_samples is not None and self.max_samples > 0:
            self.entries = self.entries[: self.max_samples]
        if len(self.entries) == 0:
            raise ValueError("manifest split={} 为空: {}".format(split, self.manifest_path))

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handle"] = None
        return state

    def close(self):
        if self._h5_handle is not None:
            self._h5_handle.close()
            self._h5_handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_h5_handle(self):
        if self._h5_handle is None:
            self._h5_handle = h5py.File(str(self.h5_path), "r")
        return self._h5_handle

    def _sample_start(self, length):
        max_start = int(length) - self.window_len
        if max_start < 0:
            raise ValueError("length 小于 window_len")
        if self.split == "train":
            if get_worker_info() is not None:
                return random.randint(0, max_start)
            return self._rng.randint(0, max_start)
        return max_start // 2

    def __getitem__(self, index):
        entry = self.entries[index]
        sample_id = str(entry["sample_id"])
        length = int(entry["length"])
        start = self._sample_start(length)
        stop = start + self.window_len

        motion = self._get_h5_handle()[sample_id][start:stop]
        motion = torch.from_numpy(motion).float()
        expected = (self.window_len, NTU_JOINTS, NTU_FEATS)
        if tuple(motion.shape) != expected:
            raise ValueError("窗口 shape 必须为 {}，{} 当前为 {}".format(expected, sample_id, tuple(motion.shape)))

        raw_motion = motion.permute(1, 2, 0).unsqueeze(0).contiguous()
        check_raw_ntu_2p_motion(raw_motion, seq_len=self.window_len)
        rot6d = raw_ntu_2p_to_rot6d(raw_motion).squeeze(0).contiguous()
        check_ntu_2p_rot6d(rot6d.unsqueeze(0), seq_len=self.window_len)

        obs_motion = rot6d[:, :, : self.obs_len].contiguous()
        future = rot6d[:, :, self.obs_len :].contiguous()
        mask = torch.ones(1, 1, self.pred_len, dtype=torch.bool)

        return {
            "obs_motion": obs_motion,
            "future": future,
            "action": torch.tensor([int(entry["action"])], dtype=torch.long),
            "mask": mask,
            "length": length,
            "start": int(start),
            "sample_id": sample_id,
            "action_code": str(entry["action_code"]),
            "action_name": str(entry.get("action_name", entry["action_code"])),
            "split": self.split,
            "manifest_path": str(self.manifest_path),
            "manifest_hash": self.manifest_hash,
        }


def ntu_2p_diffusion_collate(batch):
    not_none = [item for item in batch if item is not None]
    if len(not_none) == 0:
        raise ValueError("batch 为空")

    obs_motion = torch.stack([item["obs_motion"] for item in not_none], dim=0).float()
    future = torch.stack([item["future"] for item in not_none], dim=0).float()
    action = torch.stack([item["action"] for item in not_none], dim=0).long()
    mask = torch.stack([item["mask"] for item in not_none], dim=0).bool()
    lengths = torch.as_tensor([int(item["length"]) for item in not_none], dtype=torch.long)

    expected_obs = (NTU_JOINTS, NTU_2P_ROT6D_FEATS, not_none[0]["obs_motion"].shape[-1])
    expected_future = (NTU_JOINTS, NTU_2P_ROT6D_FEATS, not_none[0]["future"].shape[-1])
    if tuple(obs_motion.shape[1:]) != expected_obs:
        raise ValueError("batch obs_motion shape 不匹配: {}".format(tuple(obs_motion.shape)))
    if tuple(future.shape[1:]) != expected_future:
        raise ValueError("batch future shape 不匹配: {}".format(tuple(future.shape)))

    meta = []
    for item in not_none:
        meta.append(
            OrderedDict(
                [
                    ("sample_id", item["sample_id"]),
                    ("start", int(item["start"])),
                    ("length", int(item["length"])),
                    ("action", int(item["action"].item())),
                    ("action_code", item["action_code"]),
                    ("action_name", item["action_name"]),
                    ("split", item["split"]),
                    ("manifest_path", item["manifest_path"]),
                    ("manifest_hash", item["manifest_hash"]),
                ]
            )
        )

    return {
        "obs_motion": obs_motion,
        "future": future,
        "action": action,
        "mask": mask,
        "lengths": lengths,
        "meta": meta,
    }


def manifest_default_path(save_dir, seed=0):
    return os.path.join(str(save_dir), "manifest_seed{}.json".format(int(seed)))
