"""把 NTU 双人 manifest 中每条完整序列的双人 xyz 预先算好存盘，供 GPU 常驻采样器随机截窗。

与训练时 on-the-fly 路径完全相同：h5 raw axis-angle -> raw_ntu_2p_to_rot6d(CPU) -> ntu_2p_rot6d_to_xyz(GPU)。
Rotation2xyz_x 的双人分支逐帧独立（平移不减首帧），整序列 FK 后再切窗与窗口 FK 等价，
差异只来自 GPU 批量矩阵乘的浮点求和顺序。
"""

import argparse
import json
import os
import platform
from collections import OrderedDict
from datetime import datetime

import h5py
import torch

from data_loaders.forecasting.ntu_2p_diffusion import load_ntu_2p_diffusion_manifest
from data_loaders.forecasting.ntu2p_xyz_seq_cache import (
    CACHE_FORMAT_VERSION,
    CACHE_REPRESENTATION,
    DEFAULT_CACHE_DIR,
    cache_file_path,
)
from model.rotation2xyz import Rotation2xyz_x
from utils.ntu_2p_rot6d import check_raw_ntu_2p_motion, ntu_2p_rot6d_to_xyz, raw_ntu_2p_to_rot6d


DEFAULT_MANIFEST = "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _device(value):
    requested = torch.device(value)
    if requested.type != "cuda":
        raise ValueError("缓存构建必须与训练同样在 CUDA 上做 FK，例如 --device cuda:0，当前为 {}".format(value))
    if not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但 CUDA 不可用: {}".format(value))
    index = torch.cuda.current_device() if requested.index is None else int(requested.index)
    requested = torch.device("cuda:{}".format(index))
    torch.cuda.set_device(requested)
    return requested


def _h5_path(manifest, split, args):
    source = manifest.get("source_paths", {})
    if split in ("train", "val"):
        return args.train_data_path or source.get("train_h5_path")
    return args.test_data_path or source.get("test_h5_path")


@torch.no_grad()
def _sequence_xyz(handle, sample_id, length, converter, device):
    motion = torch.from_numpy(handle[sample_id][:]).float()
    if int(motion.shape[0]) != int(length):
        raise ValueError("{} h5 长度 {} 与 manifest length {} 不一致".format(sample_id, int(motion.shape[0]), length))
    raw = motion.permute(1, 2, 0).unsqueeze(0).contiguous()
    check_raw_ntu_2p_motion(raw, seq_len=length)
    rot6d = raw_ntu_2p_to_rot6d(raw)
    return ntu_2p_rot6d_to_xyz(rot6d.to(device), converter=converter)[0].cpu()


def build_split(args, manifest, split, converter, device):
    entries = manifest["splits"][split]
    h5_path = _h5_path(manifest, split, args)
    lengths = torch.as_tensor([int(item["length"]) for item in entries], dtype=torch.long)
    offsets = torch.zeros_like(lengths)
    offsets[1:] = torch.cumsum(lengths, dim=0)[:-1]
    xyz = torch.empty(int(lengths.sum().item()), 2, 55, 3, dtype=torch.float32)
    with h5py.File(str(h5_path), "r") as handle:
        for index, entry in enumerate(entries):
            start = int(offsets[index].item())
            length = int(lengths[index].item())
            xyz[start : start + length] = _sequence_xyz(handle, str(entry["sample_id"]), length, converter, device)
            if index == 0 or (index + 1) % args.log_interval == 0 or index + 1 == len(entries):
                print("split={} sequences={}/{} frames={}".format(split, index + 1, len(entries), start + length))
    if not torch.isfinite(xyz).all():
        raise ValueError("{} xyz 存在非有限数值".format(split))
    config = OrderedDict(
        [
            ("format_version", CACHE_FORMAT_VERSION),
            ("dataset", "ntu120_2p_smplx"),
            ("representation", CACHE_REPRESENTATION),
            ("xyz_layout", "[F,2,55,3] 按 manifest 顺序拼接全部帧；人物 0=A,1=B；SMPL-X 55 关节；原始相机系"),
            ("fk_path", "h5 axis-angle -> raw_ntu_2p_to_rot6d(cpu) -> ntu_2p_rot6d_to_xyz(Rotation2xyz_x, betas=0)"),
            ("split", split),
            ("manifest_path", str(args.manifest_path)),
            ("manifest_hash", manifest.get("manifest_hash")),
            ("h5_path", str(h5_path)),
            ("num_sequences", len(entries)),
            ("num_frames", int(xyz.shape[0])),
            ("device", str(device)),
            ("device_name", torch.cuda.get_device_name(device)),
            ("torch_version", torch.__version__),
            ("python_version", platform.python_version()),
            ("created_at", _utc_now()),
        ]
    )
    return OrderedDict(
        [
            ("xyz", xyz),
            ("offsets", offsets),
            ("lengths", lengths),
            ("actions", torch.as_tensor([int(item["action"]) for item in entries], dtype=torch.long)),
            ("sample_ids", [str(item["sample_id"]) for item in entries]),
            ("action_codes", [str(item["action_code"]) for item in entries]),
            ("config", config),
        ]
    )


def build_cache(args):
    manifest = load_ntu_2p_diffusion_manifest(args.manifest_path)
    device = _device(args.device)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    summary = OrderedDict([("save_dir", args.save_dir), ("manifest_path", args.manifest_path)])
    summary["manifest_hash"] = manifest.get("manifest_hash")
    for split in args.splits:
        payload = build_split(args, manifest, split, converter, device)
        path = cache_file_path(args.save_dir, split)
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(payload, path)
        _write_json(path + ".json", payload["config"])
        summary[split] = OrderedDict(
            [
                ("path", path),
                ("num_sequences", payload["config"]["num_sequences"]),
                ("num_frames", payload["config"]["num_frames"]),
                ("size_mb", round(os.path.getsize(path) / 2.0 ** 20, 2)),
            ]
        )
    summary["created_at"] = _utc_now()
    _write_json(os.path.join(args.save_dir, "cache_summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--train_data_path", default=None, help="缺省取 manifest source_paths")
    parser.add_argument("--test_data_path", default=None, help="缺省取 manifest source_paths")
    parser.add_argument("--save_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_interval", type=int, default=200)
    return parser


if __name__ == "__main__":
    build_cache(build_arg_parser().parse_args())
