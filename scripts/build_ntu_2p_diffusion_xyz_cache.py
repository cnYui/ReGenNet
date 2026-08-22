import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import (
    NTU2PDiffusionForecastDataset,
    ensure_ntu_2p_diffusion_manifest,
    ntu_2p_diffusion_collate,
)
from model.rotation2xyz import Rotation2xyz_x
from utils.ntu_2p_rot6d import ntu_2p_rot6d_to_xyz


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")


def _build_dataset(args, split):
    return NTU2PDiffusionForecastDataset(
        manifest_path=args.manifest_path,
        split=split,
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )


def _convert_split(args, split, converter, device):
    dataset = _build_dataset(args, split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ntu_2p_diffusion_collate,
        drop_last=False,
    )
    obs_items = []
    target_items = []
    action_items = []
    meta = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            obs_rot6d = batch["obs_motion"].to(device)
            future_rot6d = batch["future"].to(device)
            obs_xyz = ntu_2p_rot6d_to_xyz(obs_rot6d, converter=converter)
            target_xyz = ntu_2p_rot6d_to_xyz(future_rot6d, converter=converter)
            obs_items.append(obs_xyz.cpu())
            target_items.append(target_xyz.cpu())
            action_items.append(batch["action"].cpu())
            meta.extend(batch["meta"])
            if batch_idx == 0 or (batch_idx + 1) % args.log_interval == 0:
                print("split={} converted_batches={} samples={}".format(split, batch_idx + 1, len(meta)))
    if len(meta) != len(dataset):
        raise AssertionError("{} cache 样本数应为 {}，实际为 {}".format(split, len(dataset), len(meta)))
    payload = {
        "obs_xyz": torch.cat(obs_items, dim=0),
        "target_xyz": torch.cat(target_items, dim=0),
        "actions": torch.cat(action_items, dim=0),
        "meta": meta,
        "config": OrderedDict(
            [
                ("dataset", "ntu120_2p_smplx"),
                ("representation", "two_person_rot6d_to_xyz"),
                ("manifest_path", args.manifest_path),
                ("split", split),
                ("window_len", args.window_len),
                ("obs_len", args.obs_len),
                ("pred_len", args.pred_len),
                ("num_samples", len(meta)),
                ("created_at", _utc_now()),
            ]
        ),
    }
    return payload


def build_cache(args):
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    if args.window_len != 60 or args.obs_len != 10 or args.pred_len != 50:
        raise ValueError("本 cache builder 固定 window_len=60, obs_len=10, pred_len=50")
    os.makedirs(args.save_dir, exist_ok=True)
    if args.manifest_path is None:
        args.manifest_path = os.path.join(args.save_dir, "manifest_seed{}.json".format(args.seed))
    manifest = ensure_ntu_2p_diffusion_manifest(
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        manifest_path=args.manifest_path,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite_manifest,
    )
    device = _device()
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    summary = OrderedDict()
    summary["save_dir"] = args.save_dir
    summary["manifest_path"] = args.manifest_path
    summary["manifest_hash"] = manifest.get("manifest_hash")
    summary["split_summary"] = manifest.get("split_summary")
    summary["created_at"] = _utc_now()
    for split in ("train", "val", "test"):
        payload = _convert_split(args, split, converter, device)
        path = os.path.join(args.save_dir, "{}_xyz.pt".format(split))
        torch.save(payload, path)
        _write_json(path + ".json", payload["config"])
        summary["{}_cache".format(split)] = path
        summary["{}_count".format(split)] = int(payload["config"]["num_samples"])
    _write_json(os.path.join(args.save_dir, "cache_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--test_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--save_dir", default="results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_xyz_cache")
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite_manifest", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    return parser


def main():
    build_cache(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
