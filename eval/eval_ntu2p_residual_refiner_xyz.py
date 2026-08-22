"""评估 NTU 双人 baseline-plus-residual xyz 模型。"""

import argparse
import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import (
    NTU2PDiffusionForecastDataset,
    ntu_2p_diffusion_collate,
)
from model.forecasting_ntu2p_residual_xyz import load_ntu2p_residual_refiner_checkpoint
from model.rotation2xyz import Rotation2xyz_x
from utils.fixseed import fixseed
from utils.ntu_2p_rot6d import ntu_2p_rot6d_to_xyz
from utils.ntu_smplx_2p_xyz import check_ntu_xyz, compute_ntu_xyz_metrics, copy_last_xyz


def _device(value):
    requested = torch.device(value)
    if requested.type != "cuda":
        raise ValueError("本评估入口必须使用 CUDA，例如 --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但 CUDA 不可用")
    index = torch.cuda.current_device() if requested.index is None else int(requested.index)
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("CUDA 设备索引无效: {}".format(index))
    requested = torch.device("cuda:{}".format(index))
    torch.cuda.set_device(requested)
    return requested


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _dataset(args):
    return NTU2PDiffusionForecastDataset(
        manifest_path=args.manifest_path,
        split=args.split,
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )


def _add(total, metrics, count):
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value) * float(count)


def _finalize(total, count):
    if int(count) <= 0:
        raise ValueError("评估样本数必须大于 0")
    return OrderedDict((key, value / float(count)) for key, value in total.items())


def evaluate(args):
    device = _device(args.device)
    fixseed(args.seed)
    dataset = _dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ntu_2p_diffusion_collate,
        drop_last=False,
    )
    model, checkpoint_state = load_ntu2p_residual_refiner_checkpoint(args.checkpoint, device)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    totals = {
        "model": OrderedDict(),
        "base": OrderedDict(),
        "copy_last": OrderedDict(),
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            obs_xyz = ntu_2p_rot6d_to_xyz(batch["obs_motion"].to(device), converter=converter)
            target_xyz = ntu_2p_rot6d_to_xyz(batch["future"].to(device), converter=converter)
            action = batch["action"].to(device)
            pred_xyz, base_xyz, _ = model(obs_xyz, action, return_details=True)
            copy_xyz = copy_last_xyz(obs_xyz, args.pred_len)
            check_ntu_xyz("pred_xyz", pred_xyz, seq_len=args.pred_len, num_persons=2)
            batch_size = int(obs_xyz.shape[0])
            _add(totals["model"], compute_ntu_xyz_metrics(pred_xyz, target_xyz, obs_xyz), batch_size)
            _add(totals["base"], compute_ntu_xyz_metrics(base_xyz, target_xyz, obs_xyz), batch_size)
            _add(totals["copy_last"], compute_ntu_xyz_metrics(copy_xyz, target_xyz, obs_xyz), batch_size)
            count += batch_size

    result = OrderedDict(
        [
            ("checkpoint", args.checkpoint),
            ("split", args.split),
            ("num_samples", int(count)),
            ("sample_seed", int(args.seed)),
            ("alpha", float(model.alpha.detach().cpu().item())),
            ("model_metrics", _finalize(totals["model"], count)),
            ("base_metrics", _finalize(totals["base"], count)),
            ("copy_last_metrics", _finalize(totals["copy_last"], count)),
            ("beats_base", OrderedDict()),
            ("beats_copy_last", OrderedDict()),
            ("checkpoint_step", int(checkpoint_state.get("step", -1))),
        ]
    )
    for key in ("xyz_mse", "xyz_mae", "mpjpe"):
        result["beats_base"][key] = result["model_metrics"][key] < result["base_metrics"][key]
        result["beats_copy_last"][key] = result["model_metrics"][key] < result["copy_last_metrics"][key]
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--test_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
