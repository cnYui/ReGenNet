"""导出 NTU 双人 residual refiner 的 xyz 结果，供三色骨架视频脚本使用。"""

import argparse
import json
import os
import random
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import (
    NTU2PDiffusionForecastDataset,
    ntu_2p_diffusion_collate,
)
from eval.eval_ntu2p_residual_refiner_xyz import _device
from model.forecasting_ntu2p_residual_xyz import load_ntu2p_residual_refiner_checkpoint
from model.rotation2xyz import Rotation2xyz_x
from utils.fixseed import fixseed
from utils.ntu_2p_rot6d import ntu_2p_rot6d_to_xyz
from utils.ntu_smplx_2p_xyz import check_ntu_xyz, compute_ntu_xyz_metrics, copy_last_xyz


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _add(total, metrics, count):
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value) * float(count)


def _finalize(total, count):
    if int(count) <= 0:
        raise ValueError("评估样本数必须大于 0")
    return OrderedDict((key, value / float(count)) for key, value in total.items())


def _sample_mse(pred, target):
    diff = pred - target
    return (diff * diff).reshape(diff.shape[0], -1).mean(dim=1)


def export(args):
    fixseed(args.seed)
    device = _device(args.device)
    dataset = NTU2PDiffusionForecastDataset(
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

    arrays = {key: [] for key in ("obs_xyz", "target_xyz", "pred_xyz", "base_xyz", "copy_last_xyz", "actions")}
    meta = []
    case_scores = []
    totals = {key: {} for key in ("model", "base", "copy_last")}
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
            for name, value in (("model", pred_xyz), ("base", base_xyz), ("copy_last", copy_xyz)):
                _add(totals[name], compute_ntu_xyz_metrics(value, target_xyz, obs_xyz), batch_size)
            model_mse = _sample_mse(pred_xyz, target_xyz).detach().cpu().tolist()
            copy_mse = _sample_mse(copy_xyz, target_xyz).detach().cpu().tolist()
            for offset, (model_value, copy_value) in enumerate(zip(model_mse, copy_mse)):
                case_scores.append(
                    {
                        "index": count + offset,
                        "model_xyz_mse": float(model_value),
                        "copy_last_xyz_mse": float(copy_value),
                        "improvement": float(copy_value - model_value),
                    }
                )
            arrays["obs_xyz"].append(obs_xyz.detach().cpu())
            arrays["target_xyz"].append(target_xyz.detach().cpu())
            arrays["pred_xyz"].append(pred_xyz.detach().cpu())
            arrays["base_xyz"].append(base_xyz.detach().cpu())
            arrays["copy_last_xyz"].append(copy_xyz.detach().cpu())
            arrays["actions"].append(batch["action"].detach().cpu())
            meta.extend(batch["meta"])
            count += batch_size

    if count != len(dataset) or count != len(meta):
        raise AssertionError("导出样本数不一致: dataset={}, arrays={}, meta={}".format(len(dataset), count, len(meta)))

    combined = {key: torch.cat(value, dim=0) for key, value in arrays.items()}
    for key, value in combined.items():
        if not torch.isfinite(value.float()).all():
            raise ValueError("{} 存在非有限值".format(key))

    num_selected = min(int(args.num_visualization), count)
    if args.selection == "random":
        # 按 L2 改进选例会系统性偏向 root 大位移样本，随机选例用于还原真实分布下的观感。
        indices = sorted(random.Random(args.seed).sample(range(count), num_selected))
        case_by_index = {item["index"]: item for item in case_scores}
        selected = [case_by_index[index] for index in indices]
        selection_policy = "uniform random (seed={})".format(int(args.seed))
    else:
        case_scores.sort(key=lambda item: (-item["improvement"], item["model_xyz_mse"], item["index"]))
        selected = case_scores[:num_selected]
        indices = [item["index"] for item in selected]
        selection_policy = "descending copy_last_xyz_mse - model_xyz_mse"
    source_dir = os.path.abspath(args.output_dir)
    arrays_dir = os.path.join(source_dir, "arrays")
    os.makedirs(arrays_dir, exist_ok=True)
    selected_data = OrderedDict(
        [
            ("obs_xyz", combined["obs_xyz"][indices]),
            ("target_xyz", combined["target_xyz"][indices]),
            ("pred_xyz", combined["pred_xyz"][indices]),
            ("copy_last_xyz", combined["copy_last_xyz"][indices]),
            ("actions", combined["actions"][indices]),
            ("meta", [meta[index] for index in indices]),
        ]
    )
    torch.save(selected_data, os.path.join(arrays_dir, "ntu_label_xyz_samples.pt"))

    metrics = OrderedDict(
        [
            ("checkpoint", args.checkpoint),
            ("checkpoint_step", int(checkpoint_state.get("step", -1))),
            ("split", args.split),
            ("num_samples", int(count)),
            ("sample_seed", int(args.seed)),
            ("alpha", float(model.alpha.detach().cpu().item())),
            ("model_metrics", _finalize(totals["model"], count)),
            ("base_metrics", _finalize(totals["base"], count)),
            ("copy_last_metrics", _finalize(totals["copy_last"], count)),
            ("visualization_num_samples", len(indices)),
            ("visualization_selection", selection_policy),
            ("visualization_indices", indices),
            ("visualization_cases", selected),
        ]
    )
    _write_json(os.path.join(source_dir, "metrics_val.json" if args.split == "val" else "metrics_test.json"), metrics)
    _write_json(
        os.path.join(source_dir, "export_config.json"),
        OrderedDict(
            [
                ("checkpoint", args.checkpoint),
                ("manifest_path", args.manifest_path),
                ("split", args.split),
                ("window_len", args.window_len),
                ("obs_len", args.obs_len),
                ("pred_len", args.pred_len),
                ("device", str(device)),
                ("num_samples", count),
                ("num_visualization", len(indices)),
                ("selection_policy", metrics["visualization_selection"]),
            ]
        ),
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--test_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_visualization", type=int, default=8)
    parser.add_argument("--selection", choices=("best_improvement", "random"), default="best_improvement")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    export(build_arg_parser().parse_args())
