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
from utils.ntu_smplx_2p_xyz import (
    articulation_ratios,
    check_ntu_xyz,
    compute_ntu_articulation_metrics,
    compute_ntu_xyz_metrics,
    copy_last_xyz,
)


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
    variants = ("model", "base", "copy_last")
    totals = {key: OrderedDict() for key in variants}
    articulation_totals = {key: OrderedDict() for key in variants}
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
            for key, value in (("model", pred_xyz), ("base", base_xyz), ("copy_last", copy_xyz)):
                _add(totals[key], compute_ntu_xyz_metrics(value, target_xyz, obs_xyz), batch_size)
                _add(articulation_totals[key], compute_ntu_articulation_metrics(value, target_xyz), batch_size)
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
            ("articulation_metrics", OrderedDict()),
            ("articulation_gate", OrderedDict()),
            ("checkpoint_step", int(checkpoint_state.get("step", -1))),
        ]
    )
    for key in ("xyz_mse", "xyz_mae", "mpjpe"):
        result["beats_base"][key] = result["model_metrics"][key] < result["base_metrics"][key]
        result["beats_copy_last"][key] = result["model_metrics"][key] < result["copy_last_metrics"][key]
    for key in variants:
        aggregated = _finalize(articulation_totals[key], count)
        aggregated.update(articulation_ratios(aggregated))
        result["articulation_metrics"][key] = aggregated

    # 新 gate：L2 三项必须低于 copy-last（不变），且位置域 DCT 分频带能量比值达阈值，mpjpe 相对 base 回退不超过容忍度。
    # 帧差能量/frozen 比例受 GT 拟合抖动主导（Stage 1 诊断），仅作报告不进 gate。
    model_artic = result["articulation_metrics"]["model"]
    mpjpe_regression = result["model_metrics"]["mpjpe"] / result["base_metrics"]["mpjpe"] - 1.0
    gate = result["articulation_gate"]
    gate["dct_low_ratio_threshold"] = float(args.articulation_gate_low_ratio)
    gate["dct_mid_ratio_threshold"] = float(args.articulation_gate_mid_ratio)
    gate["base_mpjpe_regression_tolerance"] = float(args.base_mpjpe_regression_tolerance)
    gate["model_dct_low_ratio"] = float(model_artic["dct_low_energy_ratio_to_target"])
    gate["model_dct_mid_ratio"] = float(model_artic["dct_mid_energy_ratio_to_target"])
    gate["model_dct_high_ratio"] = float(model_artic["dct_high_energy_ratio_to_target"])
    gate["model_energy_ratio"] = float(model_artic["articulation_energy_ratio_to_target"])
    gate["model_frozen_ratio"] = float(model_artic["frozen_ratio"])
    gate["model_mpjpe_regression_vs_base"] = float(mpjpe_regression)
    gate["passes_l2_gate"] = all(result["beats_copy_last"].values())
    gate["passes_articulation_gate"] = (
        gate["model_dct_low_ratio"] >= gate["dct_low_ratio_threshold"]
        and gate["model_dct_mid_ratio"] >= gate["dct_mid_ratio_threshold"]
    )
    gate["within_base_tolerance"] = mpjpe_regression <= gate["base_mpjpe_regression_tolerance"]
    gate["passes_full_gate"] = (
        gate["passes_l2_gate"] and gate["passes_articulation_gate"] and gate["within_base_tolerance"]
    )
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
    parser.add_argument("--articulation_gate_low_ratio", type=float, default=0.40)
    parser.add_argument("--articulation_gate_mid_ratio", type=float, default=0.10)
    parser.add_argument("--base_mpjpe_regression_tolerance", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
