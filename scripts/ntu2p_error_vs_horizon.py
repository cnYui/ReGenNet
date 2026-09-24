"""NTU2P residual refiner 误差随预测帧数的曲线：mpjpe / root / local / A-B 相对 root。

用法：--checkpoint name=path 可重复；名字以 _s<seed> 结尾的 checkpoint 按前缀聚合成均值 ± 标准差。
base（冻结独立单人）取第一个 checkpoint 的 base 输出，copy-last 由观测末帧外推，二者与 seed 无关。
"""

import argparse
import json
import re
import statistics
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import NTU2PDiffusionForecastDataset, ntu_2p_diffusion_collate
from model.forecasting_ntu2p_residual_xyz import load_ntu2p_residual_refiner_checkpoint
from model.rotation2xyz import Rotation2xyz_x
from utils.fixseed import fixseed
from utils.ntu_2p_rot6d import ntu_2p_rot6d_to_xyz
from utils.ntu_smplx_2p_xyz import copy_last_xyz, local_pose, root_positions

CURVES = ("mpjpe", "root", "local", "rel_root")
TITLES = {
    "mpjpe": "MPJPE (m)",
    "root": "root position error (m)",
    "local": "local pose error (m)",
    "rel_root": "A-B relative root error (m)",
}
KEY_FRAMES = (10, 20, 30, 50)


def _frame_errors(pred, target):
    """逐帧误差之和（按 batch 求和），便于跨 batch 累加后除以样本数。"""
    rel_pred = root_positions(pred)[:, :, 0] - root_positions(pred)[:, :, 1]
    rel_target = root_positions(target)[:, :, 0] - root_positions(target)[:, :, 1]
    return {
        "mpjpe": torch.norm(pred - target, dim=-1).mean(dim=(2, 3)).sum(dim=0),
        "root": torch.norm(root_positions(pred) - root_positions(target), dim=-1).mean(dim=2).sum(dim=0),
        "local": torch.norm(local_pose(pred) - local_pose(target), dim=-1).mean(dim=(2, 3)).sum(dim=0),
        "rel_root": torch.norm(rel_pred - rel_target, dim=-1).sum(dim=0),
    }


def compute_curves(args, checkpoints):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    fixseed(args.seed)
    dataset = NTU2PDiffusionForecastDataset(
        manifest_path=args.manifest_path,
        split=args.split,
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        window_len=args.obs_len + args.pred_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=-1,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=ntu_2p_diffusion_collate)
    models = OrderedDict((name, load_ntu2p_residual_refiner_checkpoint(path, device)[0]) for name, path in checkpoints.items())
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    names = ["copy_last", "base"] + list(models)
    totals = {name: {key: torch.zeros(args.pred_len, device=device) for key in CURVES} for name in names}
    count = 0
    with torch.no_grad():
        for batch in loader:
            obs = ntu_2p_rot6d_to_xyz(batch["obs_motion"].to(device), converter=converter)
            target = ntu_2p_rot6d_to_xyz(batch["future"].to(device), converter=converter)
            action = batch["action"].to(device)
            preds = {"copy_last": copy_last_xyz(obs, args.pred_len)}
            for index, (name, model) in enumerate(models.items()):
                pred, base, _ = model(obs, action, return_details=True)
                preds[name] = pred
                if index == 0:
                    preds["base"] = base
            for name, pred in preds.items():
                for key, value in _frame_errors(pred, target).items():
                    totals[name][key] += value
            count += int(obs.shape[0])
    return count, {name: {key: (value / count).tolist() for key, value in curves.items()} for name, curves in totals.items()}


def group_curves(curves):
    groups = OrderedDict()
    for name, value in curves.items():
        groups.setdefault(re.sub(r"_s\d+$", "", name), []).append(value)
    result = OrderedDict()
    for group, members in groups.items():
        result[group] = OrderedDict([("n", len(members))])
        for key in CURVES:
            frames = list(zip(*[member[key] for member in members]))
            result[group][key] = {
                "mean": [statistics.mean(values) for values in frames],
                "std": [statistics.stdev(values) if len(values) > 1 else 0.0 for values in frames],
            }
    return result


def plot(grouped, path):
    fig, axes = plt.subplots(1, len(CURVES), figsize=(5 * len(CURVES), 4.5))
    for ax, key in zip(axes, CURVES):
        for group, value in grouped.items():
            mean, std = value[key]["mean"], value[key]["std"]
            frames = list(range(1, len(mean) + 1))
            label = "{} (n={})".format(group, value["n"]) if value["n"] > 1 else group
            line = ax.plot(frames, mean, lw=2, label=label)[0]
            if value["n"] > 1:
                lower = [m - s for m, s in zip(mean, std)]
                upper = [m + s for m, s in zip(mean, std)]
                ax.fill_between(frames, lower, upper, color=line.get_color(), alpha=0.2)
        ax.set_title(TITLES[key])
        ax.set_xlabel("future frame (20 FPS)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path，可重复")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--manifest_path", default="results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json")
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--test_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoints = OrderedDict(item.split("=", 1) for item in args.checkpoint)
    count, curves = compute_curves(args, checkpoints)
    grouped = group_curves(curves)
    with open(args.output_json, "w") as handle:
        json.dump({"split": args.split, "num_samples": count, "checkpoints": checkpoints, "curves": curves, "grouped": grouped}, handle, indent=1)
        handle.write("\n")
    plot(grouped, args.output_png)
    for group, value in grouped.items():
        cells = []
        for frame in KEY_FRAMES:
            cells.append("f{:02d} ".format(frame) + " ".join(
                "{} {:.3f}±{:.3f}".format(key, value[key]["mean"][frame - 1], value[key]["std"][frame - 1]) for key in CURVES
            ))
        print("{:<20s} n={} | {}".format(group, value["n"], " | ".join(cells)))


if __name__ == "__main__":
    main()
