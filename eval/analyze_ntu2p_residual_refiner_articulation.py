"""定量对比 NTU2P residual refiner / base / copy-last / GT 的关节摆动幅度，验证"回归到均值"诊断。"""

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
    check_ntu_xyz,
    copy_last_xyz,
    local_pose,
    root_positions,
    velocity_with_last_obs,
)


def _device(value):
    requested = torch.device(value)
    if requested.type != "cuda":
        raise ValueError("本分析入口必须使用 CUDA，例如 --device cuda:0")
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


def _local_pose_velocity(value):
    pose = local_pose(value)
    return pose[:, 1:] - pose[:, :-1]


def _root_velocity(value):
    root = root_positions(value)
    return root[:, 1:] - root[:, :-1]


def _sum_sq(tensor):
    return float((tensor ** 2).sum().detach().cpu().item()), int(tensor.numel())


def _below_threshold_count(tensor, threshold):
    magnitude = torch.norm(tensor, dim=-1)
    return int((magnitude < threshold).sum().detach().cpu().item()), int(magnitude.numel())


def evaluate(args):
    device = _device(args.device)
    fixseed(args.seed)
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

    variants = ("model", "base", "copy_last", "target")
    articulation_sum = {key: 0.0 for key in variants}
    articulation_count = {key: 0 for key in variants}
    root_sum = {key: 0.0 for key in variants}
    root_count = {key: 0 for key in variants}
    # 局部姿态沿时间轴的 std 比帧差能量对 GT 抖动更鲁棒，作为第二个摆动幅度口径。
    temporal_std_sum = {key: 0.0 for key in variants}
    temporal_std_count = {key: 0 for key in variants}
    # 位置 MSE 拆成 root 与 local_pose 两部分，用于判断 L2 增益来自哪一分量。
    pos_mse_sum = {key: {"total": 0.0, "root": 0.0, "local": 0.0} for key in variants}
    pos_mse_count = {key: {"total": 0, "root": 0, "local": 0} for key in variants}
    # 全局速度 MSE（与训练 velocity loss 同口径），用于量化该项在总 loss 中的真实数值权重。
    global_vel_mse_sum = {key: 0.0 for key in variants}
    global_vel_mse_count = {key: 0 for key in variants}

    target_step_magnitudes = []
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            obs_xyz = ntu_2p_rot6d_to_xyz(batch["obs_motion"].to(device), converter=converter)
            target_xyz = ntu_2p_rot6d_to_xyz(batch["future"].to(device), converter=converter)
            action = batch["action"].to(device)
            pred_xyz, base_xyz, _ = model(obs_xyz, action, return_details=True)
            copy_xyz = copy_last_xyz(obs_xyz, args.pred_len)
            check_ntu_xyz("pred_xyz", pred_xyz, seq_len=args.pred_len, num_persons=2)

            values = {
                "model": pred_xyz,
                "base": base_xyz,
                "copy_last": copy_xyz,
                "target": target_xyz,
            }
            for key, value in values.items():
                pose_vel = _local_pose_velocity(value)
                sum_sq, count = _sum_sq(pose_vel)
                articulation_sum[key] += sum_sq
                articulation_count[key] += count

                root_vel = _root_velocity(value)
                sum_sq, count = _sum_sq(root_vel)
                root_sum[key] += sum_sq
                root_count[key] += count

                pose_std = local_pose(value).std(dim=1)
                temporal_std_sum[key] += float(pose_std.sum().detach().cpu().item())
                temporal_std_count[key] += int(pose_std.numel())

                sum_sq, count = _sum_sq(value - target_xyz)
                pos_mse_sum[key]["total"] += sum_sq
                pos_mse_count[key]["total"] += count
                sum_sq, count = _sum_sq(root_positions(value) - root_positions(target_xyz))
                pos_mse_sum[key]["root"] += sum_sq
                pos_mse_count[key]["root"] += count
                sum_sq, count = _sum_sq(local_pose(value) - local_pose(target_xyz))
                pos_mse_sum[key]["local"] += sum_sq
                pos_mse_count[key]["local"] += count

                sum_sq, count = _sum_sq(
                    velocity_with_last_obs(value, obs_xyz) - velocity_with_last_obs(target_xyz, obs_xyz)
                )
                global_vel_mse_sum[key] += sum_sq
                global_vel_mse_count[key] += count

            target_pose_vel = _local_pose_velocity(target_xyz)
            target_step_magnitudes.append(torch.norm(target_pose_vel, dim=-1).reshape(-1).detach().cpu())
            num_samples += int(obs_xyz.shape[0])

    articulation_energy = OrderedDict(
        (key, articulation_sum[key] / max(articulation_count[key], 1)) for key in variants
    )
    root_energy = OrderedDict((key, root_sum[key] / max(root_count[key], 1)) for key in variants)
    temporal_std = OrderedDict(
        (key, temporal_std_sum[key] / max(temporal_std_count[key], 1)) for key in variants
    )
    pos_mse = OrderedDict(
        (
            key,
            OrderedDict(
                (part, pos_mse_sum[key][part] / max(pos_mse_count[key][part], 1))
                for part in ("total", "root", "local")
            ),
        )
        for key in variants
    )
    global_vel_mse = OrderedDict(
        (key, global_vel_mse_sum[key] / max(global_vel_mse_count[key], 1)) for key in variants
    )

    target_step_magnitudes = torch.cat(target_step_magnitudes, dim=0)
    epsilon = float(target_step_magnitudes.median().item()) * args.epsilon_ratio

    # 第二遍推理计算 frozen-frame 比例（epsilon 依赖第一遍算出的 GT 位移中位数，故分两遍，避免保存全部张量占用显存）。
    frozen_ratio = OrderedDict()
    frozen_below = {key: 0 for key in variants}
    frozen_total = {key: 0 for key in variants}
    with torch.no_grad():
        for batch in loader:
            obs_xyz = ntu_2p_rot6d_to_xyz(batch["obs_motion"].to(device), converter=converter)
            target_xyz = ntu_2p_rot6d_to_xyz(batch["future"].to(device), converter=converter)
            action = batch["action"].to(device)
            pred_xyz, base_xyz, _ = model(obs_xyz, action, return_details=True)
            copy_xyz = copy_last_xyz(obs_xyz, args.pred_len)
            values = {
                "model": pred_xyz,
                "base": base_xyz,
                "copy_last": copy_xyz,
                "target": target_xyz,
            }
            for key, value in values.items():
                pose_vel = _local_pose_velocity(value)
                below, total = _below_threshold_count(pose_vel, epsilon)
                frozen_below[key] += below
                frozen_total[key] += total

    for key in variants:
        frozen_ratio[key] = frozen_below[key] / max(frozen_total[key], 1)

    result = OrderedDict(
        [
            ("checkpoint", args.checkpoint),
            ("manifest_path", args.manifest_path),
            ("split", args.split),
            ("num_samples", int(num_samples)),
            ("epsilon_ratio", float(args.epsilon_ratio)),
            ("epsilon_meters", float(epsilon)),
            ("articulation_energy_local_pose_velocity_sq", articulation_energy),
            ("root_energy_root_velocity_sq", root_energy),
            ("articulation_energy_ratio_to_target", OrderedDict(
                (key, articulation_energy[key] / articulation_energy["target"]) for key in variants
            )),
            ("root_energy_ratio_to_target", OrderedDict(
                (key, root_energy[key] / root_energy["target"]) for key in variants
            )),
            ("local_pose_temporal_std_mean", temporal_std),
            ("local_pose_temporal_std_ratio_to_target", OrderedDict(
                (key, temporal_std[key] / temporal_std["target"]) for key in variants
            )),
            ("position_mse_decomposition", pos_mse),
            ("global_velocity_mse_vs_target", global_vel_mse),
            ("frozen_joint_frame_ratio_below_epsilon", frozen_ratio),
        ]
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
    parser.add_argument("--epsilon_ratio", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
