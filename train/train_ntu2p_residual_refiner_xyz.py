"""训练继承 independent single-person baseline 的 NTU 双人 xyz residual refiner。"""

import argparse
import copy
import json
import math
import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import (
    NTU2PDiffusionForecastDataset,
    ntu_2p_diffusion_collate,
)
from model.forecasting_ntu2p_residual_xyz import (
    MODEL_TYPE,
    NTU2PResidualRefinerXYZ,
    count_parameters,
    load_base_model_from_checkpoint,
)
from model.rotation2xyz import Rotation2xyz_x
from utils.fixseed import fixseed
from utils.ntu_2p_rot6d import ntu_2p_rot6d_to_xyz
from utils.ntu_smplx_2p_xyz import (
    acceleration_with_last_obs,
    check_ntu_xyz,
    copy_last_xyz,
    dct_band_energies,
    horizon_slices,
    interaction_pair_distances,
    local_pose,
    root_positions,
    velocity_with_last_obs,
)


DATASET = "ntu120_2p"

# 顺序即求和顺序；改变顺序会改变浮点结果，破坏与历史 run 的逐位等价。
LOSS_TERM_KEYS = (
    "mse",
    "mae",
    "root",
    "local",
    "velocity",
    "acceleration",
    "long",
    "final",
    "inter",
    "local_velocity",
    "articulation_energy",
    "temporal_std",
    "dct_low_amplitude",
    "dct_mid_amplitude",
)


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
        raise ValueError("本训练入口必须使用 CUDA，例如 --device cuda:0，当前为 {}".format(value))
    if not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但 CUDA 不可用: {}".format(value))
    index = torch.cuda.current_device() if requested.index is None else int(requested.index)
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("CUDA 设备索引无效: {}".format(index))
    requested = torch.device("cuda:{}".format(index))
    torch.cuda.set_device(requested)
    return requested


def _dataset(args, split):
    return NTU2PDiffusionForecastDataset(
        manifest_path=args.manifest_path,
        split=split,
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples if split == "train" else args.eval_max_samples,
        seed=args.seed,
    )


def _loader(args, dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size if shuffle else args.eval_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=ntu_2p_diffusion_collate,
        drop_last=False,
    )


def _batch_xyz(batch, converter, device):
    obs_xyz = ntu_2p_rot6d_to_xyz(batch["obs_motion"].to(device), converter=converter)
    target_xyz = ntu_2p_rot6d_to_xyz(batch["future"].to(device), converter=converter)
    action = batch["action"].to(device)
    check_ntu_xyz("obs_xyz", obs_xyz, seq_len=10, num_persons=2)
    check_ntu_xyz("target_xyz", target_xyz, seq_len=50, num_persons=2)
    return obs_xyz, target_xyz, action


def _local_pose_velocity_with_last_obs(value, obs):
    full = torch.cat((obs[:, -1:], value), dim=1)
    pose = local_pose(full)
    return pose[:, 1:] - pose[:, :-1]


def _articulation_energy(value, obs):
    # 每个 (person, joint) 的局部姿态帧间摆动能量，对时间取均值后与相位无关，用于对抗均值坍缩。
    velocity = _local_pose_velocity_with_last_obs(value, obs)
    return (velocity * velocity).sum(dim=-1).mean(dim=1)


def _term_weights(args):
    return OrderedDict(
        [
            ("mse", 1.0),
            ("mae", float(args.mae_loss_weight)),
            ("root", float(args.root_loss_weight)),
            ("local", float(args.local_pose_loss_weight)),
            ("velocity", float(args.velocity_loss_weight)),
            ("acceleration", float(args.acceleration_loss_weight)),
            ("long", float(args.long_loss_weight)),
            ("final", float(args.final_frame_loss_weight)),
            ("inter", float(args.inter_loss_weight)),
            ("local_velocity", float(getattr(args, "local_velocity_loss_weight", 0.0))),
            ("articulation_energy", float(getattr(args, "articulation_energy_loss_weight", 0.0))),
            ("temporal_std", float(getattr(args, "temporal_std_loss_weight", 0.0))),
            ("dct_low_amplitude", float(getattr(args, "dct_low_amplitude_loss_weight", 0.0))),
            ("dct_mid_amplitude", float(getattr(args, "dct_mid_amplitude_loss_weight", 0.0))),
        ]
    )


def _raw_terms(pred, target, obs, args, keys):
    mse = torch.nn.functional.mse_loss
    _, _, long_slice = horizon_slices(args.pred_len)
    terms = OrderedDict()
    for key in keys:
        if key == "mse":
            terms[key] = mse(pred, target)
        elif key == "mae":
            terms[key] = torch.nn.functional.l1_loss(pred, target)
        elif key == "root":
            terms[key] = mse(root_positions(pred), root_positions(target))
        elif key == "local":
            terms[key] = mse(local_pose(pred), local_pose(target))
        elif key == "velocity":
            terms[key] = mse(velocity_with_last_obs(pred, obs), velocity_with_last_obs(target, obs))
        elif key == "acceleration":
            terms[key] = mse(acceleration_with_last_obs(pred, obs), acceleration_with_last_obs(target, obs))
        elif key == "long":
            terms[key] = mse(pred[:, long_slice], target[:, long_slice])
        elif key == "final":
            terms[key] = mse(pred[:, -1], target[:, -1])
        elif key == "inter":
            terms[key] = mse(interaction_pair_distances(pred), interaction_pair_distances(target))
        elif key == "local_velocity":
            terms[key] = mse(_local_pose_velocity_with_last_obs(pred, obs), _local_pose_velocity_with_last_obs(target, obs))
        elif key == "articulation_energy":
            terms[key] = mse(_articulation_energy(pred, obs), _articulation_energy(target, obs))
        elif key == "temporal_std":
            # 幅度型（开根号）损失：能量型在 Δ≈0 处梯度消失，幅度型梯度不消失（Stage 1 诊断）。
            terms[key] = mse(local_pose(pred).std(dim=1), local_pose(target).std(dim=1))
        elif key in ("dct_low_amplitude", "dct_mid_amplitude"):
            band = key.split("_")[1]
            eps = float(getattr(args, "amplitude_eps", 1e-6))
            pred_amplitude = torch.sqrt(dct_band_energies(pred)[band] + eps)
            target_amplitude = torch.sqrt(dct_band_energies(target)[band] + eps)
            terms[key] = mse(pred_amplitude, target_amplitude)
        else:
            raise ValueError("未知 loss 项: {}".format(key))
    return terms


def _loss_terms(pred, delta, target, obs, args, scales=None):
    """scales 为 None 时与历史实现逐位等价；否则每项除以 copy-last 在训练集上的同名误差。"""
    weights = _term_weights(args)
    active = [key for key in LOSS_TERM_KEYS if weights[key] > 0]
    terms = _raw_terms(pred, target, obs, args, active)
    loss = None
    for key in active:
        term = terms[key] if scales is None else terms[key] / float(scales[key])
        loss = weights[key] * term if loss is None else loss + weights[key] * term
    delta_reg = (delta * delta).mean()
    loss = loss + float(args.delta_reg_weight) * delta_reg
    return loss, delta_reg


def _estimate_copy_last_scales(args, converter, device):
    """以 copy-last 的各项训练集误差作归一化常量，使每个 loss 项都表示'相对 copy-last 的比例'。"""
    dataset = _dataset(args, "train")
    loader = _loader(args, dataset, shuffle=False)
    sums = OrderedDict((key, 0.0) for key in LOSS_TERM_KEYS)
    count = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.scale_estimate_batches > 0 and index >= args.scale_estimate_batches:
                break
            obs_xyz, target_xyz, _ = _batch_xyz(batch, converter, device)
            copy_xyz = copy_last_xyz(obs_xyz, args.pred_len)
            terms = _raw_terms(copy_xyz, target_xyz, obs_xyz, args, LOSS_TERM_KEYS)
            batch_size = int(obs_xyz.shape[0])
            for key, value in terms.items():
                sums[key] += float(value.detach().cpu().item()) * batch_size
            count += batch_size
    if count <= 0:
        raise ValueError("归一化常量估计样本数为 0")
    return OrderedDict((key, max(value / float(count), 1e-8)) for key, value in sums.items())


def _lr_at(args, step):
    """第 step 次更新（0 起）使用的学习率；cosine_tail 只在最后一段衰减，前段与恒定学习率逐位相同。"""
    start = int(round(args.num_steps * args.lr_decay_start_frac))
    if step < start:
        return args.lr
    progress = float(step - start) / float(max(1, args.num_steps - start))
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * progress))


def _ema_model(model):
    ema = copy.deepcopy(model)
    ema.eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    return ema


@torch.no_grad()
def _update_ema(ema, model, decay):
    # 冻结的 base 参数与常量 buffer 直接复制，只对可训练参数做滑动平均。
    for ema_param, param in zip(ema.parameters(), model.parameters()):
        if param.requires_grad:
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        else:
            ema_param.copy_(param.detach())
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _save_checkpoint(args, model, optimizer, step, save_dir=None):
    save_dir = save_dir or args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model{:09d}.pt".format(int(step)))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": MODEL_TYPE,
            "model_config": model.config(),
            "baseline_checkpoint": args.baseline_checkpoint,
            "representation": "ntu2p_residual_refiner_xyz",
            "protocol": "ntu120_2p_o10_p50",
            "manifest_path": args.manifest_path,
            "step": int(step),
            "seed": int(args.seed),
            "num_params": int(args.num_params),
            "device": str(args.device),
            "created_at": _utc_now(),
        },
        path,
    )
    if optimizer is not None:
        torch.save(
            {"optimizer_state_dict": optimizer.state_dict(), "step": int(step)},
            os.path.join(save_dir, "opt{:09d}.pt".format(int(step))),
        )
    return path


def _append_log(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args):
    if args.dataset != DATASET:
        raise ValueError("dataset 必须是 {}".format(DATASET))
    if args.window_len != 60 or args.obs_len != 10 or args.pred_len != 50:
        raise ValueError("当前协议固定 window_len=60, obs_len=10, pred_len=50")
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")

    fixseed(args.seed)
    device = _device(args.device)
    args.device = str(device)
    os.makedirs(args.save_dir, exist_ok=True)
    train_dataset = _dataset(args, "train")
    train_loader = _loader(args, train_dataset, shuffle=True)
    base_model, base_state = load_base_model_from_checkpoint(args.baseline_checkpoint, device)
    model = NTU2PResidualRefinerXYZ(
        base_model=base_model,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        num_actions=args.num_actions,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        alpha=1.0,
        freeze_base=not args.unfreeze_base,
        ramp_mode=args.ramp_mode,
        ramp_saturate_frames=args.ramp_saturate_frames,
        future_pos_mode=args.future_pos_mode,
    ).to(device)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    # 用独立的数据集实例估计常量，避免消耗训练集的随机窗口采样流。
    loss_scales = _estimate_copy_last_scales(args, converter, device) if args.loss_scale_normalize else None
    args.loss_scales = loss_scales
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.num_params = count_parameters(model)
    args.base_num_params = sum(param.numel() for param in model.base_model.parameters())
    args.base_checkpoint_step = int(base_state.get("step", -1))
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))
    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    print(
        "Training NTU2P residual refiner: params={} base_params={} device={} base_step={} freeze_base={}".format(
            args.num_params,
            args.base_num_params,
            device,
            args.base_checkpoint_step,
            model.freeze_base,
        )
    )

    ema = _ema_model(model) if args.ema_decay > 0 else None
    ema_dir = os.path.join(args.save_dir, "ema")
    step = 0
    checkpoint = None
    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break
            model.train()
            obs_xyz, target_xyz, action = _batch_xyz(batch, converter, device)
            pred, base, delta = model(obs_xyz, action, return_details=True)
            loss, delta_reg = _loss_terms(pred, delta, target_xyz, obs_xyz, args, scales=loss_scales)
            if not torch.isfinite(loss):
                raise ValueError("训练 loss 非有限")
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            if args.lr_schedule != "constant":
                for group in optimizer.param_groups:
                    group["lr"] = _lr_at(args, step)
            optimizer.step()
            if ema is not None:
                _update_ema(ema, model, args.ema_decay)
            step += 1
            record = OrderedDict(
                [
                    ("step", int(step)),
                    ("train_loss", float(loss.detach().cpu().item())),
                    ("delta_reg", float(delta_reg.detach().cpu().item())),
                    ("alpha", float(model.alpha.detach().cpu().item())),
                    ("lr", float(optimizer.param_groups[0]["lr"])),
                    ("device", str(device)),
                    ("base_checkpoint_step", int(args.base_checkpoint_step)),
                    ("created_at", _utc_now()),
                ]
            )
            if step == 1 or step % args.log_interval == 0:
                print(
                    "step[{}]: loss[{:.6f}] delta_reg[{:.6f}] alpha[{:.6f}]".format(
                        step, record["train_loss"], record["delta_reg"], record["alpha"]
                    )
                )
            if step % args.save_interval == 0 or step == args.num_steps:
                checkpoint = _save_checkpoint(args, model, optimizer, step)
                record["checkpoint"] = checkpoint
                if ema is not None:
                    record["ema_checkpoint"] = _save_checkpoint(args, ema, None, step, save_dir=ema_dir)
            _append_log(log_path, record)
    if checkpoint is None:
        checkpoint = _save_checkpoint(args, model, optimizer, step)
        if ema is not None:
            _save_checkpoint(args, ema, None, step, save_dir=ema_dir)
    print("Training finished. final_checkpoint={}".format(checkpoint))
    return checkpoint


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--test_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--num_actions", type=int, default=26)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--velocity_loss_weight", type=float, default=0.2)
    parser.add_argument("--acceleration_loss_weight", type=float, default=0.1)
    parser.add_argument("--mae_loss_weight", type=float, default=0.1)
    parser.add_argument("--root_loss_weight", type=float, default=1.0)
    parser.add_argument("--local_pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--long_loss_weight", type=float, default=0.2)
    parser.add_argument("--final_frame_loss_weight", type=float, default=0.2)
    parser.add_argument("--delta_reg_weight", type=float, default=0.01)
    parser.add_argument("--inter_loss_weight", type=float, default=0.0)
    # 以下为摆动恢复实验新增项，默认值均保持历史行为。
    parser.add_argument("--local_velocity_loss_weight", type=float, default=0.0)
    parser.add_argument("--articulation_energy_loss_weight", type=float, default=0.0)
    parser.add_argument("--temporal_std_loss_weight", type=float, default=0.0)
    parser.add_argument("--dct_low_amplitude_loss_weight", type=float, default=0.0)
    parser.add_argument("--dct_mid_amplitude_loss_weight", type=float, default=0.0)
    parser.add_argument("--amplitude_eps", type=float, default=1e-6)
    parser.add_argument("--loss_scale_normalize", action="store_true")
    parser.add_argument("--scale_estimate_batches", type=int, default=0, help="0 表示遍历整个训练集")
    parser.add_argument("--ramp_mode", choices=("linear", "saturate"), default="linear")
    parser.add_argument("--ramp_saturate_frames", type=int, default=5)
    parser.add_argument("--future_pos_mode", choices=("learned_zero", "sinusoidal"), default="learned_zero")
    parser.add_argument("--lr", type=float, default=3e-4)
    # 降低终点抖动的两个开关，默认关闭时训练逐位不变。
    parser.add_argument("--lr_schedule", choices=("constant", "cosine_tail"), default="constant")
    parser.add_argument("--lr_decay_start_frac", type=float, default=0.8)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--ema_decay", type=float, default=0.0, help="0 表示关闭；影子权重存到 save_dir/ema/")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unfreeze_base", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
