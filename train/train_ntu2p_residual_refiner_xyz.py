"""训练继承 independent single-person baseline 的 NTU 双人 xyz residual refiner。"""

import argparse
import json
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
    horizon_slices,
    interaction_pair_distances,
    local_pose,
    root_positions,
    velocity_with_last_obs,
)


DATASET = "ntu120_2p"


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


def _loss_terms(pred, delta, target, obs, args):
    loss = torch.nn.functional.mse_loss(pred, target)
    loss = loss + float(args.mae_loss_weight) * torch.nn.functional.l1_loss(pred, target)
    if args.root_loss_weight > 0:
        loss = loss + float(args.root_loss_weight) * torch.nn.functional.mse_loss(root_positions(pred), root_positions(target))
    if args.local_pose_loss_weight > 0:
        loss = loss + float(args.local_pose_loss_weight) * torch.nn.functional.mse_loss(local_pose(pred), local_pose(target))
    if args.velocity_loss_weight > 0:
        loss = loss + float(args.velocity_loss_weight) * torch.nn.functional.mse_loss(
            velocity_with_last_obs(pred, obs), velocity_with_last_obs(target, obs)
        )
    if args.acceleration_loss_weight > 0:
        loss = loss + float(args.acceleration_loss_weight) * torch.nn.functional.mse_loss(
            acceleration_with_last_obs(pred, obs), acceleration_with_last_obs(target, obs)
        )
    _, _, long_slice = horizon_slices(args.pred_len)
    if args.long_loss_weight > 0:
        loss = loss + float(args.long_loss_weight) * torch.nn.functional.mse_loss(pred[:, long_slice], target[:, long_slice])
    if args.final_frame_loss_weight > 0:
        loss = loss + float(args.final_frame_loss_weight) * torch.nn.functional.mse_loss(pred[:, -1], target[:, -1])
    if args.inter_loss_weight > 0:
        pred_dist = interaction_pair_distances(pred)
        target_dist = interaction_pair_distances(target)
        loss = loss + float(args.inter_loss_weight) * torch.nn.functional.mse_loss(pred_dist, target_dist)
    delta_reg = (delta * delta).mean()
    loss = loss + float(args.delta_reg_weight) * delta_reg
    return loss, delta_reg


def _save_checkpoint(args, model, optimizer, step):
    path = os.path.join(args.save_dir, "model{:09d}.pt".format(int(step)))
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
    torch.save(
        {"optimizer_state_dict": optimizer.state_dict(), "step": int(step)},
        os.path.join(args.save_dir, "opt{:09d}.pt".format(int(step))),
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
    ).to(device)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
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

    step = 0
    checkpoint = None
    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break
            model.train()
            obs_xyz, target_xyz, action = _batch_xyz(batch, converter, device)
            pred, base, delta = model(obs_xyz, action, return_details=True)
            loss, delta_reg = _loss_terms(pred, delta, target_xyz, obs_xyz, args)
            if not torch.isfinite(loss):
                raise ValueError("训练 loss 非有限")
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            step += 1
            record = OrderedDict(
                [
                    ("step", int(step)),
                    ("train_loss", float(loss.detach().cpu().item())),
                    ("delta_reg", float(delta_reg.detach().cpu().item())),
                    ("alpha", float(model.alpha.detach().cpu().item())),
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
            _append_log(log_path, record)
    if checkpoint is None:
        checkpoint = _save_checkpoint(args, model, optimizer, step)
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
    parser.add_argument("--lr", type=float, default=3e-4)
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
