"""训练参数共享的独立单人 NTU xyz 预测 baseline。"""

import argparse
import json
import os
from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import (
    NTU2PDiffusionForecastDataset,
    ntu_2p_diffusion_collate,
)
from eval.eval_ntu_2p_forecasting_diffusion import evaluate_ntu2p_forecasting_diffusion
from model.forecasting_ntu_xyz import NTULabelXYZTransformer, count_parameters
from model.rotation2xyz import Rotation2xyz_x
from utils.fixseed import fixseed
from utils.ntu_2p_rot6d import ntu_2p_rot6d_to_xyz


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _device(value):
    requested = torch.device(value)
    if requested.type != "cuda":
        raise ValueError("独立单人 baseline 必须使用 CUDA，例如 --device cuda:0，当前为 {}".format(value))
    if not torch.cuda.is_available():
        raise RuntimeError("独立单人 baseline 指定了 CUDA，但 CUDA 不可用: {}".format(value))
    index = torch.cuda.current_device() if requested.index is None else int(requested.index)
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("CUDA 设备索引超出范围: {}，可用数量为 {}".format(index, torch.cuda.device_count()))
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


def _build_model(args):
    return NTULabelXYZTransformer(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        num_actions=args.num_actions,
        num_persons=1,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        velocity_loss_weight=args.velocity_loss_weight,
        continuity_loss_weight=args.continuity_loss_weight,
        first_step_loss_weight=args.first_step_loss_weight,
        mae_loss_weight=args.mae_loss_weight,
        root_loss_weight=args.root_loss_weight,
        local_pose_loss_weight=args.local_pose_loss_weight,
        mpjpe_loss_weight=args.mpjpe_loss_weight,
        short_loss_weight=args.short_loss_weight,
        mid_loss_weight=args.mid_loss_weight,
        long_loss_weight=args.long_loss_weight,
        final_frame_loss_weight=args.final_frame_loss_weight,
        acceleration_loss_weight=args.acceleration_loss_weight,
        relative_root_loss_weight=0.0,
        relative_velocity_loss_weight=0.0,
        key_joint_relation_loss_weight=0.0,
        contact_loss_weight=0.0,
        action_feature_loss_weight=0.0,
        action_logit_loss_weight=0.0,
    )


def _single_person_batch(batch, converter, device):
    obs_pair = ntu_2p_rot6d_to_xyz(batch["obs_motion"].to(device), converter=converter)
    target_pair = ntu_2p_rot6d_to_xyz(batch["future"].to(device), converter=converter)
    batch_size = int(obs_pair.shape[0])
    obs_single = torch.cat((obs_pair[:, :, 0:1], obs_pair[:, :, 1:2]), dim=0)
    target_single = torch.cat((target_pair[:, :, 0:1], target_pair[:, :, 1:2]), dim=0)
    action = batch["action"].to(device)
    action_single = torch.cat((action, action), dim=0)
    return obs_single, target_single, action_single, batch_size


def _save_checkpoint(args, model, optimizer, step):
    path = os.path.join(args.save_dir, "model{:09d}.pt".format(step))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model.model_type,
            "model_config": model.config(),
            "representation": "independent_single_person_xyz",
            "num_persons": 1,
            "person_shared_parameters": True,
            "protocol": "ntu120_2p_o10_p50",
            "device": str(args.device),
            "cuda_runtime": {
                "device": str(args.device),
                "device_name": torch.cuda.get_device_name(torch.device(args.device)),
                "torch_version": torch.__version__,
            },
            "manifest_path": args.manifest_path,
            "step": int(step),
            "seed": int(args.seed),
            "num_params": int(args.num_params),
        },
        path,
    )
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "step": int(step)}, os.path.join(args.save_dir, "opt{:09d}.pt".format(step)))
    return path


def _eval(args, checkpoint, split):
    eval_args = SimpleNamespace(
        mode="independent_single_person",
        manifest_path=args.manifest_path,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        split=split,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_samples=args.eval_max_samples,
        seed=args.seed,
        sample_seeds="",
        checkpoint=checkpoint,
        save_dir=args.save_dir,
        noise_schedule=None,
        timestep_respacing="",
        use_ddim=False,
        guidance_scale=1.0,
        save_arrays=False,
        save_array_limit=0,
        semantic_eval=False,
        action_classifier_path=None,
        diversity_pairs=1000,
        progress=False,
    )
    return evaluate_ntu2p_forecasting_diffusion(eval_args)


def run(args):
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    fixseed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = _device(args.device)
    args.device = str(device)
    args.cuda_runtime = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
    }
    train_dataset = _dataset(args, "train")
    train_loader = _loader(args, train_dataset, shuffle=True)
    model = _build_model(args).to(device)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.num_params = count_parameters(model)
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))
    print("Training independent single-person baseline: params={} device={}".format(args.num_params, device))

    step = 0
    checkpoint = None
    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break
            model.train()
            obs, target, action, _ = _single_person_batch(batch, converter, device)
            loss = model.training_loss(obs, target, action)
            if not torch.isfinite(loss):
                raise ValueError("独立单人训练 loss 非有限")
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_interval == 0:
                print("step[{}]: train_loss[{:.6f}]".format(step, float(loss.detach().cpu())))
            if step % args.save_interval == 0 or step == args.num_steps:
                checkpoint = _save_checkpoint(args, model, optimizer, step)
            if args.eval_interval > 0 and (step % args.eval_interval == 0 or step == args.num_steps):
                if checkpoint is None:
                    checkpoint = _save_checkpoint(args, model, optimizer, step)
                _eval(args, checkpoint, "val")
    if checkpoint is None:
        checkpoint = _save_checkpoint(args, model, optimizer, step)
    _eval(args, checkpoint, "val")
    print("Training finished. final_checkpoint={}".format(checkpoint))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--test_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--num_actions", type=int, default=26)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--decoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--velocity_loss_weight", type=float, default=0.2)
    parser.add_argument("--continuity_loss_weight", type=float, default=0.0)
    parser.add_argument("--first_step_loss_weight", type=float, default=0.0)
    parser.add_argument("--mae_loss_weight", type=float, default=0.1)
    parser.add_argument("--root_loss_weight", type=float, default=1.0)
    parser.add_argument("--local_pose_loss_weight", type=float, default=1.0)
    parser.add_argument("--mpjpe_loss_weight", type=float, default=0.0)
    parser.add_argument("--short_loss_weight", type=float, default=0.0)
    parser.add_argument("--mid_loss_weight", type=float, default=0.0)
    parser.add_argument("--long_loss_weight", type=float, default=0.2)
    parser.add_argument("--final_frame_loss_weight", type=float, default=0.2)
    parser.add_argument("--acceleration_loss_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
