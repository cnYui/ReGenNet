import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loaders.forecasting import (
    NTULabelForecastDataset,
    NTULabelXYZCacheDataset,
    ntu_label_forecasting_collate,
    ntu_label_xyz_cache_collate,
)
from eval.eval_ntu_label_xyz import evaluate_ntu_label_xyz
from model.forecasting_ntu_xyz import NTULabelXYZTransformer, count_parameters
from model.rotation2xyz import Rotation2xyz_x
from utils.fixseed import fixseed
from utils.ntu_smplx_2p_xyz import ntu_rotvec_2p_to_xyz


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)


def _append_train_log(args, record):
    with open(os.path.join(args.save_dir, "train_log.jsonl"), "a") as f:
        f.write(json.dumps(record, sort_keys=False, ensure_ascii=False))
        f.write("\n")


def _prepare_save_dir(args):
    if os.path.exists(args.save_dir):
        has_files = len(os.listdir(args.save_dir)) > 0
        if has_files and not args.overwrite and args.resume_checkpoint is None:
            raise FileExistsError("save_dir 已存在且非空: {}".format(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)


def _build_dataset(args, split):
    cache_path = args.train_xyz_cache if split == "train" else args.eval_xyz_cache
    if cache_path is not None:
        return NTULabelXYZCacheDataset(cache_path, max_samples=args.max_samples if split == "train" else args.eval_max_samples)
    data_path = args.train_data_path if split == "train" else args.eval_data_path
    return NTULabelForecastDataset(
        h5_path=data_path,
        split=split,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )


def _build_loader(args):
    dataset = _build_dataset(args, "train")
    collate_fn = ntu_label_xyz_cache_collate if args.train_xyz_cache is not None else ntu_label_forecasting_collate
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )


def _build_model(args):
    return NTULabelXYZTransformer(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        num_actions=args.num_actions,
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
        relative_root_loss_weight=args.relative_root_loss_weight,
        relative_velocity_loss_weight=args.relative_velocity_loss_weight,
        key_joint_relation_loss_weight=args.key_joint_relation_loss_weight,
        contact_loss_weight=args.contact_loss_weight,
        contact_threshold=args.contact_threshold,
        action_feature_loss_weight=args.action_feature_loss_weight,
        action_logit_loss_weight=args.action_logit_loss_weight,
    )


def _checkpoint_paths(args, step):
    return (
        os.path.join(args.save_dir, "model{:09d}.pt".format(step)),
        os.path.join(args.save_dir, "opt{:09d}.pt".format(step)),
    )


def _save_checkpoint(args, model, optimizer, step):
    model_path, opt_path = _checkpoint_paths(args, step)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model.model_type,
            "model_config": model.config(),
            "num_params": args.num_params,
            "step": int(step),
            "seed": int(args.seed),
        },
        model_path,
    )
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "step": int(step)}, opt_path)
    return model_path, opt_path


def _load_resume(args, model, optimizer, device):
    if args.resume_checkpoint is None:
        return 0
    state = torch.load(args.resume_checkpoint, map_location=device)
    if state.get("model_type") != model.model_type:
        raise ValueError("resume checkpoint model_type 不一致")
    model.load_state_dict(state["model_state_dict"])
    step = int(state.get("step", 0))
    opt_path = os.path.join(os.path.dirname(args.resume_checkpoint), "opt{:09d}.pt".format(step))
    if os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(opt_state["optimizer_state_dict"])
    return step


def _load_action_classifier(args, device):
    use_action_loss = float(args.action_feature_loss_weight) > 0.0 or float(args.action_logit_loss_weight) > 0.0
    if not use_action_loss:
        return None, None
    if args.action_classifier_path is None:
        raise ValueError("启用 action loss 时必须提供 --action_classifier_path")
    from eval.action_xyz_classifier import load_xyz_action_classifier

    classifier, normalizer, _ = load_xyz_action_classifier(args.action_classifier_path, device=device)
    return classifier, normalizer


def _train_step(model, converter, batch, args, device, action_classifier=None, action_normalizer=None):
    if "obs_xyz" in batch:
        obs_xyz = batch["obs_xyz"].to(device)
        target_xyz = batch["target_xyz"].to(device)
    else:
        with torch.no_grad():
            obs_xyz = ntu_rotvec_2p_to_xyz(batch["obs_motion"].to(device), device=device, converter=converter)
            target_xyz = ntu_rotvec_2p_to_xyz(batch["future"].to(device), device=device, converter=converter)
    action = batch["action"].to(device)
    return model.training_loss(
        obs_xyz,
        target_xyz,
        action,
        action_classifier=action_classifier,
        action_normalizer=action_normalizer,
    )


def _eval_args(args, checkpoint_path):
    return SimpleNamespace(
        mode="checkpoint",
        data_path=args.eval_data_path,
        split="test",
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_samples=args.eval_max_samples,
        seed=args.seed,
        checkpoint=checkpoint_path,
        save_dir=args.save_dir,
        xyz_cache=args.eval_xyz_cache,
        save_arrays=False,
        save_array_limit=0,
        action_classifier_path=args.action_classifier_path,
        semantic_eval=False,
    )


def _evaluate(args, model, checkpoint_path, step, device):
    summary = evaluate_ntu_label_xyz(
        _eval_args(args, checkpoint_path),
        model=model,
        checkpoint_state={"step": int(step)},
        device=device,
    )
    return summary


def run_training(args):
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    fixseed(args.seed)
    _prepare_save_dir(args)
    device = _device()
    train_loader = _build_loader(args)
    model = _build_model(args).to(device)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    action_classifier, action_normalizer = _load_action_classifier(args, device)
    args.num_params = count_parameters(model)
    args.effective_batch_size = int(args.batch_size * max(1, args.grad_accum_steps))
    args.created_at = _utc_now()
    args.device = str(device)
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))
    step = _load_resume(args, model, optimizer, device)

    optimizer.zero_grad()
    recent_losses = []
    accum_batches = 0
    latest_checkpoint = None
    print(
        "Training NTU two-person xyz model: params={} device={} effective_batch_size={}".format(
            args.num_params, device, args.effective_batch_size
        )
    )

    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break
            model.train()
            loss = _train_step(
                model,
                converter,
                batch,
                args,
                device,
                action_classifier=action_classifier,
                action_normalizer=action_normalizer,
            )
            (loss / float(args.grad_accum_steps)).backward()
            recent_losses.append(float(loss.detach().cpu().item()))
            accum_batches += 1
            if accum_batches % args.grad_accum_steps != 0:
                continue

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            train_loss = sum(recent_losses) / float(len(recent_losses))
            recent_losses = []
            record = OrderedDict()
            record["step"] = int(step)
            record["train_loss"] = train_loss
            record["lr"] = args.lr
            record["effective_batch_size"] = args.effective_batch_size
            record["model_num_params"] = args.num_params
            record["seed"] = args.seed

            if step == 1 or step % args.log_interval == 0:
                print("step[{}]: train_loss[{:.6f}]".format(step, train_loss))

            save_due = step % args.save_interval == 0 or step == args.num_steps
            eval_due = args.eval_interval > 0 and (step % args.eval_interval == 0 or step == args.num_steps)
            if save_due or eval_due:
                model_path, opt_path = _save_checkpoint(args, model, optimizer, step)
                latest_checkpoint = model_path
                record["checkpoint"] = model_path
                record["optimizer"] = opt_path

            if eval_due:
                summary = _evaluate(args, model, latest_checkpoint, step, device)
                record["test_xyz_mse"] = summary["model_metrics"]["xyz_mse"]
                record["test_xyz_mae"] = summary["model_metrics"]["xyz_mae"]
                record["test_mpjpe"] = summary["model_metrics"]["mpjpe"]
                record["test_long_xyz_mse"] = summary["model_metrics"]["long_xyz_mse"]
                record["test_final_frame_error"] = summary["model_metrics"]["final_frame_error"]
                record["copy_last_xyz_mse"] = summary["copy_last_metrics"]["xyz_mse"]
                record["copy_last_xyz_mae"] = summary["copy_last_metrics"]["xyz_mae"]
                record["copy_last_mpjpe"] = summary["copy_last_metrics"]["mpjpe"]
                record["beats_copy_last"] = summary["beats_copy_last"]

            _append_train_log(args, record)

    if latest_checkpoint is None:
        latest_checkpoint, _ = _save_checkpoint(args, model, optimizer, step)
    _evaluate(args, model, latest_checkpoint, step, device)
    print("Training finished. final_checkpoint={}".format(latest_checkpoint))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--eval_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--train_xyz_cache", default=None)
    parser.add_argument("--eval_xyz_cache", default=None)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
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
    parser.add_argument("--relative_root_loss_weight", type=float, default=0.2)
    parser.add_argument("--relative_velocity_loss_weight", type=float, default=0.1)
    parser.add_argument("--key_joint_relation_loss_weight", type=float, default=0.0)
    parser.add_argument("--contact_loss_weight", type=float, default=0.0)
    parser.add_argument("--contact_threshold", type=float, default=0.15)
    parser.add_argument("--action_feature_loss_weight", type=float, default=0.0)
    parser.add_argument("--action_logit_loss_weight", type=float, default=0.0)
    parser.add_argument("--action_classifier_path", default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    run_training(args)


if __name__ == "__main__":
    main()
