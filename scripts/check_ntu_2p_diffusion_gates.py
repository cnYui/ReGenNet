import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import (
    NTU2PDiffusionForecastDataset,
    assert_manifest_no_sample_id_leak,
    ensure_ntu_2p_diffusion_manifest,
    ntu_2p_diffusion_collate,
    scan_ntu_2p_diffusion_entries,
)
from diffusion.resample import create_named_schedule_sampler
from eval.eval_ntu_2p_forecasting_diffusion import evaluate_ntu2p_forecasting_diffusion
from model.forecasting_ntu_2p_diffusion import MODEL_TYPE, NTU2PForecastingDiffusionDecoder, count_parameters
from sample.sample_ntu_2p_forecasting_diffusion import build_sampling_diffusion, sample_diffusion_batch
from train.train_ntu_2p_forecasting_diffusion import (
    build_diffusion,
    diffusion_config_from_args,
    diffusion_train_step,
)
from utils.fixseed import fixseed
from utils.ntu_2p_rot6d import (
    NTU_2P_PERSON_ORDER,
    NTU_2P_REPRESENTATION,
    check_ntu_2p_rot6d,
    interaction_loss,
    join_ntu_2p_rot6d,
    ntu_2p_rot6d_to_xyz,
    split_ntu_2p_rot6d,
)


DEFAULT_TRAIN_H5 = "dataset/ntu120/smplx/conditioned/xsub.train.h5"
DEFAULT_TEST_H5 = "dataset/ntu120/smplx/conditioned/xsub.test.h5"


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


def _ensure_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError("{} 存在 NaN 或 Inf".format(name))


def _assert_close_zero(name, tensor, atol=1e-6):
    value = float(tensor.detach().abs().max().cpu().item())
    if value > float(atol):
        raise AssertionError("{} 应接近 0，当前 max_abs={}".format(name, value))


def _assert_positive(name, tensor, min_value=1e-9):
    value = float(tensor.detach().cpu().item())
    if value <= float(min_value):
        raise AssertionError("{} 应为正，当前为 {}".format(name, value))


def _build_manifest(args):
    manifest_path = args.manifest_path or os.path.join(args.save_dir, "manifest_seed{}.json".format(args.seed))
    manifest = ensure_ntu_2p_diffusion_manifest(
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        manifest_path=manifest_path,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite_manifest,
    )
    args.manifest_path = str(manifest_path)
    args.manifest_hash = manifest.get("manifest_hash")
    return manifest


def _build_dataset(args, split, max_samples):
    return NTU2PDiffusionForecastDataset(
        manifest_path=args.manifest_path,
        split=split,
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=max_samples,
        seed=args.seed,
    )


def _build_batch(args, split="train", max_samples=2):
    dataset = _build_dataset(args, split, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=ntu_2p_diffusion_collate,
        drop_last=False,
    )
    return dataset, next(iter(loader))


def _train_args(args, inter_loss_weight):
    return SimpleNamespace(
        dataset="ntu120_2p",
        body_model="smplx",
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        noise_schedule="cosine",
        timestep_respacing="",
        schedule_sampler="uniform",
        one_step_noise_prob=0.0,
        inter_loss_weight=float(inter_loss_weight),
        vel_threshold=0.01,
    )


def _build_small_model(args, device):
    return NTU2PForecastingDiffusionDecoder(
        model_type=MODEL_TYPE,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        window_len=args.window_len,
        latent_dim=args.latent_dim,
        obs_encoder_layers=1,
        decoder_layers=1,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=0.0,
        cond_mask_prob=0.0,
    ).to(device)


def _save_gate_checkpoint(args, model, optimizer, step):
    save_root = os.path.join(args.save_dir, "checkpoint_smoke")
    os.makedirs(save_root, exist_ok=True)
    model_path = os.path.join(save_root, "model{:09d}.pt".format(step))
    opt_path = os.path.join(save_root, "opt{:09d}.pt".format(step))
    train_args = _train_args(args, inter_loss_weight=0.0)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": MODEL_TYPE,
            "model_config": model.config(),
            "num_params": count_parameters(model),
            "step": int(step),
            "seed": int(args.seed),
            "diffusion_config": diffusion_config_from_args(train_args),
            "train_protocol": {
                "dataset": "ntu120_2p",
                "window_len": int(args.window_len),
                "obs_len": int(args.obs_len),
                "pred_len": int(args.pred_len),
                "representation": NTU_2P_REPRESENTATION,
                "person_order": NTU_2P_PERSON_ORDER,
                "inter_loss_weight": 0.0,
            },
            "representation": NTU_2P_REPRESENTATION,
            "person_order": NTU_2P_PERSON_ORDER,
            "manifest_path": args.manifest_path,
            "manifest_hash": args.manifest_hash,
            "created_at": _utc_now(),
        },
        model_path,
    )
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "step": int(step)}, opt_path)
    return model_path


def _gate_scan_and_manifest(args, summary):
    train_scan = scan_ntu_2p_diffusion_entries(args.train_data_path, window_len=args.window_len)
    test_scan = scan_ntu_2p_diffusion_entries(args.test_data_path, window_len=args.window_len)
    manifest = _build_manifest(args)
    overlaps = assert_manifest_no_sample_id_leak(manifest)
    summary["scan"] = {
        "train_kept": len(train_scan["entries"]),
        "test_kept": len(test_scan["entries"]),
        "train_raw": train_scan["raw_count"],
        "test_raw": test_scan["raw_count"],
    }
    summary["manifest"] = {
        "path": args.manifest_path,
        "hash": args.manifest_hash,
        "split_summary": manifest["split_summary"],
        "overlaps": overlaps,
    }


def _gate_dataset_and_representation(args, summary, device):
    dataset, batch = _build_batch(args, split="train", max_samples=args.max_samples)
    obs = batch["obs_motion"].to(device)
    future = batch["future"].to(device)
    full = torch.cat([obs, future], dim=-1)
    check_ntu_2p_rot6d(obs, seq_len=args.obs_len)
    check_ntu_2p_rot6d(future, seq_len=args.pred_len)
    check_ntu_2p_rot6d(full, seq_len=args.window_len)
    person_a, person_b = split_ntu_2p_rot6d(full)
    joined = join_ntu_2p_rot6d(person_a, person_b)
    if not torch.allclose(joined, full):
        raise AssertionError("Person A/B split/join round-trip 失败")

    from model.rotation2xyz import Rotation2xyz_x

    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    xyz = ntu_2p_rot6d_to_xyz(full, converter=converter)
    summary["dataset_representation"] = {
        "dataset_len": len(dataset),
        "obs_shape": list(obs.shape),
        "future_shape": list(future.shape),
        "full_shape": list(full.shape),
        "xyz_shape": list(xyz.shape),
    }
    return batch, converter


def _gate_qsample_and_losses(args, summary, batch, converter, device):
    train_args = _train_args(args, inter_loss_weight=1.0)
    diffusion = build_diffusion(train_args)
    future = batch["future"].to(device)
    t_low = torch.zeros(future.shape[0], device=device, dtype=torch.long)
    t_high = torch.full((future.shape[0],), int(diffusion.num_timesteps) - 1, device=device, dtype=torch.long)
    low = diffusion.q_sample(future, t_low, noise=torch.randn_like(future))
    high = diffusion.q_sample(future, t_high, noise=torch.randn_like(future))
    _ensure_finite("q_sample_low", low)
    _ensure_finite("q_sample_high", high)

    zero_terms = interaction_loss(future, future, converter=converter)
    for key, value in zero_terms.items():
        _assert_close_zero("zero_{}".format(key), value)

    trans_pred = future.clone()
    trans_pred[:, 55, 0, :] = trans_pred[:, 55, 0, :] + 0.1
    trans_terms = interaction_loss(trans_pred, future, converter=converter)
    _assert_positive("translation joint_mse", trans_terms["joint_mse"])
    _assert_positive("translation trans_mse", trans_terms["trans_mse"])
    _assert_close_zero("translation orient_mse", trans_terms["orient_mse"], atol=1e-5)

    joint_pred = future.clone()
    joint_pred[:, 1, 0, :] = joint_pred[:, 1, 0, :] + 0.05
    joint_terms = interaction_loss(joint_pred, future, converter=converter)
    _assert_positive("body joint joint_mse", joint_terms["joint_mse"])
    _assert_close_zero("body joint trans_mse", joint_terms["trans_mse"], atol=1e-5)
    _assert_close_zero("body joint orient_mse", joint_terms["orient_mse"], atol=1e-5)

    orient_pred = future.clone()
    orient_pred[:, 0, 0, :] = orient_pred[:, 0, 0, :] + 0.05
    orient_terms = interaction_loss(orient_pred, future, converter=converter)
    _assert_positive("root orientation orient_mse", orient_terms["orient_mse"])

    grad_pred = orient_pred.detach().clone().requires_grad_(True)
    grad_terms = interaction_loss(grad_pred, future, converter=converter)
    l_dm = ((grad_pred - future) ** 2).mean()
    l_all = l_dm + grad_terms["inter_loss"]
    l_all.backward()
    if grad_pred.grad is None or not torch.isfinite(grad_pred.grad).all():
        raise AssertionError("L_all.backward 后 prediction grad 必须 finite 且非空")
    if float(grad_pred.grad.detach().abs().sum().cpu().item()) <= 0.0:
        raise AssertionError("L_all.backward 后 prediction grad 不能全 0")

    summary["qsample_loss"] = {
        "q_low_shape": list(low.shape),
        "q_high_shape": list(high.shape),
        "zero_inter_loss": float(zero_terms["inter_loss"].detach().cpu().item()),
        "translation_inter_loss": float(trans_terms["inter_loss"].detach().cpu().item()),
        "body_joint_inter_loss": float(joint_terms["inter_loss"].detach().cpu().item()),
        "orientation_inter_loss": float(orient_terms["inter_loss"].detach().cpu().item()),
        "grad_abs_sum": float(grad_pred.grad.detach().abs().sum().cpu().item()),
    }
    return diffusion


def _gate_train_checkpoint_and_sampling(args, summary, batch, device):
    train_args = _train_args(args, inter_loss_weight=0.0)
    diffusion = build_diffusion(train_args)
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
    model = _build_small_model(args, device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    losses = []
    for _ in range(2):
        model.train()
        optimizer.zero_grad()
        loss, metrics = diffusion_train_step(model, diffusion, schedule_sampler, batch, train_args, device, converter=None)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise AssertionError("训练梯度非有限: {}".format(name))
        optimizer.step()
        losses.append(metrics)

    checkpoint_path = _save_gate_checkpoint(args, model, optimizer, step=2)
    reloaded = _build_small_model(args, device)
    state = torch.load(checkpoint_path, map_location=device)
    reloaded.load_state_dict(state["model_state_dict"])
    reloaded.eval()

    ddim2 = build_sampling_diffusion(noise_schedule="cosine", timestep_respacing="ddim2", body_model="smplx")
    ddpm2 = build_sampling_diffusion(noise_schedule="cosine", timestep_respacing="2", body_model="smplx")
    with torch.no_grad():
        sample_ddim = sample_diffusion_batch(reloaded, ddim2, batch, device, use_ddim=True, guidance_scale=1.0, progress=False)
        sample_ddpm = sample_diffusion_batch(reloaded, ddpm2, batch, device, use_ddim=False, guidance_scale=1.0, progress=False)
    for name, sample in (("ddim2", sample_ddim), ("ddpm2", sample_ddpm)):
        _ensure_finite(name, sample)
        if float(sample.detach().abs().sum().cpu().item()) <= 0.0:
            raise AssertionError("{} 输出不能全 0".format(name))
        obs_last = batch["obs_motion"].to(device)[..., -1:].expand_as(sample)
        if torch.allclose(sample, obs_last):
            raise AssertionError("{} 输出不能逐元素复制 obs 最后一帧".format(name))

    summary["train_checkpoint_sampling"] = {
        "checkpoint_path": checkpoint_path,
        "losses": losses,
        "ddim2_shape": list(sample_ddim.shape),
        "ddpm2_shape": list(sample_ddpm.shape),
    }
    return checkpoint_path


def _gate_eval(args, summary, checkpoint_path):
    eval_dir = os.path.join(args.save_dir, "eval_smoke")
    eval_args = SimpleNamespace(
        mode="diffusion",
        manifest_path=args.manifest_path,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        split="val",
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        batch_size=1,
        num_workers=0,
        max_samples=min(2, args.max_samples),
        seed=args.seed,
        sample_seeds=str(args.seed),
        checkpoint=checkpoint_path,
        save_dir=eval_dir,
        noise_schedule=None,
        timestep_respacing="ddim2",
        use_ddim=True,
        guidance_scale=1.0,
        save_arrays=False,
        save_array_limit=0,
        semantic_eval=False,
        action_classifier_path=None,
        diversity_pairs=1000,
        progress=False,
    )
    result = evaluate_ntu2p_forecasting_diffusion(eval_args)
    summary["eval_smoke"] = {
        "save_dir": eval_dir,
        "model_metrics": result["per_seed"][0]["model_metrics"],
        "copy_last_metrics": result["per_seed"][0]["copy_last_metrics"],
    }


def run_gates(args):
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    if args.window_len != 60 or args.obs_len != 10 or args.pred_len != 50:
        raise ValueError("阶段 0 gate 固定 window_len=60, obs_len=10, pred_len=50")
    fixseed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = _device()
    summary = OrderedDict()
    summary["created_at"] = _utc_now()
    summary["device"] = str(device)
    summary["save_dir"] = args.save_dir

    _gate_scan_and_manifest(args, summary)
    batch, converter = _gate_dataset_and_representation(args, summary, device)
    _gate_qsample_and_losses(args, summary, batch, converter, device)
    checkpoint_path = _gate_train_checkpoint_and_sampling(args, summary, batch, device)
    _gate_eval(args, summary, checkpoint_path)

    summary["pass"] = True
    _write_json(os.path.join(args.save_dir, "gate_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", default=DEFAULT_TRAIN_H5)
    parser.add_argument("--test_data_path", default=DEFAULT_TEST_H5)
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--save_dir", default="results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates")
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite_manifest", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=128)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_gates(args)


if __name__ == "__main__":
    main()
