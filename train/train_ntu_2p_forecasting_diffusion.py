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
    ensure_ntu_2p_diffusion_manifest,
    load_ntu_2p_diffusion_manifest,
    manifest_default_path,
    ntu_2p_diffusion_collate,
)
from diffusion import gaussian_diffusion as gd
from diffusion.resample import create_named_schedule_sampler
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.forecasting_ntu_2p_diffusion import (
    DIFFUSION_STEPS,
    MODEL_TYPE,
    NTU2PForecastingDiffusionDecoder,
    count_parameters,
)
from utils.fixseed import fixseed
from utils.ntu_2p_rot6d import (
    NTU_2P_PERSON_ORDER,
    NTU_2P_REPRESENTATION,
    NTU_2P_ROT6D_FEATS,
    interaction_loss,
)


DATASET = "ntu120_2p"
NUM_ACTIONS = 26
NJOINTS = 56
NFEATS = NTU_2P_ROT6D_FEATS
DEFAULT_TRAIN_H5 = "dataset/ntu120/smplx/conditioned/xsub.train.h5"
DEFAULT_TEST_H5 = "dataset/ntu120/smplx/conditioned/xsub.test.h5"


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _resolve_cuda_device(device_name):
    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device 必须是有效的 CUDA 设备，例如 cuda:0，当前为 {}".format(device_name)) from error
    if device.type != "cuda":
        raise ValueError("NTU 双人 diffusion 训练只支持 CUDA，当前 device={}".format(device_name))
    if not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 训练，但 torch.cuda.is_available() 为 false；请确认 GPU 已暴露给当前进程。")
    device_index = 0 if device.index is None else int(device.index)
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(
            "请求的 CUDA 设备 {} 不可用；当前可见 GPU 数为 {}".format(device_name, torch.cuda.device_count())
        )
    device = torch.device("cuda:{}".format(device_index))
    torch.cuda.set_device(device)
    return device


def _cuda_runtime_metadata(device):
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "visible_device_count": int(torch.cuda.device_count()),
    }


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def _append_train_log(args, record):
    path = os.path.join(args.save_dir, "train_log.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=False, ensure_ascii=False))
        f.write("\n")


def _is_stage_output(filename):
    if filename in ("args.json", "train_log.jsonl"):
        return True
    return (filename.startswith("model") or filename.startswith("opt")) and filename.endswith(".pt")


def _clear_stage_outputs(save_dir):
    for filename in os.listdir(save_dir):
        if _is_stage_output(filename):
            path = os.path.join(save_dir, filename)
            if os.path.isfile(path):
                os.remove(path)


def _prepare_save_dir(args):
    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    if os.path.exists(args.save_dir):
        has_files = len(os.listdir(args.save_dir)) > 0
        if has_files and args.resume_checkpoint is None and not args.overwrite:
            raise FileExistsError("save_dir [{}] already exists. 使用 --overwrite 或更换 save_dir。".format(args.save_dir))
        if has_files and args.resume_checkpoint is None and args.overwrite:
            _clear_stage_outputs(args.save_dir)
    else:
        os.makedirs(args.save_dir)


def _ensure_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError("{} 存在 NaN 或 Inf".format(name))


def _build_manifest(args):
    manifest_path = args.manifest_path or manifest_default_path(args.save_dir, seed=args.seed)
    manifest = ensure_ntu_2p_diffusion_manifest(
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        manifest_path=manifest_path,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_actions=NUM_ACTIONS,
        overwrite=args.overwrite_manifest,
    )
    args.manifest_path = str(manifest_path)
    args.manifest_hash = manifest.get("manifest_hash")
    return manifest


def _build_dataset(args, split):
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


def _build_loader(args, split, shuffle, batch_size):
    dataset = _build_dataset(args, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=ntu_2p_diffusion_collate,
        drop_last=False,
    )


def _build_model(args):
    return NTU2PForecastingDiffusionDecoder(
        model_type=args.model_type,
        njoints=NJOINTS,
        nfeats=NFEATS,
        num_actions=NUM_ACTIONS,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        window_len=args.window_len,
        latent_dim=args.latent_dim,
        obs_encoder_layers=args.obs_encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        activation=args.activation,
        cond_mask_prob=args.cond_mask_prob,
        causal_future_mask=args.causal_future_mask,
        data_rep="rot6d",
        body_model=args.body_model,
        dataset=args.dataset,
        representation=NTU_2P_REPRESENTATION,
        person_order=NTU_2P_PERSON_ORDER,
        init_rot2xyz=False,
    )


def _timestep_respacing(value):
    if value is None or value == "":
        return [DIFFUSION_STEPS]
    return value


def diffusion_config_from_args(args):
    return {
        "steps": DIFFUSION_STEPS,
        "noise_schedule": args.noise_schedule,
        "timestep_respacing": args.timestep_respacing,
        "sigma_small": True,
        "model_mean_type": "START_X",
        "model_var_type": "FIXED_SMALL",
        "loss_type": "MSE",
        "rescale_timesteps": False,
        "data_rep": "rot6d",
        "num_person": 2,
        "representation": NTU_2P_REPRESENTATION,
        "body_model": args.body_model,
    }


def build_diffusion(args):
    betas = gd.get_named_beta_schedule(args.noise_schedule, DIFFUSION_STEPS, 1.0)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(DIFFUSION_STEPS, _timestep_respacing(args.timestep_respacing)),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        lambda_rcxyz=0.0,
        lambda_vel=0.0,
        lambda_fc=0.0,
        lambda_orient=0.0,
        lambda_body=0.0,
        lambda_transl=0.0,
        data_rep="rot6d",
        num_person=2,
        body_model=args.body_model,
        vel_threshold=args.vel_threshold,
    )


def _masked_l2_per_sample(a, b, mask):
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError("masked_l2 输入 shape 不一致: {} vs {}".format(tuple(a.shape), tuple(b.shape)))
    if mask.dim() != 4 or mask.shape[0] != a.shape[0] or mask.shape[-1] != a.shape[-1]:
        raise ValueError("mask 必须是 [B,1,1,T] 或 [B,1,T] collate 后等价 shape，当前为 {}".format(tuple(mask.shape)))
    mask = mask.to(device=a.device, dtype=a.dtype)
    loss = ((a - b) ** 2) * mask
    loss = loss.sum(dim=(1, 2, 3))
    denom = mask.sum(dim=(1, 2, 3)) * a.shape[1] * a.shape[2]
    return loss / denom.clamp_min(1.0)


def _zero_inter_terms(device):
    zero = torch.tensor(0.0, device=device)
    return OrderedDict(
        [
            ("joint_mse", zero),
            ("orient_mse", zero),
            ("trans_mse", zero),
            ("inter_loss", zero),
        ]
    )


def _compute_losses(pred, target, mask, args, converter):
    _ensure_finite("pred", pred)
    _ensure_finite("target", target)
    _ensure_finite("mask", mask.float())
    rot_mse_per_sample = _masked_l2_per_sample(pred, target, mask)
    rot_mse = rot_mse_per_sample
    inter_terms = _zero_inter_terms(pred.device)
    if float(args.inter_loss_weight) > 0.0:
        inter_terms = interaction_loss(pred, target, converter=converter)
    total_per_sample = rot_mse + float(args.inter_loss_weight) * inter_terms["inter_loss"]
    terms = OrderedDict()
    terms["loss"] = total_per_sample
    terms["rot_mse"] = rot_mse
    for key, value in inter_terms.items():
        terms[key] = value
    return terms


def _batch_to_model_kwargs(batch, device):
    future = batch["future"].to(device)
    obs_motion = batch["obs_motion"].to(device)
    action = batch["action"].to(device)
    mask = batch["mask"].to(device)

    _ensure_finite("future", future)
    _ensure_finite("obs_motion", obs_motion)
    _ensure_finite("action", action.float())
    _ensure_finite("mask", mask.float())

    return future, {"obs_motion": obs_motion, "action": action, "mask": mask}


def diffusion_train_step(model, diffusion, schedule_sampler, batch, args, device, converter=None):
    future, y = _batch_to_model_kwargs(batch, device)
    batch_size = int(future.shape[0])
    t, weights = schedule_sampler.sample(batch_size, device)
    noise = torch.randn_like(future)
    x_t = diffusion.q_sample(future, t, noise=noise)
    _ensure_finite("x_t", x_t)

    pred = model(x_t, t, y)
    terms = _compute_losses(pred, future, y["mask"], args, converter)
    loss = (terms["loss"] * weights).mean()
    if not torch.isfinite(loss):
        raise ValueError("训练 loss 为非有限数值: {}".format(float(loss.detach().cpu().item())))

    metrics = OrderedDict()
    metrics["train_loss"] = float(loss.detach().cpu().item())
    metrics["rot_mse"] = float(terms["rot_mse"].detach().mean().cpu().item())
    metrics["joint_mse"] = float(terms["joint_mse"].detach().cpu().item())
    metrics["orient_mse"] = float(terms["orient_mse"].detach().cpu().item())
    metrics["trans_mse"] = float(terms["trans_mse"].detach().cpu().item())
    metrics["inter_loss"] = float(terms["inter_loss"].detach().cpu().item())
    metrics["t_mean"] = float(t.float().mean().detach().cpu().item())
    return loss, metrics


def _assert_gradients_finite(model):
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        has_grad = True
        if not torch.isfinite(param.grad).all():
            raise ValueError("参数梯度非有限: {}".format(name))
    if not has_grad:
        raise ValueError("模型没有任何参数梯度")


def _checkpoint_paths(args, step):
    return (
        os.path.join(args.save_dir, "model{:09d}.pt".format(int(step))),
        os.path.join(args.save_dir, "opt{:09d}.pt".format(int(step))),
    )


def _train_protocol(args):
    return {
        "dataset": args.dataset,
        "window_len": int(args.window_len),
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "num_actions": NUM_ACTIONS,
        "target": "future50",
        "condition": "obs10 + action",
        "mean_type": "START_X",
        "loss_type": "MSE",
        "timestep_sampling": "uniform",
        "one_step_noise_prob": 0.0,
        "inter_loss_weight": float(args.inter_loss_weight),
        "representation": NTU_2P_REPRESENTATION,
        "person_order": NTU_2P_PERSON_ORDER,
    }


def _save_checkpoint(args, model, optimizer, step):
    manifest = load_ntu_2p_diffusion_manifest(args.manifest_path)
    model_path, opt_path = _checkpoint_paths(args, step)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "model_config": model.config(),
            "num_params": int(args.num_params),
            "step": int(step),
            "seed": int(args.seed),
            "diffusion_config": diffusion_config_from_args(args),
            "train_protocol": _train_protocol(args),
            "representation": NTU_2P_REPRESENTATION,
            "person_order": NTU_2P_PERSON_ORDER,
            "manifest_path": str(args.manifest_path),
            "manifest_hash": manifest.get("manifest_hash"),
            "manifest_split_summary": manifest.get("split_summary"),
            "cuda_runtime": args.cuda_runtime,
            "created_at": _utc_now(),
        },
        model_path,
    )
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "step": int(step)}, opt_path)
    return model_path, opt_path


def _core_config_keys():
    return (
        "njoints",
        "nfeats",
        "num_actions",
        "obs_len",
        "pred_len",
        "window_len",
        "latent_dim",
        "obs_encoder_layers",
        "decoder_layers",
        "num_heads",
        "ff_size",
        "representation",
        "person_order",
    )


def _validate_resume_config(args, state, model):
    if state.get("model_type") != args.model_type:
        raise ValueError("resume model_type={} 与当前 model_type={} 不一致".format(state.get("model_type"), args.model_type))
    checkpoint_config = state.get("model_config", {})
    current_config = model.config()
    for key in _core_config_keys():
        if checkpoint_config.get(key) != current_config.get(key):
            raise ValueError("resume config mismatch: {} checkpoint={} current={}".format(key, checkpoint_config.get(key), current_config.get(key)))
    if state.get("manifest_hash") != args.manifest_hash:
        raise ValueError("resume manifest_hash 与当前 manifest 不一致")


def _load_resume(args, model, optimizer, device):
    if args.resume_checkpoint is None:
        return 0
    if not os.path.exists(args.resume_checkpoint):
        raise FileNotFoundError(args.resume_checkpoint)
    state = torch.load(args.resume_checkpoint, map_location=device)
    _validate_resume_config(args, state, model)
    model.load_state_dict(state["model_state_dict"])
    step = int(state.get("step", 0))
    opt_path = os.path.join(os.path.dirname(args.resume_checkpoint), "opt{:09d}.pt".format(step))
    if os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(opt_state["optimizer_state_dict"])
    else:
        print("warning: optimizer checkpoint not found: {}".format(opt_path))
    return step


def _save_args(args):
    serializable = dict(vars(args))
    serializable["created_at"] = _utc_now()
    serializable["train_protocol"] = _train_protocol(args)
    serializable["diffusion_config"] = diffusion_config_from_args(args)
    _write_json(os.path.join(args.save_dir, "args.json"), serializable)


def _validate_args(args):
    if args.dataset != DATASET:
        raise ValueError("dataset 必须是 {}".format(DATASET))
    if args.model_type != MODEL_TYPE:
        raise ValueError("model_type 必须是 {}".format(MODEL_TYPE))
    if args.body_model != "smplx":
        raise ValueError("body_model 必须是 smplx")
    if int(args.window_len) != 60 or int(args.obs_len) != 10 or int(args.pred_len) != 50:
        raise ValueError("本入口固定 window_len=60, obs_len=10, pred_len=50")
    if int(args.obs_len) + int(args.pred_len) != int(args.window_len):
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    if args.noise_schedule != "cosine":
        raise ValueError("正式协议固定 noise_schedule=cosine")
    if args.schedule_sampler != "uniform":
        raise ValueError("正式协议固定 schedule_sampler=uniform")
    if args.timestep_respacing not in ("", None):
        raise ValueError("训练协议固定完整 1000 timestep，不使用 respacing")
    if float(args.one_step_noise_prob) != 0.0:
        raise ValueError("正式协议固定 one_step_noise_prob=0")
    if float(args.inter_loss_weight) < 0.0:
        raise ValueError("inter_loss_weight 不能为负")


def _make_log_record(args, metrics_list, step):
    record = OrderedDict()
    record["step"] = int(step)
    for key in ("train_loss", "rot_mse", "joint_mse", "orient_mse", "trans_mse", "inter_loss", "t_mean"):
        record[key] = sum(float(item[key]) for item in metrics_list) / float(len(metrics_list))
    record["lr"] = float(args.lr)
    record["effective_batch_size"] = int(args.effective_batch_size)
    record["device"] = args.device
    record["cuda_device_name"] = args.cuda_runtime["device_name"]
    record["model_num_params"] = int(args.num_params)
    record["seed"] = int(args.seed)
    record["manifest_path"] = args.manifest_path
    record["manifest_hash"] = args.manifest_hash
    record["created_at"] = _utc_now()
    return record


def run_training(args):
    _validate_args(args)
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    args.effective_batch_size = int(args.batch_size * args.grad_accum_steps)
    fixseed(args.seed)
    device = _resolve_cuda_device(args.device)
    _prepare_save_dir(args)
    _build_manifest(args)

    train_loader = _build_loader(args, split="train", shuffle=True, batch_size=args.batch_size)
    model = _build_model(args).to(device)
    diffusion = build_diffusion(args)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    converter = None
    if float(args.inter_loss_weight) > 0.0:
        from model.rotation2xyz import Rotation2xyz_x

        converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")

    args.num_params = count_parameters(model)
    args.device = str(device)
    args.cuda_runtime = _cuda_runtime_metadata(device)
    args.model_config = model.config()
    args.diffusion_num_timesteps = int(diffusion.num_timesteps)
    _save_args(args)

    step = _load_resume(args, model, optimizer, device)
    optimizer.zero_grad()
    print(
        "Training NTU2P forecasting diffusion: params={} device={} gpu={} effective_batch_size={} inter_loss_weight={} resume_step={}".format(
            args.num_params,
            device,
            args.cuda_runtime["device_name"],
            args.effective_batch_size,
            args.inter_loss_weight,
            step,
        )
    )

    accum_batches = 0
    recent_metrics = []
    latest_checkpoint = None
    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break
            model.train()
            loss, metrics = diffusion_train_step(model, diffusion, schedule_sampler, batch, args, device, converter=converter)
            (loss / float(args.grad_accum_steps)).backward()
            recent_metrics.append(metrics)
            accum_batches += 1
            if accum_batches % args.grad_accum_steps != 0:
                continue

            _assert_gradients_finite(model)
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            record = _make_log_record(args, recent_metrics, step)
            recent_metrics = []
            if step == 1 or step % args.log_interval == 0:
                print("step[{}]: train_loss[{:.6f}] rot_mse[{:.6f}] inter_loss[{:.6f}]".format(step, record["train_loss"], record["rot_mse"], record["inter_loss"]))

            save_due = step % args.save_interval == 0 or step == args.num_steps
            if save_due:
                model_path, opt_path = _save_checkpoint(args, model, optimizer, step)
                latest_checkpoint = model_path
                record["checkpoint"] = model_path
                record["optimizer"] = opt_path
            _append_train_log(args, record)

    if latest_checkpoint is None:
        latest_checkpoint, _ = _save_checkpoint(args, model, optimizer, step)
    print("Training finished. final_checkpoint={}".format(latest_checkpoint))
    return latest_checkpoint


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--train_data_path", default=DEFAULT_TRAIN_H5)
    parser.add_argument("--test_data_path", default=DEFAULT_TEST_H5)
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--model_type", default=MODEL_TYPE)
    parser.add_argument("--body_model", default="smplx")
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--obs_encoder_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--cond_mask_prob", type=float, default=0.1)
    parser.add_argument("--causal_future_mask", action="store_true")
    parser.add_argument("--inter_loss_weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--eval_max_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_manifest", action="store_true")
    parser.add_argument("--noise_schedule", default="cosine")
    parser.add_argument("--timestep_respacing", default="")
    parser.add_argument("--schedule_sampler", default="uniform")
    parser.add_argument("--one_step_noise_prob", type=float, default=0.0)
    parser.add_argument("--vel_threshold", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=1)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
