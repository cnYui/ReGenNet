import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import (
    NTU2PDiffusionForecastDataset,
    ntu_2p_diffusion_collate,
)
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.forecasting_ntu_2p_diffusion import (
    DIFFUSION_STEPS,
    NTU2PClassifierFreeSampleModel,
    create_ntu_2p_diffusion_model_from_config,
)
from utils.fixseed import fixseed
from utils.ntu_2p_rot6d import check_ntu_2p_rot6d, ntu_2p_rot6d_to_xyz


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


def _timestep_respacing(value):
    if value is None or value == "":
        return [DIFFUSION_STEPS]
    return value


def build_sampling_diffusion(noise_schedule="cosine", timestep_respacing="ddim50", body_model="smplx"):
    betas = gd.get_named_beta_schedule(noise_schedule, DIFFUSION_STEPS, 1.0)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(DIFFUSION_STEPS, _timestep_respacing(timestep_respacing)),
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
        body_model=body_model,
        vel_threshold=0.01,
    )


def load_ntu2p_diffusion_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if "model_state_dict" not in state:
        raise ValueError("checkpoint 缺少 model_state_dict")
    if "model_config" not in state:
        raise ValueError("checkpoint 缺少 model_config")
    model = create_ntu_2p_diffusion_model_from_config(state["model_config"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


def sample_diffusion_batch(
    model,
    diffusion,
    batch,
    device,
    use_ddim=True,
    guidance_scale=1.0,
    progress=False,
):
    obs_motion = batch["obs_motion"].to(device)
    future = batch["future"].to(device)
    action = batch["action"].to(device)
    mask = batch["mask"].to(device)
    _ensure_finite("obs_motion", obs_motion)
    _ensure_finite("future", future)
    _ensure_finite("action", action.float())

    sample_model = model
    y = {
        "obs_motion": obs_motion,
        "action": action,
        "mask": mask,
        "scale": torch.ones(obs_motion.shape[0], device=device) * float(guidance_scale),
    }
    if float(guidance_scale) != 1.0:
        sample_model = NTU2PClassifierFreeSampleModel(model, guidance_scale=guidance_scale)

    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
    generated = sample_fn(
        sample_model,
        tuple(future.shape),
        clip_denoised=False,
        model_kwargs={"y": y},
        device=device,
        progress=progress,
    )
    check_ntu_2p_rot6d(generated, seq_len=future.shape[-1])
    return generated


def _build_dataset(args):
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


def _build_loader(args, dataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ntu_2p_diffusion_collate,
        drop_last=False,
    )


def sample_ntu2p_forecasting_diffusion(args):
    fixseed(args.seed)
    device = _device()
    model, state = load_ntu2p_diffusion_checkpoint(args.checkpoint, device)
    diffusion_config = state.get("diffusion_config", {})
    noise_schedule = args.noise_schedule or diffusion_config.get("noise_schedule", "cosine")
    timestep_respacing = args.timestep_respacing
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = "ddim50" if args.use_ddim else ""
    diffusion = build_sampling_diffusion(
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
        body_model=state.get("model_config", {}).get("body_model", "smplx"),
    )
    dataset = _build_dataset(args)
    loader = _build_loader(args, dataset)

    os.makedirs(args.save_dir, exist_ok=True)
    converter = None
    if args.save_xyz:
        from model.rotation2xyz import Rotation2xyz_x

        converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")

    saved = {
        "obs_rot6d": [],
        "real_future_rot6d": [],
        "generated_future_rot6d": [],
        "actions": [],
        "meta": [],
    }
    if args.save_xyz:
        saved.update({"obs_xyz": [], "real_future_xyz": [], "generated_future_xyz": []})

    num_samples = 0
    with torch.no_grad():
        for batch in loader:
            generated = sample_diffusion_batch(
                model=model,
                diffusion=diffusion,
                batch=batch,
                device=device,
                use_ddim=args.use_ddim,
                guidance_scale=args.guidance_scale,
                progress=args.progress,
            )
            obs = batch["obs_motion"].to(device)
            future = batch["future"].to(device)
            action = batch["action"].to(device)
            batch_size = int(obs.shape[0])
            num_samples += batch_size

            saved["obs_rot6d"].append(obs.detach().cpu())
            saved["real_future_rot6d"].append(future.detach().cpu())
            saved["generated_future_rot6d"].append(generated.detach().cpu())
            saved["actions"].append(action.detach().cpu())
            saved["meta"].extend(batch["meta"])
            if args.save_xyz:
                saved["obs_xyz"].append(ntu_2p_rot6d_to_xyz(obs, converter=converter).detach().cpu())
                saved["real_future_xyz"].append(ntu_2p_rot6d_to_xyz(future, converter=converter).detach().cpu())
                saved["generated_future_xyz"].append(ntu_2p_rot6d_to_xyz(generated, converter=converter).detach().cpu())

    if num_samples != len(dataset):
        raise AssertionError("采样样本数应为 {}，实际为 {}".format(len(dataset), num_samples))

    output = OrderedDict()
    for key, value in saved.items():
        if key == "meta":
            output[key] = value
        else:
            output[key] = torch.cat(value, dim=0)
            if torch.is_tensor(output[key]):
                _ensure_finite(key, output[key].float())
    output["sampling_config"] = OrderedDict(
        [
            ("checkpoint", args.checkpoint),
            ("checkpoint_step", state.get("step")),
            ("manifest_path", args.manifest_path),
            ("manifest_hash", state.get("manifest_hash")),
            ("split", args.split),
            ("window_len", args.window_len),
            ("obs_len", args.obs_len),
            ("pred_len", args.pred_len),
            ("num_samples", int(num_samples)),
            ("use_ddim", bool(args.use_ddim)),
            ("timestep_respacing", timestep_respacing),
            ("noise_schedule", noise_schedule),
            ("guidance_scale", float(args.guidance_scale)),
            ("seed", int(args.seed)),
            ("created_at", _utc_now()),
        ]
    )
    torch.save(output, os.path.join(args.save_dir, "samples.pt"))
    _write_json(os.path.join(args.save_dir, "sampling_config.json"), output["sampling_config"])
    print(json.dumps(output["sampling_config"], indent=2, sort_keys=False, ensure_ascii=False))
    return output


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--test_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_schedule", default=None)
    parser.add_argument("--timestep_respacing", default="")
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--save_xyz", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    sample_ntu2p_forecasting_diffusion(args)


if __name__ == "__main__":
    main()
