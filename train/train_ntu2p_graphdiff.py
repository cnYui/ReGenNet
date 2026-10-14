"""图/相位/扩散候选的同预算训练，复用 v2 的数据、主损失和 EMA。

确定性部署端点与带噪系数重建共享纠正器；无扩散控制保持相同调用次数与目标。
"""

import json
import os
import time

import torch
from torch.optim import AdamW

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, NTU2PXYZTrainSampler
from model.forecasting_ntu2p_graphdiff import NTU2PContactGraphDiffusion, build_training_diffusion
from model.forecasting_ntu2p_v2 import load_ntu2p_model_checkpoint
from train.train_ntu2p_residual_refiner_xyz import _ema_model, _update_ema, _append_log, _write_json, _loss_terms
from train.train_ntu2p_v2 import build_arg_parser as v2_parser, _device, _estimate_scales, _base_loss_args, _loss_joints, _extra_loss
from utils.fixseed import fixseed
from utils.ntu2p_residual_codec import glitch_mask
from utils.ntu_smplx_2p_xyz import copy_last_xyz


CANDIDATES = {"C1": (False, False, False), "C2": (True, False, False),
              "C3": (True, True, False), "C4": (True, True, True),
              "F1": (False, False, False), "F2": (True, True, False), "F3": (True, True, True)}
FINE_TUNE_CANDIDATES = frozenset(("F1", "F2", "F3"))
PRIOR_LR_MULT = 0.1


def check_args(args):
    if args.window_len != 60 or args.obs_len != 10 or args.pred_len != 50:
        raise ValueError("协议固定 obs10/future50")
    if (args.arch != "intermixer" or args.unfreeze_base or args.augment_mirror_prob != 0
            or args.retrieval_bank or args.retrieval_k or args.lr_schedule != "constant"
            or args.base_lr_mult != 1.0):
        raise ValueError("图纠正器固定InterMixer、无增广/检索与恒定学习率；底座是否微调由候选控制，不能静默忽略继承开关")
    if args.num_steps <= 0 or args.batch_size <= 0 or args.save_interval <= 0 or args.log_interval <= 0:
        raise ValueError("训练步数、batch与日志/保存间隔必须正数")
    if not 0 < args.ema_decay < 1 or args.lr <= 0 or args.clip_grad_norm <= 0:
        raise ValueError("必须使用有效EMA、正学习率和梯度裁剪阈值")
    if min(args.contact_loss_weight, args.bilateral_loss_weight, args.dm_loss_weight) < 0:
        raise ValueError("辅助损失权重不能负数")
    if args.candidate in FINE_TUNE_CANDIDATES and (args.lr != 3e-4 or args.batch_size != 8):
        raise ValueError("端到端F候选固定纠正器lr=3e-4、InterMixer lr=3e-5与batch8")


def build_optimizer(model, args):
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not model.fine_tune_prior:
        return AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    prior_ids = {id(param) for param in model.prior.parameters()}
    groups = [
        dict(params=[param for param in trainable if id(param) not in prior_ids],
             lr=args.lr, name="corrector"),
        dict(params=[param for param in trainable if id(param) in prior_ids],
             lr=args.lr * PRIOR_LR_MULT, name="prior"),
    ]
    return AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def save_checkpoint(path, model, args, step):
    torch.save(dict(model_type=model.model_type, model_config=model.config(), model_state_dict=model.state_dict(),
                    step=int(step), seed=int(args.seed), manifest_path=args.manifest_path,
                    prior_checkpoint=args.prior_checkpoint, candidate=args.candidate,
                    protocol="ntu120_2p_o10_p50", device=str(args.device)), path)


def estimate_new_scales(model, cache, args):
    from utils.ntu2p_contact_graph import contact_distance_loss, bilateral_phase_loss

    sampler = NTU2PXYZTrainSampler(cache, 16, seed=9107)
    residuals = []
    contact, phase = [], []
    was_training = model.training
    # 尺度来自同一初始预测；禁用底座 dropout，避免微调开关改变尺度与训练随机流。
    model.eval()
    with torch.no_grad():
        for _ in range(32):
            batch = sampler.sample()
            obs, target, action = batch["obs_xyz"], batch["target_xyz"], batch["action"]
            y = model.prepare(obs, action)
            coeffs = model.codec.target_coeffs(obs, y["draft"], target, y["frames"])
            clean = ~glitch_mask(torch.cat((obs, target), 1))
            if bool(clean.any()):
                residuals.append(coeffs[clean])
            copy = copy_last_xyz(obs, 50)
            contact.append(contact_distance_loss(copy, target))
            phase.append(bilateral_phase_loss(copy, target, obs))
    model.train(was_training)
    values = torch.cat(residuals, 0)
    # 仅训练残差估计；5%通道组尺度下限防止近静止系数把去噪梯度放大。
    sigma = values.pow(2).mean(0).sqrt()
    floor = 0.05 * sigma.pow(2).mean().sqrt()
    model.sigma.copy_(sigma.clamp_min(max(float(floor), 1e-3)))
    return dict(contact=max(float(torch.stack(contact).mean()), 1e-8),
                bilateral=max(float(torch.stack(phase).mean()), 1e-8))


def run(args):
    from utils.ntu2p_contact_graph import contact_distance_loss, bilateral_phase_loss

    check_args(args)
    if os.path.exists(os.path.join(args.save_dir, "args.json")):
        raise ValueError("run目录已有记录，不能覆盖历史训练")
    fixseed(args.seed)
    device = _device(args)
    args.device = str(device)
    os.makedirs(args.save_dir, exist_ok=True)
    prior, prior_state = load_ntu2p_model_checkpoint(args.prior_checkpoint, device)
    if prior_state.get("manifest_path") != args.manifest_path:
        raise ValueError("先验与训练manifest不一致，禁止跨协议复用权重")
    cache = NTU2PXYZSeqCache.load(args.cache_dir, "train", args.manifest_path, device)
    graph, phase, diffusion = CANDIDATES[args.candidate]
    args.fine_tune_prior = args.candidate in FINE_TUNE_CANDIDATES
    model = NTU2PContactGraphDiffusion(prior, graph=graph, phase=phase, diffusion=diffusion,
                                     hidden=args.graph_hidden, num_blocks=args.graph_blocks,
                                     fine_tune_prior=args.fine_tune_prior).to(device)
    sampler = NTU2PXYZTrainSampler(cache, args.batch_size, seed=args.seed)
    scales = _estimate_scales(args, cache)
    new_scales = estimate_new_scales(model, cache, args)
    args.loss_scales, args.new_loss_scales = scales, new_scales
    args.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.model_config = model.config()
    args.forward_calls_per_step = 2
    optimizer = build_optimizer(model, args)
    args.optimizer_groups = [dict(name=group.get("name", "corrector"), lr=group["lr"],
                                  num_params=sum(param.numel() for param in group["params"]))
                             for group in optimizer.param_groups]
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))
    ema = _ema_model(model)
    diffusion_process = build_training_diffusion()
    loss_args = _base_loss_args(args)
    # 扩散随机数不影响dropout和数据采样；全部候选均生成同形状噪声保持配对。
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 2389)
    log = os.path.join(args.save_dir, "train_log.jsonl")
    start_time = time.monotonic()
    print("候选{}：{}可训练参数，{}，双forward/step".format(args.candidate, args.num_params, device), flush=True)
    for step in range(1, args.num_steps + 1):
        model.train()
        batch = sampler.sample()
        obs, target, action = batch["obs_xyz"], batch["target_xyz"], batch["action"]
        y = model.prepare(obs, action)
        details = model.point_details(obs, action, prepared=y)
        primary, regularizer = _loss_terms(*(_loss_joints(v, args) for v in
                                           (details["pred"], details["delta"], target, obs)), loss_args, scales)
        extra, _, _ = _extra_loss(details, target, obs, args, scales)
        primary = primary + extra if extra is not None else primary
        contact = contact_distance_loss(details["pred"], target) / new_scales["contact"]
        bilateral = bilateral_phase_loss(details["pred"], target, obs) / new_scales["bilateral"]
        with torch.no_grad():
            x0 = model.codec.target_coeffs(obs, y["draft"], target, y["frames"]) / model.sigma
            timesteps = torch.randint(0, 1000, (int(obs.shape[0]),), device=device, generator=noise_generator)
            noise = torch.randn(tuple(x0.shape), dtype=x0.dtype, device=device, generator=noise_generator)
            noisy = diffusion_process.q_sample(x0, timesteps, noise=noise) if diffusion else x0
            clean = ~glitch_mask(torch.cat((obs, target), 1))
        prediction = model.predict_coeffs(noisy, timesteps, y)
        per_sample = (prediction - x0).pow(2).flatten(1).mean(1)
        dm = (per_sample * clean.to(per_sample.dtype)).sum() / clean.sum().clamp_min(1)
        loss = primary + args.contact_loss_weight * contact + args.bilateral_loss_weight * bilateral + args.dm_loss_weight * dm
        if not bool(torch.isfinite(loss)):
            raise ValueError("训练loss非有限，停止")
        optimizer.zero_grad()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_grad_norm)
        if not bool(torch.isfinite(grad)):
            raise ValueError("训练梯度非有限，停止")
        optimizer.step()
        _update_ema(ema, model, args.ema_decay)
        record = dict(step=step, loss=float(loss.detach()), primary=float(primary.detach()),
                      contact=float(contact.detach()), bilateral=float(bilateral.detach()), dm=float(dm.detach()),
                      grad_norm=float(grad), elapsed_s=time.monotonic() - start_time,
                      device=str(device), candidate=args.candidate)
        if step == 1 or step % args.log_interval == 0:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if step % args.save_interval == 0 or step == args.num_steps:
            for name, value in (("raw", model), ("ema", ema)):
                directory = args.save_dir if name == "raw" else os.path.join(args.save_dir, name)
                os.makedirs(directory, exist_ok=True)
                save_checkpoint(os.path.join(directory, "model{:09d}.pt".format(step)), value, args, step)
        _append_log(log, record)
    _write_json(os.path.join(args.save_dir, "completed.json"), dict(step=args.num_steps, elapsed_s=time.monotonic() - start_time))
    print("训练完成：{}".format(args.save_dir), flush=True)


def build_arg_parser():
    parser = v2_parser()
    parser.description = "NTU2P 接触事件图与相位 DCT 扩散训练"
    parser.add_argument("--prior_checkpoint", required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--graph_hidden", type=int, default=96)
    parser.add_argument("--graph_blocks", type=int, default=3)
    parser.add_argument("--contact_loss_weight", type=float, default=0.1)
    parser.add_argument("--bilateral_loss_weight", type=float, default=0.1)
    parser.add_argument("--dm_loss_weight", type=float, default=0.1)
    parser.set_defaults(arch="intermixer", kin_proj=True, leg_stream=True, loss_joint_subset="hand2",
                        canonical_ab_fallback="camera_x", root_loss_weight=0.1, inter_loss_weight=0.01,
                        loss_scale_normalize=True, local_velocity_loss_weight=1.0,
                        dct_low_amplitude_loss_weight=1.0, dct_mid_amplitude_loss_weight=1.0,
                        dct_mid_exclude_legs=True, leg_gait_loss_weight=0.5, foot_loss_weight=0.05,
                        ema_decay=0.999, ramp_mode="saturate", ramp_saturate_frames=5,
                        save_interval=5000, log_interval=100)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
