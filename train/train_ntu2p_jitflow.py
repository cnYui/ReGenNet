"""NTU2P JiT 流匹配训练入口（J1–J4）。

设计：docs/ai/context/20261018-201532-ntu2p-jit-flow-rl-design-and-plan.md 第 3 节。
复用 v2 入口的数据管线（GPU 常驻缓存 + 随机截窗采样器）、损失配方与归一化常数、EMA 与 checkpoint 布局；
每步两次前向：
1. 确定性端点（t=0、z=0，自条件模型按概率先跑一遍无梯度估计）：施加与 FD2 相同的完整损失配方；
2. 随机 t 去噪：JiT 的 x 预测 + v-loss，root 与局部两组各取均值再平均（root 只占 1.4% 的元素，
   按元素平均会让整体平移几乎不受去噪监督），权重 --flow_loss_weight。
命令行先给 FD2 的配方参数（S2_5_RECIPE + FD2 增量），--arch/--leg_stream 对本入口无意义，忽略。
"""

import json
import os
import random
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, NTU2PXYZTrainSampler
from model.forecasting_ntu2p_jitflow import NTU2PJiTFlow, NUM_LOCAL, V_DENOM_MIN, count_parameters
from train.train_ntu2p_residual_refiner_xyz import _append_log, _ema_model, _loss_terms, _update_ema, _utc_now, _write_json
from train.train_ntu2p_v2 import (
    _base_loss_args,
    _device,
    _estimate_scales,
    _extra_loss,
    _loss_joints,
    _save_checkpoint,
    build_arg_parser as build_v2_arg_parser,
    old_scale_starts,
)
from utils.fixseed import fixseed
from utils.ntu_smplx_2p_xyz import XYZ_COORD_DIM


PIPELINE = "ntu2p_xyz_seq_cache"
ARCH = "jit_flow"


def check_args(args):
    if args.window_len != 60 or args.obs_len != 10 or args.pred_len != 50:
        raise ValueError("当前协议固定 window_len=60, obs_len=10, pred_len=50")
    if args.contact_graph and not args.self_cond:
        raise ValueError("--contact_graph 需要 --self_cond")
    if args.causal_cross and args.denoiser != "stg":
        raise ValueError("--causal_cross 只支持 --denoiser stg")
    if not args.kin_proj:
        raise ValueError("流模型的解码固定经过骨架投影，FD2 配方须带 --kin_proj（投影前输出的辅助项）")
    if not 0.0 <= args.sc_prob <= 1.0:
        raise ValueError("--sc_prob 必须在 [0,1]")


def estimate_coef_scales(args, cache, model_kwargs, device):
    """GT 系数在 train 上的 root/局部两组尺度：逐窗口 RMS 的中位数。

    窗口与损失常数估计相同（每序列一个 seed 固定的起点）。不用全体 RMS：train 中 S013C001P028R001A010 含 30 m
    拟合跳变，单个窗口就把局部尺度抬到 126（中位数量级 0.1），归一化后正常样本的系数几乎为 0。
    """
    probe = NTU2PJiTFlow(**dict(model_kwargs, root_scale=1.0, local_scale=1.0)).to(device)
    window_len = args.obs_len + args.pred_len
    seq_index = torch.arange(len(cache), dtype=torch.long)
    start = old_scale_starts(cache, args.seed, window_len)
    per_window = ([], [])
    with torch.no_grad():
        for begin in range(0, len(cache), args.eval_batch_size):
            end = min(begin + args.eval_batch_size, len(cache))
            window = cache.gather_windows(seq_index[begin:end], start[begin:end], window_len)
            obs, target = window[:, : args.obs_len], window[:, args.obs_len :]
            action = cache.actions[seq_index[begin:end].to(cache.device)]
            prep = probe.prepare(obs, action)
            for index, value in enumerate(probe.encode_target(prep, target)):
                per_window[index].append(value.double().pow(2).flatten(1).mean(dim=1).sqrt().cpu())
    return tuple(max(float(torch.cat(values).median().item()), 1e-6) for values in per_window)


def model_kwargs_from_args(args):
    kwargs = OrderedDict(
        [
            ("denoiser", args.denoiser),
            ("self_cond", args.self_cond),
            ("contact_graph", args.contact_graph),
            ("causal_cross", args.causal_cross),
            ("obs_len", args.obs_len),
            ("pred_len", args.pred_len),
            ("num_actions", args.num_actions),
            ("ramp_mode", args.ramp_mode),
            ("ramp_saturate_frames", args.ramp_saturate_frames),
            ("canonical_ab_fallback", args.canonical_ab_fallback),
        ]
    )
    # 显式给出的结构参数不过滤：拼错时构造函数直接报错。
    kwargs.update(json.loads(args.flow_kwargs))
    return kwargs


def sample_t(batch_size, args, device):
    """JiT：logit(t) ~ N(mean, std²)，t=1 为干净数据；--t_uniform_mix 的比例改从 U(0,1) 抽取。

    JiT 默认 N(−0.8, 0.8²) 抽到 t>0.9 的概率约 1e-4；J1 诊断显示该区间去噪误差回升到 t=0 的水平，
    而 ODE 末段速度按 1/(1−t) 放大，样本发散（修订记录见 20261018 修订文档）。
    """
    t = torch.sigmoid(torch.randn(batch_size, device=device) * float(args.t_logit_std) + float(args.t_logit_mean))
    mix = float(getattr(args, "t_uniform_mix", 0.0))
    if mix <= 0:
        return t
    uniform = torch.rand(batch_size, device=device)
    return torch.where(torch.rand(batch_size, device=device) < mix, uniform, t)


def _maybe_self_cond(model, prep, z_root, z_local, t, args):
    """自条件：按概率先跑一遍无梯度估计作为条件（Chen et al. 的做法），否则不给条件。"""
    if not model.self_cond or random.random() >= float(args.sc_prob):
        return None
    with torch.no_grad():
        root, local = model.denoise(prep, z_root, z_local, t)
    return root.detach(), local.detach()


def endpoint_loss(model, prep, obs_xyz, target_xyz, args, loss_args, scales):
    """确定性端点 + FD2 完整损失配方；返回 (loss, delta_reg, details, extra_values, extra_weighted)。"""
    batch_size = int(obs_xyz.shape[0])
    z_root = obs_xyz.new_zeros(batch_size, 2, model.root_k, XYZ_COORD_DIM)
    z_local = obs_xyz.new_zeros(batch_size, 2, model.local_k, NUM_LOCAL, XYZ_COORD_DIM)
    t = obs_xyz.new_zeros(batch_size)
    sc = _maybe_self_cond(model, prep, z_root, z_local, t, args)
    root, local = model.denoise(prep, z_root, z_local, t, sc=sc)
    pred, free, delta = model.decode_xyz(prep, root, local)
    details = OrderedDict([("pred", pred), ("pred_free", free), ("delta", delta)])
    loss, delta_reg = _loss_terms(*(_loss_joints(value, args) for value in (pred, delta, target_xyz, obs_xyz)), loss_args, scales=scales)
    extra, values, weighted = _extra_loss(details, target_xyz, obs_xyz, args, scales)
    if extra is not None:
        loss = loss + extra
    return loss, delta_reg, details, values, weighted


def flow_loss(model, prep, x_root, x_local, args):
    """JiT v-loss：v̂ = (x̂ − z)/max(1−t, 0.05)，v = x − ε；root 与局部两组均值的平均。"""
    batch_size = int(x_root.shape[0])
    t = sample_t(batch_size, args, x_root.device)
    noise_root, noise_local = torch.randn_like(x_root), torch.randn_like(x_local)
    t_root = t.view(batch_size, 1, 1, 1)
    t_local = t.view(batch_size, 1, 1, 1, 1)
    z_root = t_root * x_root + (1.0 - t_root) * noise_root
    z_local = t_local * x_local + (1.0 - t_local) * noise_local
    sc = _maybe_self_cond(model, prep, z_root, z_local, t, args)
    root, local = model.denoise(prep, z_root, z_local, t, sc=sc)
    v_root = (root - z_root) / (1.0 - t_root).clamp_min(V_DENOM_MIN)
    v_local = (local - z_local) / (1.0 - t_local).clamp_min(V_DENOM_MIN)
    mse = torch.nn.functional.mse_loss
    return 0.5 * (mse(v_root, x_root - noise_root) + mse(v_local, x_local - noise_local))


def run(args):
    check_args(args)
    fixseed(args.seed)
    device = _device(args)
    args.device = str(device)
    args.pipeline = PIPELINE
    args.arch = ARCH
    os.makedirs(args.save_dir, exist_ok=True)
    cache = NTU2PXYZSeqCache.load(args.cache_dir, "train", manifest_path=args.manifest_path, device=device)
    sampler = NTU2PXYZTrainSampler(cache, args.batch_size, seed=args.seed, obs_len=args.obs_len, pred_len=args.pred_len)
    kwargs = model_kwargs_from_args(args)
    # 系数尺度与损失常数都只读 train，且在构建正式模型之前完成：探针模型在 fork 的随机流里，不改变正式初始化。
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        root_scale, local_scale = estimate_coef_scales(args, cache, kwargs, device)
    args.coef_scales = OrderedDict([("root", root_scale), ("local", local_scale)])
    model = NTU2PJiTFlow(**dict(kwargs, root_scale=root_scale, local_scale=local_scale)).to(device)
    loss_scales = _estimate_scales(args, cache) if args.loss_scale_normalize else None
    args.loss_scales = loss_scales
    loss_args = _base_loss_args(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.num_params = count_parameters(model)
    args.model_config = model.config()
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))
    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    print("Training NTU2P JiT flow denoiser={} self_cond={} contact_graph={} causal_cross={}: params={} scales=({:.5f},{:.5f}) device={}".format(
        args.denoiser, args.self_cond, args.contact_graph, args.causal_cross, args.num_params, root_scale, local_scale, device))
    # Python random 只决定自条件的开关；独立 seed，不碰 torch 全局随机流。
    random.seed(args.seed)
    ema = _ema_model(model) if args.ema_decay > 0 else None
    ema_dir = os.path.join(args.save_dir, "ema")
    checkpoint = None
    for step in range(1, args.num_steps + 1):
        model.train()
        batch = sampler.sample()
        obs_xyz, target_xyz, action = batch["obs_xyz"], batch["target_xyz"], batch["action"]
        prep = model.prepare(obs_xyz, action)
        with torch.no_grad():
            x_root, x_local = model.encode_target(prep, target_xyz)
        loss_ep, delta_reg, _, extra_values, extra_weighted = endpoint_loss(model, prep, obs_xyz, target_xyz, args, loss_args, loss_scales)
        loss_v = flow_loss(model, prep, x_root, x_local, args)
        loss = loss_ep + float(args.flow_loss_weight) * loss_v
        if not torch.isfinite(loss):
            raise ValueError("训练 loss 非有限")
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        if ema is not None:
            _update_ema(ema, model, args.ema_decay)
        record = OrderedDict(
            [
                ("step", int(step)),
                ("train_loss", float(loss.detach().cpu().item())),
                ("endpoint_loss", float(loss_ep.detach().cpu().item())),
                ("flow_loss", float(loss_v.detach().cpu().item())),
                ("delta_reg", float(delta_reg.detach().cpu().item())),
                ("lr", float(optimizer.param_groups[0]["lr"])),
                ("device", str(device)),
                ("created_at", _utc_now()),
            ]
        )
        for name, value in extra_values.items():
            record[name] = float(value.detach().cpu().item())
            record[name + "_share"] = float((extra_weighted[name] / loss).detach().cpu().item())
        if step == 1 or step % args.log_interval == 0:
            print("step[{}]: loss[{:.5f}] endpoint[{:.5f}] flow[{:.5f}]".format(step, record["train_loss"], record["endpoint_loss"], record["flow_loss"]), flush=True)
        if step % args.save_interval == 0 or step == args.num_steps:
            checkpoint = _save_checkpoint(args, model, optimizer, step)
            record["checkpoint"] = checkpoint
            if ema is not None:
                record["ema_checkpoint"] = _save_checkpoint(args, ema, None, step, save_dir=ema_dir)
        _append_log(log_path, record)
    print("Training finished. final_checkpoint={}".format(checkpoint))
    return checkpoint


def build_arg_parser():
    # 继承 v2 入口的全部参数（FD2 损失配方、EMA、数据管线、CUDA 检查等），同名参数含义一致。
    parser = build_v2_arg_parser()
    parser.description = "NTU2P JiT 流匹配训练入口"
    for action in parser._actions:
        if action.dest == "baseline_checkpoint":
            # 流模型不含冻结单人 base；该参数只原样记进 checkpoint 供评估脚本计算 base 指标。
            action.required = False
    parser.add_argument("--denoiser", choices=("mixer", "stg"), default="mixer")
    parser.add_argument("--self_cond", action="store_true", help="自条件：上一轮 x̂ 作为输入（接触图的几何来源）")
    parser.add_argument("--contact_graph", action="store_true", help="J2/J4：由自条件估计计算的双人手—躯干接触图")
    parser.add_argument("--causal_cross", action="store_true", help="J4：跨人注意力时间因果掩码（仅 stg）")
    parser.add_argument("--flow_loss_weight", type=float, default=1.0)
    parser.add_argument("--t_logit_mean", type=float, default=-0.8, help="JiT 默认 −0.8")
    parser.add_argument("--t_logit_std", type=float, default=0.8, help="JiT 默认 0.8")
    parser.add_argument("--sc_prob", type=float, default=0.5, help="自条件训练时使用上一轮估计的概率")
    parser.add_argument("--t_uniform_mix", type=float, default=0.0, help="按此比例改从 U(0,1) 抽 t（0 为 JiT 原分布）")
    parser.add_argument("--flow_kwargs", default="{}", help="传给 NTU2PJiTFlow 的额外结构参数（JSON）")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
