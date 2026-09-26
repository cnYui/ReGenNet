"""NTU2P v2 统一训练入口：在 GPU 常驻 xyz 序列缓存管线上训练旧 refiner（A0）、规范系 refiner（A1–A5/A7）与 InterMixer（A6）。

设计：docs/ai/context/20260925-121154-ntu2p-v2-architecture-exploration-design-and-plan.md 第 3–5 节。
损失、学习率、EMA、checkpoint 布局直接复用 train/train_ntu2p_residual_refiner_xyz.py（不复制实现），
与旧入口的差别只在数据来源：缓存 + 随机截窗采样器，其随机流与旧 DataLoader 不同，新旧管线之间不能逐步配对，
因此 A0 在新管线上重跑作为所有候选的同管线、同 seed 对照。

loss 一律在相机系上对 pred/target 计算（与旧实现一致）；规范化只发生在模型内部。
步态变体 GL/GH 的腿部损失在 utils/ntu2p_gait_losses.py，走 `_extra_loss` 路径（训练日志逐步记录 *_share）。
"""

import copy
import json
import os
import random
from collections import OrderedDict

import torch
from torch.optim import AdamW

from data_loaders.forecasting.ntu2p_retrieval_bank import NTU2PRetrievalBank, performer_ids
from data_loaders.forecasting.ntu2p_xyz_seq_cache import DEFAULT_CACHE_DIR, NTU2PXYZSeqCache, NTU2PXYZTrainSampler
from eval.analyze_ntu2p_naturalness import slide_reference
from model.forecasting_ntu2p_residual_xyz import NTU2PResidualRefinerXYZ, count_parameters, load_base_model_from_checkpoint
from model.forecasting_ntu2p_v2 import NTU2PCanonRefiner, accepted_kwargs, forward_details
from train.train_ntu2p_residual_refiner_xyz import (
    DATASET,
    LOSS_TERM_KEYS,
    _append_log,
    _device as _cuda_device,
    _ema_model,
    _loss_terms,
    _lr_at,
    _raw_terms,
    _update_ema,
    _utc_now,
    _write_json,
    build_arg_parser as build_refiner_arg_parser,
)
from utils.fixseed import fixseed
from utils.ntu2p_canonical import NTU2PSceneAugment, estimate_up
from utils.ntu2p_gait_losses import DEFAULT_LEG_GAIT_LOWPASS_K, gait_loss_terms, gait_loss_weights
from utils.ntu_smplx_2p_xyz import copy_last_xyz


PIPELINE = "ntu2p_xyz_seq_cache"
ARCHS = ("refiner", "canon_refiner", "intermixer")
CANON_ONLY_SWITCHES = ("canonical", "disp_channel", "role_embed")


def _device(args):
    requested = torch.device(args.device)
    if requested.type == "cpu" and args.allow_cpu_for_smoke_test:
        # 仅冒烟测试：显式开关 + 写入 args.json，正式训练仍走旧入口的 CUDA 强制检查，不会静默回退。
        print("WARNING: --allow_cpu_for_smoke_test，本 run 在 CPU 上运行，仅用于冒烟测试")
        return requested
    return _cuda_device(args.device)


def _refiner_kwargs(args):
    return OrderedDict(
        [
            ("obs_len", args.obs_len),
            ("pred_len", args.pred_len),
            ("num_actions", args.num_actions),
            ("latent_dim", args.latent_dim),
            ("num_heads", args.num_heads),
            ("encoder_layers", args.encoder_layers),
            ("decoder_layers", args.decoder_layers),
            ("dim_feedforward", args.dim_feedforward),
            ("dropout", args.dropout),
            ("alpha", 1.0),
            ("freeze_base", not args.unfreeze_base),
            ("ramp_mode", args.ramp_mode),
            ("ramp_saturate_frames", args.ramp_saturate_frames),
            ("future_pos_mode", args.future_pos_mode),
            ("root_head_mode", args.root_head_mode),
            ("root_dct_k", args.root_dct_k),
        ]
    )


def _build_refiner(args, device, bank):
    base_model, base_state = load_base_model_from_checkpoint(args.baseline_checkpoint, device)
    return NTU2PResidualRefinerXYZ(base_model=base_model, **_refiner_kwargs(args)), base_state


def _retrieval_anchor_kwargs(args, bank):
    kwargs = json.loads(args.retrieval_anchor_kwargs)
    if bank is None:
        return kwargs
    # 与检索库保持一致：查询键的关节数；τ 以 train 查询 top-k 距离中位数初始化（库构建时已为同一 k 算好则直接用）。
    kwargs.setdefault("key_joints", int(bank.key_joints))
    if "init_tau" not in kwargs:
        tau = bank.config.get("median_topk_dist") if int(bank.config.get("median_topk_k", -1)) == int(args.retrieval_k) else None
        kwargs["init_tau"] = float(tau if tau is not None else bank.median_neighbor_distance(k=args.retrieval_k))
    return kwargs


def _build_canon_refiner(args, device, bank):
    base_model, base_state = load_base_model_from_checkpoint(args.baseline_checkpoint, device)
    model = NTU2PCanonRefiner(
        base_model=base_model,
        canonical=args.canonical,
        disp_channel=args.disp_channel,
        role_embed=args.role_embed,
        mirror_embed=args.mirror_embed,
        retrieval_k=args.retrieval_k,
        kin_proj=args.kin_proj,
        retrieval_anchor_kwargs=_retrieval_anchor_kwargs(args, bank),
        canonical_ab_fallback=args.canonical_ab_fallback,
        **_refiner_kwargs(args)
    )
    return model, base_state


def _build_intermixer(args, device, bank):
    from model.forecasting_ntu2p_intermixer import NTU2PInterMixer

    candidates = OrderedDict(
        [
            ("obs_len", args.obs_len),
            ("pred_len", args.pred_len),
            ("num_actions", args.num_actions),
            ("ramp_mode", args.ramp_mode),
            ("ramp_saturate_frames", args.ramp_saturate_frames),
            ("kin_proj", args.kin_proj),
            ("mirror_embed", args.mirror_embed),
            ("canonical_ab_fallback", args.canonical_ab_fallback),
            ("leg_stream", args.leg_stream),
        ]
    )
    kwargs = accepted_kwargs(NTU2PInterMixer, candidates)
    for switch in ("kin_proj", "mirror_embed", "leg_stream"):
        if candidates[switch] and switch not in kwargs:
            raise ValueError("NTU2PInterMixer 不支持 --{}".format(switch))
    # 显式给出的参数不过滤：拼错或不支持时应直接报错。
    kwargs.update(json.loads(args.intermixer_kwargs))
    return NTU2PInterMixer(**kwargs), None


BUILDERS = OrderedDict([("refiner", _build_refiner), ("canon_refiner", _build_canon_refiner), ("intermixer", _build_intermixer)])


def _check_args(args):
    if args.dataset != DATASET:
        raise ValueError("dataset 必须是 {}".format(DATASET))
    if args.window_len != 60 or args.obs_len != 10 or args.pred_len != 50:
        raise ValueError("当前协议固定 window_len=60, obs_len=10, pred_len=50")
    if args.arch != "canon_refiner":
        used = [name for name in CANON_ONLY_SWITCHES if getattr(args, name)]
        if args.retrieval_bank or args.retrieval_k:
            used.append("retrieval")
        if used:
            raise ValueError("--arch {} 不支持 {}".format(args.arch, used))
    if args.arch == "refiner" and (args.kin_proj or args.mirror_embed):
        raise ValueError("--arch refiner 是原样的旧类（A0 对照），不支持 v2 开关")
    if args.arch == "intermixer" and args.unfreeze_base:
        raise ValueError("InterMixer 没有冻结 base，不支持 --unfreeze_base")
    if bool(args.retrieval_bank) != (args.retrieval_k > 0):
        raise ValueError("--retrieval_bank 与 --retrieval_k>0 必须同时给出")
    if args.mirror_embed and args.augment_mirror_prob <= 0:
        raise ValueError("--mirror_embed 需要 --augment_mirror_prob > 0，否则指示恒为 0、没有意义")
    if args.base_lr_mult != 1.0 and not args.unfreeze_base:
        raise ValueError("--base_lr_mult 只在 --unfreeze_base 时有意义")
    if args.leg_stream and args.arch != "intermixer":
        raise ValueError("--leg_stream 只支持 --arch intermixer")
    if args.dct_mid_exclude_legs and args.dct_mid_amplitude_loss_weight <= 0:
        raise ValueError("--dct_mid_exclude_legs 需要 --dct_mid_amplitude_loss_weight > 0（非腿项沿用它的权重）")
    if not 2 <= args.leg_gait_lowpass_k <= args.pred_len:
        raise ValueError("--leg_gait_lowpass_k 必须在 [2,{}] 内".format(args.pred_len))


def _foot_loss(pred, target, obs):
    from utils.ntu2p_kinematic_projection import foot_skate_loss

    return foot_skate_loss(pred, target, obs, estimate_up(obs))


def _base_loss_args(args):
    """传给旧 `_loss_terms` 的参数：--dct_mid_exclude_legs 时关掉全关节 dct_mid 项，改由额外项 dct_mid_nonleg 承担。

    开关关闭时原样返回 args，旧损失路径逐位不变。
    """
    if not args.dct_mid_exclude_legs:
        return args
    loss_args = copy.copy(args)
    loss_args.dct_mid_amplitude_loss_weight = 0.0
    return loss_args


def old_scale_starts(cache, seed, window_len):
    """复现旧实现估计常量时的窗口：新建数据集实例的 random.Random(seed)，按 manifest 顺序每序列 randint 一次。"""
    rng = random.Random(int(seed))
    return torch.as_tensor([rng.randint(0, int(length) - int(window_len)) for length in cache.lengths_cpu.tolist()], dtype=torch.long)


# 主损失的关节子集。刚体手下 30 个手指关节≈手腕误差复制 15 倍，占 A6 总 loss 81%、腿只有 7.7%；
# 子集保留身体 25 个关节（索引不变，交互项的手腕索引仍有效）与每只手少数手指根关节（足以确定刚体手旋转）。
# 只作用于 s2_5 主损失及其归一化常数；脚接触、步态等额外项仍用全部关节。
LOSS_JOINT_SUBSETS = OrderedDict(
    [
        ("all", None),
        ("hand2", tuple(range(25)) + (25, 31, 40, 46)),
        ("hand5", tuple(range(25)) + (25, 28, 31, 34, 37, 40, 43, 46, 49, 52)),
    ]
)


def _loss_joints(value, args):
    index = LOSS_JOINT_SUBSETS[getattr(args, "loss_joint_subset", "all")]
    return value if index is None else value[..., list(index), :]


def _estimate_scales(args, cache):
    """copy-last 在 train 上的各项误差作归一化常量，与旧 `_estimate_copy_last_scales` 用完全相同的窗口与分批。

    旧实现是"按 manifest 顺序、每序列一个 random.Random(seed) 随机起点、eval_batch_size 分批、逐 batch 均值按
    样本数加权"的一遍遍历；这里在缓存上逐一复现这些起点，常数与旧 run 只差缓存与在线 FK 的 1e-7 级误差
    （核对：与主线 s2_5 s0 的 args.json 逐项比值 1 ± 1e-7）。不改用采样器随机起点的原因：train 中
    S013C001P028R001A010 含 30 m 拟合跳变，它的窗口是否落在跳变上会让 mse/root/inter 常数变化 2–3 倍
    （采样器起点下该窗口占 mse 总和的 59%），等于悄悄改了损失配方，A0 就不再是主线配置。
    Python random 独立于 torch 全局 RNG，不影响模型初始化与 dropout 随机流。
    foot 项例外：copy-last 静止不动，脚滑损失恒为 0 不能当尺度，改用"GT root + 冻结姿态"纯滑行参考的值，
    即权重 w 表示"相对纯滑行的脚滑能量比例"。
    步态项（leg_pos / leg_lpvel / dct_mid_nonleg）与旧项同口径，在 copy-last 上计算，即"腿不动"的误差。
    """
    window_len = args.obs_len + args.pred_len
    seq_index = torch.arange(len(cache), dtype=torch.long)
    start = old_scale_starts(cache, args.seed, window_len)
    gait_weights = gait_loss_weights(args)
    # 步态项只在开关打开时追加：开关关闭时常量的键集合与计算都与之前相同。
    keys = list(LOSS_TERM_KEYS) + (["foot"] if args.foot_loss_weight > 0 else []) + list(gait_weights)
    sums = OrderedDict((key, 0.0) for key in keys)
    count = 0
    with torch.no_grad():
        for index, begin in enumerate(range(0, len(cache), args.eval_batch_size)):
            if args.scale_estimate_batches > 0 and index >= args.scale_estimate_batches:
                break
            end = min(begin + args.eval_batch_size, len(cache))
            window = cache.gather_windows(seq_index[begin:end], start[begin:end], window_len)
            obs_xyz = window[:, : args.obs_len].contiguous()
            target_xyz = window[:, args.obs_len :].contiguous()
            copy_xyz = copy_last_xyz(obs_xyz, args.pred_len)
            terms = _raw_terms(*(_loss_joints(value, args) for value in (copy_xyz, target_xyz, obs_xyz)), args, LOSS_TERM_KEYS)
            if args.foot_loss_weight > 0:
                terms["foot"] = _foot_loss(slide_reference(obs_xyz, target_xyz), target_xyz, obs_xyz)
            if gait_weights:
                terms.update(gait_loss_terms(copy_xyz, target_xyz, args))
            batch_size = int(obs_xyz.shape[0])
            for key, value in terms.items():
                sums[key] += float(value.detach().cpu().item()) * batch_size
            count += batch_size
    if count <= 0:
        raise ValueError("归一化常量估计样本数为 0")
    return OrderedDict((key, max(value / float(count), 1e-8)) for key, value in sums.items())


def _extra_loss(details, target, obs, args, scales):
    """v2 新增损失项；scales 非 None 时与旧 `_loss_terms` 一样除以参考值（free_aux 用 mse 的常量）。

    返回 (总额外项, 各项原始值, 各项加权后的值)；加权值供训练日志记录逐步占比，A5 与步态项的权重按占比预登记，需可核对。
    """
    terms = OrderedDict()
    if args.foot_loss_weight > 0:
        terms["foot"] = (float(args.foot_loss_weight), "foot", _foot_loss(details["pred"], target, obs))
    if args.kin_proj and args.free_aux_mse_weight > 0:
        if details.get("pred_free") is None:
            raise ValueError("--kin_proj 时模型必须在 details 中返回 pred_free")
        free_mse = torch.nn.functional.mse_loss(details["pred_free"], target)
        terms["free_aux_mse"] = (float(args.free_aux_mse_weight), "mse", free_mse)
    gait_weights = gait_loss_weights(args)
    if gait_weights:
        for name, value in gait_loss_terms(details["pred"], target, args).items():
            terms[name] = (gait_weights[name], name, value)
    loss = None
    values, weighted = OrderedDict(), OrderedDict()
    for name, (weight, scale_key, value) in terms.items():
        values[name] = value
        term = value if scales is None else value / float(scales[scale_key])
        weighted[name] = weight * term
        loss = weighted[name] if loss is None else loss + weighted[name]
    return loss, values, weighted


def _optimizer(args, model):
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not (args.unfreeze_base and args.base_lr_mult != 1.0):
        # 单参数组与旧入口的 AdamW(params) 等价；lr_mult 只供学习率调度使用。
        return AdamW([{"params": trainable, "lr_mult": 1.0}], lr=args.lr, weight_decay=args.weight_decay)
    base_ids = {id(param) for param in model.base_model.parameters()}
    groups = [
        {"params": [p for p in trainable if id(p) not in base_ids], "lr_mult": 1.0},
        {"params": [p for p in trainable if id(p) in base_ids], "lr": args.lr * args.base_lr_mult, "lr_mult": args.base_lr_mult},
    ]
    return AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)


def _save_checkpoint(args, model, optimizer, step, save_dir=None):
    """与旧入口同布局；model_type/representation 取自模型，A0 的 checkpoint 可被旧评估脚本原样加载。"""
    save_dir = save_dir or args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model{:09d}.pt".format(int(step)))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model.model_type,
            "model_config": model.config(),
            "baseline_checkpoint": args.baseline_checkpoint,
            "representation": model.model_type,
            "protocol": "ntu120_2p_o10_p50",
            "manifest_path": args.manifest_path,
            "pipeline": PIPELINE,
            "arch": args.arch,
            "retrieval_bank": args.retrieval_bank,
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


def run(args):
    _check_args(args)
    fixseed(args.seed)
    device = _device(args)
    args.device = str(device)
    args.pipeline = PIPELINE
    os.makedirs(args.save_dir, exist_ok=True)
    train_cache = NTU2PXYZSeqCache.load(args.cache_dir, "train", manifest_path=args.manifest_path, device=device)
    sampler = NTU2PXYZTrainSampler(train_cache, args.batch_size, seed=args.seed, obs_len=args.obs_len, pred_len=args.pred_len)
    # 只做镜像：A/B 顺序是施动/受动（不交换），yaw/平移会被规范化消去；p=0 时完全不经过增广代码。
    augment = None
    if args.augment_mirror_prob > 0:
        augment = NTU2PSceneAugment(yaw_range_deg=0.0, swap_prob=0.0, mirror_prob=args.augment_mirror_prob)
    performers = performer_ids(train_cache.sample_ids)
    bank = None
    if args.retrieval_bank:
        bank = NTU2PRetrievalBank.load(args.retrieval_bank, train_cache, device=device)

    model, base_state = BUILDERS[args.arch](args, device, bank)
    model = model.to(device)
    loss_scales = _estimate_scales(args, train_cache) if args.loss_scale_normalize else None
    args.loss_scales = loss_scales
    loss_args = _base_loss_args(args)
    optimizer = _optimizer(args, model)
    args.num_params = count_parameters(model)
    base_model = getattr(model, "base_model", None)
    args.base_num_params = sum(param.numel() for param in base_model.parameters()) if base_model is not None else 0
    args.base_checkpoint_step = int(base_state.get("step", -1)) if base_state is not None else -1
    args.model_config = model.config()
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))
    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    print(
        "Training NTU2P v2 arch={} model_type={}: params={} base_params={} device={} freeze_base={}".format(
            args.arch, model.model_type, args.num_params, args.base_num_params, device, getattr(model, "freeze_base", None)
        )
    )

    # EMA 在注入检索库之前深拷贝，避免复制库与其引用的 train 缓存；随后两者共享同一个库。
    ema = _ema_model(model) if args.ema_decay > 0 else None
    if bank is not None:
        model.set_retrieval_bank(bank)
        if ema is not None:
            ema.set_retrieval_bank(bank)
    ema_dir = os.path.join(args.save_dir, "ema")
    checkpoint = None
    for step in range(1, args.num_steps + 1):
        model.train()
        batch = sampler.sample()
        obs_xyz, target_xyz, action = batch["obs_xyz"], batch["target_xyz"], batch["action"]
        mirror = torch.zeros(int(obs_xyz.shape[0]), dtype=torch.bool)
        if augment is not None:
            # 用采样器自带的增广随机流：开关增广时序列与起点的抽样不变，与未增广 run 可配对。
            params = augment.draw(int(obs_xyz.shape[0]), generator=sampler.augment_generator)
            obs_xyz, target_xyz = augment.apply(obs_xyz, target_xyz, params)
            mirror = params["mirror"]
        seq_index = batch["seq_index"]
        context = OrderedDict(
            [
                ("performer", performers[seq_index].to(device)),
                ("seq_index", seq_index.to(device)),
                ("mirror", mirror.to(device)),
            ]
        )
        details = forward_details(model, obs_xyz, action, context)
        loss, delta_reg = _loss_terms(
            *(_loss_joints(value, args) for value in (details["pred"], details["delta"], target_xyz, obs_xyz)),
            loss_args,
            scales=loss_scales,
        )
        extra, extra_values, extra_weighted = _extra_loss(details, target_xyz, obs_xyz, args, loss_scales)
        if extra is not None:
            loss = loss + extra
        if not torch.isfinite(loss):
            raise ValueError("训练 loss 非有限")
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        if args.lr_schedule != "constant":
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(args, step - 1) * group["lr_mult"]
        optimizer.step()
        if ema is not None:
            _update_ema(ema, model, args.ema_decay)
        alpha = getattr(model, "alpha", None)
        record = OrderedDict(
            [
                ("step", int(step)),
                ("train_loss", float(loss.detach().cpu().item())),
                ("delta_reg", float(delta_reg.detach().cpu().item())),
                ("alpha", None if alpha is None else float(alpha.detach().cpu().item())),
                ("lr", float(optimizer.param_groups[0]["lr"])),
                ("device", str(device)),
                ("base_checkpoint_step", int(args.base_checkpoint_step)),
                ("created_at", _utc_now()),
            ]
        )
        for name, value in extra_values.items():
            record[name] = float(value.detach().cpu().item())
            record[name + "_share"] = float((extra_weighted[name] / loss).detach().cpu().item())
        if step == 1 or step % args.log_interval == 0:
            print("step[{}]: loss[{:.6f}] delta_reg[{:.6f}]".format(step, record["train_loss"], record["delta_reg"]))
        if step % args.save_interval == 0 or step == args.num_steps:
            checkpoint = _save_checkpoint(args, model, optimizer, step)
            record["checkpoint"] = checkpoint
            if ema is not None:
                record["ema_checkpoint"] = _save_checkpoint(args, ema, None, step, save_dir=ema_dir)
        _append_log(log_path, record)
    print("Training finished. final_checkpoint={}".format(checkpoint))
    return checkpoint


def build_arg_parser():
    # 继承旧入口的全部参数与默认值（损失权重、ramp、EMA、学习率等），保证同名配置含义一致。
    parser = build_refiner_arg_parser()
    parser.description = "NTU2P v2 统一训练入口（缓存管线）"
    parser.add_argument("--arch", choices=ARCHS, default="refiner")
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--canonical", action="store_true", help="A1：refiner 的输入与残差在场景规范系中")
    parser.add_argument("--disp_channel", action="store_true", help="A1：obs token 加 Linear(x_t - x_last)")
    parser.add_argument("--role_embed", action="store_true", help="A1：施动/受动角色 embedding")
    parser.add_argument("--mirror_embed", action="store_true", help="A7：镜像指示 embedding（测试时恒为 0）")
    parser.add_argument(
        "--canonical_ab_fallback",
        choices=("a_facing", "camera_x"),
        default="a_facing",
        help="双人重合（A->B 不可观测）时规范系 yaw 的回退；camera_x 见 utils/ntu2p_canonical.canonical_frame",
    )
    parser.add_argument("--augment_mirror_prob", type=float, default=0.0, help="A7：训练时镜像增广概率；不做 A/B 交换")
    parser.add_argument("--retrieval_bank", default=None, help="A3：检索库路径（scripts/build_ntu2p_retrieval_bank.py 生成）")
    parser.add_argument("--retrieval_k", type=int, default=0)
    parser.add_argument("--retrieval_anchor_kwargs", default="{}", help="传给 RetrievalAnchor 的额外参数（JSON）")
    parser.add_argument("--kin_proj", action="store_true", help="A4：输出经可微骨架投影")
    parser.add_argument("--free_aux_mse_weight", type=float, default=0.1, help="A4：投影前自由输出的 mse 辅助项权重")
    parser.add_argument("--foot_loss_weight", type=float, default=0.0, help="A5：脚接触一致性损失权重")
    parser.add_argument("--loss_joint_subset", choices=tuple(LOSS_JOINT_SUBSETS), default="all",
                        help="FD：s2_5 主损失只用身体 25 关节 + 每手少数手指根关节（手指去重）")
    parser.add_argument("--base_lr_mult", type=float, default=1.0, help="A2：解冻 base 时 base 学习率相对 refiner 的倍数")
    parser.add_argument("--intermixer_kwargs", default="{}", help="A6：传给 NTU2PInterMixer 的额外参数（JSON）")
    # 步态变体 GL/GH；默认全部关闭，训练与之前逐位相同。
    parser.add_argument("--leg_gait_loss_weight", type=float, default=0.0,
                        help="GL/GH：腿局部位置 + ≤2 Hz 低通速度的有符号损失权重（两项共用，copy-last 归一化）")
    parser.add_argument("--leg_gait_lowpass_k", type=int, default=DEFAULT_LEG_GAIT_LOWPASS_K,
                        help="leg_lpvel 保留的 DCT 基个数（第 k 个为 k×0.2 Hz；默认 11 即 ≤2 Hz）")
    parser.add_argument("--dct_mid_exclude_legs", action="store_true",
                        help="GL/GH：dct_mid 幅度项只作用于 47 个非腿关节（权重沿用 --dct_mid_amplitude_loss_weight）")
    parser.add_argument("--leg_stream", action="store_true", help="GH：InterMixer 的个人朝向系腿部专用流，替换主干的腿局部输出")
    parser.add_argument(
        "--allow_cpu_for_smoke_test",
        action="store_true",
        help="仅冒烟测试：允许 --device cpu；默认关闭，正式训练必须 CUDA",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
