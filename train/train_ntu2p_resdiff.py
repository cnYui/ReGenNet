"""Track B 生成器训练：在 OOF 残差库的干净窗口上训练 v2 残差系数的条件扩散（x0 预测，归一化空间 MSE）。

设计：docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md 第 4.5–4.8 节。
- 每步：按序列均匀抽干净窗口 -> x0 = target/σ_target -> t ~ U{0..999} -> x_t = q_sample -> 以 p=0.25 给干净 root
  （输入 root 通道替换、损失 mask 掉 root 通道）-> x̂0 -> masked MSE；AdamW + 线性预热 + 恒定学习率 + 梯度裁剪 + EMA。
- 主臂不加几何/脚接触/L_inter：首帧、骨长、刚体手由结构保证；GT 接触帧的脚损失会把样本拉回 GT 相位；
  历史上 L_inter 约为 L_dm 的 10 倍、压制个体拟合（20260822 深度排查）。
- 应急臂（默认关闭，只在自然度护栏失败时按预登记启用一轮）：C1 foot（t<300 的样本解码投影后加 GT 接触帧脚滑，
  按纯滑行参考归一化）、C2 inter（投影前 body22 相对向量 + pelvis 相对平移的 MSE，按 x̂0=0 即纯 v2 的值归一化）；
  两者都需在线运行该窗口所属的折模型得到 P。
- 必须显式 --device cuda:0；CPU 只在显式 --allow_cpu_for_smoke_test 时允许。
"""

import argparse
import os
from collections import OrderedDict

import torch
from torch.optim import AdamW

from data_loaders.forecasting.ntu2p_residual_bank import NTU2PResidualBank
from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache
from diffusion.resample import UniformSampler
from eval.analyze_ntu2p_naturalness import slide_reference
from model.forecasting_ntu2p_resdiff import (
    DIFFUSION_STEPS,
    NTU2PResDiffDenoiser,
    build_training_diffusion,
    count_parameters,
    make_condition,
    save_ntu2p_resdiff_checkpoint,
    to_meters,
)
from model.forecasting_ntu2p_v2 import forward_details, load_ntu2p_model_checkpoint
from train.train_ntu2p_residual_refiner_xyz import _append_log, _ema_model, _update_ema, _utc_now, _write_json
from train.train_ntu2p_v2 import _device
from utils.fixseed import fixseed
from utils.ntu2p_canonical import estimate_up
from utils.ntu2p_kinematic_projection import SkeletonProjector, foot_skate_loss
from utils.ntu2p_naturalness import BODY_PARTS
from utils.ntu2p_residual_codec import OBS_LEN, PRED_LEN, ResidualCodec, codec_frames


WINDOW_LEN = OBS_LEN + PRED_LEN
ARMS = ("main", "insample", "foot", "inter")
# 通道分组（每人 66 通道 = 关节 0..21 各 3 维；关节 0 为 root）：只用于训练日志的分组损失。
CHANNEL_GROUPS = OrderedDict(
    [
        ("root", (0,)),
        ("legs", BODY_PARTS["legs"]),
        ("arms", BODY_PARTS["arms"]),
        ("other", tuple(j for j in range(1, 22) if j not in BODY_PARTS["legs"] and j not in BODY_PARTS["arms"])),
    ]
)
T_QUARTILES = 4
FOOT_SCALE_WINDOWS = 2048
BODY22 = 22


def _channel_mask(joints, channels=66):
    mask = torch.zeros(channels, dtype=torch.bool)
    for joint in joints:
        mask[3 * joint : 3 * joint + 3] = True
    return mask


def _lr_at(args, step):
    """第 step 次更新（1 起）：前 warmup_steps 步线性预热，之后恒定。"""
    if args.warmup_steps <= 0:
        return args.lr
    return args.lr * min(1.0, float(step) / float(args.warmup_steps))


class OnlineBase(object):
    """应急臂用：按残差库记录的折模型在线重算 P 与 GT xyz（no_grad）。"""

    def __init__(self, bank, parent_cache_dir, device):
        config = bank.config
        self.cache = NTU2PXYZSeqCache.load(parent_cache_dir, "train", manifest_path=config["parent_manifest"], device=device)
        self.models = [load_ntu2p_model_checkpoint(path, device)[0].eval() for path in config["base_checkpoints"]]
        self.mode = config["mode"]
        self.bank = bank

    @torch.no_grad()
    def windows(self, ids):
        ids = ids.to(self.bank.device)
        seq, start = self.bank.seq_index[ids].cpu(), self.bank.start[ids].cpu()
        window = self.cache.gather_windows(seq, start, WINDOW_LEN)
        obs, target = window[:, :OBS_LEN].contiguous(), window[:, OBS_LEN:].contiguous()
        action = self.bank.action[ids].to(obs.device)
        fold = self.bank.fold[ids].cpu() if self.mode == "oof" else torch.zeros_like(seq)
        base = torch.zeros_like(target)
        for model_id, model in enumerate(self.models):
            pick = torch.nonzero(fold == model_id, as_tuple=False).view(-1)
            if int(pick.numel()) > 0:
                pick = pick.to(obs.device)
                base[pick] = forward_details(model, obs[pick], action[pick])["pred"]
        return obs, target, base


def _inter_terms(pred_free, target):
    """ReGenNet 式 L_inter 的 xyz 版本：body22 双人相对关节向量 (x_A − x_B) 的 MSE（含 pelvis 相对平移）。"""
    relative_pred = pred_free[:, :, 0, :BODY22] - pred_free[:, :, 1, :BODY22]
    relative_gt = target[:, :, 0, :BODY22] - target[:, :, 1, :BODY22]
    return (relative_pred - relative_gt).pow(2).mean()


def _estimate_arm_scales(args, bank, online, codec, device):
    """应急臂归一化常数：foot 按纯滑行参考的脚滑，inter 按 x̂0=0（纯 v2）的值；在前 2048 个干净窗口上估计。"""
    scales = OrderedDict()
    ids = torch.nonzero(bank.clean.cpu(), as_tuple=False).view(-1)[:FOOT_SCALE_WINDOWS]
    foot_sum, inter_sum, count = 0.0, 0.0, 0
    with torch.no_grad():
        for begin in range(0, int(ids.numel()), 256):
            batch = ids[begin : begin + 256].to(bank.device)
            obs, target, base = online.windows(batch)
            size = int(obs.shape[0])
            if args.foot_loss_weight > 0:
                foot_sum += float(foot_skate_loss(slide_reference(obs, target), target, obs, estimate_up(obs)).item()) * size
            if args.inter_loss_weight > 0:
                inter_sum += float(_inter_terms(base, target).item()) * size
            count += size
    if args.foot_loss_weight > 0:
        scales["foot"] = max(foot_sum / max(count, 1), 1e-8)
    if args.inter_loss_weight > 0:
        scales["inter"] = max(inter_sum / max(count, 1), 1e-8)
    return scales


def _check_args(args):
    if args.arm not in ARMS:
        raise ValueError("--arm 必须是 {}".format(ARMS))
    uses_arm = args.foot_loss_weight > 0 or args.inter_loss_weight > 0
    if uses_arm and not args.parent_cache_dir:
        raise ValueError("应急臂（--foot_loss_weight/--inter_loss_weight > 0）需要 --parent_cache_dir 在线重算底座输出")
    if not 0.0 <= args.root_known_prob <= 1.0:
        raise ValueError("--root_known_prob 必须在 [0,1] 内")


def run(args):
    _check_args(args)
    fixseed(args.seed)
    device = _device(args)
    args.device = str(device)
    os.makedirs(args.save_dir, exist_ok=True)
    bank = NTU2PResidualBank.load(args.bank, device=device)
    sampler = bank.make_sampler(args.batch_size, args.seed)
    model = NTU2PResDiffDenoiser(
        stats=bank.stats,
        num_coeffs=bank.num_coeffs,
        channels=bank.channels,
        obs_len=OBS_LEN,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
    ).to(device)
    diffusion = build_training_diffusion()
    schedule = UniformSampler(diffusion)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = _ema_model(model) if args.ema_decay > 0 else None
    codec = ResidualCodec(bank.num_coeffs, PRED_LEN).to(device)
    uses_arm = args.foot_loss_weight > 0 or args.inter_loss_weight > 0
    online = OnlineBase(bank, args.parent_cache_dir, device) if uses_arm else None
    projector = SkeletonProjector().to(device) if uses_arm else None
    arm_scales = _estimate_arm_scales(args, bank, online, codec, device) if uses_arm else OrderedDict()

    args.num_params = count_parameters(model)
    args.model_config = model.config()
    args.bank_config_sha256 = bank.config_sha256()
    args.bank_summary = bank.summary()
    args.arm_scales = arm_scales
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))
    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    extra = OrderedDict(
        [
            ("codec_config", codec.config()),
            ("bank_path", args.bank),
            ("bank_config_sha256", args.bank_config_sha256),
            ("bank_mode", bank.config.get("mode")),
            ("protocol", args.protocol or bank.config.get("protocol")),
            ("arm", args.arm),
            ("seed", int(args.seed)),
            ("device", str(device)),
        ]
    )
    print("Training NTU2P Track B resdiff arm={} params={} bank={} clean={} device={}".format(
        args.arm, args.num_params, len(bank), int(bank.clean.sum().item()), device))

    group_masks = OrderedDict((name, _channel_mask(joints).to(device)) for name, joints in CHANNEL_GROUPS.items())
    checkpoint = None
    for step in range(1, args.num_steps + 1):
        model.train()
        ids = sampler.sample()
        batch = bank.batch(ids)
        x0 = batch["target"] / model.sigma_target
        size = int(x0.shape[0])
        t, _ = schedule.sample(size, device)
        x_t = diffusion.q_sample(x0, t, torch.randn_like(x0))
        root_known = torch.rand(size, device=device) < args.root_known_prob
        y = make_condition(model, batch, batch["action"], "F", root_value=x0[..., :3], root_known=root_known)
        x0_hat = model(x_t, t, y)
        mask = torch.ones_like(x0)
        mask[..., :3] = (~root_known).to(mask.dtype).view(-1, 1, 1, 1)
        squared = (x0_hat - x0).pow(2) * mask
        loss_dm = squared.sum() / mask.sum().clamp_min(1.0)
        loss = loss_dm
        arm_values, arm_weighted = OrderedDict(), OrderedDict()
        if uses_arm:
            obs, target, base = online.windows(ids.to(device))
            frames = codec_frames(obs)
            residual = codec.residual_camera(to_meters(model, x0_hat), frames)
            if args.foot_loss_weight > 0:
                # 只对低噪声样本加：t 很小时 x_t 已带 GT 相位，只要求解码侧踩实，不改变高噪声阶段对相位的采样。
                low = t < args.foot_loss_max_t
                if bool(low.any()):
                    pred = projector(base[low] + residual[low], obs[low][:, -1])
                    arm_values["foot"] = foot_skate_loss(pred, target[low], obs[low], estimate_up(obs[low]))
                    arm_weighted["foot"] = args.foot_loss_weight * arm_values["foot"] / arm_scales["foot"]
            if args.inter_loss_weight > 0:
                arm_values["inter"] = _inter_terms(base + residual, target)
                arm_weighted["inter"] = args.inter_loss_weight * arm_values["inter"] / arm_scales["inter"]
            for value in arm_weighted.values():
                loss = loss + value
        if not bool(torch.isfinite(loss)):
            raise ValueError("训练 loss 非有限（step {}）".format(step))
        lr = _lr_at(args, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm if args.clip_grad_norm > 0 else float("inf"))
        optimizer.step()
        if ema is not None:
            _update_ema(ema, model, args.ema_decay)

        should_save = step % args.save_interval == 0 or step == args.num_steps
        if step == 1 or step % args.log_interval == 0 or should_save:
            with torch.no_grad():
                record = OrderedDict([("step", int(step)), ("loss", float(loss.item())), ("loss_dm", float(loss_dm.item()))])
                per_channel = squared.sum(dim=(0, 1, 2))
                count_channel = mask.sum(dim=(0, 1, 2))
                for name, channel_mask in group_masks.items():
                    denom = float(count_channel[channel_mask].sum().item())
                    record["loss_" + name] = float(per_channel[channel_mask].sum().item()) / denom if denom > 0 else float("nan")
                per_sample = squared.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1.0)
                bucket = (t * T_QUARTILES) // DIFFUSION_STEPS
                for quartile in range(T_QUARTILES):
                    pick = bucket == quartile
                    record["loss_tq{}".format(quartile)] = float(per_sample[pick].mean().item()) if bool(pick.any()) else float("nan")
                for name, value in arm_values.items():
                    record[name] = float(value.item())
                    record[name + "_share"] = float((arm_weighted[name] / loss).item())
                record["grad_norm"] = float(grad_norm)
                record["lr"] = float(lr)
                record["root_known_fraction"] = float(root_known.float().mean().item())
                record["device"] = str(device)
                record["created_at"] = _utc_now()
            if should_save:
                path = os.path.join(args.save_dir, "model{:09d}.pt".format(step))
                checkpoint = save_ntu2p_resdiff_checkpoint(path, model, OrderedDict(extra, step=int(step), num_trainable=int(args.num_params)))
                record["checkpoint"] = checkpoint
                if ema is not None:
                    ema_path = os.path.join(args.save_dir, "ema", "model{:09d}.pt".format(step))
                    record["ema_checkpoint"] = save_ntu2p_resdiff_checkpoint(
                        ema_path, ema, OrderedDict(extra, step=int(step), num_trainable=int(args.num_params), ema_decay=float(args.ema_decay))
                    )
            _append_log(log_path, record)
            print("step[{}]: loss[{:.5f}] root[{:.4f}] legs[{:.4f}] arms[{:.4f}] lr[{:.2e}]".format(
                step, record["loss"], record["loss_root"], record["loss_legs"], record["loss_arms"], lr), flush=True)
    print("Training finished. final_checkpoint={}".format(checkpoint))
    return OrderedDict([("checkpoint", checkpoint), ("model", model), ("ema", ema)])


def build_arg_parser():
    parser = argparse.ArgumentParser(description="NTU2P Track B 残差扩散生成器训练")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--protocol", default=None, help="缺省取残差库 config 的 protocol")
    parser.add_argument("--arm", choices=ARMS, default="main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--root_known_prob", type=float, default=0.25)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow_cpu_for_smoke_test", action="store_true", help="仅冒烟测试：允许 --device cpu；正式训练必须 CUDA")
    # 应急臂：默认关闭，训练与主臂逐位相同。
    parser.add_argument("--foot_loss_weight", type=float, default=0.0, help="C1：t<foot_loss_max_t 的样本解码后的 GT 接触帧脚滑权重")
    parser.add_argument("--foot_loss_max_t", type=int, default=300)
    parser.add_argument("--inter_loss_weight", type=float, default=0.0, help="C2：投影前 body22 双人相对向量 MSE 权重")
    parser.add_argument("--parent_cache_dir", default=None, help="应急臂在线重算底座输出用的协议 train 缓存")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
