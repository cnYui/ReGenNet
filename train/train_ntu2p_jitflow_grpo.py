"""JiT-RL：在 JiT 流模型上做 GRPO 奖励微调（设计 3.5）。

策略为随机 flow-DDIM：在线性 t 网格上，x̂ = f(z_t, t)，ε̂ = (z_t − t·x̂)/(1−t)，
z_s = s·x̂ + (1−s)·(√(1−ρ²)·ε̂ + ρ·ξ)，即每步均值 μ = s·x̂ + (1−s)√(1−ρ²)·ε̂、标准差 σ = (1−s)ρ 的各向同性高斯；
末步 s=1 时 σ=0，直接落到 x̂，不计入策略似然。训练用 10 步（Flow-GRPO 的去噪步数缩减），评价仍用 ODE。

每个条件采 G 个样本，组内标准化优势；PPO-clip，对冻结参考策略的逐步高斯 KL（同一输入下的闭式解
‖μ_θ − μ_ref‖²/(2σ²)）；对数似然与 KL 按元素取均值（Flow-GRPO 实现的做法，避免 1728 维求和导致比值爆炸）。
另加确定性端点的 FD2 配方损失（权重 --sup_weight），防止共享网络让点预测读出漂移。
按元素取均值后策略梯度只有端点项的约 1/47（J4、train batch 实测梯度范数 0.082 vs 3.88；按元素求和则是 281，
反过来淹没端点项），初版 1500 步里 RL-A/RL-F 轨迹几乎相同，奖励不起作用；因此 PG+KL 项乘 --pg_scale（默认 50），
使两者初始梯度同量级。该倍数只由 train 上的梯度诊断确定，不看 val。

奖励只在训练中使用 train 的 GT（设计 3.5）：
- acc：−mpjpe（身体 25 + 4 指根，FD2 主损失子集）/ 0.17；
- rel：−双向手—躯干距离曲线 RMSE / 0.20；
- ant：−Σ relu(A_样本 − A_GT)（GT 有新接触的受动者）/ 0.03；
- phys：−（穿地深度超出 GT 的部分 / 0.01 + jerk 超出 GT 的部分 / 20）/ 2。
RL-A 只用 acc；RL-F = acc + 0.5 rel + 0.5 ant + 0.25 phys。尺度取 FD2 在 val 上的量级，只用于把各项放到可比范围。
"""

import argparse
import copy
import json
import math
import os
import random
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, NTU2PXYZTrainSampler
from model.forecasting_ntu2p_jitflow import NUM_LOCAL, _repeat, load_ntu2p_jitflow_checkpoint
from train.train_ntu2p_jitflow import endpoint_loss
from train.train_ntu2p_residual_refiner_xyz import _append_log, _ema_model, _update_ema, _utc_now, _write_json
from train.train_ntu2p_v2 import LOSS_JOINT_SUBSETS, _base_loss_args, _device
from utils.fixseed import fixseed
from utils.ntu2p_canonical import estimate_up
from utils.ntu2p_contact_graph import hand_body_distances
from utils.ntu2p_interaction_timing import anticipation_excess
from utils.ntu2p_kinematic_projection import FOOT_JOINTS, observed_ground
from utils.ntu2p_naturalness import NTU2P_FPS


REWARD_SETS = OrderedDict([("A", OrderedDict(acc=1.0)), ("F", OrderedDict(acc=1.0, rel=0.5, ant=0.5, phys=0.25))])
SCALES = OrderedDict(acc=0.17, rel=0.20, ant=0.03, penetration=0.01, jerk=20.0)
PENETRATION_TOLERANCE = 0.02
ACC_JOINTS = list(LOSS_JOINT_SUBSETS["hand2"])


def _jerk(xyz, obs):
    full = torch.cat((obs[:, -3:], xyz), dim=1)[..., :22, :].double()
    jerk = full[:, 3:] - 3 * full[:, 2:-1] + 3 * full[:, 1:-2] - full[:, :-3]
    return jerk.norm(dim=-1).mean(dim=(1, 2, 3)) * NTU2P_FPS ** 3


def _penetration(xyz, obs):
    up = estimate_up(obs)
    ground = observed_ground(obs.to(xyz.dtype), up)  # [B,2]
    feet = xyz[..., list(FOOT_JOINTS), :]
    height = (feet * up.view(-1, 1, 1, 1, 3)).sum(dim=-1) - ground.view(-1, 1, 2, 1)
    return (-height - PENETRATION_TOLERANCE).clamp_min(0.0).double().mean(dim=(1, 2, 3))


def reward_terms(sample, target, obs):
    """逐窗口奖励分项（越小越好的代价，未加权、未取负）。"""
    acc = (sample[..., ACC_JOINTS, :] - target[..., ACC_JOINTS, :]).norm(dim=-1).double().mean(dim=(1, 2, 3))
    dist_s = hand_body_distances(torch.cat((obs, sample), dim=1)).amin(dim=-1)[:, obs.shape[1]:]
    dist_t = hand_body_distances(torch.cat((obs, target), dim=1)).amin(dim=-1)[:, obs.shape[1]:]
    rel = (dist_s - dist_t).double().pow(2).mean(dim=(1, 2, 3)).sqrt()
    ant = anticipation_excess(obs, sample, target).double()
    pen = (_penetration(sample, obs) - _penetration(target, obs)).clamp_min(0.0)
    jerk = (_jerk(sample, obs) - _jerk(target, obs)).clamp_min(0.0)
    return OrderedDict(acc=acc, rel=rel, ant=ant, penetration=pen, jerk=jerk)


def total_reward(terms, weights):
    cost = 0.0
    for name, weight in weights.items():
        if name == "phys":
            value = 0.5 * (terms["penetration"] / SCALES["penetration"] + terms["jerk"] / SCALES["jerk"])
        else:
            value = terms[name] / SCALES[name]
        cost = cost + weight * value
    return -cost


def _step_stats(policy, prep, z, t, sc, s, rho):
    """一步策略：返回 (x̂, μ)，μ = s·x̂ + (1−s)√(1−ρ²)·ε̂。"""
    batch_size = int(z[0].shape[0])
    t_value = t.expand(batch_size)
    estimate = policy.denoise(prep, z[0], z[1], t_value, sc=sc)
    keep = math.sqrt(1.0 - rho * rho)
    means = []
    for x_hat, z_part in zip(estimate, z):
        shape = (batch_size,) + (1,) * (z_part.dim() - 1)
        tv = t_value.view(shape)
        eps_hat = (z_part - tv * x_hat) / (1.0 - tv)
        means.append(s * x_hat + (1.0 - s) * keep * eps_hat)
    return estimate, tuple(means)


def _log_prob(z_next, mean, sigma):
    """逐轨迹的高斯对数似然（按元素取均值，去掉常数）。"""
    parts = [((a - b) ** 2).flatten(1) for a, b in zip(z_next, mean)]
    return -torch.cat(parts, dim=1).mean(dim=1) / (2.0 * sigma * sigma)


def _kl(mean, mean_ref, sigma):
    parts = [((a - b) ** 2).flatten(1) for a, b in zip(mean, mean_ref)]
    return torch.cat(parts, dim=1).mean(dim=1) / (2.0 * sigma * sigma)


@torch.no_grad()
def rollout(policy, reference, prep, steps, rho):
    """采样一组轨迹并记录每个随机步的输入、动作、旧对数似然与参考均值。"""
    batch_size = int(prep["obs_xyz"].shape[0])
    device, dtype = prep["obs_xyz"].device, prep["obs_xyz"].dtype
    z = (torch.randn(batch_size, 2, policy.root_k, 3, device=device, dtype=dtype),
         torch.randn(batch_size, 2, policy.local_k, NUM_LOCAL, 3, device=device, dtype=dtype))
    grid = torch.linspace(0.0, 1.0, int(steps) + 1, device=device, dtype=dtype)
    records, sc = [], None
    for index in range(int(steps)):
        t, s = grid[index], grid[index + 1]
        if index == int(steps) - 1:
            z = policy.denoise(prep, z[0], z[1], t.expand(batch_size), sc=sc)
            break
        estimate, mean = _step_stats(policy, prep, z, t, sc, s, rho)
        _, mean_ref = _step_stats(reference, prep, z, t, sc, s, rho)
        sigma = float((1.0 - s) * rho)
        z_next = tuple(m + sigma * torch.randn_like(m) for m in mean)
        records.append(OrderedDict([("z", z), ("t", t), ("s", s), ("sc", sc), ("z_next", z_next), ("sigma", sigma),
                                    ("logp_old", _log_prob(z_next, mean, sigma)), ("mean_ref", mean_ref)]))
        sc = estimate if policy.self_cond else None
        z = z_next
    return z, records


def run(args):
    fixseed(args.seed)
    random.seed(args.seed)
    device = _device(args)
    args.device = str(device)
    policy, base_state = load_ntu2p_jitflow_checkpoint(args.init_checkpoint, device)
    folder = os.path.dirname(os.path.abspath(args.init_checkpoint))
    args_path = os.path.join(folder, "args.json")
    if not os.path.exists(args_path):
        # EMA 权重在 save_dir/ema/ 下，args.json 在上一级。
        args_path = os.path.join(os.path.dirname(folder), "args.json")
    base_args = json.loads(open(args_path).read())
    # 端点配方损失沿用初始模型训练时的全部损失参数与归一化常数。
    loss_ns = argparse.Namespace(**base_args)
    loss_scales = base_args.get("loss_scales")
    loss_args = _base_loss_args(loss_ns)
    reference = copy.deepcopy(policy).eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    # RL 全程关闭 dropout：采样与重算似然必须是同一个确定函数，否则同参数下比值也不为 1。
    policy.eval()
    cache = NTU2PXYZSeqCache.load(args.cache_dir, "train", manifest_path=args.manifest_path, device=device)
    sampler = NTU2PXYZTrainSampler(cache, args.batch_size, seed=args.seed + 101, obs_len=policy.obs_len, pred_len=policy.pred_len)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    ema = _ema_model(policy) if args.ema_decay > 0 else None
    weights = REWARD_SETS[args.reward]
    os.makedirs(args.save_dir, exist_ok=True)
    args.base_args = base_args
    args.reward_weights = weights
    args.reward_scales = SCALES
    _write_json(os.path.join(args.save_dir, "args.json"), vars(args))
    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    save_args = argparse.Namespace(**dict(base_args, num_params=sum(p.numel() for p in policy.parameters()), arch="jit_flow_grpo",
                                          seed=args.seed, device=str(device), manifest_path=args.manifest_path))
    from train.train_ntu2p_v2 import _save_checkpoint

    print("GRPO reward={} G={} batch={} steps={} rho={} beta={} lr={}".format(args.reward, args.group_size, args.batch_size,
                                                                                args.sample_steps, args.rho, args.beta, args.lr), flush=True)
    checkpoint = None
    for iteration in range(1, args.num_iters + 1):
        batch = sampler.sample()
        obs, target, action = batch["obs_xyz"], batch["target_xyz"], batch["action"]
        prep = policy.prepare(obs, action)
        group_prep = _repeat(prep, args.group_size)
        final, records = rollout(policy, reference, group_prep, args.sample_steps, args.rho)
        with torch.no_grad():
            sample_xyz, _, _ = policy.decode_xyz(group_prep, final[0], final[1])
            terms = reward_terms(sample_xyz, target.repeat_interleave(args.group_size, dim=0), obs.repeat_interleave(args.group_size, dim=0))
            reward = total_reward(terms, weights).view(args.batch_size, args.group_size)
            advantage = ((reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-4)).view(-1).float()
        stats = OrderedDict(pg=0.0, kl=0.0, clip_frac=0.0)
        for _ in range(args.inner_epochs):
            optimizer.zero_grad()
            for record in records:
                # 条件嵌入带梯度；每步单独 backward（控制显存），所以每步重新构建一次。
                group_train = _repeat(policy.prepare(obs, action), args.group_size)
                _, mean = _step_stats(policy, group_train, record["z"], record["t"], record["sc"], record["s"], args.rho)
                logp = _log_prob(record["z_next"], mean, record["sigma"])
                ratio = torch.exp(logp - record["logp_old"])
                clipped = ratio.clamp(1.0 - args.clip, 1.0 + args.clip)
                pg = -torch.min(ratio * advantage, clipped * advantage).mean()
                kl = _kl(mean, record["mean_ref"], record["sigma"]).mean()
                (args.pg_scale * (pg + args.beta * kl) / len(records)).backward()
                stats["pg"] += float(pg.detach()) / len(records) / args.inner_epochs
                stats["kl"] += float(kl.detach()) / len(records) / args.inner_epochs
                stats["clip_frac"] += float(((ratio - 1.0).abs() > args.clip).float().mean()) / len(records) / args.inner_epochs
            if args.sup_weight > 0:
                sup, _, _, _, _ = endpoint_loss(policy, policy.prepare(obs, action), obs, target, loss_ns, loss_args, loss_scales)
                (args.sup_weight * sup).backward()
                stats["sup"] = float(sup.detach())
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        if ema is not None:
            _update_ema(ema, policy, args.ema_decay)
        record = OrderedDict([("iteration", iteration), ("reward_mean", float(reward.mean())), ("reward_std_within", float(reward.std(dim=1).mean()))])
        for name, value in terms.items():
            record[name] = float(value.mean())
        record.update(stats)
        record["created_at"] = _utc_now()
        if iteration == 1 or iteration % args.log_interval == 0:
            print("iter[{}] reward {:.4f} acc {:.4f} rel {:.4f} ant {:.4f} kl {:.5f} clip {:.3f}".format(
                iteration, record["reward_mean"], record["acc"], record["rel"], record["ant"], record["kl"], record["clip_frac"]), flush=True)
        if iteration % args.save_interval == 0 or iteration == args.num_iters:
            checkpoint = _save_checkpoint(save_args, policy, None, iteration, save_dir=args.save_dir)
            record["checkpoint"] = checkpoint
            if ema is not None:
                record["ema_checkpoint"] = _save_checkpoint(save_args, ema, None, iteration, save_dir=os.path.join(args.save_dir, "ema"))
        _append_log(log_path, record)
    print("GRPO finished. final_checkpoint={}".format(checkpoint))
    return checkpoint


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint", required=True, help="第 1 阶段流模型的 EMA 终点")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--reward", choices=tuple(REWARD_SETS), default="F")
    parser.add_argument("--num_iters", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=8, help="每次迭代的条件数")
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--inner_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--sup_weight", type=float, default=1.0)
    parser.add_argument("--pg_scale", type=float, default=50.0, help="PG+KL 项的倍数，使其梯度与端点配方项同量级")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow_cpu_for_smoke_test", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
