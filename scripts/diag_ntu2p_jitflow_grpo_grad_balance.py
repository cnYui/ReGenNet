"""诊断（train batch）：GRPO 策略梯度（逐元素均值 / 求和对数似然）与端点配方损失的梯度范数比。

结果见 docs/ai/context/20261018-223934-ntu2p-jitflow-grpo-gradient-balance-amendment.md。
"""
import argparse, copy, json, torch
from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, NTU2PXYZTrainSampler
from model.forecasting_ntu2p_jitflow import _repeat, load_ntu2p_jitflow_checkpoint
from train.train_ntu2p_jitflow import endpoint_loss
from train.train_ntu2p_jitflow_grpo import rollout, _step_stats, reward_terms, total_reward, REWARD_SETS
from train.train_ntu2p_v2 import _base_loss_args
torch.manual_seed(0)
ck = "save/forecasting/ntu120_label/ntu2p_jitflow_subjval_J4-tmix_s0_5000/ema/model000005000.pt"
policy, _ = load_ntu2p_jitflow_checkpoint(ck, "cuda:0"); policy.eval()
ref = copy.deepcopy(policy)
base = json.load(open("save/forecasting/ntu120_label/ntu2p_jitflow_subjval_J4-tmix_s0_5000/args.json"))
ns = argparse.Namespace(**base)
cache = NTU2PXYZSeqCache.load("results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval", "train", device="cuda:0")
b = NTU2PXYZTrainSampler(cache, 4, seed=3).sample()
obs, tgt, act = b["obs_xyz"], b["target_xyz"], b["action"]
G = 8
prep = _repeat(policy.prepare(obs, act), G)
final, records = rollout(policy, ref, prep, 10, 0.7)
with torch.no_grad():
    xyz = policy.decode_xyz(prep, *final)[0]
    r = total_reward(reward_terms(xyz, tgt.repeat_interleave(G, 0), obs.repeat_interleave(G, 0)), REWARD_SETS["F"]).view(4, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1).float()
def gnorm():
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in policy.parameters() if p.grad is not None)))
for mode in ("mean", "sum"):
    policy.zero_grad()
    for rec in records:
        g = _repeat(policy.prepare(obs, act), G)
        _, mean = _step_stats(policy, g, rec["z"], rec["t"], rec["sc"], rec["s"], 0.7)
        parts = torch.cat([((a - m) ** 2).flatten(1) for a, m in zip(rec["z_next"], mean)], 1)
        logp = -(parts.mean(1) if mode == "mean" else parts.sum(1)) / (2 * rec["sigma"] ** 2)
        (-(logp * adv).mean() / len(records)).backward()
    print(mode, "pg grad norm", gnorm())
policy.zero_grad()
sup = endpoint_loss(policy, policy.prepare(obs, act), obs, tgt, ns, _base_loss_args(ns), base["loss_scales"])[0]
sup.backward(); print("sup grad norm", gnorm(), "sup", float(sup))
