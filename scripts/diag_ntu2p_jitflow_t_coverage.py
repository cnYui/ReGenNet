"""诊断（只用 train 窗口）：J1 EMA 起点按 JiT 原 t 分布 / 50% 均匀混合各微调 800 步，比较近干净段去噪误差与样本幅度。

结果见 docs/ai/context/20261018-205749-ntu2p-jitflow-t-distribution-amendment.md。
"""
import argparse, json, random, torch
from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, NTU2PXYZTrainSampler, eval_windows
from model.forecasting_ntu2p_jitflow import load_ntu2p_jitflow_checkpoint
from train.train_ntu2p_jitflow import flow_loss
ck = "save/forecasting/ntu120_label/ntu2p_jitflow_subjval_J1_s0_5000/ema/model000005000.pt"
cache = NTU2PXYZSeqCache.load("results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval", "train", device="cuda:0")
w = eval_windows(cache)
idx = torch.arange(0, len(cache), 7)[:128]
obs, tgt, act = w["obs_xyz"][idx], w["target_xyz"][idx], w["action"][idx]
def report(m, tag):
    m.eval(); torch.manual_seed(0)
    with torch.no_grad():
        prep = m.prepare(obs, act); x = m.encode_target(prep, tgt); B = obs.shape[0]
        out = {}
        for t in (0.5, 0.8, 0.9, 0.95, 0.99):
            tt = torch.full((B,), t, device="cuda:0")
            z = (t * x[0] + (1 - t) * torch.randn_like(x[0]), t * x[1] + (1 - t) * torch.randn_like(x[1]))
            xh = m.denoise(prep, *z, tt)
            out["t%.2f" % t] = round(float((xh[0] - x[0]).pow(2).mean().sqrt()), 3)
        m.reset_sampler(1); r, l = m.sample_ode(prep, steps=20)
        out["sample_root_rms"] = round(float(r.pow(2).mean().sqrt()), 3); out["data_root_rms"] = round(float(x[0].pow(2).mean().sqrt()), 3)
        out["sample_local_rms"] = round(float(l.pow(2).mean().sqrt()), 3); out["data_local_rms"] = round(float(x[1].pow(2).mean().sqrt()), 3)
        S = torch.stack([m.decode_xyz(prep, *m.sample_ode(prep, steps=20))[0] for _ in range(4)])
        out["single_mpjpe"] = round(float((S - tgt).norm(dim=-1).mean()), 4)
    print(tag, json.dumps(out))
for mix in (0.0, 0.5):
    m, _ = load_ntu2p_jitflow_checkpoint(ck, "cuda:0")
    if mix == 0.0: report(m, "J1 原始")
    m.train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    sampler = NTU2PXYZTrainSampler(cache, 8, seed=5)
    args = argparse.Namespace(t_logit_mean=-0.8, t_logit_std=0.8, t_uniform_mix=mix, sc_prob=0.5)
    torch.manual_seed(0); random.seed(0)
    for step in range(800):
        b = sampler.sample(); prep = m.prepare(b["obs_xyz"], b["action"])
        with torch.no_grad(): x = m.encode_target(prep, b["target_xyz"])
        loss = flow_loss(m, prep, *x, args); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    report(m, "微调800步 mix=%.1f" % mix)
