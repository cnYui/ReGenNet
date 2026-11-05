"""诊断：InterMixer 输出基（root K=8、局部 K=20、×saturate ramp）对 GT 的最小二乘投影是否把推人后退起点提前。

结果记录见 docs/ai/context/20261018-201532-ntu2p-jit-flow-rl-design-and-plan.md 第 2 节。
"""
import torch
from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache
from utils.ntu_smplx_2p_xyz import dct_matrix
from utils.ntu2p_contact_graph import _event_state, PUSHING_ACTION

CACHE = "results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval"
MAN = "results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json"
OBS, PRED = 10, 50
D = dct_matrix(OBS + PRED).double()
ramp = (torch.arange(PRED, dtype=torch.float64) / 5.0).clamp(max=1.0)

def proj(k):
    B = ramp[:, None] * D[:k, OBS:].T  # [T,k]
    return B @ torch.linalg.pinv(B) if hasattr(torch, "linalg") and hasattr(torch.linalg, "pinv") else B @ torch.pinverse(B)

def project(obs, fut, kr, kl):
    last = obs[:, -1:]
    delta = (fut - last).double()
    root = delta[:, :, :, :1]
    local = delta - root
    Pr, Pl = proj(kr), proj(kl)
    root_p = torch.einsum("st,btpjc->bspjc", Pr, root)
    local_p = torch.einsum("st,btpjc->bspjc", Pl, local)
    return (last.double() + root_p + local_p)

def windows(cache, push_only, stride):
    obs, fut, act = [], [], []
    for i in range(len(cache)):
        a = int(cache.actions[i])
        if push_only and a != PUSHING_ACTION:
            continue
        seq = cache.sequence(i)
        L = seq.shape[0]
        starts = range(0, L - OBS - PRED + 1, stride) if stride else [(L - OBS - PRED) // 2]
        for s in starts:
            w = seq[s:s + OBS + PRED]
            obs.append(w[:OBS]); fut.append(w[OBS:]); act.append(a)
    return torch.stack(obs).double(), torch.stack(fut).double(), torch.tensor(act)

def mpjpe(a, b):
    return float((a - b).norm(dim=-1).mean())

torch.set_grad_enabled(False)
for split, stride in (("train", 1), ("val", 1), ("val", 0)):
    cache = NTU2PXYZSeqCache.load(CACHE, split, manifest_path=MAN)
    obs, fut, act = windows(cache, True, stride)
    print("== {} stride={} push windows {}".format(split, stride, obs.shape[0]))
    for kr, kl in ((8, 20), (20, 20), (30, 30), (50, 50)):
        p = project(obs, fut, kr, kl)
        line = "K_root={:2d} K_local={:2d} oracle mpjpe={:.4f} body22={:.4f}".format(kr, kl, mpjpe(p, fut), mpjpe(p[..., :22, :], fut[..., :22, :]))
        for thr in (0.20,):
            for spd in (0.10, 0.20):
                ref = _event_state(obs, fut, thr, spd)
                got = _event_state(obs, p, thr, spd)
                m = ref["reliable"] & (got["retreat"] >= 0)
                mc = ref["reliable"] & (got["contact"] >= 0)
                dr = (got["retreat"] - ref["retreat"]).double()[m] / 20.0
                dc = (got["contact"] - ref["contact"]).double()[mc] / 20.0
                line += " | thr{:.2f} spd{:.2f} n={} retreat_bias={:+.3f}s contact_bias={:+.3f}s".format(thr, spd, int(ref["reliable"].sum()), float(dr.mean()) if dr.numel() else float("nan"), float(dc.mean()) if dc.numel() else float("nan"))
        print(line)
