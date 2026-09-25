"""自检 A3 检索库与 RetrievalAnchor（可在 CPU 上跑；--device cuda:0 时同时给出 GPU 查询耗时）。

1. 检索库：保存/加载一致；存下的规范系参数与直接重算一致；键与直接计算一致；
2. 查询：首选层邻居同动作、异受试者；全部邻居异于 exclude_seq；每条序列至多 1 个邻居；层级有序、层内距离升序；
   返回距离与直接计算一致；disp 与"邻居规范系下未来 - 观测末帧"的直接重算一致；稀有动作的回退；
3. RetrievalAnchor：初始权重 = 按 τ0 的距离软平均；β = 0 时 anchor 与 base 逐位相同；ramp[0] = 0 时首帧与 base 逐位相同；
   Δ_knn = Δ_base 时 anchor = base；tokens 形状；反向传播梯度有限；
4. batch 8、k=16 查询耗时。
"""

import argparse
import json
import os
import time
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu2p_retrieval_bank import (
    DEFAULT_BANK_DIR,
    TIER_OTHER_ACTION,
    TIER_SAME_PERFORMER,
    NTU2PRetrievalBank,
    performer_ids,
)
from data_loaders.forecasting.ntu2p_xyz_seq_cache import DEFAULT_CACHE_DIR, NTU2PXYZSeqCache, eval_windows
from model.forecasting_ntu2p_residual_xyz import build_residual_ramp
from model.ntu2p_retrieval_anchor import RetrievalAnchor
from utils.ntu2p_canonical import canonical_frame, to_canonical


DEFAULT_MANIFEST = "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"
BANK_FILE = "train_retrieval_bank.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--bank_path", default=os.path.join(DEFAULT_BANK_DIR, BANK_FILE))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_threads", type=int, default=3)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _max_abs(a, b):
    return float((a - b).abs().max().item())


def check_bank_tables(bank, fresh):
    result = OrderedDict()
    for name in ("seq_index", "start", "action", "performer", "rotation", "translation", "up"):
        result["load_vs_build_" + name] = _max_abs(getattr(bank, name).double(), getattr(fresh, name).double())
    result["load_vs_build_keys"] = _max_abs(bank.keys_centered + bank.key_mean, fresh.keys_centered + fresh.key_mean)
    obs = bank.train_cache.gather_windows(bank.seq_index, bank.start, bank.obs_len).to(bank.device)
    frame = canonical_frame(obs, **bank.canonical_kwargs)
    result["frame_rotation_max_abs"] = _max_abs(frame["rotation"], bank.rotation)
    result["frame_translation_max_abs"] = _max_abs(frame["translation"], bank.translation)
    result["frame_up_max_abs"] = _max_abs(frame["up"], bank.up)
    keys = bank.key(to_canonical(obs, frame))
    result["key_vs_direct_max_abs"] = _max_abs(keys, bank.keys_centered + bank.key_mean)
    return result


def check_neighbors(bank, obs_canon, action, performer, exclude_seq, out, prefix):
    result = OrderedDict()
    index, tier = out["index"], out["tier"]
    primary = tier == 0
    result[prefix + "primary_same_action"] = bool((bank.action[index] == action.view(-1, 1))[primary].all())
    if performer is not None:
        result[prefix + "primary_other_performer"] = bool((bank.performer[index] != performer.view(-1, 1))[primary].all())
        same_perf = (bank.performer[index] == performer.view(-1, 1)) == ((tier & TIER_SAME_PERFORMER) != 0)
        result[prefix + "tier_bit_same_performer_consistent"] = bool(same_perf.all())
    other_action = (bank.action[index] != action.view(-1, 1)) == ((tier & TIER_OTHER_ACTION) != 0)
    result[prefix + "tier_bit_other_action_consistent"] = bool(other_action.all())
    if exclude_seq is not None:
        result[prefix + "never_exclude_seq"] = bool((bank.seq_index[index] != exclude_seq.view(-1, 1)).all())
    seq = bank.seq_index[index].sort(dim=1)[0]
    result[prefix + "one_window_per_sequence"] = bool((seq[:, 1:] != seq[:, :-1]).all())
    # 默认 other_action 回退的层级代价：异动作 1、同受试者 2；代价不降、同代价内距离不降。
    cost = ((tier & TIER_OTHER_ACTION) != 0).long() + 2 * ((tier & TIER_SAME_PERFORMER) != 0).long()
    dist = out["dist"]
    same_cost = cost[:, 1:] == cost[:, :-1]
    result[prefix + "tier_order_nondecreasing"] = bool((cost[:, 1:] >= cost[:, :-1]).all())
    result[prefix + "dist_sorted_within_tier"] = bool((dist[:, 1:] >= dist[:, :-1] - 1e-6)[same_cost].all())
    direct = bank.key(obs_canon).unsqueeze(1) - (bank.keys_centered + bank.key_mean)[index]
    result[prefix + "dist_vs_direct_max_abs"] = _max_abs(direct.pow(2).sum(-1).sqrt(), dist)
    result[prefix + "fallback_frac"] = float(out["fallback"].float().mean().item())
    return result


def check_displacement(bank, out):
    """用重算的邻居规范系直接求 disp，对比检索返回值；并记录首个未来帧的位移量（= 邻居一帧运动，按定义不为 0）。"""
    index = out["index"][:, :4].reshape(-1)
    window = bank.train_cache.gather_windows(
        bank.seq_index[index], bank.start[index], bank.obs_len + bank.pred_len
    ).to(bank.device)
    obs = window[:, : bank.obs_len]
    frame = canonical_frame(obs, **bank.canonical_kwargs)
    canon = to_canonical(window, frame)
    direct = canon[:, bank.obs_len :] - canon[:, bank.obs_len - 1 : bank.obs_len]
    got = out["disp"][:, :4].reshape(direct.shape)
    first_step = torch.norm(got[:, 0], dim=-1)
    return OrderedDict(
        [
            ("disp_vs_direct_max_abs", _max_abs(got, direct)),
            ("disp_first_future_frame_mean_norm", float(first_step.mean().item())),
            ("disp_first_future_frame_vs_one_step_motion_max_abs", _max_abs(got[:, 0], canon[:, bank.obs_len] - canon[:, bank.obs_len - 1])),
        ]
    )


def check_anchor(bank, obs_canon, out, device):
    torch.manual_seed(0)
    tau0 = float(bank.config.get("median_topk_dist", 0.074))
    k = int(out["dist"].shape[1])
    anchor = RetrievalAnchor(init_tau=tau0).to(device)
    batch = int(obs_canon.shape[0])
    base = obs_canon[:, -1:] + 0.01 * torch.randn(batch, 50, 2, 55, 3, device=device).cumsum(dim=1)
    ramp = build_residual_ramp(50, "saturate", 5).to(device)
    result = OrderedDict()
    res = anchor(obs_canon, base, out["disp"], out["dist"], ramp, out["tier"])
    result["tokens_shape"] = list(res["tokens"].shape)
    result["tokens_shape_ok"] = list(res["tokens"].shape) == [2 * k, batch, anchor.d_model]
    result["init_weights_vs_distance_softmax_max_abs"] = _max_abs(res["weights"], torch.softmax(-out["dist"] / tau0, dim=1))
    result["first_frame_equals_base"] = bool(torch.equal(res["anchor"][:, 0], base[:, 0]))
    with torch.no_grad():
        anchor.beta_root.normal_()
        anchor.beta_local.normal_()
    res = anchor(obs_canon, base, out["disp"], out["dist"], ramp, out["tier"])
    result["first_frame_equals_base_random_beta"] = bool(torch.equal(res["anchor"][:, 0], base[:, 0]))
    result["later_frames_differ_random_beta"] = bool((res["anchor"][:, 5:] != base[:, 5:]).any())
    (res["anchor"].pow(2).mean() + res["tokens"].pow(2).mean()).backward()
    result["grads_finite"] = all(
        param.grad is not None and bool(torch.isfinite(param.grad).all()) for param in anchor.parameters()
    )
    result["score_head_receives_grad"] = bool(anchor.score_mlp[-1].weight.grad.abs().sum() > 0)
    with torch.no_grad():
        anchor.beta_root.zero_()
        anchor.beta_local.zero_()
        res = anchor(obs_canon, base, out["disp"], out["dist"], ramp, out["tier"])
        result["beta_zero_anchor_equals_base"] = bool(torch.equal(res["anchor"], base))
        anchor.beta_root.normal_()
        anchor.beta_local.normal_()
        same = (base - obs_canon[:, -1:]).unsqueeze(1).expand_as(out["disp"]).contiguous()
        res = anchor(obs_canon, base, same, out["dist"], ramp, out["tier"])
        result["knn_equals_base_anchor_vs_base_max_abs"] = _max_abs(res["anchor"], base)
    return result


def time_query(bank, obs_canon, action, performer, k, repeats=20):
    for _ in range(3):
        bank.query(obs_canon, action, performer, k=k)
    if bank.device.type == "cuda":
        torch.cuda.synchronize(bank.device)
    begin = time.time()
    for _ in range(repeats):
        bank.query(obs_canon, action, performer, k=k)
    if bank.device.type == "cuda":
        torch.cuda.synchronize(bank.device)
    return (time.time() - begin) / repeats * 1000.0


def main():
    args = parse_args()
    torch.set_num_threads(int(args.num_threads))
    torch.set_grad_enabled(True)
    device = torch.device(args.device)
    train = NTU2PXYZSeqCache.load(args.cache_dir, "train", manifest_path=args.manifest_path, device=device)
    bank = NTU2PRetrievalBank.load(args.bank_path, train, device=device)
    fresh = NTU2PRetrievalBank.build(
        train,
        stride=bank.config["stride"],
        key_joints=bank.config["key_joints"],
        velocity_weight=bank.config["velocity_weight"],
        canonical_kwargs=bank.config["canonical_kwargs"],
        device=device,
    )
    report = OrderedDict()
    report["tables"] = check_bank_tables(bank, fresh)

    val = NTU2PXYZSeqCache.load(args.cache_dir, "val", manifest_path=args.manifest_path, device=device)
    windows = eval_windows(val)
    obs_canon, _, _ = bank.canonicalize(windows["obs_xyz"])
    action = windows["action"].to(device)
    performer = performer_ids(val.sample_ids).to(device)
    with torch.no_grad():
        out = bank.query(obs_canon, action, performer, k=args.k)
        report["val_excl"] = check_neighbors(bank, obs_canon, action, performer, None, out, "")
        report["displacement"] = check_displacement(bank, out)

        lengths = train.lengths_cpu
        seq = torch.arange(len(train), dtype=torch.long)
        start = (lengths - (bank.obs_len + bank.pred_len)) // 2
        train_obs = train.gather_windows(seq, start, bank.obs_len).to(device)
        train_canon, _, _ = bank.canonicalize(train_obs)
        train_action = train.actions.to(device)
        seq = seq.to(device)
        out_train = bank.query_train(train_canon, train_action, seq, k=args.k)
        report["train_query"] = check_neighbors(
            bank, train_canon, train_action, bank.seq_performers[seq], seq, out_train, ""
        )

        # 稀有动作：train 中受试者最少的动作，用其唯一受试者的序列做训练查询，必然触发回退。
        counts = [(int(torch.unique(bank.performer[bank.action == a]).numel()), a) for a in torch.unique(bank.action).tolist()]
        rare_action = min(counts)[1]
        rare_seq = torch.nonzero(train_action == rare_action, as_tuple=False).view(-1)[:1]
        rare = bank.query_train(train_canon[rare_seq], train_action[rare_seq], rare_seq, k=args.k)
        report["rare_action"] = OrderedDict(
            [
                ("action", int(rare_action)),
                ("performers_in_bank", int(min(counts)[0])),
                ("tiers", rare["tier"][0].tolist()),
                ("fallback", bool(rare["fallback"][0].item())),
            ]
        )
        rare_sp = bank.query_train(train_canon[rare_seq], train_action[rare_seq], rare_seq, k=args.k, fallback="same_performer")
        report["rare_action"]["tiers_same_performer_mode"] = rare_sp["tier"][0].tolist()

    report["anchor"] = check_anchor(bank, obs_canon[:8], OrderedDict((key, value[:8]) for key, value in out.items()), device)
    with torch.no_grad():
        report["query_ms_batch8_k{}".format(args.k)] = time_query(bank, obs_canon[:8], action[:8], performer[:8], args.k)
    report["device"] = str(device)

    failures = []
    for section, values in report.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, bool) and not value and key != "fallback":
                failures.append("{}.{}".format(section, key))
            if key.endswith("max_abs") and float(value) > 1e-4:
                failures.append("{}.{}={}".format(section, key, value))
    report["failures"] = failures
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    if failures:
        raise SystemExit("自检未通过: {}".format(failures))


if __name__ == "__main__":
    main()
