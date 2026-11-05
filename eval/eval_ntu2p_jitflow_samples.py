"""JiT 流模型的单样本评价：K 个独立 ODE 样本逐个评价后取平均（设计 4.1/4.3 的 `sample` 读出）。

逐样本：paired xyz_mse/xyz_mae/mpjpe、交互时序（interaction_timing_block）、接触距离、自然度（jerk、穿地、滑行、
步数、步态相位、骨长）；Energy Score 按 utils/ntu2p_probabilistic_metrics 的逐帧逐关节定义，第二项用相邻样本对
(k, k+1) 的无偏估计流式计算，避免同时保存 K 份全 val 预测。外层循环样本、内层循环 batch：第 k 个样本的噪声是
模型 CPU generator 的第 k 段，固定 seed 逐位可复现。
"""

import argparse
import json
import os
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, eval_windows, iter_eval_batches
from eval.eval_ntu2p_residual_refiner_xyz import _add, _finalize
from eval.eval_ntu2p_v2 import naturalness_block
from model.forecasting_ntu2p_jitflow import load_ntu2p_jitflow_checkpoint
from train.train_ntu2p_v2 import _device
from utils.ntu2p_contact_graph import contact_metrics
from utils.ntu2p_interaction_timing import interaction_timing_block
from utils.ntu_smplx_2p_xyz import articulation_ratios, compute_ntu_articulation_metrics, compute_ntu_xyz_metrics, copy_last_xyz


SUMMARY_NATURAL = ("jerk_body", "foot_penetration_ratio", "slide_frame_ratio", "step_count", "gait_sep_corr_f01_10",
                   "gait_sep_corr_f11_20", "bone_rel_err_body", "skate_gt_contact_speed_walk", "leg_swing_amp")


def _mean_dicts(items):
    keys = [key for key, value in items[0].items() if isinstance(value, (int, float)) and value is not None]
    out = OrderedDict()
    for key in keys:
        values = [item[key] for item in items if item.get(key) is not None]
        out[key] = sum(values) / len(values) if values else None
    return out


def evaluate(args):
    device = _device(args)
    cache = NTU2PXYZSeqCache.load(args.cache_dir, args.split, manifest_path=args.manifest_path, device=device)
    windows = eval_windows(cache)
    model, state = load_ntu2p_jitflow_checkpoint(args.checkpoint, device)
    model.nfe_steps, model.solver = int(args.nfe), args.solver
    model.reset_sampler(args.sample_seed)
    obs_all = windows["obs_xyz"].cpu()
    target_all = windows["target_xyz"].cpu()
    actions = windows["action"].cpu()
    # naturalness_block 需要 GT 与 copy-last 作参照（腿摆幅比、前 0.5 s 步态 RMSE 比）。
    copy_all = copy_last_xyz(obs_all, int(target_all.shape[1]))
    per_sample = []
    es_first, es_pair, es_count = 0.0, 0.0, 0
    previous = None
    exported = []
    with torch.no_grad():
        for k in range(int(args.num_samples)):
            preds, totals, artic, count = [], OrderedDict(), OrderedDict(), 0
            for batch in iter_eval_batches(windows, args.batch_size):
                prep = model.prepare(batch["obs_xyz"], batch["action"])
                pred = model.samples_xyz(prep, 1)[0]
                size = int(pred.shape[0])
                _add(totals, compute_ntu_xyz_metrics(pred, batch["target_xyz"], batch["obs_xyz"]), size)
                _add(artic, compute_ntu_articulation_metrics(pred, batch["target_xyz"]), size)
                preds.append(pred.cpu())
                count += size
            pred_all = torch.cat(preds, dim=0)
            metrics = _finalize(totals, count)
            articulation = _finalize(artic, count)
            articulation.update(articulation_ratios(articulation))
            for key in ("dct_low", "dct_mid", "dct_high"):
                metrics[key + "_ratio"] = articulation[key + "_energy"] / max(articulation[key + "_energy_target"], 1e-12)
            timing = interaction_timing_block(obs_all, OrderedDict(sample=pred_all), target_all, actions)
            metrics["primary_excess_antic"] = timing["primary_excess_antic"]["sample"]
            metrics["missed_contact_total"] = sum(timing["all_{:.2f}".format(t)]["sample"]["missed_count"] for t in (0.20, 0.25, 0.30))
            contact = contact_metrics(obs_all, target_all, OrderedDict(sample=pred_all), actions)
            metrics["contact_distance_rmse_all"] = contact["sample"]["all"]["distance_rmse"]
            metrics["contact_distance_rmse_push"] = contact["sample"]["pushing"]["distance_rmse"]
            if k < args.natural_samples:
                natural = naturalness_block(obs_all, target_all, OrderedDict([("sample", pred_all), ("copy_last", copy_all), ("GT", target_all)]))["sample"]
                for key in SUMMARY_NATURAL:
                    metrics[key] = natural.get(key)
            first = (pred_all.double() - target_all.double()).norm(dim=-1)
            es_first += float(first.sum())
            es_count += int(first.numel())
            if previous is not None:
                es_pair += float((pred_all.double() - previous.double()).norm(dim=-1).sum())
            previous = pred_all
            if k < args.export_samples:
                exported.append(pred_all)
            per_sample.append(metrics)
            print("sample {} mpjpe {:.5f} excess_antic {}".format(k, metrics["mpjpe"], metrics["primary_excess_antic"]), flush=True)
    num_pairs = int(args.num_samples) - 1
    energy = es_first / es_count - (0.5 * es_pair / (num_pairs * es_count) if num_pairs > 0 else 0.0)
    result = OrderedDict(
        [
            ("checkpoint", args.checkpoint),
            ("split", args.split),
            ("num_windows", int(obs_all.shape[0])),
            ("num_samples", int(args.num_samples)),
            ("nfe_steps", int(args.nfe)),
            ("solver", args.solver),
            ("sample_seed", int(args.sample_seed)),
            ("checkpoint_step", int(state.get("step", -1))),
            ("mean_over_samples", _mean_dicts(per_sample)),
            ("energy_score", energy),
            ("per_sample", per_sample),
        ]
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    if args.export_arrays and exported:
        methods = OrderedDict(("sample{}".format(index), value) for index, value in enumerate(exported))
        torch.save(OrderedDict([("obs_xyz", obs_all), ("target_xyz", target_all), ("methods", methods), ("actions", actions),
                                ("meta", windows["meta"]), ("source", {"checkpoint": args.checkpoint, "split": args.split})]), args.export_arrays)
    summary = result["mean_over_samples"]
    print(json.dumps(OrderedDict((key, summary.get(key)) for key in ("xyz_mse", "xyz_mae", "mpjpe", "primary_excess_antic", "jerk_body")), ensure_ascii=False))
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--natural_samples", type=int, default=4, help="前几个样本另算完整自然度（CPU double，较慢）")
    parser.add_argument("--export_samples", type=int, default=3)
    parser.add_argument("--nfe", type=int, default=20)
    parser.add_argument("--solver", choices=("heun", "euler"), default="heun")
    parser.add_argument("--sample_seed", type=int, default=2718)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--export_arrays", default=None)
    parser.add_argument("--allow_cpu_for_smoke_test", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
