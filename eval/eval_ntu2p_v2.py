"""NTU2P v2 统一评估：旧 refiner / 规范系 refiner / InterMixer 任意 checkpoint 在缓存管线的 val/test 中心窗口上评估。

输出 JSON 与 eval/eval_ntu2p_residual_refiner_xyz.py 同结构（model/base/copy_last 指标、beats_*、articulation_metrics、
articulation_gate），另加：
- naturalness：步行子集 GT 站定帧脚速、滑行帧比、迈步相位相关、骨长误差、分部位 mpjpe 等（utils/ntu2p_naturalness.py），
  以及步行人左右踝前后分离与 GT 的逐段相关/RMSE（gait_*，前 0.5 s 另报相对 copy-last 的 RMSE 比）；
- extra_variant_metrics：模型 details 里的 pred_free（骨架投影前）与 anchor（检索锚点）的指标；
  --posthoc_root_blend β 时另报免训练参考"模型输出 + 事后只混合检索 root"（A3 的对照线）。
base 一律由冻结独立单人 base checkpoint 计算（不取模型内部的 base），不同架构的 gate 口径一致；
全部指标在相机系上计算（xyz_mae 等按分量的指标不具旋转不变性）。
"""

import argparse
import json
import os
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu2p_retrieval_bank import NTU2PRetrievalBank, performer_ids
from data_loaders.forecasting.ntu2p_xyz_seq_cache import DEFAULT_CACHE_DIR, NTU2PXYZSeqCache, eval_windows, iter_eval_batches
from eval.eval_ntu2p_residual_refiner_xyz import _add, _finalize
from model.forecasting_ntu2p_residual_xyz import build_residual_ramp, load_base_model_from_checkpoint
from model.forecasting_ntu2p_v2 import forward_details, independent_base_forward, load_ntu2p_model_checkpoint, rotate_to_camera
from train.train_ntu2p_v2 import PIPELINE, _device
from utils.fixseed import fixseed
from utils.ntu2p_canonical import to_canonical
from utils.ntu2p_naturalness import GAIT_BINS, aggregate_stats, compute_naturalness_stats, estimate_scene_frame
from utils.ntu_smplx_2p_xyz import (
    articulation_ratios,
    check_ntu_xyz,
    compute_ntu_articulation_metrics,
    compute_ntu_xyz_metrics,
    copy_last_xyz,
)


DEFAULT_MANIFEST = "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"
DEFAULT_BASELINE = "save/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_cuda_retrain_s0_5000/model000005000.pt"
DEFAULT_RETRIEVAL_BANK = "results/forecasting/ntu120_label/ntu2p_retrieval_bank/train_retrieval_bank.pt"
EXTRA_VARIANTS = ("pred_free", "anchor", "posthoc_root_blend")
L2_KEYS = ("xyz_mse", "xyz_mae", "mpjpe")
# 步态相位（utils/ntu2p_naturalness.gait_phase_stats）：步行人左右踝前后分离与 GT 的逐段相关/RMSE、先迈脚准确率。
GAIT_BIN_TAGS = tuple("f{:02d}_{:02d}".format(first, last) for first, last in GAIT_BINS)
GAIT_PHASE_KEYS = (
    tuple("gait_sep_{}_{}{}".format(kind, tag, suffix) for suffix in ("", "_moving", "_starting") for kind in ("corr", "rmse") for tag in GAIT_BIN_TAGS)
    + ("gait_lead_foot_acc", "gait_lead_foot_acc_moving", "gait_lead_foot_acc_starting", "gait_walk_count", "gait_moving_count")
)
# 汇报用的自然度指标；完整指标请对 --export_arrays 的输出运行 eval/analyze_ntu2p_naturalness.py --pred_file。
NATURALNESS_KEYS = (
    "skate_gt_contact_speed_walk",
    "skate_gt_contact_violation_walk",
    "skate_self_ratio_walk",
    "skate_gt_contact_speed",
    "slide_frame_ratio",
    "rel_fwd_vel_corr_with_gt",
    "leg_swing_amp",
    "stance_fraction",
    "lr_forward_vel_corr",
    "step_count",
    "bone_rel_err_body",
    "bone_rel_err_body_max",
    "bone_rel_err_legs",
    "bone_abs_err_body",
    "mpjpe_body22",
    "local_mpjpe_body22",
    "mpjpe_legs",
    "mpjpe_torso",
    "mpjpe_arms",
    "mpjpe_head",
    "mpjpe_fingers",
    "local_mpjpe_legs",
    "local_mpjpe_arms",
    "local_mpjpe_fingers",
    "foot_penetration_ratio",
    "foot_float_ratio",
    "jerk_body",
    "heading_err_deg",
    "root_distance_abs_err",
    "min_interperson_dist_abs_err",
    "walking_person_fraction",
) + GAIT_PHASE_KEYS
# "前 0.5 s 是否比腿不动更差"：模型 RMSE / copy-last RMSE，只报可预测的前两段。
GAIT_RMSE_RATIO_TAGS = ("f01_10", "f11_20")
# 分块计算自然度（double、逐样本），限制 test 1253 条时的峰值内存。
NATURALNESS_CHUNK = 64


def _write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _windows(cache, max_samples):
    windows = eval_windows(cache)
    if max_samples is None or int(max_samples) <= 0:
        return windows
    count = int(max_samples)
    return OrderedDict((key, value[:count]) for key, value in windows.items())


def articulation_gate(result, args):
    """与 eval/eval_ntu2p_residual_refiner_xyz.py 的 gate 逐项相同（旧脚本中为内联代码，这里抽成函数）。"""
    model_artic = result["articulation_metrics"]["model"]
    mpjpe_regression = result["model_metrics"]["mpjpe"] / result["base_metrics"]["mpjpe"] - 1.0
    gate = OrderedDict()
    gate["dct_low_ratio_threshold"] = float(args.articulation_gate_low_ratio)
    gate["dct_mid_ratio_threshold"] = float(args.articulation_gate_mid_ratio)
    gate["base_mpjpe_regression_tolerance"] = float(args.base_mpjpe_regression_tolerance)
    gate["model_dct_low_ratio"] = float(model_artic["dct_low_energy_ratio_to_target"])
    gate["model_dct_mid_ratio"] = float(model_artic["dct_mid_energy_ratio_to_target"])
    gate["model_dct_high_ratio"] = float(model_artic["dct_high_energy_ratio_to_target"])
    gate["model_energy_ratio"] = float(model_artic["articulation_energy_ratio_to_target"])
    gate["model_frozen_ratio"] = float(model_artic["frozen_ratio"])
    gate["model_mpjpe_regression_vs_base"] = float(mpjpe_regression)
    gate["passes_l2_gate"] = all(result["beats_copy_last"].values())
    gate["passes_articulation_gate"] = (
        gate["model_dct_low_ratio"] >= gate["dct_low_ratio_threshold"]
        and gate["model_dct_mid_ratio"] >= gate["dct_mid_ratio_threshold"]
    )
    gate["within_base_tolerance"] = mpjpe_regression <= gate["base_mpjpe_regression_tolerance"]
    gate["passes_full_gate"] = gate["passes_l2_gate"] and gate["passes_articulation_gate"] and gate["within_base_tolerance"]
    return gate


def naturalness_block(obs, target, variants, chunk=NATURALNESS_CHUNK):
    """variants: name -> [N,50,2,55,3]（CPU）；场景几何只由 obs + GT 决定，各方法共用，与分析脚本一致。"""
    parts = OrderedDict((name, OrderedDict()) for name in variants)
    total = int(obs.shape[0])
    for begin in range(0, total, int(chunk)):
        end = min(begin + int(chunk), total)
        obs_chunk, target_chunk = obs[begin:end], target[begin:end]
        frame = estimate_scene_frame(torch.cat((obs_chunk.double(), target_chunk.double()), dim=1))
        for name, value in variants.items():
            stats = compute_naturalness_stats(value[begin:end], target_chunk, obs_chunk, frame)
            for key, pair in stats.items():
                parts[name].setdefault(key, []).append(pair)
    block = OrderedDict()
    for name, per_key in parts.items():
        merged = OrderedDict((key, (torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs]))) for key, pairs in per_key.items())
        aggregated = aggregate_stats(merged)
        block[name] = OrderedDict((key, aggregated[key]) for key in NATURALNESS_KEYS)
    gt_swing = block["GT"]["leg_swing_amp"]
    for name in block:
        block[name]["leg_swing_amp_ratio_to_gt"] = block[name]["leg_swing_amp"] / gt_swing if gt_swing > 0 else float("nan")
    for tag in GAIT_RMSE_RATIO_TAGS:
        key = "gait_sep_rmse_{}".format(tag)
        reference = block["copy_last"][key]
        for name in block:
            block[name][key + "_ratio_to_copy_last"] = block[name][key] / reference if reference > 0 else float("nan")
    block["num_walking_person_windows"] = block["GT"]["walking_person_fraction"] * 2.0 * total
    block["num_gait_walk_persons"] = block["GT"]["gait_walk_count"] * total
    block["num_gait_moving_persons"] = block["GT"]["gait_moving_count"] * total
    return block


def _load_bank(args, state, device, needed):
    """检索模型或事后 root 混合需要检索库；路径优先用命令行，其次 checkpoint 记录，最后默认库。"""
    if not needed:
        return None, None
    path = args.retrieval_bank or state.get("retrieval_bank") or DEFAULT_RETRIEVAL_BANK
    train_cache = NTU2PXYZSeqCache.load(args.cache_dir, "train", manifest_path=args.manifest_path, device=device)
    return NTU2PRetrievalBank.load(path, train_cache, device=device), path


def posthoc_root_blend_module(bank, beta, obs_len, pred_len, device):
    """免训练参考：RetrievalAnchor 取 β_root=β、β_local=0，打分 MLP 末层零初始化 → 按 τ0 的距离软平均。

    与 A3 初始锚点同一公式，只是把"base"换成被评估模型的输出：root 向检索邻居的平均 root 位移挪 β，
    局部姿态不动。其余投影层的随机初始化不影响输出（打分恒为 0，token 不使用）。
    """
    from model.ntu2p_retrieval_anchor import RetrievalAnchor

    with torch.random.fork_rng(devices=[]):
        module = RetrievalAnchor(
            pred_len=pred_len,
            obs_len=obs_len,
            key_joints=int(bank.key_joints),
            init_tau=float(bank.config["median_topk_dist"]),
            root_beta_init=float(beta),
            local_beta_init=0.0,
        )
    return module.to(device).eval()


def posthoc_root_blend(module, bank, obs_xyz, pred_xyz, action, performer, k, ramp):
    obs_canon, _, frame = bank.canonicalize(obs_xyz)
    pred_canon = to_canonical(pred_xyz, frame)
    neighbors = bank.query(obs_canon, action, performer=performer, k=int(k))
    anchor = module(obs_canon, pred_canon, neighbors["disp"], neighbors["dist"], ramp, tier=neighbors["tier"])["anchor"]
    # 只旋回残差再加原预测：首帧（ramp=0）严格不变。
    return pred_xyz + rotate_to_camera(anchor - pred_canon, frame)


def evaluate(args):
    device = _device(args)
    fixseed(args.seed)
    cache = NTU2PXYZSeqCache.load(args.cache_dir, args.split, manifest_path=args.manifest_path, device=device)
    windows = _windows(cache, args.max_samples)
    model, checkpoint_state = load_ntu2p_model_checkpoint(args.checkpoint, device)
    model.eval()
    if args.override_canonical_ab_fallback:
        # 诊断用：只替换推理期的规范系回退，权重不变；结果 JSON 记录覆盖值。
        if not hasattr(model, "canonical_ab_fallback"):
            raise ValueError("该模型不使用场景规范系，不能覆盖 canonical_ab_fallback")
        model.canonical_ab_fallback = args.override_canonical_ab_fallback
    needs_retrieval = int(getattr(model, "retrieval_k", 0)) > 0
    bank, bank_path = _load_bank(args, checkpoint_state, device, needs_retrieval or args.posthoc_root_blend > 0)
    if needs_retrieval:
        model.set_retrieval_bank(bank)
    blend_module = None
    if args.posthoc_root_blend > 0:
        blend_module = posthoc_root_blend_module(bank, args.posthoc_root_blend, args.obs_len, args.pred_len, device)
        # 与主线 s2_5 的输出 ramp 相同（saturate 5 帧）。
        blend_ramp = build_residual_ramp(args.pred_len, "saturate", 5).to(device)
    base_model, _ = load_base_model_from_checkpoint(args.baseline_checkpoint, device)

    variants = ("model", "base", "copy_last")
    totals = {key: OrderedDict() for key in variants}
    articulation_totals = {key: OrderedDict() for key in variants}
    extra_totals = OrderedDict()
    extra_articulation = OrderedDict()
    arrays = OrderedDict((key, []) for key in ("obs_xyz", "target_xyz", "model", "base", "actions") + EXTRA_VARIANTS)
    meta = []
    count = 0
    with torch.no_grad():
        for batch in iter_eval_batches(windows, args.batch_size):
            obs_xyz, target_xyz, action = batch["obs_xyz"], batch["target_xyz"], batch["action"]
            # 所有 split 同一规则：检索排除同受试者（val 受试者全在 train 中，test 天然不重叠）。
            performer = performer_ids([item["sample_id"] for item in batch["meta"]]).to(device)
            context = OrderedDict([("performer", performer)])
            details = forward_details(model, obs_xyz, action, context)
            if blend_module is not None:
                details["posthoc_root_blend"] = posthoc_root_blend(
                    blend_module, bank, obs_xyz, details["pred"], action, performer, args.posthoc_k, blend_ramp
                )
            pred_xyz = details["pred"]
            base_xyz = independent_base_forward(base_model, obs_xyz, action)
            copy_xyz = copy_last_xyz(obs_xyz, args.pred_len)
            check_ntu_xyz("pred_xyz", pred_xyz, seq_len=args.pred_len, num_persons=2)
            batch_size = int(obs_xyz.shape[0])
            for key, value in (("model", pred_xyz), ("base", base_xyz), ("copy_last", copy_xyz)):
                _add(totals[key], compute_ntu_xyz_metrics(value, target_xyz, obs_xyz), batch_size)
                _add(articulation_totals[key], compute_ntu_articulation_metrics(value, target_xyz), batch_size)
            for key in EXTRA_VARIANTS:
                if details.get(key) is not None:
                    _add(extra_totals.setdefault(key, OrderedDict()), compute_ntu_xyz_metrics(details[key], target_xyz, obs_xyz), batch_size)
                    _add(extra_articulation.setdefault(key, OrderedDict()), compute_ntu_articulation_metrics(details[key], target_xyz), batch_size)
                    arrays[key].append(details[key].detach().cpu())
            for key, value in (("obs_xyz", obs_xyz), ("target_xyz", target_xyz), ("model", pred_xyz), ("base", base_xyz), ("actions", action)):
                arrays[key].append(value.detach().cpu())
            meta.extend(batch["meta"])
            count += batch_size

    result = OrderedDict(
        [
            ("checkpoint", args.checkpoint),
            ("split", args.split),
            ("num_samples", int(count)),
            ("sample_seed", int(args.seed)),
            ("alpha", float(model.alpha.detach().cpu().item()) if hasattr(model, "alpha") else None),
            ("model_metrics", _finalize(totals["model"], count)),
            ("base_metrics", _finalize(totals["base"], count)),
            ("copy_last_metrics", _finalize(totals["copy_last"], count)),
            ("beats_base", OrderedDict()),
            ("beats_copy_last", OrderedDict()),
            ("articulation_metrics", OrderedDict()),
            ("articulation_gate", OrderedDict()),
            ("checkpoint_step", int(checkpoint_state.get("step", -1))),
            ("model_type", getattr(model, "model_type", None)),
            ("arch", checkpoint_state.get("arch", "refiner")),
            ("pipeline", PIPELINE),
            ("baseline_checkpoint", args.baseline_checkpoint),
            ("retrieval_bank", bank_path),
            ("posthoc_root_blend", OrderedDict([("beta", float(args.posthoc_root_blend)), ("k", int(args.posthoc_k))])),
            ("device", str(device)),
        ]
    )
    for key in L2_KEYS:
        result["beats_base"][key] = result["model_metrics"][key] < result["base_metrics"][key]
        result["beats_copy_last"][key] = result["model_metrics"][key] < result["copy_last_metrics"][key]
    for key in variants:
        aggregated = _finalize(articulation_totals[key], count)
        aggregated.update(articulation_ratios(aggregated))
        result["articulation_metrics"][key] = aggregated
    result["articulation_gate"] = articulation_gate(result, args)
    result["canonical_ab_fallback"] = getattr(model, "canonical_ab_fallback", None)
    result["extra_variant_metrics"] = OrderedDict()
    for key, total in extra_totals.items():
        aggregated = _finalize(extra_articulation[key], count)
        aggregated.update(articulation_ratios(aggregated))
        result["extra_variant_metrics"][key] = OrderedDict([("model_metrics", _finalize(total, count)), ("articulation_metrics", aggregated)])

    data = OrderedDict((key, torch.cat(value, dim=0)) for key, value in arrays.items() if value)
    obs_all, target_all = data["obs_xyz"], data["target_xyz"]
    if not args.skip_naturalness:
        natural_variants = OrderedDict(
            [("model", data["model"]), ("base", data["base"]), ("copy_last", copy_last_xyz(obs_all, args.pred_len)), ("GT", target_all)]
        )
        if "posthoc_root_blend" in data:
            natural_variants["posthoc_root_blend"] = data["posthoc_root_blend"]
        result["naturalness"] = naturalness_block(obs_all, target_all, natural_variants)
    if args.export_arrays:
        # 与 eval/analyze_ntu2p_naturalness.py --pred_file、sample/render_ntu2p_review_sheet.py --arrays 的格式兼容。
        methods = OrderedDict((key, data[key]) for key in ("model", "base") + EXTRA_VARIANTS if key in data)
        export = OrderedDict(
            [
                ("obs_xyz", obs_all),
                ("target_xyz", target_all),
                ("methods", methods),
                ("actions", data["actions"].reshape(-1)),
                ("meta", meta),
                ("source", {"checkpoint": args.checkpoint, "split": args.split, "manifest_path": args.manifest_path}),
            ]
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.export_arrays)), exist_ok=True)
        torch.save(export, args.export_arrays)
        result["export_arrays"] = args.export_arrays
    _write_json(args.output, result)
    summary = OrderedDict((key, result["model_metrics"][key]) for key in L2_KEYS)
    print(json.dumps(OrderedDict([("model", summary), ("gate", result["articulation_gate"]["passes_full_gate"])]), ensure_ascii=False))
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--baseline_checkpoint", default=DEFAULT_BASELINE, help="计算 base 指标与 gate 的冻结独立单人 base")
    parser.add_argument("--retrieval_bank", default=None, help="检索库；缺省用 checkpoint 中记录的路径，再缺省用默认库")
    parser.add_argument("--posthoc_root_blend", type=float, default=0.0, help="β>0 时另报'模型输出 + 事后只混合检索 root'的免训练参考")
    parser.add_argument("--posthoc_k", type=int, default=16)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=-1, help="只评估按 manifest 顺序的前 N 条（验证用）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--articulation_gate_low_ratio", type=float, default=0.40)
    parser.add_argument("--articulation_gate_mid_ratio", type=float, default=0.10)
    parser.add_argument("--base_mpjpe_regression_tolerance", type=float, default=0.05)
    parser.add_argument("--skip_naturalness", action="store_true")
    parser.add_argument("--override_canonical_ab_fallback", choices=("a_facing", "camera_x"), default=None,
                        help="诊断：推理时替换规范系的双人重合回退（不改权重）")
    parser.add_argument("--export_arrays", default=None, help="保存 obs/target/methods/actions/meta 供自然度分析与审查图")
    parser.add_argument("--allow_cpu_for_smoke_test", action="store_true", help="仅冒烟测试：允许 --device cpu")
    return parser


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
