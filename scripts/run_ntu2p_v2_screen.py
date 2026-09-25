"""NTU2P v2 架构筛选驱动（Stage 1–3）：训练、终点 val 评估（EMA 与 raw）、按预登记规则汇总判断。

对应设计：docs/ai/context/20260925-121154-ntu2p-v2-architecture-exploration-design-and-plan.md 第 4 节。
- Stage 1：A0 旧 refiner（新管线对照）、A1 规范系 refiner、A2 = A1 + 解冻 base、A3 = A1 + 检索锚点、A6 InterMixer；
- Stage 2：在 --stage2_base 指定的底座上叠加 A4 骨架投影、A5 脚接触损失、A7 镜像增广 + 指示；
- Stage 3：--configs 指定的采纳组合 10000 step，另可用 --test_config 只对最终方案评估一次 test
  （固定 10000 step EMA 终点，且须在 summary_10000.json 中判为候选采纳）。
配置名用 "-" 串接：底座（A0/A1/A2/A3/A6）后接增量（A4、A5f<权重>、A7、F、GL<权重>、GH<权重>），如 A1-A4-A5f0.35。
步态增量 GL（腿部相位敏感损失）/ GH（个人朝向系腿部专用流 + 同款损失）必须放在最后，按步态预登记规则判断（decide）。
全部 run 为 s2_5 配方 + EMA 0.999、batch 8、seed 0/1/2；报告 ema/ 终点，与参照配置逐 seed 配对。
可断点续跑：终点 checkpoint 与评估 JSON 已存在则跳过；--workers N 并行 N 个 run。
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.run_ntu2p_articulation_stage1 import BASELINE, MANIFEST, _log

SAVE_ROOT = "save/forecasting/ntu120_label"
SUMMARY_DIR = "results/forecasting/ntu120_label/ntu2p_v2_screen"
DEFAULT_RETRIEVAL_BANK = "results/forecasting/ntu120_label/ntu2p_retrieval_bank/train_retrieval_bank.pt"
SEEDS = (0, 1, 2)
SCREEN_STEPS = 5000
FINAL_STEPS = 10000
CONTROL = "A0"
# 免训练参考线："A0 输出 + 事后只把 root 向检索邻居挪 β"（带 ramp）；A3 须同时报告相对它的差值。
POSTHOC_BLEND_BETA = 0.5
POSTHOC_LABEL = "A0+root混合{:g}（免训练参考）".format(POSTHOC_BLEND_BETA)

# 与主线 ntu2p_residual_refiner_xyz_artic_ema_s2_5_dct1_root01_s0_10000/args.json 相同的损失配方与训练设置。
S2_5_RECIPE = [
    "--loss_scale_normalize",
    "--local_velocity_loss_weight", "1.0",
    "--ramp_mode", "saturate",
    "--ramp_saturate_frames", "5",
    "--dct_low_amplitude_loss_weight", "1.0",
    "--dct_mid_amplitude_loss_weight", "1.0",
    "--root_loss_weight", "0.1",
    "--inter_loss_weight", "0.01",
    "--ema_decay", "0.999",
    "--batch_size", "8",
    "--save_interval", "1000",
    "--log_interval", "100",
]
CANON = ["--arch", "canon_refiner", "--canonical", "--disp_channel", "--role_embed"]
STAGE1_CONFIGS = ("A0", "A1", "A2", "A3", "A6")
# 以自然度为主要目标的增量：L2 只要求不劣化超过 0.5%，自然度须明显改善（设计 4.2）。
NATURALNESS_ADDONS = ("A4", "A5")
# 步态增量：目标是"步行者两腿前后交替迈步"，按步态相位指标判断（gait_criteria）。
GAIT_ADDONS = ("GL", "GH")

# 预登记采纳规则（设计 4.2）；"明显改善"的具体阈值为本驱动的实现口径，汇总表中写明。
L2_ADOPT_DELTA_PCT = -0.8
NATURAL_L2_TOLERANCE_PCT = 0.5
NATURAL_VETO_RATIO = 1.10
A4_BONE_RATIO = 0.5
A5_SKATE_RATIO = 0.9
# 设计 A5"初始权重使其占总 loss 约 5%"：初始 = 冻结 base 输出（A1 底座），按训练同口径（_estimate_scales 的纯滑行
# foot 常数 ≈ 0.0211、s2_5 配方、train 采样器 seed 0/1/2 各 300 步）算逐步占比 w·f/(L + w·f)。0.35 时逐步占比中位数
# 4.3–5.0%、去掉 top 1% 大 loss 步后的均值比 5.6–5.7%，两种稳健口径都最接近 5%；不用全体均值比，train 中
# S013C001P028R001A010 的 30 m 跳变窗口让单步 L 达 486，会把它拉偏。w=1.0 时逐步占比 11–13%，远超 5%。
# A6 初始等于 copy-last、脚滑近 0，此口径对它不适用。实际占比见训练日志的 foot_share。
A5_FOOT_WEIGHT = 0.35
L2_KEYS = ("mpjpe", "xyz_mse", "xyz_mae")
NATURAL_KEYS = (
    "skate_gt_contact_speed_walk",
    "slide_frame_ratio",
    "rel_fwd_vel_corr_with_gt",
    "bone_rel_err_body",
    "leg_swing_amp_ratio_to_gt",
    "mpjpe_legs",
    "mpjpe_arms",
    "gait_sep_corr_f01_10",
    "gait_sep_corr_f11_20",
    "gait_sep_corr_f21_30",
    "gait_sep_corr_f31_50",
    "gait_sep_rmse_f01_10",
    "gait_sep_rmse_f01_10_ratio_to_copy_last",
    "gait_sep_rmse_f11_20",
    "gait_sep_rmse_f31_50",
    "gait_sep_corr_f01_10_moving",
    "gait_sep_corr_f11_20_moving",
    "gait_lead_foot_acc",
    "stance_fraction",
    "step_count",
    "local_mpjpe_legs",
)
# 旧评估 JSON 缺这个键时按新评估重算（驱动会复用已有 run，只补评估）。
GAIT_EVAL_KEY = "gait_sep_corr_f01_10"
# 重算不改模型输出：L2 与旧 JSON 的差超过它说明评估口径被意外改动。
REEVAL_L2_TOLERANCE = 1e-9

# 步态增量（GL/GH）的预登记规则，均对参照 R（去掉最后一个步态 token 的配置）按 seed 配对：
# A. 主判据：前 0.5 s 相位由观测决定（只看腿的 MLP 在已在走的人上 f1-20 相关 0.87-0.92），应明显延续；
GAIT_CORR01_MIN_DELTA = 0.15
GAIT_CORR01_MIN_ABS = 0.25
GAIT_CORR11_MIN_DELTA = 0.10
GAIT_CORR11_MIN_ABS = 0.35
GAIT_CORR11_MIN_POSITIVE = 2
# 前 0.5 s 的分离误差不能比"腿不动"（copy-last）更差。
GAIT_RMSE01_MAX_RATIO_TO_COPY_LAST = 1.00
# 脚接触：A5 底座已修过脚滑，只要求不升；A6-F 底座要求改善 10%。滑行帧比容差与自然度否决项相同。
GAIT_SKATE_MAX_RATIO_A5 = 1.00
GAIT_SKATE_MAX_RATIO = 0.90
GAIT_SLIDE_MAX_RATIO = 1.10
# B. 护栏：L2 与 A4/A5 同为 +0.5%；f31-50 为多模态区，只要求不变差超过 5%。
GAIT_LONG_RMSE_MAX_RATIO = 1.05
# GH 相对 GL（配对）的增量门槛：达不到时采纳更简单的 GL。
GH_OVER_GL_CORR_DELTA = 0.05
GH_OVER_GL_MPJPE_PCT = -0.3
# "不确定"时预登记只允许补跑一次，且只用这个权重。
GAIT_RERUN_WEIGHT = 0.25
GAIT_NEXT_ROUND = "下一轮候选：手指去重 α=1/15 + GL w≈0.06，起步者交给 Track B residual diffusion"

_print_lock = threading.Lock()


def _say(message):
    with _print_lock:
        _log(message)


def base_config_args(name, retrieval_bank):
    table = OrderedDict(
        [
            ("A0", ["--arch", "refiner"]),
            ("A1", list(CANON)),
            ("A2", CANON + ["--unfreeze_base", "--base_lr_mult", "0.1"]),
            ("A3", CANON + ["--retrieval_bank", retrieval_bank, "--retrieval_k", "16"]),
            ("A6", ["--arch", "intermixer"]),
        ]
    )
    if name not in table:
        raise ValueError("未知底座配置 {}，可选 {}".format(name, list(table)))
    return table[name]


def addon_args(token):
    if token == "A4":
        return ["--kin_proj"]
    if token == "A7":
        return ["--augment_mirror_prob", "0.5", "--mirror_embed"]
    if token == "F":
        return ["--canonical_ab_fallback", "camera_x"]
    if token.startswith("A5f"):
        return ["--foot_loss_weight", str(float(token[3:]))]
    if token.startswith("GL"):
        return ["--leg_gait_loss_weight", str(float(token[2:])), "--dct_mid_exclude_legs"]
    if token.startswith("GH"):
        return ["--leg_stream", "--leg_gait_loss_weight", str(float(token[2:])), "--dct_mid_exclude_legs"]
    raise ValueError("未知增量 {}（可选 A4、A5f<权重>、A7、F、GL<权重>、GH<权重>）".format(token))


def config_args(name, retrieval_bank):
    tokens = name.split("-")
    gait = [index for index, token in enumerate(tokens) if token[:2] in GAIT_ADDONS]
    # 步态规则以"去掉最后一个 token"的配置为参照，步态 token 不在最后会让参照里混进另一个步态变体。
    if len(gait) > 1 or (gait and gait[0] != len(tokens) - 1):
        raise ValueError("{}：步态增量（GL/GH）至多一个且必须放在最后".format(name))
    args = list(base_config_args(tokens[0], retrieval_bank))
    for token in tokens[1:]:
        args += addon_args(token)
    return args


def reference_config(name):
    """L2 与 A4/A5 改善判据的配对参照：底座配置对 A0；叠加增量的配置对去掉最后一个增量后的配置。
    自然度否决项不用它，固定对 A0（见 natural_veto）。"""
    tokens = name.split("-")
    return CONTROL if len(tokens) == 1 else "-".join(tokens[:-1])


def last_addon(name):
    tokens = name.split("-")
    return None if len(tokens) == 1 else tokens[-1][:2]


def run_dir(name, seed, steps, save_root=SAVE_ROOT):
    return os.path.join(save_root, "ntu2p_v2_{}_s{}_{}".format(name, seed, steps))


def eval_json(checkpoint_dir, steps, split="val"):
    return os.path.join(checkpoint_dir, "eval_v2_{}_{:09d}.json".format(split, steps))


def _run(command, log_path, dry_run):
    if dry_run:
        _say("DRY RUN: " + " ".join(command))
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") or "."
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as handle:
        handle.write("$ {}\n".format(" ".join(command)))
        handle.flush()
        subprocess.run(command, check=True, env=env, stdout=handle, stderr=subprocess.STDOUT)


def _eval_command(opts, checkpoint, output, split="val", export=None):
    command = [sys.executable, "eval/eval_ntu2p_v2.py", "--checkpoint", checkpoint, "--output", output,
               "--split", split, "--manifest_path", MANIFEST, "--baseline_checkpoint", BASELINE, "--device", opts.device]
    if export:
        command += ["--export_arrays", export]
    return command + opts.smoke_args


def _needs_eval(output, posthoc):
    if not os.path.exists(output):
        return True
    result = _load(output)
    if GAIT_EVAL_KEY not in result.get("naturalness", {}).get("model", {}):
        return True
    return posthoc and "posthoc_root_blend" not in result.get("extra_variant_metrics", {})


def _reeval_mismatch(previous, output):
    """重算前后模型 L2 的最大绝对差；超过容差时返回说明，否则 None。"""
    current = _load(output)
    diff = max(abs(current["model_metrics"][key] - previous["model_metrics"][key]) for key in L2_KEYS)
    _say("re-eval {}：L2 与旧 JSON 最大差 {:.3e}".format(output, diff))
    return None if diff < REEVAL_L2_TOLERANCE else "{} 重算后 L2 最大差 {:.3e} ≥ {:g}".format(output, diff, REEVAL_L2_TOLERANCE)


def _job(name, seed, steps, opts):
    save_dir = run_dir(name, seed, steps, opts.save_root)
    final = os.path.join(save_dir, "model{:09d}.pt".format(steps))
    ema_final = os.path.join(save_dir, "ema", "model{:09d}.pt".format(steps))
    log_path = os.path.join(save_dir, "driver.log")
    if os.path.exists(final) and os.path.exists(ema_final):
        _say("skip train {} (final checkpoints exist)".format(save_dir))
    else:
        if name.split("-")[0] == "A3" and not opts.dry_run and not os.path.exists(opts.retrieval_bank):
            raise FileNotFoundError("A3 需要检索库 {}（先运行检索库构建脚本）".format(opts.retrieval_bank))
        stale = os.path.join(save_dir, "train_log.jsonl")
        if os.path.exists(stale) and not opts.dry_run:
            # 未完成的旧 run 从头重训；旧日志改名保留，避免与新日志混在一起。
            shutil.move(stale, stale + ".incomplete")
        command = [sys.executable, "train/train_ntu2p_v2.py", "--save_dir", save_dir, "--num_steps", str(steps),
                   "--seed", str(seed), "--manifest_path", MANIFEST, "--baseline_checkpoint", BASELINE,
                   "--device", opts.device] + S2_5_RECIPE + config_args(name, opts.retrieval_bank) + opts.smoke_args + opts.extra_train_args
        _say("train {} s{} {}".format(name, seed, steps))
        _run(command, log_path, opts.dry_run)
    # 每个 run 训完立即评估 EMA 与 raw 终点；seed 0 的 EMA 终点另导出数组，供审查图与自然度分析。
    mismatches = []
    for subdir in ("ema", ""):
        checkpoint_dir = os.path.join(save_dir, subdir) if subdir else save_dir
        output = eval_json(checkpoint_dir, steps)
        posthoc = bool(subdir) and name == CONTROL
        if not _needs_eval(output, posthoc):
            continue
        previous = None
        if os.path.exists(output) and not opts.dry_run:
            # 补评估会覆盖旧 JSON：先留一份，供核对与回溯。
            previous = _load(output)
            shutil.copyfile(output, output[: -len(".json")] + ".pre_gait.json")
        export = os.path.join(checkpoint_dir, "val_arrays_{:09d}.pt".format(steps)) if (subdir and seed == SEEDS[0]) else None
        checkpoint = os.path.join(checkpoint_dir, "model{:09d}.pt".format(steps))
        command = _eval_command(opts, checkpoint, output, export=export)
        if posthoc:
            command += ["--posthoc_root_blend", str(POSTHOC_BLEND_BETA), "--retrieval_bank", opts.retrieval_bank]
        _run(command, log_path, opts.dry_run)
        if previous is not None:
            mismatch = _reeval_mismatch(previous, output)
            if mismatch:
                mismatches.append(mismatch)
    if mismatches:
        raise ValueError("；".join(mismatches))
    _say("done {} s{} {}".format(name, seed, steps))
    return save_dir


def run_jobs(configs, seeds, steps, opts):
    jobs = [(name, seed) for name in configs for seed in seeds]
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(opts.workers))) as pool:
        futures = OrderedDict((pool.submit(_job, name, seed, steps, opts), (name, seed)) for name, seed in jobs)
        for future in as_completed(futures):
            name, seed = futures[future]
            try:
                future.result()
            except Exception as error:  # 单个 run 失败不影响其它 run；结束时统一报告并以非零码退出。
                failures.append((name, seed, repr(error)))
                _say("FAILED {} s{}: {!r}".format(name, seed, error))
    return failures


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _pct(value, reference):
    return 100.0 * (value / reference - 1.0)


def _mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def collect(configs, seeds, steps, save_root):
    results = OrderedDict()
    for name in configs:
        per_seed = OrderedDict()
        for seed in seeds:
            save_dir = run_dir(name, seed, steps, save_root)
            ema_path, raw_path = eval_json(os.path.join(save_dir, "ema"), steps), eval_json(save_dir, steps)
            if os.path.exists(ema_path) and os.path.exists(raw_path):
                per_seed[seed] = {"ema": _load(ema_path), "raw": _load(raw_path)}
        if len(per_seed) == len(seeds):
            results[name] = per_seed
    return results


def config_stats(per_seed):
    seeds = list(per_seed)
    ema = [per_seed[s]["ema"] for s in seeds]
    stats = OrderedDict()
    for key in L2_KEYS:
        values = [r["model_metrics"][key] for r in ema]
        stats[key] = _mean_std(values)
        stats[key + "_per_seed"] = values
    stats["raw_mpjpe"] = _mean_std([per_seed[s]["raw"]["model_metrics"]["mpjpe"] for s in seeds])
    stats["gate"] = sum(bool(r["articulation_gate"]["passes_full_gate"]) for r in ema)
    stats["dct_mid"] = statistics.mean(r["articulation_gate"]["model_dct_mid_ratio"] for r in ema)
    stats["natural"] = OrderedDict()
    # 逐 seed 值供"3/3 seed"类判据配对使用。
    stats["natural_per_seed"] = OrderedDict()
    if all("naturalness" in r for r in ema):
        for key in NATURAL_KEYS:
            # 旧评估 JSON 没有步态相位键：缺失时不报，汇总表显示"—"，步态判断报"缺指标"。
            if all(key in r["naturalness"]["model"] for r in ema):
                values = [r["naturalness"]["model"][key] for r in ema]
                stats["natural"][key] = statistics.mean(values)
                stats["natural_per_seed"][key] = values
    return stats


def posthoc_stats(per_seed):
    """A0 的 EMA 评估里附带的事后 root 混合参考；缺失（旧 JSON）时返回 None。"""
    blocks = [per_seed[s]["ema"].get("extra_variant_metrics", {}).get("posthoc_root_blend") for s in per_seed]
    if any(block is None for block in blocks):
        return None
    stats = OrderedDict()
    for key in L2_KEYS:
        values = [block["model_metrics"][key] for block in blocks]
        stats[key] = _mean_std(values)
        stats[key + "_per_seed"] = values
    stats["dct_mid"] = statistics.mean(block["articulation_metrics"]["dct_mid_energy_ratio_to_target"] for block in blocks)
    naturals = [per_seed[s]["ema"].get("naturalness", {}).get("posthoc_root_blend") for s in per_seed]
    stats["natural"] = OrderedDict()
    if all(natural is not None for natural in naturals):
        for key in NATURAL_KEYS:
            if all(key in natural for natural in naturals):
                stats["natural"][key] = statistics.mean(natural[key] for natural in naturals)
    return stats


def paired_delta(stats, reference):
    return OrderedDict(
        (key, [_pct(v, r) for v, r in zip(stats[key + "_per_seed"], reference[key + "_per_seed"])]) for key in L2_KEYS
    )


def natural_veto(stats, control_stats):
    """设计 4.2 的自然度否决项，固定以 A0 为参照：对上一级比较会让叠加配置每层再差 10% 仍不被否决。
    返回 (是否通过, 理由列表)。"""
    if control_stats is None:
        return False, ["A0 未完成，无法判否决项"]
    natural, a0_natural = stats["natural"], control_stats["natural"]
    if not (natural and a0_natural):
        return False, ["缺自然度指标"]
    reasons = []
    for key in ("skate_gt_contact_speed_walk", "slide_frame_ratio"):
        if natural[key] > NATURAL_VETO_RATIO * a0_natural[key]:
            reasons.append("{} 比 A0 差 >{:.0f}%（否决）".format(key, 100 * (NATURAL_VETO_RATIO - 1)))
    # 骨长与脚滑同用 10% 容差：严格"不高于 A0"会被 3 seed 的噪声误否。
    if natural["bone_rel_err_body"] > NATURAL_VETO_RATIO * a0_natural["bone_rel_err_body"]:
        reasons.append("骨长相对误差比 A0 高 >{:.0f}%（否决）".format(100 * (NATURAL_VETO_RATIO - 1)))
    return not reasons, reasons


def natural_paired_delta(stats, reference, key):
    """自然度指标逐 seed 的绝对差（相关系数等有界量不用百分比）。"""
    return [v - r for v, r in zip(stats["natural_per_seed"][key], reference["natural_per_seed"][key])]


def _signed_list(values, digits=3):
    return "/".join("{:+.{}f}".format(value, digits) for value in values)


def _ratio(value, reference):
    """仅用于汇报；判据用乘法形式比较，参照为 0（如几乎不动的模型没有滑行帧）时不除零。"""
    if reference > 0:
        return value / reference
    return 1.0 if value <= 0 else float("inf")


GAIT_DECISION_KEYS = (
    "gait_sep_corr_f01_10",
    "gait_sep_corr_f11_20",
    "gait_sep_rmse_f01_10_ratio_to_copy_last",
    "gait_sep_rmse_f31_50",
    "skate_gt_contact_speed_walk",
    "slide_frame_ratio",
)


def gait_criteria(name, stats, reference_stats):
    """步态增量的主判据 A 与护栏 B（gate 与 A0 否决项由 decide 统一判）。

    返回 (A 全过, B 中除 L2 外全过, L2 护栏通过, 理由列表)。
    """
    natural, ref_natural = stats["natural"], reference_stats["natural"]
    missing = [key for key in GAIT_DECISION_KEYS if key not in natural or key not in ref_natural]
    if missing:
        return False, False, False, ["缺步态指标 {}（需用新评估重算）".format(missing)]
    reasons = []
    d01 = natural_paired_delta(stats, reference_stats, "gait_sep_corr_f01_10")
    a1 = (statistics.mean(d01) >= GAIT_CORR01_MIN_DELTA and natural["gait_sep_corr_f01_10"] >= GAIT_CORR01_MIN_ABS
          and all(value > 0 for value in d01))
    reasons.append("corr_f01_10 {:.3f}，Δ {:+.3f}（逐 seed {}；要求 Δ≥+{:.2f}、绝对值≥{:.2f}、逐 seed 全正）：{}".format(
        natural["gait_sep_corr_f01_10"], statistics.mean(d01), _signed_list(d01), GAIT_CORR01_MIN_DELTA, GAIT_CORR01_MIN_ABS, a1))
    d11 = natural_paired_delta(stats, reference_stats, "gait_sep_corr_f11_20")
    a2 = (statistics.mean(d11) >= GAIT_CORR11_MIN_DELTA and natural["gait_sep_corr_f11_20"] >= GAIT_CORR11_MIN_ABS
          and sum(value > 0 for value in d11) >= GAIT_CORR11_MIN_POSITIVE)
    reasons.append("corr_f11_20 {:.3f}，Δ {:+.3f}（逐 seed {}；要求 Δ≥+{:.2f}、绝对值≥{:.2f}、至少 {} 个 seed 为正）：{}".format(
        natural["gait_sep_corr_f11_20"], statistics.mean(d11), _signed_list(d11), GAIT_CORR11_MIN_DELTA, GAIT_CORR11_MIN_ABS,
        GAIT_CORR11_MIN_POSITIVE, a2))
    rmse_ratio = natural["gait_sep_rmse_f01_10_ratio_to_copy_last"]
    a3 = rmse_ratio <= GAIT_RMSE01_MAX_RATIO_TO_COPY_LAST
    reasons.append("rmse_f01_10/copy-last {:.3f}（要求 ≤ {:.2f}）：{}".format(rmse_ratio, GAIT_RMSE01_MAX_RATIO_TO_COPY_LAST, a3))
    has_a5 = any(token.startswith("A5") for token in reference_config(name).split("-"))
    skate_limit = GAIT_SKATE_MAX_RATIO_A5 if has_a5 else GAIT_SKATE_MAX_RATIO
    skate, ref_skate = natural["skate_gt_contact_speed_walk"], ref_natural["skate_gt_contact_speed_walk"]
    slide, ref_slide = natural["slide_frame_ratio"], ref_natural["slide_frame_ratio"]
    a4 = skate <= skate_limit * ref_skate and slide <= GAIT_SLIDE_MAX_RATIO * ref_slide
    reasons.append("步行脚速 ×{:.3f}（{}底座要求 ≤ {:.2f}）、滑行帧比 ×{:.3f}（要求 ≤ {:.2f}）：{}".format(
        _ratio(skate, ref_skate), "A5 " if has_a5 else "", skate_limit, _ratio(slide, ref_slide), GAIT_SLIDE_MAX_RATIO, a4))
    delta = paired_delta(stats, reference_stats)
    l2_ok = all(statistics.mean(delta[key]) <= NATURAL_L2_TOLERANCE_PCT for key in L2_KEYS)
    reasons.append("L2 配对 Δ% {}（各项要求 ≤ +{:.1f}%）：{}".format(
        "、".join("{} {:+.2f}".format(key, statistics.mean(delta[key])) for key in L2_KEYS), NATURAL_L2_TOLERANCE_PCT, l2_ok))
    long_rmse, ref_long_rmse = natural["gait_sep_rmse_f31_50"], ref_natural["gait_sep_rmse_f31_50"]
    guard_ok = long_rmse <= GAIT_LONG_RMSE_MAX_RATIO * ref_long_rmse
    reasons.append("rmse_f31_50 ×{:.3f}（要求 ≤ {:.2f}）：{}".format(_ratio(long_rmse, ref_long_rmse), GAIT_LONG_RMSE_MAX_RATIO, guard_ok))
    return a1 and a2 and a3 and a4, guard_ok, l2_ok, reasons


def gh_vs_gl(gh_stats, gl_stats):
    """GH 相对同底座 GL 的配对增量；返回 (GH 是否值得其结构, 理由)。"""
    if "gait_sep_corr_f01_10" not in gh_stats["natural_per_seed"] or "gait_sep_corr_f01_10" not in gl_stats["natural_per_seed"]:
        return False, "缺步态指标"
    d_corr = natural_paired_delta(gh_stats, gl_stats, "gait_sep_corr_f01_10")
    d_mpjpe = paired_delta(gh_stats, gl_stats)["mpjpe"]
    corr_ok = statistics.mean(d_corr) >= GH_OVER_GL_CORR_DELTA and all(value > 0 for value in d_corr)
    mpjpe_ok = statistics.mean(d_mpjpe) <= GH_OVER_GL_MPJPE_PCT
    reason = "Δcorr_f01_10 {:+.3f}（逐 seed {}；要求 ≥+{:.2f} 且全正）：{}；Δmpjpe {:+.2f}%（逐 seed {}；要求 ≤ {:.1f}%）：{}".format(
        statistics.mean(d_corr), _signed_list(d_corr), GH_OVER_GL_CORR_DELTA, corr_ok,
        statistics.mean(d_mpjpe), _signed_list(d_mpjpe, 2), GH_OVER_GL_MPJPE_PCT, mpjpe_ok)
    return corr_ok or mpjpe_ok, reason


def gh_gl_pairs(names):
    """同底座、同权重的 (GH 配置, GL 配置)。"""
    pairs = []
    for name in names:
        tokens = name.split("-")
        if len(tokens) > 1 and tokens[-1].startswith("GH"):
            partner = "-".join(tokens[:-1] + ["GL" + tokens[-1][2:]])
            if partner in names:
                pairs.append((name, partner))
    return pairs


def gait_rerun_config(name):
    """"不确定"的步态配置按预登记补跑的配置名；本身已是 w=0.25 补跑时返回 None（只允许补跑一次）。"""
    tokens = name.split("-")
    if float(tokens[-1][2:]) == GAIT_RERUN_WEIGHT:
        return None
    return "-".join(tokens[:-1] + ["{}{:g}".format(tokens[-1][:2], GAIT_RERUN_WEIGHT)])


def gait_uncertain_label(name):
    rerun = gait_rerun_config(name)
    if rerun is None:
        return "不确定（步态判据通过、仅 L2 越界；已是 w={:g} 补跑，按预登记不再补跑）".format(GAIT_RERUN_WEIGHT)
    return "不确定（步态判据通过、仅 L2 越界：允许补跑一次 w={:g}，即 {}）".format(GAIT_RERUN_WEIGHT, rerun)


def gh_gl_choice(gh_name, gl_name, decisions, worth):
    """GH/GL 配对按预登记规则的取舍；返回 (说明, 允许补跑的配置列表)。"""
    gh_ok = decisions.get(gh_name, {}).get("candidate_adopt", False)
    gl_ok = decisions.get(gl_name, {}).get("candidate_adopt", False)
    if gh_ok and gl_ok:
        return (gh_name if worth else gl_name + "（GH 结构无足够增量，采纳更简单的 GL）"), []
    if gh_ok or gl_ok:
        return (gh_name if gh_ok else gl_name), []
    # 都未候选时"不确定"仍有一次预登记补跑，不能并入"两者均未通过"，否则读 choice 的人或脚本会丢掉这次补跑。
    uncertain = [name for name in (gl_name, gh_name) if decisions.get(name, {}).get("uncertain", False)]
    reruns = [gait_rerun_config(name) for name in uncertain if gait_rerun_config(name) is not None]
    if reruns:
        return "{} 不确定（步态判据通过、仅 L2 越界）：按预登记补跑一次 w={:g}（{}）".format(
            "、".join(uncertain), GAIT_RERUN_WEIGHT, "、".join(reruns)), reruns
    if uncertain:
        return "{} 不确定但已是 w={:g} 补跑：按预登记不再调权重（{}）".format("、".join(uncertain), GAIT_RERUN_WEIGHT, GAIT_NEXT_ROUND), []
    return "两者均未通过（本轮不再调权重；{}）".format(GAIT_NEXT_ROUND), []


def decide(name, stats, reference_stats, control_stats, blend_stats=None):
    """设计 4.2 的预登记规则；返回 (是否候选采纳, 是否不确定, 理由列表)。自然度结论仍需看审查图后才能最终采纳。

    "不确定"只在其它判据都通过、仅 L2 判据落在 seed 噪声内时给出（改善方向但未达阈值，或 A4/A5 逐 seed 区间跨过
    容许线）：3 seed 下这类结果既不能判有效也不能判无效，应带入更长训练或更多 seed 复核，而不是判无效。
    """
    reasons = []
    if name.split("-")[0] == "A3" and blend_stats is not None:
        # 只报告不参与判断：检索锚点若不优于免训练的事后 root 混合，学习到的部分就没有增量。
        blend_delta = statistics.mean(paired_delta(stats, blend_stats)["mpjpe"])
        reasons.append("相对{} Δmpjpe {:+.2f}%".format(POSTHOC_LABEL, blend_delta))
    delta = paired_delta(stats, reference_stats)
    mean_delta = statistics.mean(delta["mpjpe"])
    delta_range = "逐 seed [{:+.2f}, {:+.2f}]".format(min(delta["mpjpe"]), max(delta["mpjpe"]))
    gate_ok = stats["gate"] == len(delta["mpjpe"])
    reasons.append("gate {}/{}".format(stats["gate"], len(delta["mpjpe"])))
    veto_ok, veto_reasons = natural_veto(stats, control_stats)
    reasons += veto_reasons
    natural, ref_natural = stats["natural"], reference_stats["natural"]
    kind = last_addon(name)
    if kind in GAIT_ADDONS:
        # A 与 B 都过才候选采纳；A 与非 L2 护栏都过、仅 L2 越界时判"不确定"（允许补跑一次 w=0.25）。
        gait_ok, guard_ok, l2_ok, gait_reasons = gait_criteria(name, stats, reference_stats)
        reasons += gait_reasons
        others_ok = gait_ok and guard_ok and gate_ok and veto_ok
        return others_ok and l2_ok, others_ok and not l2_ok, reasons
    if kind in NATURALNESS_ADDONS:
        l2_ok = mean_delta <= NATURAL_L2_TOLERANCE_PCT
        l2_noise = not l2_ok and min(delta["mpjpe"]) <= NATURAL_L2_TOLERANCE_PCT
        reasons.append("Δmpjpe {:+.2f}%，{}（允许 ≤ +{:.1f}%）".format(mean_delta, delta_range, NATURAL_L2_TOLERANCE_PCT))
        improved = False
        if natural and ref_natural:
            if kind == "A4":
                improved = natural["bone_rel_err_body"] <= A4_BONE_RATIO * ref_natural["bone_rel_err_body"]
                reasons.append("骨长误差 {:.4f} vs {:.4f}（要求 ≤ {:.0%}）".format(
                    natural["bone_rel_err_body"], ref_natural["bone_rel_err_body"], A4_BONE_RATIO))
            else:
                improved = any(natural[k] <= A5_SKATE_RATIO * ref_natural[k] for k in ("skate_gt_contact_speed_walk", "slide_frame_ratio"))
                reasons.append("步行脚速/滑行帧比至少一项改善 ≥ {:.0%}：{}".format(1 - A5_SKATE_RATIO, improved))
        others_ok = gate_ok and veto_ok and improved
    else:
        all_negative = all(value < 0 for value in delta["mpjpe"])
        l2_ok = mean_delta <= L2_ADOPT_DELTA_PCT and all_negative
        reasons.append("Δmpjpe {:+.2f}%，{}（要求 ≤ {:.1f}% 且逐 seed 为负：{}）".format(
            mean_delta, delta_range, L2_ADOPT_DELTA_PCT, all_negative))
        for key in ("xyz_mse", "xyz_mae"):
            if stats[key][0] > reference_stats[key][0]:
                l2_ok = False
                reasons.append("{} 均值上升".format(key))
        l2_noise = not l2_ok and mean_delta < 0
        others_ok = gate_ok and veto_ok
    ok = others_ok and l2_ok
    return ok, others_ok and l2_noise, reasons


def write_summary(results, steps, seeds, summary_dir, filename):
    stats = OrderedDict((name, config_stats(per_seed)) for name, per_seed in results.items())
    blend = posthoc_stats(results[CONTROL]) if CONTROL in results else None
    rows, decisions = [], OrderedDict()
    header = ["配置", "参照", "mpjpe (EMA)", "Δ% vs A0 逐 seed", "Δ% vs 参照", "xyz_mse", "xyz_mae", "raw mpjpe",
              "gate", "dct_mid"] + list(NATURAL_KEYS) + ["判断"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    control = stats.get(CONTROL)
    for name, s in stats.items():
        reference = reference_config(name)
        delta_a0 = paired_delta(s, control) if control is not None else None
        cells = [name, "—" if name == CONTROL else reference, "{:.5f} ± {:.5f}".format(*s["mpjpe"])]
        cells.append("—" if delta_a0 is None or name == CONTROL else " / ".join("{:+.2f}".format(v) for v in delta_a0["mpjpe"]))
        verdict = "对照"
        if name != CONTROL:
            if reference in stats:
                ok, uncertain, reasons = decide(name, s, stats[reference], control, blend)
                decisions[name] = OrderedDict([("reference", reference), ("candidate_adopt", ok), ("uncertain", uncertain),
                                               ("reasons", reasons)])
                cells.append("{:+.2f}".format(statistics.mean(paired_delta(s, stats[reference])["mpjpe"])))
                uncertain_label = gait_uncertain_label(name) if last_addon(name) in GAIT_ADDONS else "不确定（L2 落在 seed 噪声内，需更长训练复核）"
                label = "候选采纳（待审查图）" if ok else (uncertain_label if uncertain else "不采纳")
                verdict = label + "：" + "；".join(reasons)
            else:
                # Stage 3 的叠加配置常缺上一级的同步数 run：无法判 L2 增量，但 A0 否决项仍可判。
                veto_ok, veto_reasons = natural_veto(s, control)
                cells.append("参照未完成")
                verdict = "参照 {} 未完成（无采纳判断；A0 否决项{}）".format(reference, "通过" if veto_ok else "：" + "；".join(veto_reasons))
        else:
            cells.append("—")
        cells += ["{:.5f}".format(s["xyz_mse"][0]), "{:.5f}".format(s["xyz_mae"][0]), "{:.5f}".format(s["raw_mpjpe"][0]),
                  "{}/{}".format(s["gate"], len(s["mpjpe_per_seed"])), "{:.3f}".format(s["dct_mid"])]
        cells += ["{:.4f}".format(s["natural"][key]) if key in s["natural"] else "—" for key in NATURAL_KEYS]
        cells.append(verdict)
        lines.append("| " + " | ".join(cells) + " |")
        rows.append(OrderedDict([("config", name), ("reference", reference), ("stats", s), ("delta_vs_a0", delta_a0)]))
        if name == CONTROL and blend is not None:
            delta = paired_delta(blend, s)["mpjpe"]
            cells = [POSTHOC_LABEL, "A0", "{:.5f} ± {:.5f}".format(*blend["mpjpe"]), " / ".join("{:+.2f}".format(v) for v in delta), "—",
                     "{:.5f}".format(blend["xyz_mse"][0]), "{:.5f}".format(blend["xyz_mae"][0]), "—", "—", "{:.3f}".format(blend["dct_mid"])]
            cells += ["{:.4f}".format(blend["natural"][key]) if key in blend["natural"] else "—" for key in NATURAL_KEYS]
            cells.append("参考线（K=16、排除同受试者、saturate ramp）")
            lines.append("| " + " | ".join(cells) + " |")
            rows.append(OrderedDict([("config", POSTHOC_LABEL), ("reference", CONTROL), ("stats", blend)]))
    for gh_name, gl_name in gh_gl_pairs(list(stats)):
        worth, reason = gh_vs_gl(stats[gh_name], stats[gl_name])
        choice, reruns = gh_gl_choice(gh_name, gl_name, decisions, worth)
        adopt = any(decisions.get(name, {}).get("candidate_adopt", False) for name in (gh_name, gl_name))
        # 这一行的参照是 GL 不是 A0："Δ% vs A0 逐 seed"列留空，GH 对 GL 的逐 seed Δmpjpe 只写在判断理由里。
        d_mpjpe = paired_delta(stats[gh_name], stats[gl_name])["mpjpe"]
        cells = ["GH vs GL 配对：{}".format(gh_name), gl_name, "—", "—", "{:+.2f}".format(statistics.mean(d_mpjpe))]
        cells += ["—"] * (len(header) - len(cells) - 1)
        cells.append(("待审查图后采纳 {}" if adopt else "{}").format(choice) + "；GH 相对 GL：" + reason)
        lines.append("| " + " | ".join(cells) + " |")
        decisions["GH vs GL: " + gh_name] = OrderedDict([("gl", gl_name), ("gh_worth_structure", worth), ("choice", choice),
                                                          ("reruns", reruns), ("reason", reason)])
    lines += [
        "",
        "{} step，EMA 终点，val 198 条，seed {}。Δ% 为逐 seed 配对的相对变化（负为改善）。".format(steps, "/".join(str(seed) for seed in seeds)),
        "采纳规则（设计 4.2）：相对参照（底座对 A0，叠加配置对上一级）Δmpjpe 均值 ≤ {:.1f}% 且逐 seed 为负、"
        "xyz_mse/xyz_mae 均值不升；gate 全过；否决项固定对 A0：步行子集 GT 站定帧脚速与滑行帧比不比 A0 差超过 {:.0f}%、"
        "骨长相对误差不比 A0 高超过 {:.0f}%。".format(L2_ADOPT_DELTA_PCT, 100 * (NATURAL_VETO_RATIO - 1), 100 * (NATURAL_VETO_RATIO - 1)),
        "A4/A5（自然度目标，对上一级）：Δmpjpe ≤ +{:.1f}%；A4 要求骨长误差 ≤ 上一级的 {:.0%}，A5 要求步行脚速或滑行帧比"
        "至少改善 {:.0%}（后两个阈值为本驱动的实现口径）。任何候选都要看审查图（随机 12 例 + 位移最大 6 例）后才能最终采纳。".format(
            NATURAL_L2_TOLERANCE_PCT, A4_BONE_RATIO, 1 - A5_SKATE_RATIO),
        "噪声口径：主线 const-EMA 在 5000 step 的 val mpjpe seed 标准差 0.0043（2.3%），10000 step 为 0.0014（0.8%）；"
        "同 seed 配对不降噪（rootdct 对 const-EMA 的配对 Δmpjpe 标准差在 4000–10000 step 为 0.7–2.4%，5000 step 为 2.35%）。"
        "本轮 Stage 1/2 用 10000 step 筛选；\"不确定\"表示其它判据均通过、仅 L2 落在噪声内，应加 seed 复核而不是判无效。",
    ]
    if any(last_addon(name) in GAIT_ADDONS for name in stats):
        lines += [
            "步态增量 GL/GH（对参照 R = 去掉最后一个步态 token 的配置，逐 seed 配对；指标见 utils/ntu2p_naturalness.gait_phase_stats，"
            "步行人 = GT 未来 pelvis 水平位移 ≥ 0.5 m，s(t) = 左右踝沿行进方向的前后分离）：",
            "A. 主判据全过：corr_f01_10 均值 Δ ≥ +{:.2f}、绝对值 ≥ {:.2f}、逐 seed Δ 全正；corr_f11_20 均值 Δ ≥ +{:.2f}、绝对值 ≥ {:.2f}、"
            "至少 {} 个 seed Δ 为正；gait_sep_rmse_f01_10 / copy-last ≤ {:.2f}；步行 GT 站定帧脚速 ≤ {:.2f}× R（A5 底座）或 ≤ {:.2f}× R"
            "（无 A5 底座），且滑行帧比 ≤ {:.2f}× R。".format(
                GAIT_CORR01_MIN_DELTA, GAIT_CORR01_MIN_ABS, GAIT_CORR11_MIN_DELTA, GAIT_CORR11_MIN_ABS, GAIT_CORR11_MIN_POSITIVE,
                GAIT_RMSE01_MAX_RATIO_TO_COPY_LAST, GAIT_SKATE_MAX_RATIO_A5, GAIT_SKATE_MAX_RATIO, GAIT_SLIDE_MAX_RATIO),
            "B. 护栏全过：mpjpe/xyz_mse/xyz_mae 逐 seed 配对 Δ% 均值各 ≤ +{:.1f}%；gate 全过；gait_sep_rmse_f31_50 ≤ {:.2f}× R；"
            "A0 否决项通过。A、B 全过判\"候选采纳（待审查图）\"；A 与非 L2 护栏全过、仅 L2 越界判\"不确定\"（允许补跑一次 w={:g}，"
            "补跑本身再判\"不确定\"时不再补跑）；否则不采纳。".format(NATURAL_L2_TOLERANCE_PCT, GAIT_LONG_RMSE_MAX_RATIO, GAIT_RERUN_WEIGHT),
            "C. 审查图（seed 0，Stage 1 的 18 例 + walk_096、walk_070，行 GT / R / GL / GH）：位移最大的 6 个步行者中至少 4 个在 f1-30 "
            "内预测 s(t) 至少变号一次且先后与 GT 一致；俯视图为离散落脚点、无连续拖痕；非步行者 GT 静止时腿无可见晃动。",
            "GL 与 GH 都通过时，GH 须相对 GL 配对满足 Δcorr_f01_10 均值 ≥ +{:.2f} 且逐 seed 全正，或 Δmpjpe ≤ {:.1f}%，"
            "否则采纳更简单的 GL。两者都未候选采纳时，判\"不确定\"的一方按上条补跑（\"GH vs GL 配对\"行列出补跑配置，JSON 的 reruns "
            "字段相同），两者都不是\"不确定\"才写\"两者均未通过\"。配对行的\"参照\"为 GL、\"Δ% vs 参照\"为 GH 对 GL 的 Δmpjpe 均值，"
            "逐 seed 值见判断理由。只报告不判定：moving/starting 分组、gait_lead_foot_acc、local_mpjpe_legs、leg_swing_amp_ratio_to_gt、"
            "step_count、stance_fraction。".format(GH_OVER_GL_CORR_DELTA, GH_OVER_GL_MPJPE_PCT),
        ]
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, filename + ".json"), "w") as handle:
        json.dump(OrderedDict([("steps", steps), ("rows", rows), ("decisions", decisions)]), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(os.path.join(summary_dir, filename + ".md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    _say("summary written: {}/{}.md".format(summary_dir, filename))
    return decisions


def check_test_allowed(name, steps, opts):
    """test 只允许看一次，且只对 Stage 3 判为候选采纳的最终方案：先核对判断与全部终点，任何一项不满足就拒绝。"""
    if name == CONTROL:
        raise ValueError("--test_config 是最终方案，A0 已作为对照自动评估")
    if steps != FINAL_STEPS:
        raise ValueError("test 只在最终预算 {} step 的终点上评估，收到 --steps {}".format(FINAL_STEPS, steps))
    summary_path = os.path.join(opts.summary_dir, "summary_{}.json".format(steps))
    decision = _load(summary_path)["decisions"].get(name) if os.path.exists(summary_path) else None
    if not (decision and decision["candidate_adopt"]):
        detail = "；".join(decision["reasons"]) if decision else "无判断：先完成 Stage 3，叠加配置需同时训练上一级 {}".format(
            reference_config(name))
        raise ValueError("{} 在 {} 中不是候选采纳（{}）".format(name, summary_path, detail))
    missing = [path for label in (name, CONTROL) for seed in SEEDS
               for path in [os.path.join(run_dir(label, seed, steps, opts.save_root), "ema", "model{:09d}.pt".format(steps))]
               if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("缺少 EMA 终点：{}".format(missing))


def run_test(name, steps, opts):
    """只对最终方案（与 A0 对照）的 EMA 终点评估一次 test；seed 0 导出数组供审查图。
    固定用 SEEDS 而不是 --seeds：最终数字按协议是 3 seed。"""
    check_test_allowed(name, steps, opts)
    tables = OrderedDict()
    for label in (name, CONTROL):
        per_seed = []
        for seed in SEEDS:
            ema_dir = os.path.join(run_dir(label, seed, steps, opts.save_root), "ema")
            output = eval_json(ema_dir, steps, split="test")
            if not os.path.exists(output):
                export = os.path.join(ema_dir, "test_arrays_{:09d}.pt".format(steps)) if seed == SEEDS[0] else None
                checkpoint = os.path.join(ema_dir, "model{:09d}.pt".format(steps))
                _run(_eval_command(opts, checkpoint, output, split="test", export=export), os.path.join(ema_dir, "driver_test.log"), opts.dry_run)
            if not opts.dry_run:
                per_seed.append(_load(output))
        tables[label] = per_seed
    if opts.dry_run:
        return
    lines = ["| 配置 | mpjpe | xyz_mse | xyz_mae | gate | 步行脚速 | 滑行帧比 | 骨长误差 |", "|---|---|---|---|---|---|---|---|"]
    for label, results in tables.items():
        def mean_std(key, block="model_metrics"):
            return "{:.5f} ± {:.5f}".format(*_mean_std([r[block][key] for r in results]))
        natural = [r["naturalness"]["model"] for r in results]
        lines.append("| {} | {} | {} | {} | {}/{} | {:.4f} | {:.4f} | {:.4f} |".format(
            label, mean_std("mpjpe"), mean_std("xyz_mse"), mean_std("xyz_mae"),
            sum(bool(r["articulation_gate"]["passes_full_gate"]) for r in results), len(results),
            statistics.mean(n["skate_gt_contact_speed_walk"] for n in natural), statistics.mean(n["slide_frame_ratio"] for n in natural),
            statistics.mean(n["bone_rel_err_body"] for n in natural)))
    os.makedirs(opts.summary_dir, exist_ok=True)
    with open(os.path.join(opts.summary_dir, "test_{}_{}.md".format(name, steps)), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    _say("test table written: {}/test_{}_{}.md".format(opts.summary_dir, name, steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--configs", nargs="*", default=None, help="Stage 1 默认 A0 A1 A2 A3 A6；Stage 3 必填（采纳组合）")
    parser.add_argument("--stage2_base", default="A1", help="Stage 2 的底座配置（Stage 1 选出的最佳结构）")
    parser.add_argument("--foot_weight", type=float, default=A5_FOOT_WEIGHT,
                        help="Stage 2 中 A5 的 --foot_loss_weight；默认值的标定口径见 A5_FOOT_WEIGHT 注释（底座为 A6 时不适用，需重标）")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=None, help="默认 Stage 1/2 为 5000、Stage 3 为 10000")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retrieval_bank", default=DEFAULT_RETRIEVAL_BANK)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_root", default=SAVE_ROOT)
    parser.add_argument("--summary_dir", default=SUMMARY_DIR)
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--test_config", default=None,
                        help="Stage 3 判断后单独调用：只评估该配置与 A0 的 10000 step EMA test 终点；要求 summary_10000.json 判为候选采纳")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_cpu_for_smoke_test", action="store_true", help="仅冒烟测试：透传给训练与评估入口")
    parser.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[], help="仅冒烟测试：追加到训练命令末尾")
    opts = parser.parse_args()
    opts.smoke_args = ["--allow_cpu_for_smoke_test"] if opts.allow_cpu_for_smoke_test else []
    # --test_config 不依赖 --stage：否则默认 Stage 1 会让唯一一次 test 落在 5000 step 的筛选 checkpoint 上。
    steps = opts.steps or (FINAL_STEPS if (opts.stage == 3 or opts.test_config) else SCREEN_STEPS)

    if opts.test_config:
        run_test(opts.test_config, steps, opts)
        return
    if opts.stage == 1:
        configs = opts.configs or list(STAGE1_CONFIGS)
    elif opts.stage == 2:
        base = opts.stage2_base
        addons = ["A4", "A5f{:g}".format(opts.foot_weight), "A7"]
        configs = opts.configs or [CONTROL, base] + ["{}-{}".format(base, addon) for addon in addons]
    else:
        if not opts.configs:
            raise ValueError("Stage 3 需要 --configs 指定采纳的组合")
        configs = [CONTROL] + [name for name in opts.configs if name != CONTROL]
    for name in configs:
        config_args(name, opts.retrieval_bank)  # 先校验配置名，避免跑到一半才报错。
    configs = list(OrderedDict.fromkeys([CONTROL] + configs if opts.stage != 1 else configs))

    failures = [] if opts.summary_only else run_jobs(configs, opts.seeds, steps, opts)
    if not opts.dry_run:
        # 汇总覆盖同一 step 数下所有已完成的配置（Stage 1 与 Stage 2 同为 5000 step，合在一张表里）。
        known = sorted({entry[len("ntu2p_v2_"):].rsplit("_s", 1)[0] for entry in os.listdir(opts.save_root)
                        if entry.startswith("ntu2p_v2_") and entry.endswith("_{}".format(steps))})
        ordered = list(OrderedDict.fromkeys([CONTROL] + configs + known))
        filename = "summary" if steps == SCREEN_STEPS else "summary_{}".format(steps)
        write_summary(collect(ordered, opts.seeds, steps, opts.save_root), steps, opts.seeds, opts.summary_dir, filename)
    if failures:
        _say("{} 个 run 失败：{}".format(len(failures), failures))
        sys.exit(1)
    _say("all done")


if __name__ == "__main__":
    main()
