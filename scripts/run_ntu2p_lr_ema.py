"""NTU2P residual refiner 学习率余弦尾部衰减 × 权重 EMA 的 2×2 对比：训练、val 评估、等价性核对与预登记判断。

对应计划：docs/ai/context/20260924-154319-ntu2p-lr-cosine-ema-plan.md
const-raw 复用已有的 final 10k run；可断点续跑。判断完成后用 --test_variant 单独评估选中变体的 test。
"""

import argparse
import json
import os
import statistics
from collections import OrderedDict

from scripts.run_ntu2p_articulation_stage1 import _evaluate, _log, _write_summary
from scripts.run_ntu2p_articulation_stage2 import _row
from scripts.run_ntu2p_articulation_stage3 import _save_dir, _train
from scripts.run_ntu2p_final_10k import CONFIG, _evaluate_test, _split_rows, _write_final

SUMMARY_DIR = "results/forecasting/ntu120_label/ntu2p_lr_ema"
STEPS = 10000
SEEDS = (0, 1, 2)
EMA = ["--ema_decay", "0.999"]
COSINE = ["--lr_schedule", "cosine_tail", "--lr_decay_start_frac", "0.8", "--lr_min", "3e-5"]
DECAY_START = 8000

# 新训练的 run：prefix → 额外参数。const-raw 来自已有 final run，不重训。
TRAIN_RUNS = OrderedDict([("ema", EMA), ("cosema", COSINE + EMA)])
# 变体 → (run prefix, 权重子目录)；顺序即"越简单越优先"的顺序（const-raw 为对照）。
VARIANTS = OrderedDict(
    [
        ("const-raw", ("final", "")),
        ("const-EMA", ("ema", "ema")),
        ("cos-raw", ("cosema", "")),
        ("cos-EMA", ("cosema", "ema")),
    ]
)

# 预登记阈值：J 至少减半，L 不比对照差出 1 个 seed 标准差，gate 全过；J 相差不到该值时取更简单者。
J_REDUCTION = 0.5
J_TIE = 0.0005


def _variant_dir(variant, seed):
    prefix, subdir = VARIANTS[variant]
    return os.path.join(_save_dir(CONFIG, seed, STEPS, prefix), subdir)


def _collect(dry_run):
    curves = OrderedDict()
    for variant in VARIANTS:
        curves[variant] = OrderedDict()
        for seed in SEEDS:
            curves[variant][seed] = OrderedDict(_evaluate(_variant_dir(variant, seed), dry_run))
    return curves


def _equivalence(curves):
    """原始权重的逐位等价：EMA 不扰动训练；余弦衰减开始前轨迹与恒定学习率相同。"""
    checks = [("const-EMA 训练的 raw 权重", "ema", STEPS), ("cos 训练的 raw 权重", "cosema", DECAY_START)]
    lines = ["| 核对 | seed | 比较的 checkpoint | 逐位相同 |", "|---|---:|---|---|"]
    passed = True
    for label, prefix, last_step in checks:
        for seed in SEEDS:
            raw = OrderedDict(_evaluate(_save_dir(CONFIG, seed, STEPS, prefix), dry_run=False))
            reference = curves["const-raw"][seed]
            steps = [step for step in reference if step <= last_step]
            same = all(raw[step]["model_metrics"] == reference[step]["model_metrics"] for step in steps)
            passed = passed and same
            lines.append("| {} vs const-raw | {} | {}–{} | {} |".format(label, seed, steps[0], steps[-1], "✓" if same else "✗"))
    return passed, lines


def _stats(curves):
    stats = OrderedDict()
    for variant, per_seed in curves.items():
        mpjpe = {seed: {step: r["model_metrics"]["mpjpe"] for step, r in steps.items()} for seed, steps in per_seed.items()}
        ends = [per_seed[seed][STEPS] for seed in SEEDS]
        endpoint = [mpjpe[seed][STEPS] for seed in SEEDS]
        stats[variant] = OrderedDict(
            [
                ("J", statistics.mean(abs(mpjpe[s][STEPS] - mpjpe[s][STEPS - 1000]) for s in SEEDS)),
                ("J_per_seed", [abs(mpjpe[s][STEPS] - mpjpe[s][STEPS - 1000]) for s in SEEDS]),
                ("tail_std", statistics.mean(statistics.stdev([mpjpe[s][t] for t in (STEPS - 2000, STEPS - 1000, STEPS)]) for s in SEEDS)),
                ("L_mean", statistics.mean(endpoint)),
                ("L_std", statistics.stdev(endpoint)),
                ("L_per_seed", endpoint),
                ("xyz_mse", statistics.mean(r["model_metrics"]["xyz_mse"] for r in ends)),
                ("xyz_mae", statistics.mean(r["model_metrics"]["xyz_mae"] for r in ends)),
                ("dct_mid", statistics.mean(r["articulation_gate"]["model_dct_mid_ratio"] for r in ends)),
                ("frozen", statistics.mean(r["articulation_gate"]["model_frozen_ratio"] for r in ends)),
                ("gate", sum(bool(r["articulation_gate"]["passes_full_gate"]) for r in ends)),
            ]
        )
    return stats


def _decide(stats):
    control = stats["const-raw"]
    eligible = [
        variant for variant, s in stats.items()
        if variant != "const-raw"
        and s["J"] <= J_REDUCTION * control["J"]
        and s["L_mean"] <= control["L_mean"] + control["L_std"]
        and s["gate"] == len(SEEDS)
    ]
    if not eligible:
        return "const-raw", eligible
    best_j = min(stats[v]["J"] for v in eligible)
    # eligible 保持 VARIANTS 的简单度顺序，第一个进入平局带的即为选中者。
    return next(v for v in eligible if stats[v]["J"] <= best_j + J_TIE), eligible


def _write_decision(stats, chosen, eligible, equivalence_lines, passed):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    with open(os.path.join(SUMMARY_DIR, "decision.json"), "w") as handle:
        json.dump({"stats": stats, "eligible": eligible, "chosen": chosen, "equivalence_passed": passed}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    lines = [
        "| 变体 | J = mean\\|m10k−m9k\\| | J 逐 seed | 8k–10k std | 终点 mpjpe | 逐 seed | xyz_mse | xyz_mae | dct_mid | frozen | gate |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for variant, s in stats.items():
        lines.append(
            "| {} | {:.5f} | {} | {:.5f} | {:.5f} ± {:.5f} | {} | {:.5f} | {:.5f} | {:.3f} | {:.3f} | {}/{} |".format(
                variant, s["J"], " / ".join("{:.5f}".format(v) for v in s["J_per_seed"]), s["tail_std"],
                s["L_mean"], s["L_std"], " / ".join("{:.5f}".format(v) for v in s["L_per_seed"]),
                s["xyz_mse"], s["xyz_mae"], s["dct_mid"], s["frozen"], s["gate"], len(SEEDS),
            )
        )
    lines += ["", "满足采纳条件：{}；选中：**{}**".format(", ".join(eligible) or "无", chosen)]
    with open(os.path.join(SUMMARY_DIR, "decision.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(SUMMARY_DIR, "equivalence.md"), "w") as handle:
        handle.write("\n".join(equivalence_lines + ["", "全部通过：{}".format(passed)]) + "\n")


def _run_test(variant):
    tables = OrderedDict()
    for name in (variant, "const-raw"):
        results = [(seed, _evaluate_test(os.path.join(_variant_dir(name, seed), "model{:09d}.pt".format(STEPS)), False)) for seed in SEEDS]
        tables["test (1253) {}".format(name)] = _split_rows(results, [], label=name)
    _write_final(tables, SUMMARY_DIR)
    _log("test done; table at {}/final.md".format(SUMMARY_DIR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--test_variant", choices=[v for v in VARIANTS if v != "const-raw"], default=None,
                        help="判断完成后单独调用：只评估选中变体的 test 终点")
    args = parser.parse_args()
    if args.test_variant:
        _run_test(args.test_variant)
        return

    for prefix, extra_args in TRAIN_RUNS.items():
        for seed in SEEDS:
            _log("=== {}_{}_s{}_{} ===".format(prefix, CONFIG, seed, STEPS))
            save_dir = _train(CONFIG, seed, STEPS, args.dry_run, prefix=prefix, extra_args=extra_args)
            # 每个 run 训完立即评估 raw 与 EMA，便于中途查看；后续汇总只读已有 JSON。
            _evaluate(save_dir, args.dry_run)
            _evaluate(os.path.join(save_dir, "ema"), args.dry_run)
    curves = _collect(args.dry_run)
    if args.dry_run:
        _log("dry run done")
        return
    _write_summary([_row("{}_s{}".format(v, seed), step, r) for v, per_seed in curves.items() for seed, steps in per_seed.items() for step, r in steps.items()], SUMMARY_DIR)
    passed, equivalence_lines = _equivalence(curves)
    stats = _stats(curves)
    chosen, eligible = _decide(stats)
    _write_decision(stats, chosen, eligible, equivalence_lines, passed)
    _log("all done; equivalence_passed={} chosen={} ; see {}".format(passed, chosen, SUMMARY_DIR))


if __name__ == "__main__":
    main()
