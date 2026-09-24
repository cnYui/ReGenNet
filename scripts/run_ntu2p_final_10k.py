"""NTU2P residual refiner 最终数字：s2_5 配置 × 3 seed × 10000 step，终点 checkpoint 的 val / test 评估与汇总。

对应计划：docs/ai/context/20260924-113322-ntu2p-final-10k-3seed-test-eval-plan.md
复用 Stage 1/2/3 驱动的训练、评估与汇总函数；可断点续跑。
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import OrderedDict

from scripts.run_ntu2p_articulation_stage1 import MANIFEST, SAVE_ROOT, _evaluate, _log, _run, _write_summary
from scripts.run_ntu2p_articulation_stage2 import _row
from scripts.run_ntu2p_articulation_stage3 import _train

SUMMARY_DIR = "results/forecasting/ntu120_label/ntu2p_final_10k"
CONFIG = "s2_5_dct1_root01"
PREFIX = "final"

# 只作 test 对照、不参与任何选择：摆动恢复前的 control 与 Stage 3 单 seed 长训终点。
REFERENCES = OrderedDict(
    [
        ("control_inter001_s0_5000", os.path.join(SAVE_ROOT, "ntu2p_residual_refiner_xyz_inter001_s0_5000", "model000005000.pt")),
        ("s3_s2_5_s0_20000", os.path.join(SAVE_ROOT, "ntu2p_residual_refiner_xyz_artic_s3_s2_5_dct1_root01_s0_20000", "model000020000.pt")),
    ]
)

L2_KEYS = ("xyz_mse", "xyz_mae", "mpjpe")
ARTIC_KEYS = (
    ("dct_low", "dct_low_energy_ratio_to_target"),
    ("dct_mid", "dct_mid_energy_ratio_to_target"),
    ("dct_high", "dct_high_energy_ratio_to_target"),
    ("frozen", "frozen_ratio"),
)


def _evaluate_test(checkpoint, dry_run):
    step = re.search(r"model(\d+)\.pt$", checkpoint).group(1)
    output = os.path.join(os.path.dirname(checkpoint), "eval_test_{}.json".format(step))
    if not os.path.exists(output):
        command = [
            sys.executable, "eval/eval_ntu2p_residual_refiner_xyz.py",
            "--manifest_path", MANIFEST, "--checkpoint", checkpoint, "--output", output,
            "--split", "test", "--device", "cuda:0",
        ]
        if dry_run:
            _log("DRY RUN: " + " ".join(command))
            return None
        _run(command)
    with open(output) as handle:
        return json.load(handle)


def _variant_row(label, result, variant="model"):
    metrics = result["{}_metrics".format(variant)]
    artic = result["articulation_metrics"][variant]
    row = OrderedDict([("run", label)])
    for key in L2_KEYS:
        row[key] = metrics[key]
    row["mpjpe_vs_base"] = metrics["mpjpe"] / result["base_metrics"]["mpjpe"] - 1.0
    row["mpjpe_vs_copy_last"] = metrics["mpjpe"] / result["copy_last_metrics"]["mpjpe"] - 1.0
    for name, key in ARTIC_KEYS:
        row[name] = artic[key]
    row["full_gate"] = result["articulation_gate"]["passes_full_gate"] if variant == "model" else "-"
    return row


def _mean_std_row(label, rows):
    row = OrderedDict([("run", label)])
    for key, value in rows[0].items():
        if isinstance(value, float):
            values = [r[key] for r in rows]
            row[key] = "{:.5f} ± {:.5f}".format(statistics.mean(values), statistics.stdev(values))
        elif key == "full_gate":
            row[key] = "{}/{}".format(sum(bool(r[key]) for r in rows), len(rows))
    return row


def _split_rows(seed_results, reference_results):
    seed_rows = [_variant_row("final_s{}_{}".format(seed, result["checkpoint_step"]), result) for seed, result in seed_results]
    rows = seed_rows + [_mean_std_row("final mean ± std (n={})".format(len(seed_rows)), seed_rows)]
    anchor = seed_results[0][1]
    rows.append(_variant_row("base (frozen independent)", anchor, "base"))
    rows.append(_variant_row("copy-last", anchor, "copy_last"))
    for name, result in reference_results:
        rows.append(_variant_row("ref: " + name, result))
    gt_frozen = anchor["articulation_metrics"]["model"]["frozen_ratio_target"]
    rows.append(OrderedDict([("run", "GT"), ("frozen", gt_frozen)]))
    return rows


def _write_final(tables):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    with open(os.path.join(SUMMARY_DIR, "final.json"), "w") as handle:
        json.dump(tables, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    header = ["run"] + list(L2_KEYS) + ["mpjpe_vs_base", "mpjpe_vs_copy_last"] + [name for name, _ in ARTIC_KEYS] + ["full_gate"]
    lines = []
    for split, rows in tables.items():
        lines += ["## {}".format(split), "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for row in rows:
            cells = []
            for key in header:
                value = row.get(key, "")
                cells.append("{:.5f}".format(value) if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    with open(os.path.join(SUMMARY_DIR, "final.md"), "w") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    curve_rows = []
    endpoints = {"val": [], "test": []}
    for seed in args.seeds:
        name = "final_{}_s{}_{}".format(CONFIG, seed, args.steps)
        _log("=== {} ===".format(name))
        save_dir = _train(CONFIG, seed, args.steps, args.dry_run, prefix=PREFIX)
        val_results = _evaluate(save_dir, args.dry_run)
        for step, result in val_results:
            curve_rows.append(_row(name, step, result))
        if curve_rows:
            _write_summary(curve_rows, SUMMARY_DIR)
        test_result = _evaluate_test(os.path.join(save_dir, "model{:09d}.pt".format(args.steps)), args.dry_run)
        if val_results and test_result is not None:
            endpoints["val"].append((seed, dict(val_results)[args.steps]))
            endpoints["test"].append((seed, test_result))
        _log("finished {}".format(name))

    references = [(name, _evaluate_test(path, args.dry_run)) for name, path in REFERENCES.items()]
    if args.dry_run or len(endpoints["test"]) < 2:
        _log("skip final table (dry run or fewer than 2 finished seeds)")
        return
    tables = OrderedDict(
        [
            ("val (198)", _split_rows(endpoints["val"], [])),
            ("test (1253)", _split_rows(endpoints["test"], references)),
        ]
    )
    _write_final(tables)
    _log("all done; summary at {}".format(SUMMARY_DIR))


if __name__ == "__main__":
    main()
