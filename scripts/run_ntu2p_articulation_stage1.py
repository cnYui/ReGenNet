"""串行执行 NTU2P residual refiner 摆动恢复 Stage 1 单因子实验，并对每个 checkpoint 评估、汇总。

对应计划：docs/ai/context/20260902-145142-ntu2p-articulation-recovery-training-plan.md
可断点续跑：最终 checkpoint 已存在则跳过训练，已有评估 JSON 则跳过评估。
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime

MANIFEST = "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"
BASELINE = "save/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_cuda_retrain_s0_5000/model000005000.pt"
SAVE_ROOT = "save/forecasting/ntu120_label"
SUMMARY_DIR = "results/forecasting/ntu120_label/ntu2p_articulation_stage1"

# 与 ntu2p_residual_refiner_xyz_inter001_s0_5000/args.json 一致的公共条件；被测因子之外全部不变。
COMMON_TRAIN_ARGS = [
    "--manifest_path", MANIFEST,
    "--baseline_checkpoint", BASELINE,
    "--device", "cuda:0",
    "--seed", "0",
    "--inter_loss_weight", "0.01",
    "--save_interval", "1000",
    "--log_interval", "100",
]

RUNS = OrderedDict(
    [
        ("s1_1_noreg", {"steps": 5000, "args": ["--delta_reg_weight", "0.0"], "hypothesis": "H2 残差正则压小"}),
        ("s1_2_long20k", {"steps": 20000, "args": [], "hypothesis": "H1 训练不足"}),
        ("s1_3_ramp_sat5", {"steps": 5000, "args": ["--ramp_mode", "saturate", "--ramp_saturate_frames", "5"], "hypothesis": "H3a 线性 ramp 抑制"}),
        ("s1_4_sinpos", {"steps": 5000, "args": ["--future_pos_mode", "sinusoidal"], "hypothesis": "H3b 零初始化位置编码"}),
        ("s1_5_scalenorm", {"steps": 5000, "args": ["--loss_scale_normalize"], "hypothesis": "H4 loss 尺度失衡"}),
        ("s1_6_localvel", {"steps": 5000, "args": ["--loss_scale_normalize", "--local_velocity_loss_weight", "1.0"], "hypothesis": "H4 局部姿态速度"}),
        ("s1_7_energy", {"steps": 5000, "args": ["--loss_scale_normalize", "--articulation_energy_loss_weight", "1.0"], "hypothesis": "H5 相位无关能量匹配"}),
    ]
)

CONTROL = {
    "run": "control_inter001",
    "save_dir": os.path.join(SAVE_ROOT, "ntu2p_residual_refiner_xyz_inter001_s0_5000"),
}


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message):
    print("[{}] {}".format(_now(), message), flush=True)


def _run(command):
    env = dict(os.environ)
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") or "."
    _log("$ " + " ".join(command))
    subprocess.run(command, check=True, env=env)


def _save_dir(name, steps):
    return os.path.join(SAVE_ROOT, "ntu2p_residual_refiner_xyz_artic_{}_s0_{}".format(name, steps))


def _checkpoints(save_dir):
    paths = sorted(glob.glob(os.path.join(save_dir, "model*.pt")))
    return [(int(re.search(r"model(\d+)\.pt$", path).group(1)), path) for path in paths]


def _train(name, spec, dry_run):
    save_dir = _save_dir(name, spec["steps"])
    final = os.path.join(save_dir, "model{:09d}.pt".format(spec["steps"]))
    if os.path.exists(final):
        _log("skip train {} (final checkpoint exists)".format(name))
        return save_dir
    command = [sys.executable, "train/train_ntu2p_residual_refiner_xyz.py", "--save_dir", save_dir, "--num_steps", str(spec["steps"])]
    command += COMMON_TRAIN_ARGS + spec["args"]
    if dry_run:
        _log("DRY RUN: " + " ".join(command))
    else:
        _run(command)
    return save_dir


def _evaluate(save_dir, dry_run, suffix=""):
    results = []
    for step, path in _checkpoints(save_dir):
        output = os.path.join(save_dir, "eval_val{}_{:09d}.json".format(suffix, step))
        if not os.path.exists(output):
            command = [
                sys.executable, "eval/eval_ntu2p_residual_refiner_xyz.py",
                "--manifest_path", MANIFEST, "--checkpoint", path, "--output", output, "--device", "cuda:0",
            ]
            if dry_run:
                _log("DRY RUN: " + " ".join(command))
                continue
            _run(command)
        with open(output) as handle:
            results.append((step, json.load(handle)))
    return results


def _row(run, step, result):
    model = result["model_metrics"]
    artic = result["articulation_metrics"]["model"]
    gate = result["articulation_gate"]
    return OrderedDict(
        [
            ("run", run),
            ("step", int(step)),
            ("xyz_mse", model["xyz_mse"]),
            ("xyz_mae", model["xyz_mae"]),
            ("mpjpe", model["mpjpe"]),
            ("energy_ratio", artic["articulation_energy_ratio_to_target"]),
            ("root_energy_ratio", artic["root_energy_ratio_to_target"]),
            ("std_ratio", artic["local_pose_temporal_std_ratio_to_target"]),
            ("frozen_ratio", artic["frozen_ratio"]),
            ("mpjpe_reg_vs_base", gate["model_mpjpe_regression_vs_base"]),
            ("l2_gate", gate["passes_l2_gate"]),
            ("artic_gate", gate["passes_articulation_gate"]),
            ("full_gate", gate["passes_full_gate"]),
        ]
    )


def _write_summary(rows, summary_dir=SUMMARY_DIR):
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "summary.json"), "w") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    header = list(rows[0].keys()) if rows else []
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = []
        for key in header:
            value = row[key]
            cells.append("{:.5f}".format(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(summary_dir, "summary.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="只执行指定 run 名称")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_control", action="store_true")
    args = parser.parse_args()

    rows = []
    if not args.skip_control:
        _log("evaluate control checkpoints")
        for step, result in _evaluate(CONTROL["save_dir"], args.dry_run):
            rows.append(_row(CONTROL["run"], step, result))
        _write_summary(rows)

    for name, spec in RUNS.items():
        if args.only and name not in args.only:
            continue
        _log("=== {} ({}) ===".format(name, spec["hypothesis"]))
        save_dir = _train(name, spec, args.dry_run)
        for step, result in _evaluate(save_dir, args.dry_run):
            rows.append(_row(name, step, result))
        _write_summary(rows)
        _log("finished {}".format(name))
    _log("all done; summary at {}".format(SUMMARY_DIR))


if __name__ == "__main__":
    main()
