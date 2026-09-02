"""NTU2P 摆动恢复 Stage 3：胜出配置的多 seed 复现与 20000 step 长训。

对应设计：docs/ai/context/20260902-173335-ntu2p-articulation-recovery-stage2-design-and-plan.md（Stage 3 触发条件）
复用 Stage 1/2 驱动函数；可断点续跑。
"""

import argparse
import os
from collections import OrderedDict

from scripts.run_ntu2p_articulation_stage1 import COMMON_TRAIN_ARGS, SAVE_ROOT, _evaluate, _log, _run, _write_summary
from scripts.run_ntu2p_articulation_stage2 import COMBO, _row

import sys

SUMMARY_DIR = "results/forecasting/ntu120_label/ntu2p_articulation_stage3"

CONFIGS = {
    "s2_5_dct1_root01": COMBO + ["--dct_low_amplitude_loss_weight", "1.0", "--dct_mid_amplitude_loss_weight", "1.0", "--root_loss_weight", "0.1"],
    "s2_3_dct1": COMBO + ["--dct_low_amplitude_loss_weight", "1.0", "--dct_mid_amplitude_loss_weight", "1.0"],
    "s2_6_std1_dct1": COMBO + ["--temporal_std_loss_weight", "1.0", "--dct_low_amplitude_loss_weight", "1.0", "--dct_mid_amplitude_loss_weight", "1.0"],
}


def _save_dir(config, seed, steps):
    return os.path.join(SAVE_ROOT, "ntu2p_residual_refiner_xyz_artic_s3_{}_s{}_{}".format(config, seed, steps))


def _train(config, seed, steps, dry_run):
    save_dir = _save_dir(config, seed, steps)
    final = os.path.join(save_dir, "model{:09d}.pt".format(steps))
    if os.path.exists(final):
        _log("skip train {} (final checkpoint exists)".format(save_dir))
        return save_dir
    common = [arg for arg in COMMON_TRAIN_ARGS]
    seed_index = common.index("--seed")
    common[seed_index + 1] = str(seed)
    command = [sys.executable, "train/train_ntu2p_residual_refiner_xyz.py", "--save_dir", save_dir, "--num_steps", str(steps)]
    command += common + CONFIGS[config]
    if dry_run:
        _log("DRY RUN: " + " ".join(command))
    else:
        _run(command)
    return save_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), default="s2_5_dct1_root01")
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--long_steps", type=int, default=20000, help="0 表示不跑长训")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    rows = []
    jobs = [(seed, 5000) for seed in args.seeds]
    if args.long_steps > 0:
        jobs.append((0, args.long_steps))
    for seed, steps in jobs:
        name = "s3_{}_s{}_{}".format(args.config, seed, steps)
        _log("=== {} ===".format(name))
        save_dir = _train(args.config, seed, steps, args.dry_run)
        for step, result in _evaluate(save_dir, args.dry_run):
            rows.append(_row(name, step, result))
        _write_summary(rows, SUMMARY_DIR)
        _log("finished {}".format(name))
    _log("all done; summary at {}".format(SUMMARY_DIR))


if __name__ == "__main__":
    main()
