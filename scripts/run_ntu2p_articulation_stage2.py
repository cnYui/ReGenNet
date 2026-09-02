"""串行执行 NTU2P residual refiner 摆动恢复 Stage 2：幅度型损失 + 组合因子，按 DCT 分频带指标评估。

对应设计：docs/ai/context/20260902-173335-ntu2p-articulation-recovery-stage2-design-and-plan.md
复用 Stage 1 驱动的训练/评估/汇总函数；可断点续跑。
"""

import argparse
import os
from collections import OrderedDict

from scripts.run_ntu2p_articulation_stage1 import CONTROL, SAVE_ROOT, _evaluate, _log, _train, _write_summary

SUMMARY_DIR = "results/forecasting/ntu120_label/ntu2p_articulation_stage2"

# Stage 1 结论：局部姿态速度损失给出最佳 L2、饱和 ramp 是唯一弱有效的结构因子；两者作为 Stage 2 公共底座。
COMBO = [
    "--loss_scale_normalize",
    "--local_velocity_loss_weight", "1.0",
    "--ramp_mode", "saturate",
    "--ramp_saturate_frames", "5",
]

RUNS = OrderedDict(
    [
        ("s2_0_combo", {"steps": 5000, "args": COMBO, "hypothesis": "底座对照：归一化 + 局部速度 + 饱和 ramp"}),
        ("s2_1_std1", {"steps": 5000, "args": COMBO + ["--temporal_std_loss_weight", "1.0"], "hypothesis": "位置域时间 std 幅度匹配 w=1"}),
        ("s2_2_std3", {"steps": 5000, "args": COMBO + ["--temporal_std_loss_weight", "3.0"], "hypothesis": "位置域时间 std 幅度匹配 w=3"}),
        ("s2_3_dct1", {"steps": 5000, "args": COMBO + ["--dct_low_amplitude_loss_weight", "1.0", "--dct_mid_amplitude_loss_weight", "1.0"], "hypothesis": "DCT low+mid 频带幅度匹配 w=1"}),
        ("s2_4_dct3", {"steps": 5000, "args": COMBO + ["--dct_low_amplitude_loss_weight", "3.0", "--dct_mid_amplitude_loss_weight", "3.0"], "hypothesis": "DCT low+mid 频带幅度匹配 w=3"}),
        ("s2_5_dct1_root01", {"steps": 5000, "args": COMBO + ["--dct_low_amplitude_loss_weight", "1.0", "--dct_mid_amplitude_loss_weight", "1.0", "--root_loss_weight", "0.1"], "hypothesis": "DCT w=1 + root 降权（root 梯度约为 mse 的 10 倍）"}),
        ("s2_6_std1_dct1", {"steps": 5000, "args": COMBO + ["--temporal_std_loss_weight", "1.0", "--dct_low_amplitude_loss_weight", "1.0", "--dct_mid_amplitude_loss_weight", "1.0"], "hypothesis": "std + DCT 叠加 w=1"}),
    ]
)


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
            ("dct_low_ratio", artic["dct_low_energy_ratio_to_target"]),
            ("dct_mid_ratio", artic["dct_mid_energy_ratio_to_target"]),
            ("dct_high_ratio", artic["dct_high_energy_ratio_to_target"]),
            ("std_ratio", artic["local_pose_temporal_std_ratio_to_target"]),
            ("energy_ratio", artic["articulation_energy_ratio_to_target"]),
            ("frozen_ratio", artic["frozen_ratio"]),
            ("mpjpe_reg_vs_base", gate["model_mpjpe_regression_vs_base"]),
            ("l2_gate", gate["passes_l2_gate"]),
            ("artic_gate", gate["passes_articulation_gate"]),
            ("full_gate", gate["passes_full_gate"]),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_control", action="store_true")
    args = parser.parse_args()

    rows = []
    if not args.skip_control:
        # 对照组与 Stage 1 各 run 的最终 checkpoint 用新指标重评，输出带 _dct 后缀，不覆写旧评估文件。
        references = [(CONTROL["run"], CONTROL["save_dir"])]
        for name in ("s1_3_ramp_sat5", "s1_6_localvel", "s1_7_energy"):
            references.append((name, os.path.join(SAVE_ROOT, "ntu2p_residual_refiner_xyz_artic_{}_s0_5000".format(name))))
        for run, save_dir in references:
            _log("re-evaluate {} with DCT metrics".format(run))
            for step, result in _evaluate(save_dir, args.dry_run, suffix="_dct"):
                if step == 5000:
                    rows.append(_row(run, step, result))
        _write_summary(rows, SUMMARY_DIR)

    for name, spec in RUNS.items():
        if args.only and name not in args.only:
            continue
        _log("=== {} ({}) ===".format(name, spec["hypothesis"]))
        save_dir = _train(name, spec, args.dry_run)
        for step, result in _evaluate(save_dir, args.dry_run):
            rows.append(_row(name, step, result))
        _write_summary(rows, SUMMARY_DIR)
        _log("finished {}".format(name))
    _log("all done; summary at {}".format(SUMMARY_DIR))


if __name__ == "__main__":
    main()
