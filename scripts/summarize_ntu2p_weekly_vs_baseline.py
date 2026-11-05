"""NTU2P 周报汇总：最新主线 vs 初始 baseline 的全方位对比（表 + bootstrap + 分层误差 + 图）。

输入：
- --eval_dir：`eval/eval_ntu2p_v2.py --split test` 的输出，命名 `<里程碑>_s<seed>_test.json`；
  同一份 JSON 内已含该次评估的 base（冻结独立单人 baseline）、copy-last、GT 的 L2 / 摆动 / 自然度指标，
  因此 baseline 与各里程碑在同一批 1253 条 test 样本、同一指标实现下比较。
- --arrays：最新主线某个 seed 用 `--export_arrays` 导出的数组，用于配对 bootstrap、分 horizon / 分动作 / 逐样本分析。

只汇报，不参与选型：test 在各里程碑选型时已用过，本脚本不得反过来调整任何配置。
"""

import argparse
import json
import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.analyze_ntu2p_naturalness import NTU2P_ACTION_NAMES
from utils.ntu2p_naturalness import PELVIS

FPS = 20.0  # 50 帧 = 2.5 s（与既有文档 horizon 口径一致）
BOOTSTRAP_ROUNDS = 10000

# (key, 展示名, 评估文件前缀)；copy_last / base 取自任一 JSON 的同名块，不单独评估。
MILESTONES = (
    ("copy_last", "copy-last", None),
    ("base", "独立单人 baseline（初始）", None),
    ("refiner_first", "首版 residual refiner", "refiner_first"),
    ("artic", "+ 摆动恢复（10k）", "artic"),
    ("ema", "+ 权重 EMA", "ema"),
    ("rootdct", "+ root DCT 头", "rootdct"),
    ("gh", "v2 InterMixer（GH）", "gh"),
    ("fd2", "v2 + FD2（最新主线）", "fd2"),
)
LATEST, BASE, COPY = "fd2", "base", "copy_last"

# 配色按角色固定：基线橙、最新蓝、中间里程碑浅蓝、copy-last 中性灰、GT 黑。
C_LATEST, C_BASE, C_MID, C_COPY, C_GT, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#86b6ef", "#8a8985", "#0b0b0b", "#0b0b0b", "#52514e"

L2_KEYS = ("xyz_mse", "xyz_mae", "mpjpe")
DECOMP_KEYS = (
    ("root_translation_error", "root（骨盆）平移误差"),
    ("local_pose_error", "局部姿态误差（相对 root）"),
    ("short_xyz_mse", "short horizon xyz_mse"),
    ("mid_xyz_mse", "mid horizon xyz_mse"),
    ("long_xyz_mse", "long horizon xyz_mse"),
    ("final_frame_error", "终帧（第 50 帧）误差"),
    ("velocity_error", "速度误差"),
    ("acceleration_error", "加速度误差"),
)
RELATION_KEYS = (
    ("relative_root_distance_error", "两人相对 root 距离误差"),
    ("relative_root_velocity_error", "两人相对 root 速度误差"),
    ("key_joint_relation_error", "关键关节跨人关系误差"),
    ("contact_error", "接触帧跨人距离误差"),
    ("inter_person_distance_consistency", "两人距离一致性"),
)
NATURAL_KEYS = (
    ("bone_rel_err_body", "身体骨长相对误差", "lower"),
    ("bone_rel_err_body_max", "身体骨长最大偏差", "lower"),
    ("slide_frame_ratio", "滑行帧比（脚与骨盆锁定平移）", "lower"),
    ("skate_gt_contact_speed_walk", "步行 GT 站定帧上的预测脚速（m/s）", "lower"),
    ("step_count", "步数（GT 约 2.19）", "gt"),
    ("leg_swing_amp_ratio_to_gt", "腿摆幅 / GT", "gt"),
    ("gait_sep_corr_f01_10", "左右踝前后分离相位相关 第 1–10 帧", "higher"),
    ("gait_sep_corr_f11_20", "左右踝前后分离相位相关 第 11–20 帧", "higher"),
    ("gait_sep_corr_f21_30", "左右踝前后分离相位相关 第 21–30 帧", "higher"),
    ("gait_lead_foot_acc", "先迈脚正确率", "higher"),
    ("foot_penetration_ratio", "脚穿地帧比", "lower"),
    ("foot_float_ratio", "脚悬空帧比", "gt"),
    ("heading_err_deg", "朝向误差（度）", "lower"),
    ("mpjpe_legs", "腿 mpjpe", "lower"),
    ("mpjpe_arms", "臂 mpjpe", "lower"),
    ("mpjpe_torso", "躯干 mpjpe", "lower"),
    ("mpjpe_head", "头 mpjpe", "lower"),
)


def set_style():
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cfcec8",
            "axes.labelcolor": C_MUTED,
            "xtick.color": C_MUTED,
            "ytick.color": C_MUTED,
            "text.color": C_INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#ecebe6",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "font.size": 10.5,
        }
    )


def load_runs(eval_dir):
    runs = OrderedDict()
    for key, _, prefix in MILESTONES:
        if prefix is None:
            continue
        files = sorted(name for name in os.listdir(eval_dir) if name.startswith(prefix + "_s") and name.endswith("_test.json"))
        # 前缀 "ema_s" 不会误匹配 "...": 文件名形如 ema_s0_test.json
        runs[key] = [json.load(open(os.path.join(eval_dir, name))) for name in files]
        runs[key + "_files"] = files
    return runs


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    # 样本标准差（ddof=1），与既有结果文档 "0.1805 ± 0.0005" 的口径一致。
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), std


def fmt(values, digits=4):
    mean, std = mean_std(values)
    if np.isnan(mean):
        return "—"  # 例如 copy-last 没有左右踝分离，相位相关无定义
    if len(values) == 1:
        return "{:.{d}f}".format(mean, d=digits)
    return "{:.{d}f} ± {:.{d}f}".format(mean, std, d=digits)


def pct(new, ref):
    return 100.0 * (new / ref - 1.0)


def metric_block(runs, key, group, name):
    """返回 list[float]：某里程碑（或 base / copy_last）在若干 seed 上的某个指标。"""
    if key in (BASE, COPY):
        source = runs[LATEST][0]
        block = source["base_metrics" if key == BASE else "copy_last_metrics"] if group == "metrics" else None
        if group == "artic":
            return [source["articulation_metrics"]["base" if key == BASE else "copy_last"][name]]
        if group == "natural":
            return [source["naturalness"]["base" if key == BASE else "copy_last"][name]]
        return [block[name]]
    items = runs[key]
    if group == "metrics":
        return [item["model_metrics"][name] for item in items]
    if group == "artic":
        return [item["articulation_metrics"]["model"][name] for item in items]
    return [item["naturalness"]["model"][name] for item in items]


def check_reference_consistency(runs):
    """各 JSON 的 base / copy-last 应逐位相同（同一 baseline 权重、同一 test 窗口）。"""
    reference = runs[LATEST][0]
    worst = 0.0
    for key, _, prefix in MILESTONES:
        if prefix is None:
            continue
        for item in runs[key]:
            for block in ("base_metrics", "copy_last_metrics"):
                for name in L2_KEYS:
                    worst = max(worst, abs(item[block][name] - reference[block][name]))
            if item["num_samples"] != reference["num_samples"]:
                raise ValueError("test 样本数不一致")
    return worst


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def table_main(runs, summary):
    base_mpjpe = metric_block(runs, BASE, "metrics", "mpjpe")[0]
    copy_mpjpe = metric_block(runs, COPY, "metrics", "mpjpe")[0]
    rows = []
    previous = None
    for key, label, prefix in MILESTONES:
        values = OrderedDict((name, metric_block(runs, key, "metrics", name)) for name in L2_KEYS)
        mp = mean_std(values["mpjpe"])[0]
        seeds = 1 if prefix is None else len(runs[key])
        step = "—" if previous is None else "{:+.1f}%".format(pct(mp, previous))
        previous = mp
        rows.append(
            [
                label,
                seeds if prefix else "—",
                fmt(values["xyz_mse"], 5),
                fmt(values["xyz_mae"], 5),
                fmt(values["mpjpe"], 5),
                "—" if key == COPY else "{:+.1f}%".format(pct(mp, copy_mpjpe)),
                "—" if key in (COPY, BASE) else "{:+.1f}%".format(pct(mp, base_mpjpe)),
                step,
            ]
        )
        summary["milestones"][key] = OrderedDict((name, mean_std(values[name])) for name in L2_KEYS)
    header = ["方法", "seed 数", "xyz_mse", "xyz_mae", "mpjpe", "mpjpe vs copy-last", "mpjpe vs baseline", "较上一行"]
    return md_table(header, rows)


def table_group(runs, group_keys, group, digits=4):
    shown = [("copy_last", "copy-last"), ("base", "baseline")] + [(k, l) for k, l, p in MILESTONES if p and k in ("artic", "rootdct", "gh", "fd2")]
    header = ["指标"] + [label for _, label in shown] + ["最新 vs baseline"]
    rows = []
    for name, label in group_keys:
        cells = [fmt(metric_block(runs, key, group, name), digits) for key, _ in shown]
        base_value = mean_std(metric_block(runs, BASE, group, name))[0]
        latest_value = mean_std(metric_block(runs, LATEST, group, name))[0]
        change = "{:+.1f}%".format(pct(latest_value, base_value)) if base_value else "—"
        rows.append([label] + cells + [change])
    return md_table(header, rows)


def table_articulation(runs):
    shown = [("copy_last", "copy-last"), ("base", "baseline")] + [(k, l) for k, l, p in MILESTONES if p and k in ("refiner_first", "artic", "ema", "rootdct", "gh", "fd2")]
    header = ["指标"] + [label for _, label in shown] + ["GT"]
    spec = (
        ("dct_low_energy_ratio_to_target", "≤1 Hz 慢动作能量 / GT", None),
        ("dct_mid_energy_ratio_to_target", "1.2–2 Hz 步态频段能量 / GT", None),
        ("dct_high_energy_ratio_to_target", "高频（抖动）能量 / GT", None),
        ("frozen_ratio", "关节几乎不动的比例（越接近 GT 越好）", "frozen_ratio_target"),
    )
    rows = []
    for name, label, target in spec:
        cells = [fmt(metric_block(runs, key, "artic", name), 3) for key, _ in shown]
        gt = "1.000" if target is None else "{:.3f}".format(runs[LATEST][0]["articulation_metrics"]["model"][target])
        rows.append([label] + cells + [gt])
    gate = ["摆动 gate（dct_low≥0.40 且 dct_mid≥0.10）通过 seed 数"]
    for key, _ in shown:
        if key in (COPY, BASE):
            source = runs[LATEST][0]
            art = source["articulation_metrics"]["copy_last" if key == COPY else "base"]
            passed = art["dct_low_energy_ratio_to_target"] >= 0.40 and art["dct_mid_energy_ratio_to_target"] >= 0.10
            gate.append("通过" if passed else "未通过")
        else:
            gate.append("{}/{}".format(sum(bool(item["articulation_gate"]["passes_articulation_gate"]) for item in runs[key]), len(runs[key])))
    gate.append("—")
    rows.append(gate)
    return md_table(header, rows)


def table_natural(runs):
    shown = [("copy_last", "copy-last"), ("base", "baseline")] + [(k, l) for k, l, p in MILESTONES if p and k in ("rootdct", "gh", "fd2")]
    header = ["指标"] + [label for _, label in shown] + ["GT"]
    gt_block = runs[LATEST][0]["naturalness"]["GT"]
    rows = []
    for name, label, _ in NATURAL_KEYS:
        cells = [fmt(metric_block(runs, key, "natural", name), 3) for key, _ in shown]
        gt = gt_block.get(name)
        rows.append([label] + cells + ["—" if gt is None else "{:.3f}".format(gt)])
    return md_table(header, rows)


def per_sample_mpjpe(pred, target):
    return torch.linalg.norm(pred.double() - target.double(), dim=-1).mean(dim=(1, 2, 3))


def array_analysis(arrays_path, runs, summary):
    data = torch.load(arrays_path, map_location="cpu")
    obs, target = data["obs_xyz"], data["target_xyz"]
    preds = OrderedDict(
        [
            (LATEST, data["methods"]["model"]),
            (BASE, data["methods"]["base"]),
            (COPY, obs[:, -1:].expand(-1, target.shape[1], -1, -1, -1).contiguous()),
        ]
    )
    sample = OrderedDict((key, per_sample_mpjpe(value, target).numpy()) for key, value in preds.items())
    # 与评估 JSON 对账：每样本 mpjpe 的均值应等于 model_metrics.mpjpe。
    reference = runs[LATEST][0]
    drift = abs(sample[LATEST].mean() - reference["model_metrics"]["mpjpe"])
    summary["array_vs_json_mpjpe_drift"] = float(drift)

    count = len(sample[LATEST])
    rng = np.random.default_rng(0)
    index = rng.integers(0, count, size=(BOOTSTRAP_ROUNDS, count))
    boot = OrderedDict()
    for ref_key in (BASE, COPY):
        new_mean = sample[LATEST][index].mean(axis=1)
        ref_mean = sample[ref_key][index].mean(axis=1)
        rel = 100.0 * (new_mean / ref_mean - 1.0)
        diff = new_mean - ref_mean
        boot[ref_key] = OrderedDict(
            [
                ("rel_pct_mean", float(rel.mean())),
                ("rel_pct_ci95", [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))]),
                ("abs_diff_mean", float(diff.mean())),
                ("abs_diff_ci95", [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]),
                ("win_rate", float((sample[LATEST] < sample[ref_key]).mean())),
                ("median_sample_reduction_pct", float(np.median(100.0 * (sample[LATEST] / sample[ref_key] - 1.0)))),
            ]
        )
    summary["bootstrap"] = boot

    # 分 horizon：整体 / root / 局部姿态。
    def per_frame(pred):
        return torch.linalg.norm(pred.double() - target.double(), dim=-1).mean(dim=(0, 2, 3)).numpy()

    def root_local(pred):
        root_err = torch.linalg.norm(pred[..., PELVIS, :].double() - target[..., PELVIS, :].double(), dim=-1).mean(dim=(0, 2)).numpy()
        pred_local = pred.double() - pred[..., PELVIS : PELVIS + 1, :].double()
        target_local = target.double() - target[..., PELVIS : PELVIS + 1, :].double()
        local_err = torch.linalg.norm(pred_local - target_local, dim=-1).mean(dim=(0, 2, 3)).numpy()
        return root_err, local_err

    horizon = OrderedDict()
    for key, value in preds.items():
        root_err, local_err = root_local(value)
        horizon[key] = OrderedDict([("total", per_frame(value)), ("root", root_err), ("local", local_err)])
    summary["horizon_s"] = OrderedDict()
    marks = (5, 10, 20, 30, 40, 50)
    for key, value in horizon.items():
        summary["horizon_s"][key] = OrderedDict(
            (str(frame / FPS) + "s", OrderedDict((part, float(series[frame - 1])) for part, series in value.items())) for frame in marks
        )

    # 分动作：每类的 mpjpe（latest / baseline / copy-last）与样本数。
    actions = data["actions"].reshape(-1).numpy()
    per_action = []
    for action in range(len(NTU2P_ACTION_NAMES)):
        mask = actions == action
        if mask.sum() == 0:
            continue
        row = OrderedDict([("action", NTU2P_ACTION_NAMES[action]), ("n", int(mask.sum()))])
        for key in (LATEST, BASE, COPY):
            row[key] = float(sample[key][mask].mean())
        row["vs_base_pct"] = pct(row[LATEST], row[BASE])
        row["vs_copy_pct"] = pct(row[LATEST], row[COPY])
        per_action.append(row)
    summary["per_action"] = per_action
    return sample, horizon, per_action


def table_bootstrap(summary):
    boot = summary["bootstrap"]
    rows = []
    for key, label in ((BASE, "vs 独立单人 baseline"), (COPY, "vs copy-last")):
        item = boot[key]
        rows.append(
            [
                label,
                "{:+.2f}%".format(item["rel_pct_mean"]),
                "[{:+.2f}%, {:+.2f}%]".format(*item["rel_pct_ci95"]),
                "{:+.4f}".format(item["abs_diff_mean"]),
                "[{:+.4f}, {:+.4f}]".format(*item["abs_diff_ci95"]),
                "{:.1f}%".format(100 * item["win_rate"]),
                "{:+.1f}%".format(item["median_sample_reduction_pct"]),
            ]
        )
    header = ["对比", "mpjpe 相对变化", "95% CI", "mpjpe 绝对差（m）", "95% CI", "逐样本胜率", "逐样本中位变化"]
    return md_table(header, rows)


def table_horizon(summary):
    header = ["时间点", "copy-last", "baseline", "最新主线", "最新 vs baseline", "root：baseline → 最新", "局部姿态：baseline → 最新"]
    rows = []
    for tag in summary["horizon_s"][LATEST]:
        latest, base, copy = (summary["horizon_s"][key][tag] for key in (LATEST, BASE, COPY))
        rows.append(
            [
                tag,
                "{:.4f}".format(copy["total"]),
                "{:.4f}".format(base["total"]),
                "{:.4f}".format(latest["total"]),
                "{:+.1f}%".format(pct(latest["total"], base["total"])),
                "{:.3f} → {:.3f}".format(base["root"], latest["root"]),
                "{:.3f} → {:.3f}".format(base["local"], latest["local"]),
            ]
        )
    return md_table(header, rows)


def table_action(per_action):
    header = ["动作", "样本数", "copy-last", "baseline", "最新主线", "最新 vs baseline"]
    rows = [
        [item["action"], item["n"], "{:.3f}".format(item[COPY]), "{:.3f}".format(item[BASE]), "{:.3f}".format(item[LATEST]), "{:+.1f}%".format(item["vs_base_pct"])]
        for item in sorted(per_action, key=lambda x: x["vs_base_pct"])
    ]
    return md_table(header, rows)


# ---------------------------------------------------------------- 图
def fig_milestones(runs, summary, path):
    labels, values, errors, colors, notes = [], [], [], [], []
    base_mpjpe = metric_block(runs, BASE, "metrics", "mpjpe")[0]
    for key, label, prefix in MILESTONES:
        mean, std = mean_std(metric_block(runs, key, "metrics", "mpjpe"))
        labels.append(label)
        values.append(mean)
        errors.append(std)
        colors.append({COPY: C_COPY, BASE: C_BASE, LATEST: C_LATEST}.get(key, C_MID))
        notes.append("{:.4f}".format(mean) + ("" if key in (COPY, BASE) else "  ({:+.1f}% vs baseline)".format(pct(mean, base_mpjpe))))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    positions = np.arange(len(labels))[::-1]
    ax.barh(positions, values, color=colors, height=0.62)
    multi = [i for i, err in enumerate(errors) if err > 0]  # 单 seed / 参照行没有标准差，不画误差棒
    ax.errorbar([values[i] for i in multi], [positions[i] for i in multi], xerr=[errors[i] for i in multi], fmt="none", ecolor=C_INK, elinewidth=1.2, capsize=3)
    for pos, value, note in zip(positions, values, notes):
        ax.text(value + 0.004, pos, note, va="center", fontsize=9.5, color=C_INK)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.43)
    ax.set_xlabel("test mpjpe（米，越低越好；误差棒为 3 个 seed 的标准差，首版 refiner 仅 1 seed）")
    ax.grid(axis="y", visible=False)
    ax.set_title("从初始 baseline 到最新主线：test 1253 条上的 mpjpe", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_horizon(horizon, path):
    time = (np.arange(1, 51)) / FPS
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    style = ((COPY, "copy-last", C_COPY, "-"), (BASE, "独立单人 baseline", C_BASE, "-"), (LATEST, "最新主线", C_LATEST, "-"))
    for ax, part, title in zip(axes[:2], ("total", "root"), ("整体 mpjpe 随预测时长", "root（骨盆）误差随预测时长")):
        for key, label, color, ls in style:
            ax.plot(time, horizon[key][part], color=color, lw=2, ls=ls, label=label)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("预测时间（s）")
        ax.set_ylabel("误差（m）")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="upper left")
    ax = axes[2]
    for part, label, ls in (("total", "整体", "-"), ("root", "root", "--"), ("local", "局部姿态", ":")):
        reduction = 100.0 * (1.0 - horizon[LATEST][part] / np.maximum(horizon[BASE][part], 1e-9))
        ax.plot(time, reduction, color=C_LATEST, lw=2, ls=ls, label=label)
    ax.set_title("最新主线相对 baseline 的误差降幅", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("预测时间（s）")
    ax.set_ylabel("降低百分比（%）")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_action(per_action, path):
    ordered = sorted(per_action, key=lambda x: x["vs_base_pct"], reverse=True)
    fig, ax = plt.subplots(figsize=(9.4, 8.2))
    positions = np.arange(len(ordered))
    gap = 0.2
    ax.barh(positions + gap, [item[BASE] for item in ordered], height=0.36, color=C_BASE, label="独立单人 baseline")
    ax.barh(positions - gap, [item[LATEST] for item in ordered], height=0.36, color=C_LATEST, label="最新主线")
    for pos, item in zip(positions, ordered):
        ax.text(max(item[BASE], item[LATEST]) + 0.006, pos, "{:+.0f}%  (n={})".format(item["vs_base_pct"], item["n"]), va="center", fontsize=8.8, color=C_INK)
    ax.set_yticks(positions)
    ax.set_yticklabels([item["action"] for item in ordered], fontsize=9.5)
    ax.set_xlim(0, 0.52)
    ax.set_xlabel("该动作的 test mpjpe（米，越低越好）")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("分动作：最新主线相对 baseline 的 mpjpe 变化（seed 0）", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_scatter(sample, summary, path):
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    x, y = sample[BASE], sample[LATEST]
    # 个别坏样本（GT 跟踪失败）会把坐标轴拉得很长，轴按 99.5 分位截取并标注轴外点数。
    limit = float(np.percentile(np.maximum(x, y), 99.5)) * 1.08
    outside = int(((x > limit) | (y > limit)).sum())
    ax.scatter(x, y, s=9, color=C_LATEST, alpha=0.4, linewidths=0)
    ax.plot([0, limit], [0, limit], color=C_MUTED, lw=1.2, ls="--")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("独立单人 baseline 的样本 mpjpe（米）")
    ax.set_ylabel("最新主线的样本 mpjpe（米）")
    win = summary["bootstrap"][BASE]["win_rate"]
    ax.text(0.04 * limit, 0.93 * limit, "对角线以下 = 最新主线更好\n{:.1f}% 的样本更好\n（{} 个样本在轴外）".format(100 * win, outside), fontsize=10, va="top", color=C_INK)
    ax.set_title("逐样本配对（1253 条，seed 0）", loc="left", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_motion(runs, path):
    natural_gt = runs[LATEST][0]["naturalness"]["GT"]
    specs = []
    for name, label in (
        ("dct_low_energy_ratio_to_target", "≤1 Hz 慢动作能量 / GT"),
        ("dct_mid_energy_ratio_to_target", "1.2–2 Hz 步态频段能量 / GT"),
    ):
        specs.append((label, [metric_block(runs, key, "artic", name)[0] for key in (BASE, LATEST)]))
    specs.append(("腿摆幅 / GT", [metric_block(runs, key, "natural", "leg_swing_amp_ratio_to_gt")[0] for key in (BASE, LATEST)]))
    specs.append(("步数 / GT", [metric_block(runs, key, "natural", "step_count")[0] / natural_gt["step_count"] for key in (BASE, LATEST)]))
    specs.append(("迈步相位相关 第 1–10 帧（GT=1）", [metric_block(runs, key, "natural", "gait_sep_corr_f01_10")[0] for key in (BASE, LATEST)]))
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    positions = np.arange(len(specs))[::-1]
    gap = 0.2
    base_values = [100 * item[1][0] for item in specs]
    latest_values = [100 * item[1][1] for item in specs]
    ax.barh(positions + gap, base_values, height=0.36, color=C_BASE, label="独立单人 baseline")
    ax.barh(positions - gap, latest_values, height=0.36, color=C_LATEST, label="最新主线")
    for pos, b, l in zip(positions, base_values, latest_values):
        ax.text(max(b, 0) + 1.5, pos + gap, "{:.0f}%".format(b), va="center", fontsize=9, color=C_INK)
        ax.text(max(l, 0) + 1.5, pos - gap, "{:.0f}%".format(l), va="center", fontsize=9, color=C_INK)
    ax.axvline(100, color=C_GT, lw=1.2, ls="--")
    ax.text(100.8, positions.max() + 0.62, "GT = 100%", fontsize=9, color=C_INK)
    ax.set_yticks(positions)
    ax.set_yticklabels([item[0] for item in specs])
    ax.set_xlim(0, 125)
    ax.set_xlabel("占 GT 的百分比（越接近 100% 越像真实动作；copy-last 全为 0）")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("动作摆动与步态：不再整体平移、四肢不动", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--arrays", required=True, help="最新主线 seed 0 的 --export_arrays 输出")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_style()

    runs = load_runs(args.eval_dir)
    summary = OrderedDict([("milestones", OrderedDict()), ("seed_files", OrderedDict())])
    for key, _, prefix in MILESTONES:
        if prefix:
            summary["seed_files"][key] = runs[key + "_files"]
    summary["reference_consistency_max_abs_diff"] = check_reference_consistency(runs)

    sample, horizon, per_action = array_analysis(args.arrays, runs, summary)
    tables = OrderedDict(
        [
            ("main", table_main(runs, summary)),
            ("decomposition", table_group(runs, DECOMP_KEYS, "metrics", 4)),
            ("relation", table_group(runs, RELATION_KEYS, "metrics", 4)),
            ("articulation", table_articulation(runs)),
            ("natural", table_natural(runs)),
            ("bootstrap", table_bootstrap(summary)),
            ("horizon", table_horizon(summary)),
            ("action", table_action(per_action)),
        ]
    )
    with open(os.path.join(args.output_dir, "tables.md"), "w") as handle:
        for name, text in tables.items():
            handle.write("## {}\n\n{}\n\n".format(name, text))
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    fig_milestones(runs, summary, os.path.join(args.output_dir, "fig1_milestones_mpjpe.png"))
    fig_horizon(horizon, os.path.join(args.output_dir, "fig2_horizon.png"))
    fig_action(per_action, os.path.join(args.output_dir, "fig3_per_action.png"))
    fig_scatter(sample, summary, os.path.join(args.output_dir, "fig4_per_sample_scatter.png"))
    fig_motion(runs, os.path.join(args.output_dir, "fig5_motion_fidelity.png"))
    print(json.dumps({"array_vs_json_mpjpe_drift": summary["array_vs_json_mpjpe_drift"], "reference_consistency_max_abs_diff": summary["reference_consistency_max_abs_diff"]}))


if __name__ == "__main__":
    main()
