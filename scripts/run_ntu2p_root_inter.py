"""NTU2P residual refiner：root 轨迹 DCT 头与 inter 权重重测（Stage A）、条件触发的组合（Stage B）与选中配置的 test。

对应计划：docs/ai/context/20260925-110531-ntu2p-root-dct-head-inter-weight-plan.md
全部配置为 s2_5 + EMA、10000 step × seed 0/1/2，报告 EMA 终点；对照为已有 const-EMA run，不重训。可断点续跑。
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
from scripts.run_ntu2p_lr_ema import EMA, SEEDS, STEPS, _stats

SUMMARY_DIR = "results/forecasting/ntu120_label/ntu2p_root_inter"
CONTROL = "const-EMA"
ROOT = ["--root_head_mode", "dct", "--root_dct_k", "5"]
# 与最低 mpjpe 相差不到该值的 inter 权重视为并列，取较小者（对训练目标改动更小）。
TIE = 0.0005


def _inter(weight):
    # 覆盖公共参数里的 --inter_loss_weight 0.01：argparse 取最后一次出现的值，以 run 目录 args.json 为准。
    return ["--inter_loss_weight", str(weight)]


INTER_WEIGHTS = OrderedDict([("inter10", 10.0), ("inter3", 3.0), ("inter1", 1.0)])
# 顺序即 Stage A 的执行顺序（按信息量）：root 头优先，inter 从最强剂量开始。
STAGE_A = OrderedDict([("rootdct", ROOT)] + [(name, _inter(weight)) for name, weight in INTER_WEIGHTS.items()])
EXTRA_KEYS = (
    ("root", "root_translation_error"),
    ("local", "local_pose_error"),
    ("relroot", "relative_root_distance_error"),
    ("keyjoint", "key_joint_relation_error"),
    ("contact", "contact_error"),
)


def _combo_name(inter):
    return "rootdct_" + inter


def _combo_args(inter):
    return ROOT + _inter(INTER_WEIGHTS[inter])


def _variant_dir(name, seed):
    prefix = "ema" if name == CONTROL else name
    return os.path.join(_save_dir(CONFIG, seed, STEPS, prefix), "ema")


def _run_config(name, extra_args, dry_run):
    for seed in SEEDS:
        _log("=== {}_{}_s{}_{} ===".format(name, CONFIG, seed, STEPS))
        save_dir = _train(CONFIG, seed, STEPS, dry_run, prefix=name, extra_args=EMA + extra_args)
        # 每个 run 训完立即评估 raw 与 EMA，便于中途查看；汇总只读已有 JSON。
        _evaluate(save_dir, dry_run)
        _evaluate(os.path.join(save_dir, "ema"), dry_run)


def _collect(names, dry_run):
    curves = OrderedDict()
    for name in names:
        per_seed = OrderedDict((seed, OrderedDict(_evaluate(_variant_dir(name, seed), dry_run))) for seed in SEEDS)
        # 只汇总三个 seed 都跑完的配置；_stats 需要终点前 2000 step 的点。
        if all(step in steps for steps in per_seed.values() for step in (STEPS - 2000, STEPS - 1000, STEPS)):
            curves[name] = per_seed
    return curves


def _extra_means(per_seed):
    ends = [per_seed[seed][STEPS]["model_metrics"] for seed in SEEDS]
    return OrderedDict((short, statistics.mean(end[key] for end in ends)) for short, key in EXTRA_KEYS)


def _adopt(s, control):
    return (
        s["L_mean"] <= control["L_mean"] - control["L_std"]
        and s["xyz_mse"] <= control["xyz_mse"]
        and s["xyz_mae"] <= control["xyz_mae"]
        and s["gate"] == len(SEEDS)
    )


def _decide(stats):
    control = stats[CONTROL]
    adopted = [name for name in STAGE_A if name in stats and _adopt(stats[name], control)]
    root = "rootdct" if "rootdct" in adopted else None
    inters = [name for name in adopted if name in INTER_WEIGHTS]
    inter = None
    if inters:
        best = min(stats[name]["L_mean"] for name in inters)
        inter = min((name for name in inters if stats[name]["L_mean"] <= best + TIE), key=lambda name: INTER_WEIGHTS[name])
    if root and inter:
        combo = _combo_name(inter)
        if combo not in stats:
            return adopted, root, inter, "pending-stage-b"
        better_single = min((root, inter), key=lambda name: stats[name]["L_mean"])
        chosen = combo if _adopt(stats[combo], control) and stats[combo]["L_mean"] <= stats[better_single]["L_mean"] else better_single
        return adopted, root, inter, chosen
    return adopted, root, inter, root or inter or CONTROL


def _write_decision(stats, extras, decision):
    adopted, root, inter, chosen = decision
    control = stats[CONTROL]
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    with open(os.path.join(SUMMARY_DIR, "decision.json"), "w") as handle:
        json.dump({"stats": stats, "extras": extras, "adopted": adopted, "root": root, "inter": inter, "chosen": chosen},
                  handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    header = ["配置", "mpjpe", "Δ vs 对照", "逐 seed", "xyz_mse", "xyz_mae", "gate", "dct_mid", "frozen", "J"]
    header += [short for short, _ in EXTRA_KEYS] + ["采纳条件"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, s in stats.items():
        cells = [
            name,
            "{:.5f} ± {:.5f}".format(s["L_mean"], s["L_std"]),
            "—" if name == CONTROL else "{:+.5f}".format(s["L_mean"] - control["L_mean"]),
            " / ".join("{:.5f}".format(v) for v in s["L_per_seed"]),
            "{:.5f}".format(s["xyz_mse"]),
            "{:.5f}".format(s["xyz_mae"]),
            "{}/{}".format(s["gate"], len(SEEDS)),
            "{:.3f}".format(s["dct_mid"]),
            "{:.3f}".format(s["frozen"]),
            "{:.5f}".format(s["J"]),
        ]
        cells += ["{:.4f}".format(extras[name][short]) for short, _ in EXTRA_KEYS]
        cells.append("对照" if name == CONTROL else ("✓" if name in adopted else "✗"))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "采纳条件：mpjpe ≤ 对照 − {:.5f}（1 个对照 seed 标准差），xyz_mse / xyz_mae 不高于对照，gate {}/{}。".format(control["L_std"], len(SEEDS), len(SEEDS)),
        "root：{}；inter：{}；最终选中：**{}**".format(root or "未采纳", inter or "未采纳", chosen),
    ]
    with open(os.path.join(SUMMARY_DIR, "decision.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _summarize(dry_run):
    names = [CONTROL] + list(STAGE_A) + [_combo_name(inter) for inter in INTER_WEIGHTS]
    curves = _collect(names, dry_run)
    if dry_run or CONTROL not in curves:
        _log("skip summary (dry run or control missing)")
        return None
    _write_summary([_row("{}_s{}".format(name, seed), step, r) for name, per_seed in curves.items()
                    for seed, steps in per_seed.items() for step, r in steps.items()], SUMMARY_DIR)
    stats = _stats(curves)
    extras = OrderedDict((name, _extra_means(per_seed)) for name, per_seed in curves.items())
    decision = _decide(stats)
    _write_decision(stats, extras, decision)
    _log("summary done; adopted={} chosen={} ; see {}".format(decision[0], decision[3], SUMMARY_DIR))
    return decision


def _run_test(name):
    tables = OrderedDict()
    for label in (name, CONTROL):
        results = [(seed, _evaluate_test(os.path.join(_variant_dir(label, seed), "model{:09d}.pt".format(STEPS)), False)) for seed in SEEDS]
        tables["test (1253) {}".format(label)] = _split_rows(results, [], label=label)
    _write_final(tables, SUMMARY_DIR)
    _log("test done; table at {}/final.md".format(SUMMARY_DIR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stage_b", action="store_true", help="Stage A 判断 root 与某个 inter 权重都采纳时，训练两者的组合")
    parser.add_argument("--test_config", default=None, help="判断完成后单独调用：只评估选中配置的 test 终点")
    args = parser.parse_args()
    if args.test_config:
        _run_test(args.test_config)
        return
    if args.stage_b:
        decision = _summarize(args.dry_run)
        if decision is None or decision[3] != "pending-stage-b":
            _log("Stage B 未触发：{}".format("无判断结果" if decision is None else "root={} inter={}".format(decision[1], decision[2])))
            return
        _run_config(_combo_name(decision[2]), _combo_args(decision[2]), args.dry_run)
        _summarize(args.dry_run)
        return
    for name, extra_args in STAGE_A.items():
        _run_config(name, extra_args, args.dry_run)
    _summarize(args.dry_run)


if __name__ == "__main__":
    main()
