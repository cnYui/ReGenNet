"""JiT 流匹配候选（J1–J4、RL 变体）与 FD2 同预算对照的训练、val 评价与预登记判定。

设计与预登记：docs/ai/context/20261018-201532-ntu2p-jit-flow-rl-design-and-plan.md 第 4–5 节。
每个流候选评价 4 个打分读出（endpoint / mean16 / median16 / medoid16，eval/eval_ntu2p_v2.py）与 16 个独立样本
（eval/eval_ntu2p_jitflow_samples.py）；参照为同 seed 的 FD2 H-val EMA 10000 终点（与上一轮相同）。
单卡最多 --workers 个训练并发；评价紧随各自训练，在同一工作线程里串行。
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("results/forecasting/ntu120_label")
SAVE = Path("save/forecasting/ntu120_label")
PYTHON = sys.executable
PROTOCOLS = {
    "subjval": dict(manifest=str(BASE / "ntu2p_subjval/manifest_subjval_seed0.json"),
                    cache=str(BASE / "ntu2p_xyz_seq_cache_subjval"), prefix="ntu2p_v2subj",
                    base=str(SAVE / "ntu2p_independent_single_person_o10_p50_subjval_s0_5000/model000005000.pt")),
    "original": dict(manifest=str(BASE / "ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"),
                     cache=str(BASE / "ntu2p_xyz_seq_cache"), prefix="ntu2p_v2",
                     base=str(SAVE / "ntu2p_independent_single_person_o10_p50_cuda_retrain_s0_5000/model000005000.pt")),
}
FD2_CONFIG = "A6-F-A5f0.05-A4-GH0.5-FD2"
# 与 scripts/run_ntu2p_v2_screen.py 的 S2_5_RECIPE + FD2 增量逐项相同（FD2 主线的 args.json 可核对）。
S2_5_RECIPE = ["--loss_scale_normalize", "--local_velocity_loss_weight", "1.0", "--ramp_mode", "saturate",
               "--ramp_saturate_frames", "5", "--dct_low_amplitude_loss_weight", "1.0", "--dct_mid_amplitude_loss_weight", "1.0",
               "--root_loss_weight", "0.1", "--inter_loss_weight", "0.01", "--ema_decay", "0.999", "--batch_size", "8",
               "--log_interval", "100"]
FD2_ADDONS = ["--canonical_ab_fallback", "camera_x", "--foot_loss_weight", "0.05", "--kin_proj", "--leg_gait_loss_weight", "0.5",
              "--dct_mid_exclude_legs", "--loss_joint_subset", "hand2"]
FLOW_CANDIDATES = OrderedDict(
    [
        ("J1", ["--denoiser", "mixer"]),
        ("J2", ["--denoiser", "mixer", "--self_cond", "--contact_graph"]),
        ("J3", ["--denoiser", "stg", "--self_cond"]),
        ("J4", ["--denoiser", "stg", "--self_cond", "--contact_graph", "--causal_cross"]),
        # 归因对照：同结构、同端点配方（含自条件两遍精炼），只去掉去噪损失；判断端点收益是否来自扩散训练。
        ("J3N", ["--denoiser", "stg", "--self_cond", "--flow_loss_weight", "0"]),
        ("J4N", ["--denoiser", "stg", "--self_cond", "--contact_graph", "--causal_cross", "--flow_loss_weight", "0"]),
    ]
)
SCORE_READOUTS = ("endpoint", "mean", "median", "medoid")
L2_KEYS = ("xyz_mse", "xyz_mae", "mpjpe")
# 预登记阈值（设计 4.3）。
L2_TOLERANCE = 1.005
MPJPE_GAIN = 0.992
JERK_RATIO, PENETRATION_DELTA = 1.2, 0.01
SAMPLE_JERK_RATIO, SAMPLE_PENETRATION_DELTA = 1.5, 0.03
DCT_HIGH_RATIO, DCT_HIGH_FLOOR = 1.2, 0.05
GAIT_CORR_DROP = 0.05
BONE_MAX = 1e-5
ANTIC_GAIN = 0.7
CONTACT_RMSE_RATIO = 1.05

_lock = threading.Lock()


def say(message):
    with _lock:
        print(message, flush=True)


def run(command, log):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH=str(ROOT))
    with open(log, "a") as handle:
        handle.write("\n命令：" + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def common_args(protocol):
    return ["--manifest_path", protocol["manifest"], "--cache_dir", protocol["cache"], "--device", "cuda:0"]


def eval_score(checkpoint, output, protocol, log, readout=None, export=None):
    command = [PYTHON, "eval/eval_ntu2p_v2.py", "--checkpoint", str(checkpoint), "--output", str(output),
               "--baseline_checkpoint", protocol["base"], "--split", "val", "--interaction_metrics"] + common_args(protocol)
    if readout is not None:
        command += ["--flow_readout", readout]
    if export:
        command += ["--export_arrays", str(export)]
    if not Path(output).exists():
        run(command, log)
    return json.loads(Path(output).read_text())


def train_flow(name, extra, run_dir, steps, seed, protocol, log):
    if (run_dir / "ema" / "model{:09d}.pt".format(steps)).exists():
        return
    command = [PYTHON, "train/train_ntu2p_jitflow.py", "--save_dir", str(run_dir), "--num_steps", str(steps), "--seed", str(seed),
               "--save_interval", str(steps), "--baseline_checkpoint", protocol["base"]] + S2_5_RECIPE + FD2_ADDONS + extra + common_args(protocol)
    run(command, log)


def train_fd2(run_dir, steps, seed, protocol, log):
    if (run_dir / "ema" / "model{:09d}.pt".format(steps)).exists():
        return
    command = [PYTHON, "train/train_ntu2p_v2.py", "--arch", "intermixer", "--leg_stream", "--save_dir", str(run_dir),
               "--num_steps", str(steps), "--seed", str(seed), "--save_interval", str(steps),
               "--baseline_checkpoint", protocol["base"]] + S2_5_RECIPE + FD2_ADDONS + common_args(protocol)
    run(command, log)


def job(opts, name, out):
    protocol = PROTOCOLS[opts.protocol]
    log = out / "{}.log".format(name)
    results = OrderedDict()
    if name == "FD2ref":
        checkpoint = SAVE / "{}_{}_s{}_10000/ema/model000010000.pt".format(protocol["prefix"], FD2_CONFIG, opts.seed)
        results["endpoint"] = eval_score(checkpoint, out / "FD2ref_endpoint.json", protocol, log, export=out / "FD2ref_arrays.pt")
        return name, results
    if name == "FD2same":
        run_dir = SAVE / "{}_{}_s{}_{}".format(protocol["prefix"], FD2_CONFIG, opts.seed, opts.steps)
        say("训练 {}".format(name))
        train_fd2(run_dir, opts.steps, opts.seed, protocol, log)
        checkpoint = run_dir / "ema/model{:09d}.pt".format(opts.steps)
        results["endpoint"] = eval_score(checkpoint, out / "FD2same_endpoint.json", protocol, log)
        return name, results
    if "-RL" in name:
        # 第 2 阶段：在第 1 阶段候选的 EMA 终点上做 GRPO（RL-A / RL-F），评价 EMA 终点。
        base_name, reward = name.split("-RL")
        init = SAVE / "ntu2p_jitflow_{}_{}{}_s{}_{}".format(opts.protocol, base_name, opts.run_suffix, opts.seed, opts.steps) / "ema/model{:09d}.pt".format(opts.steps)
        run_dir = SAVE / "ntu2p_jitflow_{}_{}{}_s{}_{}_grpo{}".format(opts.protocol, name, opts.run_suffix, opts.seed, opts.steps, opts.rl_iters)
        checkpoint = run_dir / "ema/model{:09d}.pt".format(opts.rl_iters)
        say("GRPO {}".format(name))
        if not checkpoint.exists():
            # S：同一 GRPO 循环但策略梯度倍数为 0（只剩端点配方损失的继续训练），用来区分 RL 与"多训 1500 步"。
            reward_args = ["--reward", "A", "--pg_scale", "0"] if reward == "S" else ["--reward", reward]
            run([PYTHON, "train/train_ntu2p_jitflow_grpo.py", "--init_checkpoint", str(init), "--save_dir", str(run_dir),
                 "--num_iters", str(opts.rl_iters), "--save_interval", str(opts.rl_iters), "--seed", str(opts.seed)]
                + reward_args + common_args(protocol), log)
    else:
        run_dir = SAVE / "ntu2p_jitflow_{}_{}{}_s{}_{}".format(opts.protocol, name, opts.run_suffix, opts.seed, opts.steps)
        say("训练 {}".format(name))
        train_flow(name, FLOW_CANDIDATES[name] + opts.flow_args, run_dir, opts.steps, opts.seed, protocol, log)
        checkpoint = run_dir / "ema/model{:09d}.pt".format(opts.steps)
    say("评价 {}".format(name))
    for readout in SCORE_READOUTS:
        export = out / "{}_{}_arrays.pt".format(name, readout) if readout in ("endpoint", "median") else None
        results[readout] = eval_score(checkpoint, out / "{}_{}.json".format(name, readout), protocol, log, readout=readout, export=export)
    sample_path = out / "{}_samples.json".format(name)
    if not sample_path.exists():
        run([PYTHON, "eval/eval_ntu2p_jitflow_samples.py", "--checkpoint", str(checkpoint), "--output", str(sample_path),
             "--export_arrays", str(out / "{}_samples_arrays.pt".format(name))] + common_args(protocol)[:-2], log)
    results["sample"] = json.loads(sample_path.read_text())
    say("完成 {}".format(name))
    return name, results


# ------------------------------------------------------------------ 汇总与判定


def score_row(result):
    """eval_ntu2p_v2 的输出 -> 判定所需的扁平指标。"""
    model = result["model_metrics"]
    gate = result["articulation_gate"]
    natural = result.get("naturalness", {}).get("model", {})
    timing = result["interaction_timing"]
    row = OrderedDict((key, model[key]) for key in L2_KEYS)
    row.update(
        [
            ("dct_low_ratio", gate["model_dct_low_ratio"]),
            ("dct_mid_ratio", gate["model_dct_mid_ratio"]),
            ("dct_high_ratio", gate["model_dct_high_ratio"]),
            ("jerk_body", natural.get("jerk_body")),
            ("foot_penetration_ratio", natural.get("foot_penetration_ratio")),
            ("slide_frame_ratio", natural.get("slide_frame_ratio")),
            ("step_count", natural.get("step_count")),
            ("gait_sep_corr_f01_10", natural.get("gait_sep_corr_f01_10")),
            ("bone_rel_err_body", natural.get("bone_rel_err_body")),
            ("primary_excess_antic", timing["primary_excess_antic"]["model"]),
            ("missed_contact_total", sum(timing["all_{:.2f}".format(t)]["model"]["missed_count"] for t in (0.20, 0.25, 0.30))),
            ("contact_distance_rmse_all", result["contact"]["model"]["all"]["distance_rmse"]),
            ("contact_distance_rmse_push", result["contact"]["model"]["pushing"]["distance_rmse"]),
        ]
    )
    return row


def sample_row(result):
    mean = result["mean_over_samples"]
    keys = L2_KEYS + ("dct_low_ratio", "dct_mid_ratio", "dct_high_ratio", "jerk_body", "foot_penetration_ratio", "slide_frame_ratio",
                      "step_count", "gait_sep_corr_f01_10", "bone_rel_err_body", "primary_excess_antic", "missed_contact_total",
                      "contact_distance_rmse_all", "contact_distance_rmse_push")
    row = OrderedDict((key, mean.get(key)) for key in keys)
    row["energy_score"] = result["energy_score"]
    return row


def guards(row, ref, sample=False):
    """返回未通过的护栏名列表（设计 4.3）。"""
    failed = []
    jerk_ratio, pen_delta = (SAMPLE_JERK_RATIO, SAMPLE_PENETRATION_DELTA) if sample else (JERK_RATIO, PENETRATION_DELTA)
    if row["bone_rel_err_body"] is None or row["bone_rel_err_body"] > BONE_MAX:
        failed.append("bone")
    if row["jerk_body"] is None or row["jerk_body"] > jerk_ratio * ref["jerk_body"]:
        failed.append("jerk")
    if row["foot_penetration_ratio"] is None or row["foot_penetration_ratio"] > ref["foot_penetration_ratio"] + pen_delta:
        failed.append("penetration")
    if row["dct_high_ratio"] > max(DCT_HIGH_RATIO * ref["dct_high_ratio"], DCT_HIGH_FLOOR):
        failed.append("dct_high")
    if row["dct_low_ratio"] < 0.40 or row["dct_mid_ratio"] < 0.10:
        failed.append("articulation_gate")
    if row["gait_sep_corr_f01_10"] is None or row["gait_sep_corr_f01_10"] < ref["gait_sep_corr_f01_10"] - GAIT_CORR_DROP:
        failed.append("gait_f01_10")
    return failed


def decide(row, ref, sample=False):
    l2_ok = all(row[key] <= L2_TOLERANCE * ref[key] for key in L2_KEYS)
    score = (not sample) and l2_ok and row["mpjpe"] <= MPJPE_GAIN * ref["mpjpe"]
    interaction = (row["primary_excess_antic"] is not None and row["primary_excess_antic"] <= ANTIC_GAIN * ref["primary_excess_antic"]
                   and row["missed_contact_total"] <= ref["missed_contact_total"]
                   and row["contact_distance_rmse_all"] <= CONTACT_RMSE_RATIO * ref["contact_distance_rmse_all"])
    failed = guards(row, ref, sample)
    return OrderedDict([("l2_not_worse", l2_ok), ("score_adopt", score and not failed), ("interaction_adopt", interaction and not failed),
                        ("failed_guards", failed)])


def summarize(all_results, out):
    ref = score_row(all_results["FD2ref"]["endpoint"])
    rows = OrderedDict()
    for name, results in all_results.items():
        for readout, result in results.items():
            sample = readout == "sample"
            row = sample_row(result) if sample else score_row(result)
            row["delta_mpjpe_pct"] = 100.0 * (row["mpjpe"] / ref["mpjpe"] - 1.0)
            row["decision"] = decide(row, ref, sample)
            rows["{}:{}".format(name, readout)] = row
    (out / "summary.json").write_text(json.dumps(OrderedDict([("reference", "FD2ref:endpoint"), ("rows", rows)]), ensure_ascii=False, indent=2))
    header = ["行", "xyz_mse", "xyz_mae", "mpjpe", "Δmpjpe%", "dct_mid", "jerk", "穿地", "滑行", "步数", "相位f1-10",
              "excess_antic", "漏接触", "接触RMSE", "提分", "交互", "未过护栏"]
    lines = ["# JiT 流候选 H-val 终点比较", "", "参照 FD2ref:endpoint（FD2 H-val EMA 10000 终点）；判定按设计 4.3。", "",
             "|" + "|".join(header) + "|", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]

    def fmt(value, digits=5):
        return "—" if value is None else ("{:." + str(digits) + "f}").format(value)

    for key, row in rows.items():
        decision = row["decision"]
        lines.append("|" + "|".join([
            key, fmt(row["xyz_mse"]), fmt(row["xyz_mae"]), fmt(row["mpjpe"]), "{:+.2f}".format(row["delta_mpjpe_pct"]),
            fmt(row["dct_mid_ratio"], 3), fmt(row["jerk_body"], 2), fmt(row["foot_penetration_ratio"], 4), fmt(row["slide_frame_ratio"], 3),
            fmt(row["step_count"], 3), fmt(row["gait_sep_corr_f01_10"], 3), fmt(row["primary_excess_antic"], 4),
            str(row["missed_contact_total"]), fmt(row["contact_distance_rmse_all"], 4),
            "是" if decision["score_adopt"] else "否", "是" if decision["interaction_adopt"] else "否",
            ",".join(decision["failed_guards"]) or "—"]) + "|")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="subjval")
    parser.add_argument("--candidates", nargs="+", default=["FD2ref", "FD2same"] + list(FLOW_CANDIDATES))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tag", default="screen5000")
    parser.add_argument("--rl_iters", type=int, default=1500, help="第 2 阶段 GRPO 迭代数（候选名形如 J2-RLF）")
    parser.add_argument("--run_suffix", default="", help="流候选 run 名后缀（如 -tmix）")
    parser.add_argument("--flow_args", default="", help="追加给全部流候选训练的参数，一个字符串（如 --flow_args='--t_uniform_mix 0.5'）")
    opts = parser.parse_args()
    opts.flow_args = opts.flow_args.split()
    out = BASE / "ntu2p_jitflow" / opts.protocol / opts.tag
    out.mkdir(parents=True, exist_ok=True)
    results = OrderedDict()
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futures = [pool.submit(job, opts, name, out) for name in opts.candidates]
        for future in as_completed(futures):
            name, value = future.result()
            results[name] = value
    ordered = OrderedDict((name, results[name]) for name in opts.candidates if name in results)
    summarize(ordered, out)


if __name__ == "__main__":
    main()
