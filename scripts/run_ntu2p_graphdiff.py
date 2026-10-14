"""同预算候选训练与终点评价；单卡并发可控，test 须有冻结的 H-val 采纳记录。"""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("results/forecasting/ntu120_label")
SAVE = Path("save/forecasting/ntu120_label")
PROTOCOLS = {
    "subjval": dict(manifest=str(BASE / "ntu2p_subjval/manifest_subjval_seed0.json"),
                    cache=str(BASE / "ntu2p_xyz_seq_cache_subjval"),
                    prior_prefix="ntu2p_v2subj", base="ntu2p_independent_single_person_o10_p50_subjval_s0_5000"),
    "original": dict(manifest=str(BASE / "ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"),
                     cache=str(BASE / "ntu2p_xyz_seq_cache"),
                     prior_prefix="ntu2p_v2", base="ntu2p_independent_single_person_o10_p50_cuda_retrain_s0_5000"),
}
PRIOR_CONFIG = "A6-F-A5f0.05-A4-GH0.5-FD2"
CANDIDATES = ("C1", "C2", "C3", "C4", "F1", "F2", "F3")
FINE_TUNE_CANDIDATES = frozenset(("F1", "F2", "F3"))


def run_command(command, log_path):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH=str(ROOT))
    with open(log_path, "a") as handle:
        handle.write("\n命令：" + repr(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def run_one(args, candidate, seed, out):
    protocol = PROTOCOLS[args.protocol]
    run_dir = SAVE / "ntu2p_graphdiff_{}_{}_s{}_{}".format(args.protocol, candidate, seed, args.steps)
    checkpoint = run_dir / "ema/model{:09d}.pt".format(args.steps)
    log = out / "{}_s{}.log".format(candidate, seed)
    prior = SAVE / "{}_{}_s{}_10000/ema/model000010000.pt".format(protocol["prior_prefix"], PRIOR_CONFIG, seed)
    result_path = out / "{}_s{}_{}.json".format(candidate, seed, args.split)
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if (cached.get("checkpoint") != str(checkpoint) or cached.get("checkpoint_step") != args.steps
                or cached.get("split") != args.split or cached.get("manifest_path") != protocol["manifest"]):
            raise ValueError("已有评估与请求的checkpoint/步数/split/manifest不一致，必须新建tag")
        return cached
    print("开始 {} seed{} {} {}steps".format(candidate, seed, args.protocol, args.steps), flush=True)
    common = ["--manifest_path", protocol["manifest"], "--cache_dir", protocol["cache"], "--device", "cuda:0"]
    if not checkpoint.exists():
        train = [sys.executable, "train/train_ntu2p_graphdiff.py", "--candidate", candidate,
                 "--prior_checkpoint", str(prior), "--save_dir", str(run_dir), "--num_steps", str(args.steps),
                 "--seed", str(seed), "--save_interval", str(args.steps),
                 "--baseline_checkpoint", str(SAVE / protocol["base"] / "model000005000.pt")] + common
        if candidate in FINE_TUNE_CANDIDATES:
            train += ["--lr", "0.0003", "--batch_size", "8"]
        run_command(train, log)
    saved_args = json.loads((run_dir / "args.json").read_text())
    expected = dict(candidate=candidate, seed=seed, num_steps=args.steps,
                    manifest_path=protocol["manifest"], cache_dir=protocol["cache"], prior_checkpoint=str(prior))
    if any(saved_args.get(key) != value for key, value in expected.items()):
        raise ValueError("已有训练与请求配置不一致，不能复用checkpoint")
    if candidate in FINE_TUNE_CANDIDATES:
        groups = {group["name"]: group["lr"] for group in saved_args.get("optimizer_groups", [])}
        if (not saved_args.get("fine_tune_prior") or not saved_args.get("model_config", {}).get("fine_tune_prior")
                or saved_args.get("batch_size") != 8 or saved_args.get("lr") != 3e-4
                or set(groups) != {"corrector", "prior"} or groups["corrector"] != 3e-4
                or abs(groups["prior"] - 3e-5) > 1e-12):
            raise ValueError("端到端F候选的底座微调与双学习率记录不符合预登记协议")
    evaluate = [sys.executable, "eval/eval_ntu2p_v2.py", "--checkpoint", str(checkpoint),
                "--output", str(result_path), "--baseline_checkpoint", str(SAVE / protocol["base"] / "model000005000.pt"),
                "--split", args.split, "--export_arrays", str(out / "{}_s{}_{}_arrays.pt".format(candidate, seed, args.split))] + common
    run_command(evaluate, log)
    print("完成 {} seed{}".format(candidate, seed), flush=True)
    return json.loads(result_path.read_text())


def _mean(values):
    return statistics.mean(values)


def _contact_distance(result, name):
    return result["contact"][name]["pushing"]["distance_rmse"]


def summarize(results, output, args):
    grouped = {}
    for candidate, seed, result in results:
        grouped.setdefault(candidate, []).append((seed, result))
    summary = dict(protocol=args.protocol, steps=args.steps, split=args.split, seed_count=len(args.seeds),
                   metric_scope="单次确定性端点；DDIM 单独评价", candidates={})
    lines = ["# 接触图、相位与频域扩散：终点比较", "", "协议 {}，{} step，{}，EMA 终点。".format(args.protocol, args.steps, args.split),
             "", "|候选|mpjpe 均值±标准差|Δmpjpe|xyz_mse|xyz_mae|滑行帧比|步数|推人距离RMSE|判断|", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for candidate, group in sorted(grouped.items()):
        means = {key: _mean([r["model_metrics"][key] for _, r in group]) for key in ("mpjpe", "xyz_mse", "xyz_mae")}
        prior = {key: _mean([r["extra_variant_metrics"]["prior"]["model_metrics"][key] for _, r in group]) for key in means}
        delta = {key: means[key] / prior[key] - 1.0 for key in means}
        natural = {key: _mean([r["naturalness"]["model"][key] for _, r in group]) for key in
                   ("slide_frame_ratio", "step_count", "jerk_body", "bone_rel_err_body", "foot_penetration_ratio",
                    "gait_sep_corr_f01_10", "gait_sep_corr_f01_10_moving", "gait_sep_corr_f01_10_starting")}
        ref_natural = {key: _mean([r["naturalness"]["prior"][key] for _, r in group]) for key in natural}
        contact = _mean([_contact_distance(r, "model") for _, r in group])
        ref_contact = _mean([_contact_distance(r, "prior") for _, r in group])
        safety = all(r["articulation_gate"]["passes_full_gate"] and r["model_metrics"]["first_step_error"] <= 1e-6
                     and r["naturalness"]["model"]["bone_rel_err_body"] <= 1e-5
                     and r["naturalness"]["model"]["jerk_body"] <= 1.2 * r["naturalness"]["prior"]["jerk_body"]
                     and r["naturalness"]["model"]["foot_penetration_ratio"] <= r["naturalness"]["prior"]["foot_penetration_ratio"] + 0.01
                     and r["articulation_gate"]["model_dct_high_ratio"] <= max(1.2 * r["extra_variant_metrics"]["prior"]["articulation_metrics"]["dct_high_energy_ratio_to_target"], 0.05)
                     for _, r in group)
        phase_ok = all(natural["gait_sep_corr_f01_10" + suffix] >= ref_natural["gait_sep_corr_f01_10" + suffix] - 0.05
                       for suffix in ("", "_moving", "_starting"))
        l2_ok = all(value <= 0.005 for value in delta.values())
        gt_steps = _mean([r["naturalness"]["GT"]["step_count"] for _, r in group])
        step_gain = abs(natural["step_count"] - gt_steps) <= 0.9 * abs(ref_natural["step_count"] - gt_steps)
        targeted = (natural["slide_frame_ratio"] <= 0.9 * ref_natural["slide_frame_ratio"] or step_gain or contact <= 0.9 * ref_contact)
        potential = safety and phase_ok and l2_ok and (delta["mpjpe"] <= -0.008 or targeted)
        verdict = "候选通过，待三seed与视觉审查" if potential else "不采纳"
        std = statistics.stdev([r["model_metrics"]["mpjpe"] for _, r in group]) if len(group) > 1 else 0.0
        summary["candidates"][candidate] = dict(metrics=means, mpjpe_std=std, prior_metrics=prior,
                                              delta=delta, naturalness=natural, prior_naturalness=ref_natural,
                                              contact_rmse=contact, prior_contact_rmse=ref_contact,
                                              safety=safety, phase_ok=phase_ok, potential=potential,
                                              verdict=verdict, seeds=[seed for seed, _ in group])
        lines.append("|{}|{:.6f} ± {:.6f}|{:+.2f}%|{:.6f}|{:.6f}|{:.3f}|{:.2f}|{:.3f}|{}|".format(
            candidate, means["mpjpe"], std, 100 * delta["mpjpe"], means["xyz_mse"], means["xyz_mae"],
            natural["slide_frame_ratio"], natural["step_count"], contact, verdict))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(args):
    if args.split == "test":
        if args.protocol != "original" or not args.selection_path:
            raise ValueError("test 必须使用原协议及已冻结的 H-val 采纳记录")
        selection = json.loads(Path(args.selection_path).read_text())
        if (selection.get("protocol") != "subjval" or selection.get("split") != "val"
                or selection.get("steps") != 10000 or sorted(selection.get("seeds", [])) != [0, 1, 2]
                or not selection.get("visual_review_pass")
                or not set(args.candidates).issubset(selection.get("adopted", []))):
            raise ValueError("候选未通过冻结选型/视觉审查，不能评估test")
        if args.steps != 10000 or sorted(args.seeds) != [0, 1, 2]:
            raise ValueError("正式test固定10000step与3seed终点")
    output = BASE / "ntu2p_graphdiff" / args.protocol / args.tag
    output.mkdir(parents=True, exist_ok=True)
    request = dict(protocol=args.protocol, candidates=sorted(args.candidates), seeds=sorted(args.seeds),
                   steps=args.steps, split=args.split)
    if any(candidate in FINE_TUNE_CANDIDATES for candidate in args.candidates):
        request["fine_tune_training"] = dict(candidates=sorted(set(args.candidates) & FINE_TUNE_CANDIDATES),
                                              corrector_lr=3e-4, prior_lr=3e-5, batch_size=8,
                                              reference="冻结原FD2，随checkpoint保存")
    request_path = output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("tag已有其他实验请求，不能重标历史结果")
    if not request_path.exists():
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(run_one, args, c, seed, output): (c, seed) for c in args.candidates for seed in args.seeds}
        for future in as_completed(jobs):
            candidate, seed = jobs[future]
            results.append((candidate, seed, future.result()))
    summarize(results, output, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="subjval")
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=["C1", "C2", "C3", "C4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tag", default="screen5000")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--selection_path")
    main(parser.parse_args())
