import argparse
import json
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime


LOSS_KEYS = (
    "mae_loss_weight",
    "root_loss_weight",
    "local_pose_loss_weight",
    "mpjpe_loss_weight",
    "short_loss_weight",
    "mid_loss_weight",
    "long_loss_weight",
    "final_frame_loss_weight",
    "velocity_loss_weight",
    "acceleration_loss_weight",
    "continuity_loss_weight",
    "first_step_loss_weight",
    "relative_root_loss_weight",
    "relative_velocity_loss_weight",
    "key_joint_relation_loss_weight",
    "contact_loss_weight",
    "contact_threshold",
    "action_feature_loss_weight",
    "action_logit_loss_weight",
)


BASE_CONFIG = OrderedDict(
    [
        ("mae_loss_weight", 0.0),
        ("root_loss_weight", 0.0),
        ("local_pose_loss_weight", 0.0),
        ("mpjpe_loss_weight", 0.0),
        ("short_loss_weight", 0.0),
        ("mid_loss_weight", 0.0),
        ("long_loss_weight", 0.0),
        ("final_frame_loss_weight", 0.0),
        ("velocity_loss_weight", 0.2),
        ("acceleration_loss_weight", 0.0),
        ("continuity_loss_weight", 0.0),
        ("first_step_loss_weight", 0.0),
        ("relative_root_loss_weight", 0.0),
        ("relative_velocity_loss_weight", 0.0),
        ("key_joint_relation_loss_weight", 0.0),
        ("contact_loss_weight", 0.0),
        ("contact_threshold", 0.15),
        ("action_feature_loss_weight", 0.0),
        ("action_logit_loss_weight", 0.0),
    ]
)


ACCEPTED_D_CONFIG = OrderedDict(
    [
        ("mae_loss_weight", 0.05),
        ("root_loss_weight", 0.25),
        ("local_pose_loss_weight", 0.25),
        ("mpjpe_loss_weight", 0.0),
        ("short_loss_weight", 0.0),
        ("mid_loss_weight", 0.0),
        ("long_loss_weight", 0.05),
        ("final_frame_loss_weight", 0.05),
        ("velocity_loss_weight", 0.2),
        ("acceleration_loss_weight", 0.025),
        ("continuity_loss_weight", 0.0),
        ("first_step_loss_weight", 0.0),
        ("relative_root_loss_weight", 0.025),
        ("relative_velocity_loss_weight", 0.025),
        ("key_joint_relation_loss_weight", 0.05),
        ("contact_loss_weight", 0.05),
        ("contact_threshold", 0.15),
        ("action_feature_loss_weight", 0.02),
        ("action_logit_loss_weight", 0.0),
    ]
)


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=False, ensure_ascii=False)


def _append_jsonl(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(value, sort_keys=False, ensure_ascii=False))
        f.write("\n")


def _parse_ints(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _candidate(config_id, stage, updates):
    config = OrderedDict(BASE_CONFIG)
    config.update(updates)
    return OrderedDict([("config_id", config_id), ("stage", stage), ("loss_weights", config)])


def _built_in_candidates(stage):
    if stage == "baseline":
        return [_candidate("baseline_v0", "baseline", {})]
    if stage == "stageA":
        return [
            _candidate("stageA_a1_root_local_010", "stageA", {"root_loss_weight": 0.1, "local_pose_loss_weight": 0.1}),
            _candidate("stageA_a1_root_local_025", "stageA", {"root_loss_weight": 0.25, "local_pose_loss_weight": 0.25}),
            _candidate("stageA_a1_root_local_050", "stageA", {"root_loss_weight": 0.5, "local_pose_loss_weight": 0.5}),
            _candidate("stageA_a2_long_final_005", "stageA", {"long_loss_weight": 0.05, "final_frame_loss_weight": 0.05}),
            _candidate(
                "stageA_compact_v1",
                "stageA",
                {
                    "mae_loss_weight": 0.05,
                    "root_loss_weight": 0.25,
                    "local_pose_loss_weight": 0.25,
                    "long_loss_weight": 0.05,
                    "final_frame_loss_weight": 0.05,
                    "acceleration_loss_weight": 0.025,
                    "relative_root_loss_weight": 0.025,
                    "relative_velocity_loss_weight": 0.025,
                },
            ),
        ]
    if stage == "stageA_long_final_grid":
        candidates = []
        for long_weight in (0.025, 0.05, 0.075, 0.1):
            for final_weight in (0.025, 0.05, 0.075, 0.1):
                candidates.append(
                    _candidate(
                        "stageA_lf_l{:03d}_f{:03d}".format(
                            int(long_weight * 1000), int(final_weight * 1000)
                        ),
                        "stageA",
                        {
                            "long_loss_weight": long_weight,
                            "final_frame_loss_weight": final_weight,
                        },
                    )
                )
        return candidates
    if stage == "stageB":
        base = OrderedDict(ACCEPTED_D_CONFIG)
        base["action_feature_loss_weight"] = 0.0
        base["action_logit_loss_weight"] = 0.0
        candidates = []
        for key_weight in (0.025, 0.05, 0.1):
            for contact_weight in (0.025, 0.05, 0.1):
                updates = OrderedDict(base)
                updates["key_joint_relation_loss_weight"] = key_weight
                updates["contact_loss_weight"] = contact_weight
                candidates.append(
                    _candidate(
                        "stageB_key{:03d}_contact{:03d}".format(int(key_weight * 1000), int(contact_weight * 1000)),
                        "stageB",
                        updates,
                    )
                )
        return candidates
    if stage == "stageB_from_stageA":
        stage_a_bases = [
            (
                "long_final_005",
                {"long_loss_weight": 0.05, "final_frame_loss_weight": 0.05},
            ),
            (
                "compact_v1",
                {
                    "mae_loss_weight": 0.05,
                    "root_loss_weight": 0.25,
                    "local_pose_loss_weight": 0.25,
                    "long_loss_weight": 0.05,
                    "final_frame_loss_weight": 0.05,
                    "acceleration_loss_weight": 0.025,
                    "relative_root_loss_weight": 0.025,
                    "relative_velocity_loss_weight": 0.025,
                },
            ),
        ]
        key_contact_pairs = ((0.025, 0.025), (0.05, 0.05), (0.05, 0.1), (0.1, 0.05))
        candidates = []
        for base_name, base_updates in stage_a_bases:
            for key_weight, contact_weight in key_contact_pairs:
                updates = OrderedDict(base_updates)
                updates["key_joint_relation_loss_weight"] = key_weight
                updates["contact_loss_weight"] = contact_weight
                candidates.append(
                    _candidate(
                        "stageB_{}_key{:03d}_contact{:03d}".format(
                            base_name, int(key_weight * 1000), int(contact_weight * 1000)
                        ),
                        "stageB",
                        updates,
                    )
                )
        return candidates
    if stage == "stageB_lite_from_stageA_best":
        candidates = []
        for key_weight, contact_weight in ((0.0, 0.01), (0.01, 0.0), (0.01, 0.01), (0.025, 0.01), (0.01, 0.025)):
            candidates.append(
                _candidate(
                    "stageB_lite_l050_f075_key{:03d}_contact{:03d}".format(
                        int(key_weight * 1000), int(contact_weight * 1000)
                    ),
                    "stageB",
                    {
                        "long_loss_weight": 0.05,
                        "final_frame_loss_weight": 0.075,
                        "key_joint_relation_loss_weight": key_weight,
                        "contact_loss_weight": contact_weight,
                    },
                )
            )
        return candidates
    if stage == "stageD":
        candidates = []
        for weight in (0.0, 0.005, 0.01, 0.02, 0.05):
            updates = OrderedDict(ACCEPTED_D_CONFIG)
            updates["action_feature_loss_weight"] = weight
            candidates.append(
                _candidate("stageD_feature_{:03d}".format(int(weight * 1000)), "stageD", updates)
            )
        return candidates
    if stage == "stageD_from_stageA":
        candidates = []
        for weight in (0.0, 0.005, 0.01, 0.02, 0.05):
            updates = {
                "long_loss_weight": 0.05,
                "final_frame_loss_weight": 0.05,
                "action_feature_loss_weight": weight,
                "action_logit_loss_weight": 0.0,
            }
            candidates.append(
                _candidate("stageD_from_A_feature_{:03d}".format(int(weight * 1000)), "stageD", updates)
            )
        return candidates
    if stage == "acceptedD":
        return [_candidate("acceptedD_current", "acceptedD", ACCEPTED_D_CONFIG)]
    raise ValueError("未知 stage: {}".format(stage))


def _load_candidates(args):
    if args.config_jsonl is None:
        candidates = _built_in_candidates(args.stage)
    else:
        candidates = []
        with open(args.config_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line, object_pairs_hook=OrderedDict))
    if args.candidate_ids:
        keep = set(item.strip() for item in args.candidate_ids.split(",") if item.strip())
        candidates = [item for item in candidates if item["config_id"] in keep]
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]
    if not candidates:
        raise ValueError("没有可运行候选")
    return candidates


def _run_command(command, log_path, dry_run):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if dry_run:
        _write_json(log_path + ".json", {"command": command})
        return
    with open(log_path, "w") as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError("命令失败，查看日志: {}".format(log_path))


def _train_command(args, candidate, seed, save_dir):
    command = [
        sys.executable,
        "-m",
        "train.train_ntu_label_xyz",
        "--train_xyz_cache",
        args.train_cache,
        "--eval_xyz_cache",
        args.val_cache,
        "--eval_split",
        "val",
        "--save_dir",
        save_dir,
        "--num_steps",
        str(args.num_steps),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--eval_interval",
        str(args.eval_interval),
        "--save_interval",
        str(args.save_interval),
        "--log_interval",
        str(args.log_interval),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--seed",
        str(seed),
        "--num_workers",
        str(args.num_workers),
        "--overwrite",
    ]
    if args.resume_checkpoint:
        command.extend(["--resume_checkpoint", args.resume_checkpoint])
    loss_weights = candidate["loss_weights"]
    for key in LOSS_KEYS:
        command.extend(["--" + key, str(loss_weights.get(key, BASE_CONFIG[key]))])
    uses_action = float(loss_weights.get("action_feature_loss_weight", 0.0)) > 0.0 or float(
        loss_weights.get("action_logit_loss_weight", 0.0)
    ) > 0.0
    if uses_action:
        if args.action_classifier_path is None:
            raise ValueError("{} 启用 action loss，但未提供 --action_classifier_path".format(candidate["config_id"]))
        command.extend(["--action_classifier_path", args.action_classifier_path])
    return command


def _best_checkpoint_from_log(save_dir):
    log_path = os.path.join(save_dir, "train_log.jsonl")
    best = None
    with open(log_path, "r") as f:
        for line in f:
            record = json.loads(line)
            checkpoint = record.get("checkpoint")
            metric = record.get("test_xyz_mse")
            if checkpoint is None or metric is None:
                continue
            if best is None or float(metric) < float(best["metric"]):
                best = {"checkpoint": checkpoint, "metric": float(metric), "step": int(record["step"])}
    if best is None:
        raise ValueError("训练日志没有可选 checkpoint: {}".format(log_path))
    with open(os.path.join(save_dir, "best_checkpoint.txt"), "w") as f:
        f.write(best["checkpoint"])
        f.write("\n")
    _write_json(os.path.join(save_dir, "best_checkpoint.json"), best)
    return best


def _eval_command(args, checkpoint, save_dir, candidate):
    command = [
        sys.executable,
        "-m",
        "eval.eval_ntu_label_xyz",
        "--mode",
        "checkpoint",
        "--xyz_cache",
        args.val_cache,
        "--split",
        "val",
        "--checkpoint",
        checkpoint,
        "--save_dir",
        save_dir,
        "--batch_size",
        str(args.eval_batch_size),
        "--num_workers",
        str(args.num_workers),
    ]
    loss_weights = candidate["loss_weights"]
    if args.semantic_eval:
        if args.action_classifier_path is None:
            raise ValueError("--semantic_eval 必须提供 --action_classifier_path")
        command.extend(["--semantic_eval", "--action_classifier_path", args.action_classifier_path])
    elif float(loss_weights.get("action_feature_loss_weight", 0.0)) > 0.0 and args.action_classifier_path is not None:
        command.extend(["--semantic_eval", "--action_classifier_path", args.action_classifier_path])
    return command


def _score_command(args, metrics_path, baseline_metrics, output_path):
    return [
        sys.executable,
        "-m",
        "scripts.score_ntu_xyz_candidate",
        "--candidate_metrics",
        metrics_path,
        "--baseline_metrics",
        baseline_metrics,
        "--output",
        output_path,
    ]


def _leaderboard_entry(candidate, seed, save_dir, metrics_path, score_path, best):
    metrics = _read_json(metrics_path)
    score = _read_json(score_path)
    model_metrics = metrics["model_metrics"]
    return OrderedDict(
        [
            ("created_at", _utc_now()),
            ("config_id", candidate["config_id"]),
            ("stage", candidate["stage"]),
            ("seed", int(seed)),
            ("save_dir", save_dir),
            ("best_checkpoint", best["checkpoint"]),
            ("best_step", int(best["step"])),
            ("metrics_path", metrics_path),
            ("score_path", score_path),
            ("hard_gate_pass", bool(score["hard_gate_pass"])),
            ("selection_score", score["selection_score"]),
            ("geometry_score", score["geometry_score"]),
            ("xyz_mse", model_metrics["xyz_mse"]),
            ("xyz_mae", model_metrics["xyz_mae"]),
            ("mpjpe", model_metrics["mpjpe"]),
            ("final_frame_error", model_metrics["final_frame_error"]),
            ("long_xyz_mse", model_metrics["long_xyz_mse"]),
            ("relative_root_distance_error", model_metrics["relative_root_distance_error"]),
            ("contact_error", model_metrics["contact_error"]),
            ("loss_weights", candidate["loss_weights"]),
        ]
    )


def run_search(args):
    candidates = _load_candidates(args)
    seeds = _parse_ints(args.seeds)
    os.makedirs(args.result_root, exist_ok=True)
    os.makedirs(args.save_root, exist_ok=True)
    _write_json(os.path.join(args.result_root, "run_args.json"), vars(args))
    for candidate in candidates:
        _append_jsonl(os.path.join(args.result_root, "candidates.jsonl"), candidate)

    baseline_metrics = args.baseline_metrics
    leaderboard_path = os.path.join(args.result_root, "leaderboard.json")
    if os.path.exists(leaderboard_path) and not args.reset_leaderboard:
        leaderboard = _read_json(leaderboard_path)
    else:
        leaderboard = []
    for candidate in candidates:
        for seed in seeds:
            run_id = "{}_s{}".format(candidate["config_id"], seed)
            save_dir = os.path.join(args.save_root, run_id)
            result_dir = os.path.join(args.result_root, candidate["stage"], run_id)
            os.makedirs(result_dir, exist_ok=True)
            _write_json(os.path.join(result_dir, "candidate_config.json"), candidate)

            train_command = _train_command(args, candidate, seed, save_dir)
            _write_json(os.path.join(result_dir, "train_command.json"), {"command": train_command})
            print("train {}".format(run_id))
            _run_command(train_command, os.path.join(result_dir, "train.log"), args.dry_run)
            if args.dry_run:
                continue

            best = _best_checkpoint_from_log(save_dir)
            eval_command = _eval_command(args, best["checkpoint"], result_dir, candidate)
            _write_json(os.path.join(result_dir, "eval_command.json"), {"command": eval_command})
            print("eval {}".format(run_id))
            _run_command(eval_command, os.path.join(result_dir, "eval.log"), args.dry_run)

            metrics_path = os.path.join(result_dir, "metrics_val.json")
            if baseline_metrics is None:
                baseline_metrics = metrics_path
            score_path = os.path.join(result_dir, "candidate_score.json")
            score_command = _score_command(args, metrics_path, baseline_metrics, score_path)
            _write_json(os.path.join(result_dir, "score_command.json"), {"command": score_command})
            print("score {}".format(run_id))
            _run_command(score_command, os.path.join(result_dir, "score.log"), args.dry_run)
            entry = _leaderboard_entry(candidate, seed, save_dir, metrics_path, score_path, best)
            leaderboard.append(entry)
            _write_json(leaderboard_path, leaderboard)
            _append_jsonl(os.path.join(args.result_root, "leaderboard.jsonl"), entry)
    if leaderboard:
        sorted_board = sorted(
            leaderboard,
            key=lambda item: (
                item["selection_score"] is None,
                float("inf") if item["selection_score"] is None else float(item["selection_score"]),
            ),
        )
        _write_json(os.path.join(args.result_root, "leaderboard_sorted.json"), sorted_board)
        print(json.dumps(sorted_board[: min(5, len(sorted_board))], indent=2, sort_keys=False, ensure_ascii=False))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="baseline",
        choices=(
            "baseline",
            "stageA",
            "stageA_long_final_grid",
            "stageB",
            "stageB_from_stageA",
            "stageB_lite_from_stageA_best",
            "stageD",
            "stageD_from_stageA",
            "acceptedD",
        ),
    )
    parser.add_argument("--config_jsonl", default=None)
    parser.add_argument("--candidate_ids", default=None)
    parser.add_argument("--max_candidates", type=int, default=-1)
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache", required=True)
    parser.add_argument("--baseline_metrics", default=None)
    parser.add_argument("--save_root", default="save/forecasting/ntu120_label/xyz_loss_optimal")
    parser.add_argument("--result_root", default="results/forecasting/ntu120_label/xyz_loss_optimal")
    parser.add_argument("--action_classifier_path", default=None)
    parser.add_argument("--semantic_eval", action="store_true")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--reset_leaderboard", action="store_true")
    return parser


def main():
    run_search(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
