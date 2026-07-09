import argparse
import json
import math
import os
from collections import OrderedDict
from datetime import datetime


GEOMETRY_WEIGHTS = OrderedDict(
    [
        ("xyz_mse", 0.35),
        ("mpjpe", 0.25),
        ("final_frame_error", 0.15),
        ("long_xyz_mse", 0.10),
        ("relative_root_distance_error", 0.10),
        ("contact_error", 0.05),
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


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_ratio(value, ref):
    if not _finite(value) or not _finite(ref) or abs(float(ref)) < 1e-12:
        return None
    return float(value) / float(ref)


def _weighted_score(ratios):
    total = 0.0
    used = 0.0
    for key, weight in GEOMETRY_WEIGHTS.items():
        ratio = ratios.get(key)
        if ratio is None:
            continue
        total += float(weight) * float(ratio)
        used += float(weight)
    if used <= 0.0:
        return None
    return total / used


def _semantic_score(candidate, baseline):
    candidate_semantic = candidate.get("semantic_metrics")
    baseline_semantic = baseline.get("semantic_metrics")
    if candidate_semantic is None or baseline_semantic is None:
        return None
    candidate_fid = candidate_semantic.get("fid", {}).get("model_vs_real")
    baseline_fid = baseline_semantic.get("fid", {}).get("model_vs_real")
    candidate_top1 = candidate_semantic.get("model_future", {}).get("top1_acc")
    baseline_top1 = baseline_semantic.get("model_future", {}).get("top1_acc")
    fid_ratio = _safe_ratio(candidate_fid, baseline_fid)
    if fid_ratio is None or not _finite(candidate_top1) or not _finite(baseline_top1):
        return None
    action_top1_gain = float(candidate_top1) - float(baseline_top1)
    return OrderedDict(
        [
            ("fid_ratio", fid_ratio),
            ("action_top1_gain", action_top1_gain),
            ("semantic_score", fid_ratio - action_top1_gain),
        ]
    )


def score_candidate(args):
    candidate = _read_json(args.candidate_metrics)
    baseline = _read_json(args.baseline_metrics)
    candidate_model = candidate["model_metrics"]
    baseline_model = baseline["model_metrics"]
    ratios = OrderedDict()
    for key in GEOMETRY_WEIGHTS.keys():
        ratios[key] = _safe_ratio(candidate_model.get(key), baseline_model.get(key))
    geometry_score = _weighted_score(ratios)

    beats = candidate.get("beats_copy_last", {})
    hard_gate = OrderedDict(
        [
            ("finite_metrics", all(_finite(value) for value in candidate_model.values())),
            ("beats_copy_last_xyz_mse", bool(beats.get("xyz_mse"))),
            ("beats_copy_last_xyz_mae", bool(beats.get("xyz_mae"))),
            ("beats_copy_last_mpjpe", bool(beats.get("mpjpe"))),
            (
                "first_step_error_ok",
                _finite(candidate_model.get("first_step_error"))
                and float(candidate_model.get("first_step_error")) <= float(args.first_step_tolerance),
            ),
        ]
    )
    hard_gate_pass = all(hard_gate.values())
    summary = OrderedDict(
        [
            ("created_at", _utc_now()),
            ("candidate_metrics", args.candidate_metrics),
            ("baseline_metrics", args.baseline_metrics),
            ("geometry_weights", GEOMETRY_WEIGHTS),
            ("metric_ratios", ratios),
            ("geometry_score", geometry_score),
            ("semantic_diagnostic", _semantic_score(candidate, baseline)),
            ("hard_gate", hard_gate),
            ("hard_gate_pass", hard_gate_pass),
            ("selection_score", geometry_score if hard_gate_pass else None),
        ]
    )
    _write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_metrics", required=True)
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--first_step_tolerance", type=float, default=1e-6)
    return parser


def main():
    score_candidate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
