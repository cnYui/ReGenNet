import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _read_entries(path):
    if path.endswith(".jsonl"):
        entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
    return _read_json(path)


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=False, ensure_ascii=False)


def _mean(values):
    return sum(values) / float(len(values)) if values else None


def _std(values):
    if len(values) <= 1:
        return 0.0 if values else None
    mean = _mean(values)
    return (sum((item - mean) ** 2 for item in values) / float(len(values) - 1)) ** 0.5


def _group(entries):
    groups = OrderedDict()
    for entry in entries:
        groups.setdefault(entry["config_id"], []).append(entry)
    return groups


def summarize(args):
    entries = _read_entries(args.leaderboard)
    groups = _group(entries)
    rows = []
    for config_id, items in groups.items():
        scores = [float(item["selection_score"]) for item in items if item["selection_score"] is not None]
        rows.append(
            OrderedDict(
                [
                    ("config_id", config_id),
                    ("stage", items[0]["stage"]),
                    ("runs", len(items)),
                    ("hard_gate_pass_runs", sum(1 for item in items if item["hard_gate_pass"])),
                    ("selection_score_mean", _mean(scores)),
                    ("selection_score_std", _std(scores)),
                    ("xyz_mse_mean", _mean([float(item["xyz_mse"]) for item in items])),
                    ("mpjpe_mean", _mean([float(item["mpjpe"]) for item in items])),
                    ("contact_error_mean", _mean([float(item["contact_error"]) for item in items])),
                    ("best_checkpoint", min(items, key=lambda item: item["selection_score"] or float("inf"))["best_checkpoint"]),
                ]
            )
        )
    rows = sorted(rows, key=lambda row: row["selection_score_mean"] if row["selection_score_mean"] is not None else float("inf"))
    summary = OrderedDict([("created_at", _utc_now()), ("leaderboard", args.leaderboard), ("rows", rows)])
    _write_json(args.output_json, summary)

    lines = ["# NTU xyz optimal search leaderboard", ""]
    lines.append("| rank | config | stage | runs | gate | score_mean | score_std | xyz_mse | mpjpe | contact |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                idx,
                row["config_id"],
                row["stage"],
                row["runs"],
                row["hard_gate_pass_runs"],
                "{:.6f}".format(row["selection_score_mean"]) if row["selection_score_mean"] is not None else "NA",
                "{:.6f}".format(row["selection_score_std"]) if row["selection_score_std"] is not None else "NA",
                "{:.9f}".format(row["xyz_mse_mean"]),
                "{:.9f}".format(row["mpjpe_mean"]),
                "{:.9f}".format(row["contact_error_mean"]),
            )
        )
    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print("\n".join(lines))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboard",
        default="results/forecasting/ntu120_label/xyz_loss_optimal/leaderboard.jsonl",
    )
    parser.add_argument(
        "--output_json",
        default="results/forecasting/ntu120_label/xyz_loss_optimal/leaderboard_summary.json",
    )
    parser.add_argument(
        "--output_md",
        default="results/forecasting/ntu120_label/xyz_loss_optimal/leaderboard.md",
    )
    return parser


def main():
    summarize(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
