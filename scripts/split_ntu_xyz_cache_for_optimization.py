import argparse
import json
import os
import random
from collections import OrderedDict
from datetime import datetime

import torch


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=False, ensure_ascii=False)


def _label_counts(actions, num_actions):
    counts = [0 for _ in range(int(num_actions))]
    for item in actions.view(-1).tolist():
        counts[int(item)] += 1
    return counts


def _validate_cache(payload):
    required = ("obs_xyz", "target_xyz", "actions", "meta")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("xyz cache 缺少字段: {}".format(missing))
    count = int(payload["obs_xyz"].shape[0])
    if int(payload["target_xyz"].shape[0]) != count:
        raise ValueError("target_xyz 样本数不一致")
    if int(payload["actions"].shape[0]) != count:
        raise ValueError("actions 样本数不一致")
    if len(payload["meta"]) != count:
        raise ValueError("meta 样本数不一致")
    return count


def _stratified_indices(actions, val_ratio, seed, num_actions):
    rng = random.Random(int(seed))
    labels = actions.view(-1).tolist()
    by_label = OrderedDict((idx, []) for idx in range(int(num_actions)))
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)

    train_indices = []
    val_indices = []
    per_label = []
    for label, indices in by_label.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        count = len(shuffled)
        if count <= 1:
            val_count = 0
        else:
            val_count = int(round(float(count) * float(val_ratio)))
            if val_count == 0:
                val_count = 1
            val_count = min(val_count, count - 1)
        val_part = sorted(shuffled[:val_count])
        train_part = sorted(shuffled[val_count:])
        train_indices.extend(train_part)
        val_indices.extend(val_part)
        per_label.append(
            OrderedDict(
                [
                    ("label", int(label)),
                    ("full_count", int(count)),
                    ("train_count", int(len(train_part))),
                    ("val_count", int(len(val_part))),
                ]
            )
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices, per_label


def _slice_payload(payload, indices, split_name, source_cache, source_config):
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    config = dict(source_config)
    config.update(
        {
            "source_cache": source_cache,
            "optimization_split": split_name,
            "num_samples": int(len(indices)),
            "created_at": _utc_now(),
        }
    )
    return {
        "obs_xyz": payload["obs_xyz"].index_select(0, index_tensor).contiguous(),
        "target_xyz": payload["target_xyz"].index_select(0, index_tensor).contiguous(),
        "actions": payload["actions"].index_select(0, index_tensor).contiguous(),
        "meta": [payload["meta"][idx] for idx in indices],
        "config": config,
    }


def split_cache(args):
    if not 0.0 < float(args.val_ratio) < 1.0:
        raise ValueError("--val_ratio 必须在 (0, 1) 内")
    payload = torch.load(args.input_cache, map_location="cpu")
    full_count = _validate_cache(payload)
    os.makedirs(args.output_dir, exist_ok=True)

    actions = payload["actions"].long()
    if actions.dim() == 1:
        actions = actions.view(-1, 1)
        payload["actions"] = actions
    train_indices, val_indices, per_label = _stratified_indices(
        actions=actions,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_actions=args.num_actions,
    )
    if len(train_indices) + len(val_indices) != full_count:
        raise AssertionError("split 后样本数不守恒")

    source_config = payload.get("config", {})
    train_payload = _slice_payload(payload, train_indices, "train_opt", args.input_cache, source_config)
    val_payload = _slice_payload(payload, val_indices, "val_opt", args.input_cache, source_config)
    train_path = os.path.join(args.output_dir, args.train_filename)
    val_path = os.path.join(args.output_dir, args.val_filename)
    torch.save(train_payload, train_path)
    torch.save(val_payload, val_path)

    train_counts = _label_counts(train_payload["actions"], args.num_actions)
    val_counts = _label_counts(val_payload["actions"], args.num_actions)
    summary = OrderedDict(
        [
            ("created_at", _utc_now()),
            ("input_cache", args.input_cache),
            ("output_dir", args.output_dir),
            ("train_cache", train_path),
            ("val_cache", val_path),
            ("seed", int(args.seed)),
            ("val_ratio_requested", float(args.val_ratio)),
            ("num_actions", int(args.num_actions)),
            ("full_count", int(full_count)),
            ("train_count", int(len(train_indices))),
            ("val_count", int(len(val_indices))),
            ("val_ratio_actual", float(len(val_indices)) / float(full_count)),
            ("train_label_counts", train_counts),
            ("val_label_counts", val_counts),
            ("val_missing_labels", [idx for idx, count in enumerate(val_counts) if count == 0]),
            ("train_missing_labels", [idx for idx, count in enumerate(train_counts) if count == 0]),
            ("per_label", per_label),
        ]
    )
    summary_path = os.path.join(args.output_dir, "split_summary.json")
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_cache",
        default="results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/train_xyz.pt",
    )
    parser.add_argument(
        "--output_dir",
        default="results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split",
    )
    parser.add_argument("--train_filename", default="train_opt_xyz.pt")
    parser.add_argument("--val_filename", default="val_opt_xyz.pt")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_actions", type=int, default=26)
    return parser


def main():
    split_cache(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
