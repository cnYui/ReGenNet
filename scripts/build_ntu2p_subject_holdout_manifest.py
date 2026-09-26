"""由现有 NTU2P manifest 构造"按受试者留出"的选型 val：train/val 受试者不重叠，test 不变。

现有 val 按序列随机抽取，受试者全部出现在 train 中，而 test 受试者与 train 不重叠，因此 val 会高估
能记住个人风格的模型（v2 相对 A0：val −7.8%，test −3.85%）。选择过程在看任何模型结果之前预登记，见
docs/ai/context/20260926-105152-ntu2p-subject-holdout-val-plan.md。
"""

import argparse
import copy
import random
import re
from collections import Counter, OrderedDict

from data_loaders.forecasting.ntu_2p_diffusion import (
    _split_summary,
    _write_json,
    assert_manifest_no_sample_id_leak,
    load_ntu_2p_diffusion_manifest,
    manifest_payload_hash,
)
from data_loaders.forecasting.ntu_label import NUM_ACTIONS


DEFAULT_SOURCE = "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"
DEFAULT_OUTPUT = "results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json"
# P008 占池 17%：留出它会让训练集骤减，且它在 test 中无对应的超大受试者。
DEFAULT_EXCLUDE = (8,)
# NTU60 部分为 setup S001–S017；test 与池子在这一比例上差异最大（0.555 vs 0.789）。
NTU60_MAX_SETUP = 17


def performer(sample_id):
    return int(re.search(r"P(\d{3})", sample_id).group(1))


def setup(sample_id):
    return int(re.search(r"S(\d{3})", sample_id).group(1))


def _distribution(entries):
    counts = Counter(int(item["action"]) for item in entries)
    total = float(max(len(entries), 1))
    return [counts[action] / total for action in range(NUM_ACTIONS)]


def _ntu60_share(entries):
    return sum(setup(item["sample_id"]) <= NTU60_MAX_SETUP for item in entries) / float(max(len(entries), 1))


def score_heldout(held, pool_action_counts, test_share, test_dist):
    """预登记评分：越低越像 test；池中常见（≥5 条）却在留出集中缺失的动作额外罚分。"""
    held_counts = Counter(int(item["action"]) for item in held)
    missing = sum(1 for action, count in pool_action_counts.items() if count >= 5 and held_counts[action] == 0)
    dist = _distribution(held)
    l1 = sum(abs(a - b) for a, b in zip(dist, test_dist))
    return abs(_ntu60_share(held) - test_share) + 0.5 * l1 + 0.05 * missing


def search_heldout(pool, test, exclude, min_frac, max_frac, trials, seed):
    by_performer = OrderedDict()
    for item in pool:
        by_performer.setdefault(performer(item["sample_id"]), []).append(item)
    candidates = sorted(key for key in by_performer if key not in set(exclude))
    low, high = min_frac * len(pool), max_frac * len(pool)
    pool_action_counts = Counter(int(item["action"]) for item in pool)
    test_share, test_dist = _ntu60_share(test), _distribution(test)
    rng = random.Random(int(seed))
    best = None
    for _ in range(int(trials)):
        order = list(candidates)
        rng.shuffle(order)
        chosen, size = [], 0
        for key in order:
            if size >= low:
                break
            if size + len(by_performer[key]) > high:
                continue
            chosen.append(key)
            size += len(by_performer[key])
        if not low <= size <= high:
            continue
        held = [item for key in chosen for item in by_performer[key]]
        score = score_heldout(held, pool_action_counts, test_share, test_dist)
        if best is None or score < best[0]:
            best = (score, sorted(chosen))
    if best is None:
        raise ValueError("未找到落入数量区间的受试者组合")
    return best


def build(args):
    source = load_ntu_2p_diffusion_manifest(args.source_manifest)
    pool = source["splits"]["train"] + source["splits"]["val"]
    test = source["splits"]["test"]
    score, heldout = search_heldout(pool, test, args.exclude, args.min_frac, args.max_frac, args.trials, args.seed)
    held_set = set(heldout)

    def relabel(items, split):
        result = []
        for item in sorted(items, key=lambda entry: entry["sample_id"]):
            entry = OrderedDict(item)
            entry["split"] = split
            result.append(entry)
        return result

    manifest = OrderedDict()
    for key in ("protocol", "source_paths", "source_scan"):
        manifest[key] = copy.deepcopy(source[key])
    manifest["split_config"] = OrderedDict(
        [
            ("seed", int(args.seed)),
            ("split_unit", "performer"),
            ("heldout_performers", heldout),
            ("excluded_from_heldout", sorted(int(value) for value in args.exclude)),
            ("heldout_frac_range", [float(args.min_frac), float(args.max_frac)]),
            ("search_trials", int(args.trials)),
            ("heldout_score", float(score)),
            ("source_manifest", str(args.source_manifest)),
            ("source_manifest_hash", source["manifest_hash"]),
            ("train_val_source", "xsub.train"),
            ("test_source", "xsub.test"),
        ]
    )
    manifest["splits"] = OrderedDict(
        [
            ("train", relabel([item for item in pool if performer(item["sample_id"]) not in held_set], "train")),
            ("val", relabel([item for item in pool if performer(item["sample_id"]) in held_set], "val")),
            ("test", relabel(test, "test")),
        ]
    )
    manifest["split_summary"] = OrderedDict(
        (split, _split_summary(manifest["splits"][split], NUM_ACTIONS)) for split in ("train", "val", "test")
    )
    manifest["sample_id_overlaps"] = assert_manifest_no_sample_id_leak(manifest)
    performers = {split: {performer(item["sample_id"]) for item in manifest["splits"][split]} for split in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = performers[left] & performers[right]
        if common:
            raise AssertionError("{}/{} 受试者重叠: {}".format(left, right, sorted(common)))
    manifest["manifest_path"] = str(args.output)
    manifest["manifest_hash"] = manifest_payload_hash(manifest)
    _write_json(args.output, manifest)
    summary = OrderedDict(
        [
            ("heldout_performers", heldout),
            ("score", round(score, 4)),
            ("sizes", {split: len(manifest["splits"][split]) for split in ("train", "val", "test")}),
            ("ntu60_share", {split: round(_ntu60_share(manifest["splits"][split]), 3) for split in ("train", "val", "test")}),
        ]
    )
    print(summary)
    return manifest


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude", nargs="*", type=int, default=list(DEFAULT_EXCLUDE))
    parser.add_argument("--min_frac", type=float, default=0.12)
    parser.add_argument("--max_frac", type=float, default=0.16)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    build(build_arg_parser().parse_args())
