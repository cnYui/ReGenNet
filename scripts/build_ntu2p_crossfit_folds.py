"""Track B 交叉拟合：把协议 train 按受试者分成 3 个不相交的折，生成折 manifest 与按序列切片的 xyz 缓存。

第 f 折的 train = 其余两折、val = 第 f 折、test 原样复制；折模型只在自己的 val（未见过的受试者）上前向，
得到的残差（OOF）才与部署时"底座没见过这些受试者"的情形一致（in-sample 残差按部位收缩 0.61–0.81）。
缓存直接从协议 train 缓存按序列切片，不重算 FK；只生成 train 与 val：折模型不评估 test。
设计：docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md 第 4.1 节。
"""

import argparse
import copy
import os
from collections import Counter, OrderedDict
from datetime import datetime

import torch

from data_loaders.forecasting.ntu_2p_diffusion import (
    _split_summary,
    _write_json,
    assert_manifest_no_sample_id_leak,
    load_ntu_2p_diffusion_manifest,
    manifest_payload_hash,
)
from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, cache_file_path
from data_loaders.forecasting.ntu_label import NUM_ACTIONS
from scripts.build_ntu2p_subject_holdout_manifest import performer


WINDOW_LEN = 60
FOLD_NOTE = "折模型只评估 val；test 缓存未生成"


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def assign_folds(entries, num_folds=3):
    """受试者按序列数降序（同数量按 id 升序）依次放入当前最轻的折（同重量取编号小的折）-> {performer: fold}。"""
    counts = Counter(performer(item["sample_id"]) for item in entries)
    loads = [0] * int(num_folds)
    assignment = OrderedDict()
    for key in sorted(counts, key=lambda value: (-counts[value], value)):
        fold = min(range(int(num_folds)), key=lambda index: (loads[index], index))
        assignment[key] = fold
        loads[fold] += counts[key]
    return assignment


def fold_manifest_path(output_dir, fold):
    return os.path.join(str(output_dir), "manifest_fold{}.json".format(int(fold)))


def fold_cache_dir(output_dir, fold):
    return os.path.join(str(output_dir), "cache_fold{}".format(int(fold)))


def _relabel(items, split):
    result = []
    for item in items:
        entry = OrderedDict(item)
        entry["split"] = split
        result.append(entry)
    return result


def build_fold_manifest(source, source_path, assignment, fold, num_folds, output_path):
    train_source = source["splits"]["train"]
    fold_performers = OrderedDict(
        (str(index), sorted(key for key, value in assignment.items() if value == index)) for index in range(int(num_folds))
    )
    manifest = OrderedDict()
    for key in ("protocol", "source_paths", "source_scan"):
        if key in source:
            manifest[key] = copy.deepcopy(source[key])
    manifest["split_config"] = OrderedDict(
        [
            ("split_unit", "performer_fold"),
            ("fold", int(fold)),
            ("num_folds", int(num_folds)),
            ("fold_performers", fold_performers),
            ("assignment_rule", "受试者按序列数降序（同数量按 id 升序）贪心放入当前最轻的折（同重量取编号小的折）"),
            ("source_manifest", str(source_path)),
            ("source_manifest_hash", source["manifest_hash"]),
            ("note", FOLD_NOTE),
        ]
    )
    # 保持源 manifest 中的顺序：切片缓存按同一顺序拼接，缓存与 manifest 的逐条核对才成立。
    manifest["splits"] = OrderedDict(
        [
            ("train", _relabel([item for item in train_source if assignment[performer(item["sample_id"])] != fold], "train")),
            ("val", _relabel([item for item in train_source if assignment[performer(item["sample_id"])] == fold], "val")),
            ("test", copy.deepcopy(source["splits"]["test"])),
        ]
    )
    manifest["split_summary"] = OrderedDict(
        (split, _split_summary(manifest["splits"][split], NUM_ACTIONS)) for split in ("train", "val", "test")
    )
    manifest["sample_id_overlaps"] = assert_manifest_no_sample_id_leak(manifest)
    performers = {split: {performer(item["sample_id"]) for item in manifest["splits"][split]} for split in ("train", "val")}
    if performers["train"] & performers["val"]:
        raise AssertionError("折 {} 的 train/val 受试者重叠: {}".format(fold, sorted(performers["train"] & performers["val"])))
    manifest["manifest_path"] = str(output_path)
    manifest["manifest_hash"] = manifest_payload_hash(manifest)
    _write_json(output_path, manifest)
    return manifest


def slice_cache(source_payload, positions, split, manifest, manifest_path, source_cache_path):
    """按源 train 缓存中的序列位置切片，offsets 重算；键与源缓存相同。"""
    offsets, lengths = source_payload["offsets"], source_payload["lengths"]
    chunks = [source_payload["xyz"][int(offsets[p]) : int(offsets[p]) + int(lengths[p])] for p in positions]
    xyz = torch.cat(chunks, dim=0).contiguous()
    new_lengths = lengths[positions].clone()
    new_offsets = torch.zeros_like(new_lengths)
    if int(new_lengths.numel()) > 1:
        new_offsets[1:] = torch.cumsum(new_lengths, dim=0)[:-1]
    config = OrderedDict(source_payload["config"])
    config.update(
        [
            ("split", split),
            ("manifest_path", str(manifest_path)),
            ("manifest_hash", manifest["manifest_hash"]),
            ("num_sequences", len(positions)),
            ("num_frames", int(xyz.shape[0])),
            ("derived_from", OrderedDict([("cache", str(source_cache_path)), ("source_manifest_hash", source_payload["config"].get("manifest_hash"))])),
            ("created_at", _utc_now()),
        ]
    )
    return OrderedDict(
        [
            ("xyz", xyz),
            ("offsets", new_offsets),
            ("lengths", new_lengths),
            ("actions", source_payload["actions"][positions].clone()),
            ("sample_ids", [source_payload["sample_ids"][p] for p in positions]),
            ("action_codes", [source_payload["action_codes"][p] for p in positions]),
            ("config", config),
        ]
    )


def build(args):
    source = load_ntu_2p_diffusion_manifest(args.source_manifest)
    train = source["splits"]["train"]
    assignment = assign_folds(train, args.num_folds)
    source_cache_path = cache_file_path(args.source_cache_dir, "train")
    source_payload = None if args.skip_cache else torch.load(source_cache_path, map_location="cpu")
    if source_payload is not None and list(source_payload["sample_ids"]) != [str(item["sample_id"]) for item in train]:
        raise ValueError("源 train 缓存与源 manifest 的 train 顺序不一致")
    position = {str(item["sample_id"]): index for index, item in enumerate(train)}
    os.makedirs(args.output_dir, exist_ok=True)
    folds = OrderedDict()
    all_val = []
    for fold in range(int(args.num_folds)):
        path = fold_manifest_path(args.output_dir, fold)
        manifest = build_fold_manifest(source, args.source_manifest, assignment, fold, args.num_folds, path)
        # 读回校验 hash：训练与评估入口都经这个函数加载 manifest。
        load_ntu_2p_diffusion_manifest(path)
        record = OrderedDict(
            [
                ("fold", fold),
                ("manifest_path", path),
                ("manifest_hash", manifest["manifest_hash"]),
                ("cache_dir", fold_cache_dir(args.output_dir, fold)),
                ("performers", sorted(key for key, value in assignment.items() if value == fold)),
                ("num_val_sequences", len(manifest["splits"]["val"])),
                ("num_train_sequences", len(manifest["splits"]["train"])),
                ("num_val_windows", sum(int(item["length"]) - WINDOW_LEN + 1 for item in manifest["splits"]["val"])),
            ]
        )
        all_val.extend(str(item["sample_id"]) for item in manifest["splits"]["val"])
        if source_payload is not None:
            for split in ("train", "val"):
                positions = [position[str(item["sample_id"])] for item in manifest["splits"][split]]
                payload = slice_cache(source_payload, positions, split, manifest, path, source_cache_path)
                target = cache_file_path(record["cache_dir"], split)
                os.makedirs(record["cache_dir"], exist_ok=True)
                torch.save(payload, target)
                _write_json(target + ".json", payload["config"])
                # 读回校验：缓存与折 manifest 的 sample_id/length/action 逐条一致。
                NTU2PXYZSeqCache.load(record["cache_dir"], split, manifest_path=path)
        folds[str(fold)] = record
    performer_sets = [set(record["performers"]) for record in folds.values()]
    for left in range(len(performer_sets)):
        for right in range(left + 1, len(performer_sets)):
            if performer_sets[left] & performer_sets[right]:
                raise AssertionError("折 {} 与折 {} 的受试者重叠".format(left, right))
    if set().union(*performer_sets) != {performer(item["sample_id"]) for item in train}:
        raise AssertionError("三折受试者的并集不等于源 train 的受试者")
    if sorted(all_val) != sorted(str(item["sample_id"]) for item in train):
        raise AssertionError("三折 val 的并集不等于源 train")
    summary = OrderedDict(
        [
            ("protocol_tag", args.protocol_tag),
            ("source_manifest", args.source_manifest),
            ("source_manifest_hash", source["manifest_hash"]),
            ("source_cache_dir", args.source_cache_dir),
            ("num_folds", int(args.num_folds)),
            ("assignment", OrderedDict((str(key), value) for key, value in assignment.items())),
            ("folds", folds),
            ("note", FOLD_NOTE),
            ("created_at", _utc_now()),
        ]
    )
    _write_json(os.path.join(args.output_dir, "folds.json"), summary)
    print({key: (value["num_val_sequences"], value["num_val_windows"]) for key, value in folds.items()})
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--source_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_folds", type=int, default=3)
    parser.add_argument("--protocol_tag", choices=("subjval", "original"), required=True)
    parser.add_argument("--skip_cache", action="store_true", help="只生成折 manifest 与 folds.json（核对折大小用）")
    return parser


if __name__ == "__main__":
    build(build_arg_parser().parse_args())
