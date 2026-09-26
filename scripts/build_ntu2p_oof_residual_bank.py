"""Track B 残差库：协议 train 每条序列的全部 stride-1 窗口，用"没见过该受试者"的折模型前向得到 OOF 残差系数。

- OOF（主臂）：窗口所在受试者属于第 f 折时用折模型 F_{-f}；断言该折模型 checkpoint 记录的 manifest 与折 manifest
  hash 相同、且该受试者不在其 train 中。这样残差与部署时（底座没见过 val/test 受试者）同分布。
- in-sample（诊断对照 4.9）：同一个底座（通常是 v2subj s0）在它自己的 train 窗口上前向；断言其 manifest 即协议 manifest。
跳变窗口照常入库、只打标记（clean=False）：生成器训练与统计量只用干净窗口，评估与诊断仍可看全体。
设计：docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md 第 4.3–4.4 节。
"""

import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch

from data_loaders.forecasting.ntu2p_residual_bank import BANK_FORMAT_VERSION, NTU2PResidualBank, compute_bank_stats
from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache
from data_loaders.forecasting.ntu_2p_diffusion import load_ntu_2p_diffusion_manifest
from model.forecasting_ntu2p_v2 import forward_details, load_ntu2p_model_checkpoint
from scripts.build_ntu2p_subject_holdout_manifest import performer
from train.train_ntu2p_v2 import _device
from utils.ntu2p_probabilistic_metrics import window_mpjpe
from utils.ntu2p_residual_codec import GLITCH_STEP_M, OBS_LEN, PRED_LEN, ResidualCodec, codec_frames, condition_features, glitch_mask


WINDOW_LEN = OBS_LEN + PRED_LEN


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _load_folds(folds_dir):
    with open(os.path.join(folds_dir, "folds.json")) as handle:
        return json.load(handle)


def check_oof(checkpoint, fold, folds, skip=False):
    """折模型 checkpoint 必须来自折 manifest，且该折受试者不在其 train 中；返回记录的 manifest 路径。"""
    state = torch.load(checkpoint, map_location="cpu")
    manifest_path = state.get("manifest_path")
    if skip:
        return manifest_path
    record = folds["folds"][str(fold)]
    manifest = load_ntu_2p_diffusion_manifest(manifest_path)
    if manifest["manifest_hash"] != record["manifest_hash"]:
        raise AssertionError(
            "折 {} 模型 {} 的 manifest hash {} 与折 manifest {} 不一致（不是该折的 OOF 模型）".format(
                fold, checkpoint, manifest["manifest_hash"], record["manifest_hash"]
            )
        )
    seen = {performer(item["sample_id"]) for item in manifest["splits"]["train"]}
    leaked = sorted(seen & set(record["performers"]))
    if leaked:
        raise AssertionError("折 {} 模型的 train 含该折受试者 {}".format(fold, leaked))
    return manifest_path


def check_insample(checkpoint, parent_hash, skip=False):
    state = torch.load(checkpoint, map_location="cpu")
    manifest_path = state.get("manifest_path")
    if not skip and load_ntu_2p_diffusion_manifest(manifest_path)["manifest_hash"] != parent_hash:
        raise AssertionError("in-sample 底座 {} 不是在协议 train 上训练的".format(checkpoint))
    return manifest_path


def _window_index(entries, assignment, max_sequences):
    """全部 stride-1 窗口：(seq_index, start, fold, performer, action)，按序列、起点顺序。"""
    rows = []
    count = len(entries) if max_sequences is None or max_sequences <= 0 else min(int(max_sequences), len(entries))
    for seq_index in range(count):
        item = entries[seq_index]
        key = performer(item["sample_id"])
        fold = int(assignment[str(key)]) if assignment is not None else -1
        for start in range(int(item["length"]) - WINDOW_LEN + 1):
            rows.append((seq_index, start, fold, key, int(item["action"])))
    columns = list(zip(*rows))
    return OrderedDict((name, torch.tensor(columns[i], dtype=torch.long)) for i, name in enumerate(("seq_index", "start", "fold", "performer", "action"))), count


def build(args):
    with torch.no_grad():
        return _build(args)


def _build(args):
    if args.skip_oof_assert_for_smoke and not args.allow_cpu_for_smoke_test:
        raise ValueError("--skip_oof_assert_for_smoke 只允许与 --allow_cpu_for_smoke_test 同时使用")
    if bool(args.fold_checkpoints) == bool(args.insample_checkpoint):
        raise ValueError("--fold_checkpoints 与 --insample_checkpoint 必须二选一")
    device = _device(args)
    parent = load_ntu_2p_diffusion_manifest(args.parent_manifest)
    entries = parent["splits"]["train"]
    cache = NTU2PXYZSeqCache.load(args.parent_cache_dir, "train", manifest_path=args.parent_manifest, device=device)
    folds = _load_folds(args.folds_dir)
    if folds["source_manifest_hash"] != parent["manifest_hash"]:
        raise ValueError("folds.json 的源 manifest 与 --parent_manifest 不一致")
    mode = "oof" if args.fold_checkpoints else "insample"
    if mode == "oof":
        if len(args.fold_checkpoints) != int(folds["num_folds"]):
            raise ValueError("--fold_checkpoints 数量必须等于折数 {}".format(folds["num_folds"]))
        recorded = [check_oof(path, fold, folds, args.skip_oof_assert_for_smoke) for fold, path in enumerate(args.fold_checkpoints)]
        checkpoints = list(args.fold_checkpoints)
        assignment = folds["assignment"]
    else:
        recorded = [check_insample(args.insample_checkpoint, parent["manifest_hash"], args.skip_oof_assert_for_smoke)]
        checkpoints = [args.insample_checkpoint]
        assignment = None
    models = [load_ntu2p_model_checkpoint(path, device)[0].eval() for path in checkpoints]

    index, num_sequences = _window_index(entries, assignment, args.max_sequences)
    total = int(index["seq_index"].numel())
    expected = sum(int(item["length"]) - WINDOW_LEN + 1 for item in entries[:num_sequences])
    if total != expected:
        raise AssertionError("窗口总数 {} 不等于 Σ(len−59)={}".format(total, expected))
    codec = ResidualCodec().to(device)
    fields = OrderedDict()
    fields["target"] = torch.zeros(total, 2, codec.num_coeffs, 66)
    fields["draft"] = torch.zeros(total, 2, codec.num_coeffs, 66)
    fields["obs_feats"] = torch.zeros(total, 2, OBS_LEN, 66)
    fields["rel_geom"] = torch.zeros(total, 2, 6)
    fields["v2_walk"] = torch.zeros(total, 2, dtype=torch.bool)
    fields["glitch"] = torch.zeros(total, dtype=torch.bool)
    fields["base_mpjpe"] = torch.zeros(total)
    model_of = index["fold"] if mode == "oof" else torch.zeros(total, dtype=torch.long)
    for model_id, model in enumerate(models):
        ids = torch.nonzero(model_of == model_id, as_tuple=False).view(-1)
        for begin in range(0, int(ids.numel()), int(args.batch_size)):
            batch = ids[begin : begin + int(args.batch_size)]
            window = cache.gather_windows(index["seq_index"][batch], index["start"][batch], WINDOW_LEN)
            obs, target = window[:, :OBS_LEN].contiguous(), window[:, OBS_LEN:].contiguous()
            action = index["action"][batch].to(device)
            base = forward_details(model, obs, action)["pred"]
            frames = codec_frames(obs)
            feats = condition_features(codec, obs, base, frames)
            fields["target"][batch] = codec.target_coeffs(obs, base, target, frames).cpu()
            fields["draft"][batch] = feats["draft"].cpu()
            fields["obs_feats"][batch] = feats["obs_feats"].cpu()
            fields["rel_geom"][batch] = feats["rel_geom"].cpu()
            fields["v2_walk"][batch] = feats["v2_walk"].cpu()
            fields["glitch"][batch] = glitch_mask(window, args.glitch_threshold).cpu()
            fields["base_mpjpe"][batch] = window_mpjpe(base, target).float().cpu()
        print("model {}/{} windows={}".format(model_id + 1, len(models), int(ids.numel())), flush=True)
    for name, value in index.items():
        fields[name] = value
    fields["clean"] = ~fields["glitch"]
    for name in ("target", "draft", "obs_feats", "rel_geom"):
        if not bool(torch.isfinite(fields[name]).all()):
            raise ValueError("残差库字段 {} 含非有限数值".format(name))
    stats = compute_bank_stats(fields["target"], fields["draft"], fields["obs_feats"], fields["clean"])
    fold_records = folds["folds"]
    config = OrderedDict(
        [
            ("format_version", BANK_FORMAT_VERSION),
            ("protocol", args.protocol or folds.get("protocol_tag")),
            ("mode", mode),
            ("parent_manifest", args.parent_manifest),
            ("parent_manifest_hash", parent["manifest_hash"]),
            ("parent_cache_dir", args.parent_cache_dir),
            ("folds_dir", args.folds_dir),
            ("fold_manifests", [fold_records[str(f)]["manifest_path"] for f in range(int(folds["num_folds"]))]),
            ("fold_manifest_hashes", [fold_records[str(f)]["manifest_hash"] for f in range(int(folds["num_folds"]))]),
            ("base_checkpoints", checkpoints),
            ("base_checkpoint_manifests", recorded),
            ("codec_config", codec.config()),
            ("glitch_threshold", float(args.glitch_threshold)),
            ("num_sequences", int(num_sequences)),
            ("max_sequences", int(args.max_sequences or 0)),
            ("smoke_skip_oof_assert", bool(args.skip_oof_assert_for_smoke)),
            ("device", str(device)),
            ("created_at", _utc_now()),
        ]
    )
    bank = NTU2PResidualBank(fields, stats, config, [str(item["sample_id"]) for item in entries], device="cpu")
    bank.save(args.output)
    summary = bank.summary()
    summary["output"] = args.output
    print(json.dumps(summary, ensure_ascii=False))
    return bank


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_manifest", required=True)
    parser.add_argument("--parent_cache_dir", required=True)
    parser.add_argument("--folds_dir", required=True)
    parser.add_argument("--fold_checkpoints", nargs="*", default=None, help="OOF：按折顺序给出 3 个折模型 EMA 终点")
    parser.add_argument("--insample_checkpoint", default=None, help="in-sample 对照：在协议 train 上训练的底座")
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default=None, help="缺省取 folds.json 的 protocol_tag")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--glitch_threshold", type=float, default=GLITCH_STEP_M)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow_cpu_for_smoke_test", action="store_true", help="仅冒烟测试：允许 --device cpu")
    parser.add_argument("--max_sequences", type=int, default=0, help="仅冒烟测试：只处理前 N 条序列")
    parser.add_argument("--skip_oof_assert_for_smoke", action="store_true", help="仅冒烟测试：跳过 OOF/in-sample 断言")
    return parser


if __name__ == "__main__":
    build(build_arg_parser().parse_args())
