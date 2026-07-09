import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting import (
    NTULabelForecastDataset,
    NTULabelXYZCacheDataset,
    ntu_label_forecasting_collate,
    ntu_label_xyz_cache_collate,
)
from model.forecasting_ntu_xyz import create_ntu_label_xyz_model_from_config
from model.rotation2xyz import Rotation2xyz_x
from utils.ntu_smplx_2p_xyz import (
    NTU_XYZ_METRIC_KEYS,
    compute_ntu_xyz_metrics,
    copy_last_xyz,
    ntu_rotvec_2p_to_xyz,
)


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=False, ensure_ascii=False)


def _build_dataset(args):
    if args.xyz_cache is not None:
        return NTULabelXYZCacheDataset(args.xyz_cache, max_samples=args.max_samples)
    return NTULabelForecastDataset(
        h5_path=args.data_path,
        split=args.split,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )


def _build_loader(args, dataset):
    collate_fn = ntu_label_xyz_cache_collate if args.xyz_cache is not None else ntu_label_forecasting_collate
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )


def _load_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if "model_state_dict" not in state:
        raise ValueError("checkpoint 缺少 model_state_dict")
    if "model_config" not in state:
        raise ValueError("checkpoint 缺少 model_config")
    model = create_ntu_label_xyz_model_from_config(state["model_config"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


def _empty_sums():
    return OrderedDict((key, 0.0) for key in NTU_XYZ_METRIC_KEYS)


def _add_metrics(sums, metrics, batch_size):
    if tuple(metrics.keys()) != NTU_XYZ_METRIC_KEYS:
        raise AssertionError("metrics key 不稳定")
    for key in NTU_XYZ_METRIC_KEYS:
        sums[key] += float(metrics[key]) * float(batch_size)


def _finalize(sums, num_samples):
    return OrderedDict((key, sums[key] / float(num_samples)) for key in NTU_XYZ_METRIC_KEYS)


def _load_semantic_evaluator(args, device):
    if not bool(getattr(args, "semantic_eval", False)):
        return None
    if getattr(args, "action_classifier_path", None) is None:
        raise ValueError("--semantic_eval 必须提供 --action_classifier_path")
    from eval.action_xyz_classifier import load_xyz_action_classifier

    classifier, normalizer, state = load_xyz_action_classifier(args.action_classifier_path, device=device)
    return {
        "classifier": classifier,
        "normalizer": normalizer,
        "state": state,
        "logits": {"real": [], "model": [], "copy_last": []},
        "features": {"real": [], "model": [], "copy_last": []},
        "labels": [],
    }


def _append_semantic_batch(semantic, target_xyz, pred_xyz, copy_xyz, labels):
    if semantic is None:
        return
    from eval.action_xyz_classifier import extract_xyz_action_features

    classifier = semantic["classifier"]
    normalizer = semantic["normalizer"]
    for name, value in (("real", target_xyz), ("model", pred_xyz), ("copy_last", copy_xyz)):
        logits, features = extract_xyz_action_features(classifier, value, normalizer)
        semantic["logits"][name].append(logits.detach().cpu())
        semantic["features"][name].append(features.detach().cpu())
    semantic["labels"].append(labels.detach().view(-1).cpu())


def _classification_metrics(logits, labels, num_actions):
    topk = min(5, int(logits.shape[1]))
    _, top_labels = torch.topk(logits, k=topk, dim=1)
    preds = top_labels[:, 0]
    total = int(labels.shape[0])
    class_count = [0 for _ in range(num_actions)]
    class_correct = [0 for _ in range(num_actions)]
    for target, pred in zip(labels.tolist(), preds.tolist()):
        target = int(target)
        pred = int(pred)
        class_count[target] += 1
        if pred == target:
            class_correct[target] += 1
    per_class_acc = []
    valid_acc = []
    for idx in range(num_actions):
        if class_count[idx] > 0:
            acc = float(class_correct[idx]) / float(class_count[idx])
            per_class_acc.append(acc)
            valid_acc.append(acc)
        else:
            per_class_acc.append(None)
    return OrderedDict(
        [
            ("top1_acc", float((preds == labels).float().mean().item())),
            ("top5_acc", float((top_labels == labels.unsqueeze(1)).any(dim=1).float().mean().item())),
            ("balanced_acc", sum(valid_acc) / float(len(valid_acc)) if valid_acc else 0.0),
            ("per_class_acc", per_class_acc),
            ("per_class_count", class_count),
            ("predicted_label_counts", [int((preds == idx).sum().item()) for idx in range(num_actions)]),
            ("num_samples", total),
        ]
    )


def _fid(features_a, features_b):
    from scipy import linalg

    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("FID features 必须是二维矩阵")
    if a.shape[0] < 2 or b.shape[0] < 2:
        return None
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    sigma_a = np.cov(a, rowvar=False)
    sigma_b = np.cov(b, rowvar=False)
    eps = 1e-6
    sigma_a = sigma_a + np.eye(sigma_a.shape[0]) * eps
    sigma_b = sigma_b + np.eye(sigma_b.shape[0]) * eps
    covmean = linalg.sqrtm(sigma_a.dot(sigma_b))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    value = diff.dot(diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean)
    return float(np.real(value))


def _diversity(features, num_pairs, seed):
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] < 2:
        return 0.0
    rng = np.random.RandomState(int(seed))
    pairs = int(min(max(1, int(num_pairs)), features.shape[0] * (features.shape[0] - 1)))
    distances = []
    for _ in range(pairs):
        i = rng.randint(0, features.shape[0])
        j = rng.randint(0, features.shape[0] - 1)
        if j >= i:
            j += 1
        distances.append(np.linalg.norm(features[i] - features[j]))
    return float(np.mean(distances))


def _class_wise_fid(real_features, other_features, labels, num_actions):
    values = []
    labels = np.asarray(labels, dtype=np.int64)
    for idx in range(num_actions):
        mask = labels == idx
        count = int(mask.sum())
        if count >= 2:
            values.append(_fid(real_features[mask], other_features[mask]))
        else:
            values.append(None)
    valid = [item for item in values if item is not None]
    return OrderedDict(
        [
            ("per_class_fid", values),
            ("mean_class_fid", float(np.mean(valid)) if valid else None),
            ("valid_class_count", len(valid)),
        ]
    )


def _semantic_summary(args, semantic):
    if semantic is None:
        return None
    labels = torch.cat(semantic["labels"], dim=0).long()
    logits = {key: torch.cat(value, dim=0) for key, value in semantic["logits"].items()}
    features = {key: torch.cat(value, dim=0).numpy() for key, value in semantic["features"].items()}
    labels_np = labels.numpy()
    num_actions = int(logits["real"].shape[1])
    return OrderedDict(
        [
            ("action_classifier_path", args.action_classifier_path),
            ("classifier_checkpoint_step", semantic["state"].get("step")),
            ("real_future", _classification_metrics(logits["real"], labels, num_actions)),
            ("model_future", _classification_metrics(logits["model"], labels, num_actions)),
            ("copy_last_future", _classification_metrics(logits["copy_last"], labels, num_actions)),
            (
                "fid",
                OrderedDict(
                    [
                        ("model_vs_real", _fid(features["real"], features["model"])),
                        ("copy_last_vs_real", _fid(features["real"], features["copy_last"])),
                    ]
                ),
            ),
            (
                "diversity",
                OrderedDict(
                    [
                        (
                            "real",
                            _diversity(features["real"], getattr(args, "diversity_pairs", 1000), getattr(args, "seed", 0)),
                        ),
                        (
                            "model",
                            _diversity(features["model"], getattr(args, "diversity_pairs", 1000), getattr(args, "seed", 0)),
                        ),
                        (
                            "copy_last",
                            _diversity(features["copy_last"], getattr(args, "diversity_pairs", 1000), getattr(args, "seed", 0)),
                        ),
                    ]
                ),
            ),
            (
                "class_wise_fid",
                OrderedDict(
                    [
                        ("model_vs_real", _class_wise_fid(features["real"], features["model"], labels_np, num_actions)),
                        ("copy_last_vs_real", _class_wise_fid(features["real"], features["copy_last"], labels_np, num_actions)),
                    ]
                ),
            ),
        ]
    )


def evaluate_ntu_label_xyz(args, model=None, checkpoint_state=None, device=None):
    if device is None:
        device = _device()
    dataset = _build_dataset(args)
    loader = _build_loader(args, dataset)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    semantic = _load_semantic_evaluator(args, device)

    model_sums = _empty_sums()
    copy_sums = _empty_sums()
    num_samples = 0
    saved = {"obs_xyz": [], "target_xyz": [], "pred_xyz": [], "copy_last_xyz": [], "actions": [], "meta": []}

    if model is not None:
        model.eval()

    with torch.no_grad():
        for batch in loader:
            action = batch["action"].to(device)
            if "obs_xyz" in batch:
                obs_xyz = batch["obs_xyz"].to(device)
                target_xyz = batch["target_xyz"].to(device)
            else:
                obs_xyz = ntu_rotvec_2p_to_xyz(batch["obs_motion"].to(device), device=device, converter=converter)
                target_xyz = ntu_rotvec_2p_to_xyz(batch["future"].to(device), device=device, converter=converter)
            copy_xyz = copy_last_xyz(obs_xyz, args.pred_len)
            if model is None:
                pred_xyz = copy_xyz
            else:
                pred_xyz = model(obs_xyz, action)

            batch_size = int(obs_xyz.shape[0])
            _add_metrics(model_sums, compute_ntu_xyz_metrics(pred_xyz, target_xyz, obs_xyz), batch_size)
            _add_metrics(copy_sums, compute_ntu_xyz_metrics(copy_xyz, target_xyz, obs_xyz), batch_size)
            _append_semantic_batch(semantic, target_xyz, pred_xyz, copy_xyz, action)
            num_samples += batch_size

            if args.save_arrays and len(saved["meta"]) < int(args.save_array_limit):
                remain = int(args.save_array_limit) - len(saved["meta"])
                take = min(remain, batch_size)
                saved["obs_xyz"].append(obs_xyz[:take].detach().cpu())
                saved["target_xyz"].append(target_xyz[:take].detach().cpu())
                saved["pred_xyz"].append(pred_xyz[:take].detach().cpu())
                saved["copy_last_xyz"].append(copy_xyz[:take].detach().cpu())
                saved["actions"].append(action[:take].detach().cpu())
                saved["meta"].extend(batch["meta"][:take])

    if num_samples != len(dataset):
        raise AssertionError("评估样本数应为 {}，实际为 {}".format(len(dataset), num_samples))

    model_metrics = _finalize(model_sums, num_samples)
    copy_metrics = _finalize(copy_sums, num_samples)
    summary = OrderedDict()
    summary["mode"] = args.mode
    summary["dataset"] = "ntu120_2p_smplx"
    summary["split"] = args.split
    summary["data_path"] = args.data_path
    summary["xyz_cache"] = args.xyz_cache
    summary["window_len"] = args.window_len
    summary["obs_len"] = args.obs_len
    summary["pred_len"] = args.pred_len
    summary["num_samples"] = int(num_samples)
    summary["batch_size"] = args.batch_size
    summary["metrics_keys"] = list(NTU_XYZ_METRIC_KEYS)
    summary["model_metrics"] = model_metrics
    summary["copy_last_metrics"] = copy_metrics
    summary["beats_copy_last"] = {
        "xyz_mse": model_metrics["xyz_mse"] < copy_metrics["xyz_mse"],
        "xyz_mae": model_metrics["xyz_mae"] <= copy_metrics["xyz_mae"],
        "mpjpe": model_metrics["mpjpe"] < copy_metrics["mpjpe"],
    }
    semantic_metrics = _semantic_summary(args, semantic)
    if semantic_metrics is not None:
        summary["semantic_metrics"] = semantic_metrics
    summary["checkpoint"] = args.checkpoint
    summary["checkpoint_step"] = None if checkpoint_state is None else checkpoint_state.get("step")
    summary["created_at"] = _utc_now()

    if args.save_dir is not None:
        _write_json(os.path.join(args.save_dir, "metrics_{}.json".format(args.split)), summary)
        if args.save_arrays and len(saved["meta"]) > 0:
            array_dir = os.path.join(args.save_dir, "arrays")
            os.makedirs(array_dir, exist_ok=True)
            torch.save(
                {
                    "obs_xyz": torch.cat(saved["obs_xyz"], dim=0),
                    "target_xyz": torch.cat(saved["target_xyz"], dim=0),
                    "pred_xyz": torch.cat(saved["pred_xyz"], dim=0),
                    "copy_last_xyz": torch.cat(saved["copy_last_xyz"], dim=0),
                    "actions": torch.cat(saved["actions"], dim=0),
                    "meta": saved["meta"],
                },
                os.path.join(array_dir, "ntu_label_xyz_samples.pt"),
            )
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="checkpoint", choices=("checkpoint", "copy_last"))
    parser.add_argument("--data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--xyz_cache", default=None)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=20)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_dir", default="results/forecasting/ntu120_label/xyz_eval")
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--save_array_limit", type=int, default=8)
    parser.add_argument("--semantic_eval", action="store_true")
    parser.add_argument("--action_classifier_path", default=None)
    parser.add_argument("--diversity_pairs", type=int, default=1000)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    device = _device()
    model = None
    state = None
    if args.mode == "checkpoint":
        if args.checkpoint is None:
            raise ValueError("--mode checkpoint 必须提供 --checkpoint")
        model, state = _load_checkpoint(args.checkpoint, device)
    evaluate_ntu_label_xyz(args, model=model, checkpoint_state=state, device=device)


if __name__ == "__main__":
    main()
