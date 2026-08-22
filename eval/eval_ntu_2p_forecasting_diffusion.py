import argparse
import json
import os
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import (
    NTU2PDiffusionForecastDataset,
    ntu_2p_diffusion_collate,
)
from model.forecasting_ntu_xyz import create_ntu_label_xyz_model_from_config
from sample.sample_ntu_2p_forecasting_diffusion import (
    build_sampling_diffusion,
    load_ntu2p_diffusion_checkpoint,
    sample_diffusion_batch,
)
from utils.fixseed import fixseed
from utils.ntu_2p_rot6d import (
    check_ntu_2p_rot6d,
    ntu_2p_rot6d_to_xyz,
    root_rotation_matrices,
)
from utils.ntu_smplx_2p_xyz import (
    compute_ntu_xyz_metrics,
    interaction_pair_distances,
    masked_contact_l1,
)


NTU2P_DIFFUSION_METRIC_KEYS = (
    "xyz_mse",
    "xyz_mae",
    "mpjpe",
    "first_step_error",
    "velocity_error",
    "acceleration_error",
    "final_frame_error",
    "relative_joint_vector_error",
    "relative_root_translation_error",
    "relative_orientation_error",
    "contact_error",
    "contact_mask_ratio",
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
        f.write("\n")


def _sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _timed_model_call(device, fn):
    _sync_if_cuda(device)
    started_at = time.perf_counter()
    result = fn()
    _sync_if_cuda(device)
    return result, time.perf_counter() - started_at


def _to_float(value):
    if value is None:
        return None
    return float(value.detach().cpu().item())


def _relative_joint_vector_error(pred_xyz, target_xyz):
    pred_rel = pred_xyz[:, :, 0] - pred_xyz[:, :, 1]
    target_rel = target_xyz[:, :, 0] - target_xyz[:, :, 1]
    return torch.norm(pred_rel - target_rel, dim=-1).mean()


def _relative_root_translation_error(pred_xyz, target_xyz):
    pred_rel = pred_xyz[:, :, 0, 0] - pred_xyz[:, :, 1, 0]
    target_rel = target_xyz[:, :, 0, 0] - target_xyz[:, :, 1, 0]
    return torch.norm(pred_rel - target_rel, dim=-1).mean()


def _relative_orientation_error(pred_rot6d, target_rot6d):
    if pred_rot6d is None:
        return None
    pred_a, pred_b = root_rotation_matrices(pred_rot6d)
    target_a, target_b = root_rotation_matrices(target_rot6d)
    pred_rel = torch.matmul(pred_a.transpose(-1, -2), pred_b)
    target_rel = torch.matmul(target_a.transpose(-1, -2), target_b)
    return torch.sqrt(((pred_rel - target_rel) ** 2).sum(dim=(-2, -1))).mean()


def compute_ntu2p_diffusion_metrics(pred_xyz, target_xyz, obs_xyz, pred_rot6d=None, target_rot6d=None):
    base = compute_ntu_xyz_metrics(pred_xyz, target_xyz, obs_xyz)
    pred_pair_dist = interaction_pair_distances(pred_xyz)
    target_pair_dist = interaction_pair_distances(target_xyz)
    contact_error, contact_ratio = masked_contact_l1(pred_pair_dist, target_pair_dist)

    metrics = OrderedDict()
    metrics["xyz_mse"] = float(base["xyz_mse"])
    metrics["xyz_mae"] = float(base["xyz_mae"])
    metrics["mpjpe"] = float(base["mpjpe"])
    metrics["first_step_error"] = float(base["first_step_error"])
    metrics["velocity_error"] = float(base["velocity_error"])
    metrics["acceleration_error"] = float(base["acceleration_error"])
    metrics["final_frame_error"] = float(base["final_frame_error"])
    metrics["relative_joint_vector_error"] = _to_float(_relative_joint_vector_error(pred_xyz, target_xyz))
    metrics["relative_root_translation_error"] = _to_float(_relative_root_translation_error(pred_xyz, target_xyz))
    metrics["relative_orientation_error"] = _to_float(_relative_orientation_error(pred_rot6d, target_rot6d))
    metrics["contact_error"] = _to_float(contact_error)
    metrics["contact_mask_ratio"] = _to_float(contact_ratio)
    if tuple(metrics.keys()) != NTU2P_DIFFUSION_METRIC_KEYS:
        raise AssertionError("NTU2P diffusion metrics key 不稳定")
    for key, value in metrics.items():
        if value is not None and not np.isfinite(float(value)):
            raise ValueError("{} 指标为非有限数值: {}".format(key, value))
    return metrics


def _empty_sums():
    return OrderedDict((key, 0.0) for key in NTU2P_DIFFUSION_METRIC_KEYS)


def _empty_counts():
    return OrderedDict((key, 0) for key in NTU2P_DIFFUSION_METRIC_KEYS)


def _add_metrics(sums, counts, metrics, batch_size):
    if tuple(metrics.keys()) != NTU2P_DIFFUSION_METRIC_KEYS:
        raise AssertionError("metrics key 不稳定")
    for key in NTU2P_DIFFUSION_METRIC_KEYS:
        value = metrics[key]
        if value is None:
            continue
        sums[key] += float(value) * float(batch_size)
        counts[key] += int(batch_size)


def _finalize(sums, counts):
    result = OrderedDict()
    for key in NTU2P_DIFFUSION_METRIC_KEYS:
        result[key] = None if counts[key] == 0 else sums[key] / float(counts[key])
    return result


def _mean_std(metric_list):
    result = OrderedDict()
    for key in NTU2P_DIFFUSION_METRIC_KEYS:
        values = [item[key] for item in metric_list if item[key] is not None]
        if not values:
            result[key] = {"mean": None, "std": None}
            continue
        arr = np.asarray(values, dtype=np.float64)
        result[key] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
    return result


def _parse_seeds(value):
    if value is None or str(value).strip() == "":
        return []
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


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
        "sample_ids": [],
    }


def _append_semantic_batch(semantic, target_xyz, pred_xyz, copy_xyz, labels, meta):
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
    semantic["sample_ids"].extend([item["sample_id"] for item in meta])


def _classification_metrics(logits, labels, num_actions):
    topk = min(5, int(logits.shape[1]))
    _, top_labels = torch.topk(logits, k=topk, dim=1)
    preds = top_labels[:, 0]
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
            ("num_samples", int(labels.shape[0])),
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


def _semantic_summary(args, semantic, seed):
    if semantic is None:
        return None, None
    labels = torch.cat(semantic["labels"], dim=0).long()
    logits = {key: torch.cat(value, dim=0) for key, value in semantic["logits"].items()}
    features = {key: torch.cat(value, dim=0).numpy() for key, value in semantic["features"].items()}
    num_actions = int(logits["real"].shape[1])
    summary = OrderedDict(
        [
            ("action_classifier_path", args.action_classifier_path),
            ("classifier_checkpoint_step", semantic["state"].get("step")),
            ("classifier_real_test_metrics", semantic["state"].get("real_test_metrics")),
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
                        ("real", _diversity(features["real"], args.diversity_pairs, seed)),
                        ("model", _diversity(features["model"], args.diversity_pairs, seed)),
                        ("copy_last", _diversity(features["copy_last"], args.diversity_pairs, seed)),
                    ]
                ),
            ),
        ]
    )
    payload = {
        "sample_ids": list(semantic["sample_ids"]),
        "model_features": features["model"],
    }
    return summary, payload


def _multimodality_across_seeds(payloads):
    if len(payloads) < 2:
        return None
    base_ids = payloads[0]["sample_ids"]
    for payload in payloads[1:]:
        if payload["sample_ids"] != base_ids:
            raise ValueError("多 seed multimodality 要求 sample_id 顺序一致")
    features = np.stack([payload["model_features"] for payload in payloads], axis=0)
    if features.shape[0] < 2:
        return None
    per_condition = []
    for cond_idx in range(features.shape[1]):
        cond_features = features[:, cond_idx, :]
        distances = []
        for i in range(cond_features.shape[0]):
            for j in range(i + 1, cond_features.shape[0]):
                distances.append(np.linalg.norm(cond_features[i] - cond_features[j]))
        if distances:
            per_condition.append(float(np.mean(distances)))
    return OrderedDict(
        [
            ("value", float(np.mean(per_condition)) if per_condition else None),
            ("num_conditions", int(features.shape[1])),
            ("num_seeds", int(features.shape[0])),
        ]
    )


def _build_dataset(args):
    return NTU2PDiffusionForecastDataset(
        manifest_path=args.manifest_path,
        split=args.split,
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        window_len=args.window_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )


def _build_loader(args, dataset):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ntu_2p_diffusion_collate,
        drop_last=False,
    )


def _load_direct_checkpoint(path, device):
    state = torch.load(path, map_location=device)
    if "model_state_dict" not in state or "model_config" not in state:
        raise ValueError("direct checkpoint 缺少 model_state_dict/model_config")
    model = create_ntu_label_xyz_model_from_config(state["model_config"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, state


def _load_independent_single_person_checkpoint(path, device):
    model, state = _load_direct_checkpoint(path, device)
    if int(model.num_persons) != 1:
        raise ValueError("独立单人 baseline checkpoint 必须 num_persons=1，当前为 {}".format(model.num_persons))
    if state.get("representation") not in (None, "independent_single_person_xyz"):
        raise ValueError("checkpoint representation 不是 independent_single_person_xyz")
    return model, state


def _independent_single_person_forward(model, obs_xyz, action):
    if int(obs_xyz.shape[2]) != 2:
        raise ValueError("独立单人 baseline 需要双人 obs_xyz")
    batch_size = int(obs_xyz.shape[0])
    obs_single = torch.cat((obs_xyz[:, :, 0:1], obs_xyz[:, :, 1:2]), dim=0)
    action_single = torch.cat((action, action), dim=0)
    pred_single = model(obs_single, action_single)
    if tuple(pred_single.shape[2:]) != (1, 55, 3):
        raise ValueError("单人模型输出必须为 [B,T,1,55,3]，当前为 {}".format(tuple(pred_single.shape)))
    return torch.cat((pred_single[:batch_size], pred_single[batch_size:]), dim=2)


def _copy_last_rot6d(obs_rot6d, pred_len):
    check_ntu_2p_rot6d(obs_rot6d)
    return obs_rot6d[..., -1:].expand(-1, -1, -1, int(pred_len)).contiguous()


def _evaluate_once(args, dataset, loader, device, sample_seed):
    fixseed(sample_seed)
    converter = None
    from model.rotation2xyz import Rotation2xyz_x

    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")

    model = None
    checkpoint_state = None
    diffusion = None
    if args.mode == "direct":
        if args.checkpoint is None:
            raise ValueError("--mode direct 必须提供 --checkpoint")
        model, checkpoint_state = _load_direct_checkpoint(args.checkpoint, device)
    elif args.mode == "independent_single_person":
        if args.checkpoint is None:
            raise ValueError("--mode independent_single_person 必须提供 --checkpoint")
        model, checkpoint_state = _load_independent_single_person_checkpoint(args.checkpoint, device)
    elif args.mode == "diffusion":
        if args.checkpoint is None:
            raise ValueError("--mode diffusion 必须提供 --checkpoint")
        model, checkpoint_state = load_ntu2p_diffusion_checkpoint(args.checkpoint, device)
        diffusion_config = checkpoint_state.get("diffusion_config", {})
        noise_schedule = args.noise_schedule or diffusion_config.get("noise_schedule", "cosine")
        timestep_respacing = args.timestep_respacing or ("ddim50" if args.use_ddim else "")
        diffusion = build_sampling_diffusion(
            noise_schedule=noise_schedule,
            timestep_respacing=timestep_respacing,
            body_model=checkpoint_state.get("model_config", {}).get("body_model", "smplx"),
        )

    model_sums = _empty_sums()
    model_counts = _empty_counts()
    copy_sums = _empty_sums()
    copy_counts = _empty_counts()
    semantic = _load_semantic_evaluator(args, device)
    num_samples = 0
    model_inference_seconds = 0.0
    model_inference_batches = 0
    saved = {
        "obs_xyz": [],
        "target_xyz": [],
        "pred_xyz": [],
        "copy_last_xyz": [],
        "obs_rot6d": [],
        "target_rot6d": [],
        "pred_rot6d": [],
        "copy_last_rot6d": [],
        "actions": [],
        "meta": [],
    }

    with torch.no_grad():
        for batch in loader:
            obs_rot6d = batch["obs_motion"].to(device)
            target_rot6d = batch["future"].to(device)
            action = batch["action"].to(device)
            obs_xyz = ntu_2p_rot6d_to_xyz(obs_rot6d, converter=converter)
            target_xyz = ntu_2p_rot6d_to_xyz(target_rot6d, converter=converter)
            copy_rot6d = _copy_last_rot6d(obs_rot6d, args.pred_len)
            copy_xyz = ntu_2p_rot6d_to_xyz(copy_rot6d, converter=converter)

            pred_rot6d = None
            if args.mode == "copy_last":
                pred_rot6d = copy_rot6d
                pred_xyz = copy_xyz
            elif args.mode == "direct":
                pred_xyz, elapsed = _timed_model_call(device, lambda: model(obs_xyz, action))
                model_inference_seconds += elapsed
                model_inference_batches += 1
            elif args.mode == "independent_single_person":
                pred_xyz, elapsed = _timed_model_call(
                    device,
                    lambda: _independent_single_person_forward(model, obs_xyz, action),
                )
                model_inference_seconds += elapsed
                model_inference_batches += 1
            elif args.mode == "diffusion":
                pred_rot6d, elapsed = _timed_model_call(
                    device,
                    lambda: sample_diffusion_batch(
                        model=model,
                        diffusion=diffusion,
                        batch=batch,
                        device=device,
                        use_ddim=args.use_ddim,
                        guidance_scale=args.guidance_scale,
                        progress=args.progress,
                    ),
                )
                model_inference_seconds += elapsed
                model_inference_batches += 1
                pred_xyz = ntu_2p_rot6d_to_xyz(pred_rot6d, converter=converter)
            else:
                raise ValueError("未知 mode: {}".format(args.mode))

            batch_size = int(obs_rot6d.shape[0])
            _add_metrics(
                model_sums,
                model_counts,
                compute_ntu2p_diffusion_metrics(pred_xyz, target_xyz, obs_xyz, pred_rot6d=pred_rot6d, target_rot6d=target_rot6d),
                batch_size,
            )
            _add_metrics(
                copy_sums,
                copy_counts,
                compute_ntu2p_diffusion_metrics(copy_xyz, target_xyz, obs_xyz, pred_rot6d=copy_rot6d, target_rot6d=target_rot6d),
                batch_size,
            )
            _append_semantic_batch(semantic, target_xyz, pred_xyz, copy_xyz, action, batch["meta"])
            num_samples += batch_size

            if args.save_arrays and len(saved["meta"]) < int(args.save_array_limit):
                remain = int(args.save_array_limit) - len(saved["meta"])
                take = min(remain, batch_size)
                saved["obs_xyz"].append(obs_xyz[:take].detach().cpu())
                saved["target_xyz"].append(target_xyz[:take].detach().cpu())
                saved["pred_xyz"].append(pred_xyz[:take].detach().cpu())
                saved["copy_last_xyz"].append(copy_xyz[:take].detach().cpu())
                saved["obs_rot6d"].append(obs_rot6d[:take].detach().cpu())
                saved["target_rot6d"].append(target_rot6d[:take].detach().cpu())
                saved["copy_last_rot6d"].append(copy_rot6d[:take].detach().cpu())
                if pred_rot6d is not None:
                    saved["pred_rot6d"].append(pred_rot6d[:take].detach().cpu())
                saved["actions"].append(action[:take].detach().cpu())
                saved["meta"].extend(batch["meta"][:take])

    if num_samples != len(dataset):
        raise AssertionError("评估样本数应为 {}，实际为 {}".format(len(dataset), num_samples))

    result = OrderedDict()
    result["sample_seed"] = int(sample_seed)
    result["model_metrics"] = _finalize(model_sums, model_counts)
    result["copy_last_metrics"] = _finalize(copy_sums, copy_counts)
    result["beats_copy_last"] = OrderedDict(
        [
            ("xyz_mse", result["model_metrics"]["xyz_mse"] < result["copy_last_metrics"]["xyz_mse"]),
            ("xyz_mae", result["model_metrics"]["xyz_mae"] <= result["copy_last_metrics"]["xyz_mae"]),
            ("mpjpe", result["model_metrics"]["mpjpe"] < result["copy_last_metrics"]["mpjpe"]),
        ]
    )
    result["latency"] = OrderedDict(
        [
            ("model_inference_seconds_total", float(model_inference_seconds)),
            ("model_inference_batches", int(model_inference_batches)),
            (
                "model_inference_seconds_per_sample",
                0.0 if num_samples == 0 else float(model_inference_seconds) / float(num_samples),
            ),
        ]
    )
    result["checkpoint_step"] = None if checkpoint_state is None else checkpoint_state.get("step")
    semantic_metrics, semantic_payload = _semantic_summary(args, semantic, sample_seed)
    if semantic_metrics is not None:
        result["semantic_metrics"] = semantic_metrics
        result["_semantic_payload"] = semantic_payload
    result["_saved_arrays"] = saved
    return result


def evaluate_ntu2p_forecasting_diffusion(args):
    device = _device()
    dataset = _build_dataset(args)
    loader = _build_loader(args, dataset)
    sample_seeds = _parse_seeds(args.sample_seeds)
    if args.mode != "diffusion" or len(sample_seeds) == 0:
        sample_seeds = [int(args.seed)]

    per_seed = []
    first_saved = None
    semantic_payloads = []
    for seed in sample_seeds:
        result = _evaluate_once(args, dataset, loader, device, seed)
        if first_saved is None:
            first_saved = result.pop("_saved_arrays")
        else:
            result.pop("_saved_arrays")
        if "_semantic_payload" in result:
            semantic_payloads.append(result.pop("_semantic_payload"))
        per_seed.append(result)

    summary = OrderedDict()
    summary["mode"] = args.mode
    summary["dataset"] = "ntu120_2p_smplx"
    summary["split"] = args.split
    summary["manifest_path"] = args.manifest_path
    summary["train_data_path"] = args.train_data_path
    summary["test_data_path"] = args.test_data_path
    summary["window_len"] = args.window_len
    summary["obs_len"] = args.obs_len
    summary["pred_len"] = args.pred_len
    summary["num_samples"] = len(dataset)
    summary["batch_size"] = args.batch_size
    summary["metrics_keys"] = list(NTU2P_DIFFUSION_METRIC_KEYS)
    summary["sample_seeds"] = sample_seeds
    summary["per_seed"] = per_seed
    summary["model_metrics_mean_std"] = _mean_std([item["model_metrics"] for item in per_seed])
    summary["copy_last_metrics_mean_std"] = _mean_std([item["copy_last_metrics"] for item in per_seed])
    summary["latency"] = OrderedDict(
        [
            (
                "model_inference_seconds_per_sample_mean",
                float(np.mean([item["latency"]["model_inference_seconds_per_sample"] for item in per_seed])),
            ),
            (
                "model_inference_seconds_per_sample_std",
                float(np.std([item["latency"]["model_inference_seconds_per_sample"] for item in per_seed], ddof=0)),
            ),
        ]
    )
    summary["checkpoint"] = args.checkpoint
    summary["use_ddim"] = bool(args.use_ddim)
    summary["timestep_respacing"] = args.timestep_respacing or ("ddim50" if args.mode == "diffusion" and args.use_ddim else "")
    summary["guidance_scale"] = float(args.guidance_scale)
    summary["multimodality"] = _multimodality_across_seeds(semantic_payloads) if semantic_payloads else None
    summary["created_at"] = _utc_now()

    if args.save_dir is not None:
        _write_json(os.path.join(args.save_dir, "metrics_{}.json".format(args.split)), summary)
        if args.save_arrays and first_saved is not None and len(first_saved["meta"]) > 0:
            array_dir = os.path.join(args.save_dir, "arrays")
            os.makedirs(array_dir, exist_ok=True)
            arrays = {}
            for key, values in first_saved.items():
                if key == "meta":
                    arrays[key] = values
                elif len(values) > 0:
                    arrays[key] = torch.cat(values, dim=0)
            torch.save(arrays, os.path.join(array_dir, "ntu2p_diffusion_eval_samples.pt"))
    print(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="diffusion", choices=("copy_last", "direct", "independent_single_person", "diffusion"))
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--test_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--window_len", type=int, default=60)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_seeds", default="")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_dir", default="results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_eval")
    parser.add_argument("--noise_schedule", default=None)
    parser.add_argument("--timestep_respacing", default="")
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--save_array_limit", type=int, default=8)
    parser.add_argument("--semantic_eval", action="store_true")
    parser.add_argument("--action_classifier_path", default=None)
    parser.add_argument("--diversity_pairs", type=int, default=1000)
    parser.add_argument("--progress", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.obs_len + args.pred_len != args.window_len:
        raise ValueError("obs_len + pred_len 必须等于 window_len")
    evaluate_ntu2p_forecasting_diffusion(args)


if __name__ == "__main__":
    main()
