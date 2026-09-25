"""自检 NTU2P 完整序列 xyz 缓存、GPU 采样器、场景规范化与几何增广。

1. val/test 中心窗口与原数据集 + on-the-fly FK 逐样本对比；train 按采样器给出的 (sample_id, start) 抽检；
2. 采样器同 seed 可复现、state_dict 可恢复、一个 epoch 内每条序列至多出现一次；
3. 规范化：往返误差、旋转正交性、规范系几何性质、规范系与原系指标一致性；
4. 增广：交换/镜像两次恒等、yaw 与镜像保骨长与竖直分量、规范化对 yaw/平移/镜像/交换的等变性。
"""

import argparse
import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from data_loaders.forecasting.ntu_2p_diffusion import NTU2PDiffusionForecastDataset, ntu_2p_diffusion_collate
from data_loaders.forecasting.ntu2p_xyz_seq_cache import (
    DEFAULT_CACHE_DIR,
    NTU2PXYZSeqCache,
    NTU2PXYZTrainSampler,
    eval_windows,
    iter_eval_batches,
)
from model.rotation2xyz import Rotation2xyz_x
from model.smpl import SMPLX
from utils.ntu2p_canonical import (
    DEFAULT_MIN_AB_DISTANCE,
    PELVIS,
    SMPLX_MIRROR_PERMUTATION,
    NTU2PSceneAugment,
    apply_linear,
    canonical_frame,
    estimate_up,
    from_canonical,
    mirror_joint_labels,
    swap_persons,
    to_canonical,
)
from utils.ntu_2p_rot6d import ntu_2p_rot6d_to_xyz
from utils.ntu_smplx_2p_xyz import compute_ntu_articulation_metrics, compute_ntu_xyz_metrics, copy_last_xyz


DEFAULT_MANIFEST = "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"
DEFAULT_CHECKPOINT = (
    "save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_ema_s2_5_dct1_root01_s0_10000/ema/model000010000.pt"
)
# 按坐标分量取绝对值或标准差的指标不具旋转不变性，只作记录不进判定。
NON_ROTATION_INVARIANT = ("xyz_mae", "local_pose_temporal_std", "local_pose_temporal_std_target")


def _max_abs(a, b):
    return float((a - b).abs().max().item())


def check_cache_against_dataset(args, device, converter):
    result = OrderedDict()
    for split in ("val", "test"):
        cache = NTU2PXYZSeqCache.load(args.cache_dir, split, manifest_path=args.manifest_path, device=device)
        dataset = NTU2PDiffusionForecastDataset(args.manifest_path, split)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=ntu_2p_diffusion_collate)
        max_obs = max_target = 0.0
        count = 0
        with torch.no_grad():
            for batch, new in zip(loader, iter_eval_batches(eval_windows(cache), 16)):
                if [m["sample_id"] for m in batch["meta"]] != [m["sample_id"] for m in new["meta"]]:
                    raise AssertionError("{} 样本顺序不一致".format(split))
                if [m["start"] for m in batch["meta"]] != [m["start"] for m in new["meta"]]:
                    raise AssertionError("{} 窗口起点不一致".format(split))
                if not torch.equal(batch["action"].view(-1).to(device), new["action"]):
                    raise AssertionError("{} action 不一致".format(split))
                obs = ntu_2p_rot6d_to_xyz(batch["obs_motion"].to(device), converter=converter)
                target = ntu_2p_rot6d_to_xyz(batch["future"].to(device), converter=converter)
                max_obs = max(max_obs, _max_abs(obs, new["obs_xyz"]))
                max_target = max(max_target, _max_abs(target, new["target_xyz"]))
                count += int(obs.shape[0])
        result[split] = OrderedDict([("num_samples", count), ("max_abs_obs", max_obs), ("max_abs_target", max_target)])

    cache = NTU2PXYZSeqCache.load(args.cache_dir, "train", manifest_path=args.manifest_path, device=device)
    batch = NTU2PXYZTrainSampler(cache, batch_size=args.train_spot_windows, seed=args.seed).sample()
    dataset = NTU2PDiffusionForecastDataset(args.manifest_path, "train")
    max_diff = 0.0
    for row in range(args.train_spot_windows):
        seq, start = int(batch["seq_index"][row]), int(batch["start"][row])
        dataset._sample_start = lambda length, fixed=start: fixed
        item = ntu_2p_diffusion_collate([dataset[seq]])
        same_action = int(item["action"].item()) == int(batch["action"][row])
        if item["meta"][0]["sample_id"] != cache.sample_ids[seq] or not same_action:
            raise AssertionError("train 抽检 sample_id/action 不一致")
        with torch.no_grad():
            obs = ntu_2p_rot6d_to_xyz(item["obs_motion"].to(device), converter=converter)
            target = ntu_2p_rot6d_to_xyz(item["future"].to(device), converter=converter)
        max_diff = max(max_diff, _max_abs(obs[0], batch["obs_xyz"][row]), _max_abs(target[0], batch["target_xyz"][row]))
    result["train_spot_check"] = OrderedDict([("num_windows", args.train_spot_windows), ("max_abs", max_diff)])
    return result


def check_sampler(cache):
    first = NTU2PXYZTrainSampler(cache, batch_size=32, seed=7)
    second = NTU2PXYZTrainSampler(cache, batch_size=32, seed=7)
    identical = True
    for _ in range(20):
        a, b = first.sample(), second.sample()
        identical &= torch.equal(a["seq_index"], b["seq_index"]) and torch.equal(a["start"], b["start"])
        identical &= torch.equal(a["obs_xyz"], b["obs_xyz"]) and torch.equal(a["target_xyz"], b["target_xyz"])
    state = first.state_dict()
    reference = [first.sample() for _ in range(5)]
    resumed = NTU2PXYZTrainSampler(cache, batch_size=32, seed=999)
    resumed.load_state_dict(state)
    resume_ok = all(torch.equal(ref["seq_index"], resumed.sample()["seq_index"]) for ref in reference)
    epoch = NTU2PXYZTrainSampler(cache, batch_size=8, seed=0)
    seen = torch.cat([epoch.sample()["seq_index"] for _ in range(len(cache) // 8)])
    counts = torch.bincount(seen, minlength=len(cache))
    starts = NTU2PXYZTrainSampler(cache, batch_size=4096, seed=1).sample()
    max_start = cache.lengths_cpu[starts["seq_index"]] - 60
    return OrderedDict(
        [
            ("same_seed_identical_20_steps", bool(identical)),
            ("state_dict_resume_identical", bool(resume_ok)),
            ("first_epoch_max_count", int(counts.max().item())),
            ("start_in_range", bool(((starts["start"] >= 0) & (starts["start"] <= max_start)).all())),
            ("start_hits_max_start_ratio", float((starts["start"] == max_start).float().mean())),
        ]
    )


def _metrics(pred, target, obs):
    merged = OrderedDict()
    merged.update(compute_ntu_xyz_metrics(pred, target, obs))
    merged.update(compute_ntu_articulation_metrics(pred, target))
    return merged


def _compare_metrics(original, canonical):
    rel = OrderedDict()
    for key, value in original.items():
        rel[key] = abs(value - canonical[key]) / max(abs(value), 1e-12)
    invariant = sorted((value, key) for key, value in rel.items() if key not in NON_ROTATION_INVARIANT)
    worst_rel, worst_key = invariant[-1]
    return OrderedDict(
        [
            ("max_rel_diff_invariant_metrics", worst_rel),
            ("worst_invariant_metric", [worst_key, original[worst_key], canonical[worst_key]]),
            ("rel_diff_non_invariant", OrderedDict((key, rel[key]) for key in NON_ROTATION_INVARIANT)),
            ("rel_diff_key_l2", OrderedDict((key, rel[key]) for key in ("xyz_mse", "mpjpe", "root_mse", "local_mse"))),
        ]
    )


def check_canonical(obs, target, preds):
    result = OrderedDict()
    unit_y = torch.tensor([0.0, 1.0, 0.0], device=obs.device)
    for yaw_ref in ("ab_line", "a_facing"):
        for up_mode in ("obs", "dataset"):
            frame = canonical_frame(obs, up_mode=up_mode, yaw_ref=yaw_ref)
            rot = frame["rotation"]
            eye = torch.eye(3, device=obs.device).expand_as(rot)
            gram = (rot.unsqueeze(-2) * rot.unsqueeze(-3)).sum(-1)
            obs_c = to_canonical(obs, frame)
            pelvis = obs_c[:, -1, :, PELVIS]
            up_c = apply_linear(frame["up"], rot)
            entry = OrderedDict(
                [
                    ("roundtrip_max_abs", _max_abs(from_canonical(obs_c, frame), obs)),
                    ("orthonormal_max_err", _max_abs(gram, eye)),
                    ("det_min", float((rot[:, 0] * torch.cross(rot[:, 1], rot[:, 2], dim=-1)).sum(-1).min().item())),
                    ("last_obs_pelvis_mid_max_abs", float(pelvis.mean(dim=1).abs().max().item())),
                    ("up_maps_to_plus_y_max_err", _max_abs(up_c, unit_y.expand_as(up_c))),
                ]
            )
            if yaw_ref == "ab_line":
                ab = pelvis[:, 1] - pelvis[:, 0]
                wide = ab[:, [0, 2]].norm(dim=-1) >= DEFAULT_MIN_AB_DISTANCE
                entry["ab_line_fallback_samples"] = int((~wide).sum().item())
                entry["ab_line_z_max_abs_non_fallback"] = float(ab[wide, 2].abs().max().item())
                entry["ab_line_x_min_non_fallback"] = float(ab[wide, 0].min().item())
            for name, pred in preds.items():
                entry["metrics_{}".format(name)] = _compare_metrics(
                    _metrics(pred, target, obs),
                    _metrics(to_canonical(pred, frame), to_canonical(target, frame), obs_c),
                )
            result["{}_{}".format(yaw_ref, up_mode)] = entry
    # obs 模式下规范系里重新估计的 up 应为 +Y
    frame = canonical_frame(obs)
    up_c = estimate_up(to_canonical(obs, frame), "obs", max_deviation_deg=180.0)
    fallback = (estimate_up(obs, "obs") - estimate_up(obs, "dataset")).abs().sum(-1).eq(0)
    result["fallback_to_dataset_up_ratio"] = float(fallback.float().mean())
    result["obs_up_in_canonical_minus_plus_y_max_abs_non_fallback"] = _max_abs(
        up_c[~fallback], unit_y.expand_as(up_c[~fallback])
    )
    return result


def _bone_lengths(value, parents):
    child = torch.arange(1, len(parents), device=value.device)
    parent = torch.as_tensor(parents[1:], device=value.device)
    return (value.index_select(-2, child) - value.index_select(-2, parent)).norm(dim=-1)


def _params(batch_size, yaw=0.0, swap=False, mirror=False, shift=(0.0, 0.0)):
    return OrderedDict(
        [
            ("yaw", torch.full((batch_size,), float(yaw))),
            ("swap", torch.full((batch_size,), bool(swap), dtype=torch.bool)),
            ("mirror", torch.full((batch_size,), bool(mirror), dtype=torch.bool)),
            ("shift", torch.tensor(shift).view(1, 2).expand(batch_size, 2).clone()),
        ]
    )


def check_augment(obs, target, parents):
    aug = NTU2PSceneAugment()
    batch = int(obs.shape[0])
    result = OrderedDict()
    up = estimate_up(obs)

    def vertical(value):
        return (value * up.view(-1, 1, 1, 1, 3)).sum(-1)

    once_o, once_t = aug.apply(obs, target, _params(batch, swap=True))
    twice_o, twice_t = aug.apply(once_o, once_t, _params(batch, swap=True))
    result["swap_twice_identity_max_abs"] = max(_max_abs(twice_o, obs), _max_abs(twice_t, target))

    mir_o, mir_t = aug.apply(obs, target, _params(batch, mirror=True))
    back_o, back_t = aug.apply(mir_o, mir_t, _params(batch, mirror=True))
    result["mirror_twice_identity_max_abs"] = max(_max_abs(back_o, obs), _max_abs(back_t, target))
    result["mirror_vertical_unchanged_max_abs"] = _max_abs(vertical(mir_t), vertical(mirror_joint_labels(target)))
    bones = _bone_lengths(target, parents)
    perm = list(SMPLX_MIRROR_PERMUTATION)
    permuted_bones = _bone_lengths(target.index_select(-2, torch.as_tensor(perm, device=obs.device)), parents)
    result["mirror_bone_len_vs_counterpart_max_abs"] = _max_abs(_bone_lengths(mir_t, parents), permuted_bones)
    result["mirror_bone_len_vs_same_label_max_abs"] = _max_abs(_bone_lengths(mir_t, parents), bones)

    yaw_o, yaw_t = aug.apply(obs, target, _params(batch, yaw=1.234, shift=(0.3, -0.2)))
    result["yaw_bone_len_max_abs"] = _max_abs(_bone_lengths(yaw_t, parents), bones)
    result["yaw_vertical_unchanged_max_abs"] = _max_abs(vertical(yaw_t), vertical(target))
    result["yaw_up_estimate_unchanged_max_abs"] = _max_abs(estimate_up(yaw_o), up)
    # 只取末帧双人 110 个关节；显式差分而非 cdist（后者走 matmul，在 Ampere 上默认 TF32）。
    flat = target[:, -1].reshape(batch, -1, 3)
    flat_yaw = yaw_t[:, -1].reshape(batch, -1, 3)
    result["yaw_pairwise_dist_max_abs"] = _max_abs(
        (flat.unsqueeze(1) - flat.unsqueeze(2)).norm(dim=-1),
        (flat_yaw.unsqueeze(1) - flat_yaw.unsqueeze(2)).norm(dim=-1),
    )

    frame = canonical_frame(obs)
    base_c = to_canonical(target, frame)
    yaw_c = to_canonical(yaw_t, canonical_frame(yaw_o))
    result["canonical_removes_yaw_shift_max_abs"] = _max_abs(yaw_c, base_c)
    mirror_c = to_canonical(mir_t, canonical_frame(mir_o))
    expected = mirror_joint_labels(base_c * torch.tensor([1.0, 1.0, -1.0], device=obs.device))
    result["canonical_mirror_equals_z_reflection_max_abs"] = _max_abs(mirror_c, expected)
    swap_c = to_canonical(once_t, canonical_frame(once_o))
    expected = swap_persons(base_c * torch.tensor([-1.0, 1.0, -1.0], device=obs.device))
    pelvis = obs[:, -1, :, PELVIS]
    ab = pelvis[:, 1] - pelvis[:, 0]
    wide = (ab - (ab * up).sum(-1, keepdim=True) * up).norm(dim=-1) >= 0.1
    result["canonical_swap_equals_rot180_max_abs_ab_ge_0p1m"] = _max_abs(swap_c[wide], expected[wide])
    result["swap_check_num_samples"] = int(wide.sum().item())

    generator = torch.Generator()
    generator.manual_seed(0)
    rand_o, rand_t = aug(obs, target, generator)
    result["random_aug_finite"] = bool(torch.isfinite(rand_o).all() and torch.isfinite(rand_t).all())
    return result


def run(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("自检需要 CUDA 设备，例如 --device cuda:0")
    torch.cuda.set_device(device)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    report = OrderedDict()
    report["cache_vs_dataset"] = check_cache_against_dataset(args, device, converter)
    train = NTU2PXYZSeqCache.load(args.cache_dir, "train", manifest_path=args.manifest_path, device=device)
    report["sampler"] = check_sampler(train)

    val = eval_windows(NTU2PXYZSeqCache.load(args.cache_dir, "val", manifest_path=args.manifest_path, device=device))
    train_batch = NTU2PXYZTrainSampler(train, batch_size=256, seed=args.seed + 1).sample()
    obs = torch.cat((val["obs_xyz"], train_batch["obs_xyz"]))
    target = torch.cat((val["target_xyz"], train_batch["target_xyz"]))
    action = torch.cat((val["action"], train_batch["action"]))
    preds = OrderedDict([("copy_last", copy_last_xyz(obs, target.shape[1]))])
    if args.checkpoint and os.path.exists(args.checkpoint):
        from model.forecasting_ntu2p_residual_xyz import load_ntu2p_residual_refiner_checkpoint

        model, _ = load_ntu2p_residual_refiner_checkpoint(args.checkpoint, device)
        with torch.no_grad():
            preds["residual_refiner"] = torch.cat([model(o, a) for o, a in zip(obs.split(64), action.split(64))])
    report["canonical"] = check_canonical(obs, target, preds)
    report["augment"] = check_augment(obs, target, SMPLX().parents[:55].tolist())
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="可选；用于非平凡预测的指标不变性检查")
    parser.add_argument("--output", default=os.path.join(DEFAULT_CACHE_DIR, "pipeline_check.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_spot_windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
