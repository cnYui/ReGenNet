"""NTU 双人预测"自然度"分析：脚底滑行、骨长、步态交替、分部位/分 horizon 误差、交互几何，并按 26 类动作分解。

两种输入：
1. --checkpoint：residual refiner checkpoint，在 val 上推理，比较 model / base / copy_last / GT；
2. --pred_file：新架构复用入口，.pt 内含 obs_xyz [N,10,2,55,3]、target_xyz [N,50,2,55,3]、actions [N] 或 [N,1]，
   以及 pred_xyz（单方法）或 methods: {name: [N,50,2,55,3]}（多方法），可选 meta（list[dict]，含 sample_id/action_code）。

除被评估方法外，总会附加两个参考：
- GT：target 自身，给出每个指标的数据噪声下限（如拟合抖动带来的"滑步"）；
- slide_ref：GT root 轨迹 + obs 末帧局部姿态刚性平移，即"纯滑行"合成参考，用来标定滑行类指标的另一端。
"""

import argparse
import csv
import json
import os
import re
from collections import OrderedDict

import torch

from utils.ntu2p_naturalness import (
    BODY_PARTS,
    CONTACT_HEIGHT_THRESHOLDS,
    CONTACT_SPEED_THRESHOLD,
    FLOAT_HEIGHT,
    HORIZON_BIN_FRAMES,
    PENETRATION_DEPTH,
    SLIDE_LOCK_TOLERANCE,
    SLIDE_ROOT_SPEED_MIN,
    SMOOTH_KERNEL,
    STEP_HYSTERESIS,
    WALK_DISPLACEMENT_THRESHOLD,
    aggregate_stats,
    compute_naturalness_stats,
    estimate_scene_frame,
    per_sample_values,
    to_scene_coords,
)

NTU2P_ACTION_NAMES = (
    "punching/slapping", "kicking", "pushing", "pat on back", "point finger", "hugging",
    "giving something", "touch pocket", "handshaking", "walking towards", "walking apart",
    "hit with something", "wield knife", "knock over", "grab stuff", "shoot with gun",
    "step on foot", "high-five", "cheers and drink", "carry together", "take a photo",
    "follow", "whisper", "exchange things", "support with hand", "rock-paper-scissors",
)

# 报告中的核心指标（其余全部写入 JSON）。
SUMMARY_KEYS = (
    "mpjpe", "mpjpe_body22", "local_mpjpe_body22",
    "skate_gt_contact_speed", "skate_gt_contact_violation", "skate_self_weighted_speed", "skate_self_ratio",
    "skate_gt_contact_speed_walk", "skate_gt_contact_violation_walk", "skate_self_ratio_walk",
    "foot_penetration_ratio", "foot_float_ratio",
    "bone_abs_err_body", "bone_rel_err_body", "bone_rel_err_legs", "bone_rel_err_arms", "bone_rel_err_fingers",
    "bone_rel_err_body_max", "bone_rel_err_legs_max",
    "root_distance_abs_err", "heading_err_deg", "facing_angle_abs_err_deg", "min_interperson_dist_abs_err",
    "jerk_body",
)
GAIT_KEYS = (
    "root_speed_mean", "moving_frame_fraction", "stance_fraction", "slide_frame_ratio", "foot_rel_motion_ratio",
    "foot_to_root_path_ratio", "ankle_peak_speed", "ankle_peak_to_root_speed", "lr_forward_vel_corr",
    "lr_rel_forward_vel_corr", "rel_fwd_vel_corr_with_gt", "leg_swing_amp", "swing_per_speed", "step_count",
)
ACTION_KEYS = (
    "mpjpe", "mpjpe_body22", "local_mpjpe_body22", "mpjpe_legs", "local_mpjpe_legs", "skate_gt_contact_speed",
    "skate_gt_contact_violation", "bone_rel_err_body", "heading_err_deg", "root_distance_abs_err",
)


def _device(value):
    requested = torch.device(value)
    if requested.type != "cuda":
        raise ValueError("本分析入口必须使用 CUDA，例如 --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但 CUDA 不可用")
    index = torch.cuda.current_device() if requested.index is None else int(requested.index)
    requested = torch.device("cuda:{}".format(index))
    torch.cuda.set_device(requested)
    return requested


def _write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def collect_from_checkpoint(args):
    """在 val 上跑 residual refiner，返回与 --pred_file 相同结构的 dict（CPU 张量）。"""
    from torch.utils.data import DataLoader

    from data_loaders.forecasting.ntu_2p_diffusion import NTU2PDiffusionForecastDataset, ntu_2p_diffusion_collate
    from model.forecasting_ntu2p_residual_xyz import load_ntu2p_residual_refiner_checkpoint
    from model.rotation2xyz import Rotation2xyz_x
    from utils.fixseed import fixseed
    from utils.ntu_2p_rot6d import ntu_2p_rot6d_to_xyz

    device = _device(args.device)
    fixseed(args.seed)
    dataset = NTU2PDiffusionForecastDataset(
        manifest_path=args.manifest_path,
        split=args.split,
        train_h5_path=args.train_data_path,
        test_h5_path=args.test_data_path,
        window_len=args.obs_len + args.pred_len,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    # SMPL-X FK 的显存随 batch 线性增长（batch 16 峰值约 3 GB），默认 4 以便与训练任务共用显卡。
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=ntu_2p_diffusion_collate)
    model, _ = load_ntu2p_residual_refiner_checkpoint(args.checkpoint, device)
    converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    buffers = OrderedDict((key, []) for key in ("obs_xyz", "target_xyz", "model", "base", "actions"))
    meta = []
    with torch.no_grad():
        for batch in loader:
            obs_xyz = ntu_2p_rot6d_to_xyz(batch["obs_motion"].to(device), converter=converter)
            target_xyz = ntu_2p_rot6d_to_xyz(batch["future"].to(device), converter=converter)
            pred_xyz, base_xyz, _ = model(obs_xyz, batch["action"].to(device), return_details=True)
            for key, value in (("obs_xyz", obs_xyz), ("target_xyz", target_xyz), ("model", pred_xyz), ("base", base_xyz), ("actions", batch["action"])):
                buffers[key].append(value.detach().cpu())
            meta.extend(batch["meta"])
    print("peak_cuda_memory_mb={:.0f}".format(torch.cuda.max_memory_allocated(device) / 1e6))
    data = {key: torch.cat(value, dim=0) for key, value in buffers.items()}
    return {
        "obs_xyz": data["obs_xyz"],
        "target_xyz": data["target_xyz"],
        "methods": OrderedDict([("model", data["model"]), ("base", data["base"])]),
        "actions": data["actions"].reshape(-1),
        "meta": meta,
        "source": {"checkpoint": args.checkpoint, "split": args.split, "manifest_path": args.manifest_path},
    }


def load_pred_file(path, pred_name):
    data = torch.load(path, map_location="cpu")
    for key in ("obs_xyz", "target_xyz", "actions"):
        if key not in data:
            raise ValueError("{} 缺少字段 {}".format(path, key))
    if "methods" in data:
        methods = OrderedDict((str(k), v.float()) for k, v in data["methods"].items())
    elif "pred_xyz" in data:
        methods = OrderedDict([(pred_name, data["pred_xyz"].float())])
        if "base_xyz" in data:
            methods["base"] = data["base_xyz"].float()
    else:
        raise ValueError("{} 需要 pred_xyz 或 methods".format(path))
    count = int(data["obs_xyz"].shape[0])
    meta = data.get("meta") or [{"sample_id": "idx{:04d}".format(i)} for i in range(count)]
    return {
        "obs_xyz": data["obs_xyz"].float(),
        "target_xyz": data["target_xyz"].float(),
        "methods": methods,
        "actions": torch.as_tensor(data["actions"]).reshape(-1).long(),
        "meta": meta,
        "source": {"pred_file": os.path.abspath(path)},
    }


def slide_reference(obs_xyz, target_xyz):
    """GT root 轨迹 + obs 末帧局部姿态：root 走对了但四肢完全不动的"纯滑行"合成参考。"""
    last = obs_xyz[:, -1:]
    local = last - last[..., :1, :]
    return target_xyz[..., :1, :] + local


def build_variants(data):
    obs, target = data["obs_xyz"], data["target_xyz"]
    variants = OrderedDict(data["methods"])
    if "copy_last" not in variants:
        variants["copy_last"] = obs[:, -1:].expand(-1, target.shape[1], -1, -1, -1).contiguous()
    variants["slide_ref"] = slide_reference(obs, target)
    variants["GT"] = target
    return variants


def _fmt(value, digits=4):
    if value != value:  # nan
        return "nan"
    return "{:.{}f}".format(value, digits)


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(path, result, variants):
    names = list(variants)
    overall = result["overall"]
    lines = ["# NTU2P 自然度分析", ""]
    lines.append("来源：`{}`；样本数 {}；步行人窗口数 {}（GT future root 水平净位移 ≥ {} m）。".format(
        json.dumps(result["source"], ensure_ascii=False), result["num_samples"], int(result["num_walking_person_windows"]), WALK_DISPLACEMENT_THRESHOLD))
    lines.append("")
    lines.append("参考列：`GT` 为数据噪声下限，`slide_ref` 为 GT root + 冻结姿态的纯滑行合成参考。")
    lines.append("")
    lines.append("## 核心指标（全体窗口；`_walk` 后缀为 GT 步行窗口子集）")
    lines.append("")
    lines.append(_table(["metric"] + names, [[key] + [_fmt(overall[n][key]) for n in names] for key in SUMMARY_KEYS]))
    lines.append("")
    lines.append("## 步态（仅 GT 步行窗口，逐人计算后平均）")
    lines.append("")
    lines.append(_table(["metric"] + names, [[key] + [_fmt(overall[n][key]) for n in names] for key in GAIT_KEYS]))
    lines.append("")
    lines.append("比值（方法 / GT）：")
    lines.append("")
    ratio_keys = ("root_speed_mean", "leg_swing_amp", "swing_per_speed", "ankle_peak_speed", "stance_fraction", "step_count")
    lines.append(_table(["metric"] + names, [[key] + [_fmt(overall[n][key] / overall["GT"][key], 3) for n in names] for key in ratio_keys]))
    lines.append("")
    lines.append("## 分部位 × horizon 误差（m）")
    for kind in ("mpjpe", "local_mpjpe"):
        lines.append("")
        lines.append("### {}".format(kind))
        for name in names:
            if name == "GT":
                continue
            bins = sorted({re.search(r"_f(\d\d_\d\d)$", k).group(1) for k in overall[name] if re.search(r"^{}_\w+_f\d\d_\d\d$".format(kind), k) and "__count" not in k})
            header = ["{} / part".format(name), "all"] + ["f{}".format(b) for b in bins]
            rows = []
            for part in ("all",) + tuple(BODY_PARTS) + ("body22",):
                if part == "all":
                    total = overall[name][kind]
                    per_bin = [overall[name]["{}_all_f{}".format(kind, b)] for b in bins]
                elif part == "body22":
                    rows.append([part, _fmt(overall[name][kind + "_body22"])] + ["" for _ in bins])
                    continue
                else:
                    total = overall[name]["{}_{}".format(kind, part)]
                    per_bin = [overall[name]["{}_{}_f{}".format(kind, part, b)] for b in bins]
                rows.append([part, _fmt(total)] + [_fmt(v) for v in per_bin])
            if kind == "mpjpe":
                rows.append(["root"] + [""] + [_fmt(overall[name]["root_err_f{}".format(b)]) for b in bins])
            lines.append("")
            lines.append(_table(header, rows))
    lines.append("")
    lines.append("### GT 接触帧滑步速度随 horizon（m/s）")
    lines.append("")
    skate_bins = sorted(k for k in overall[names[0]] if k.startswith("skate_gt_contact_speed_f") and "__count" not in k)
    lines.append(_table(["method"] + [k.replace("skate_gt_contact_speed_", "") for k in skate_bins], [[n] + [_fmt(overall[n][k]) for k in skate_bins] for n in names]))
    lines.append("")
    lines.append("## 按动作分解（主方法：`{}`）".format(result["primary_method"]))
    lines.append("")
    primary = result["primary_method"]
    compare = [n for n in names if n not in ("GT", "slide_ref")]
    header = ["action", "name", "n", "walk_frac"] + ["mpjpe_{}".format(n) for n in compare] + [
        "body22_{}".format(primary), "local_legs_{}".format(primary), "skate_gtc_{}".format(primary), "skate_gtc_GT",
        "bone_rel_{}".format(primary), "heading_deg_{}".format(primary), "err_share"]
    rows = []
    for code, item in result["per_action"].items():
        m = item["metrics"]
        rows.append([code, item["name"], str(item["count"]), _fmt(item["walking_person_fraction"], 2)]
                    + [_fmt(m[n]["mpjpe"]) for n in compare]
                    + [_fmt(m[primary]["mpjpe_body22"]), _fmt(m[primary]["local_mpjpe_legs"]), _fmt(m[primary]["skate_gt_contact_speed"]),
                       _fmt(m["GT"]["skate_gt_contact_speed"]), _fmt(m[primary]["bone_rel_err_body"]), _fmt(m[primary]["heading_err_deg"], 1),
                       _fmt(item["error_share"], 3)])
    lines.append(_table(header, rows))
    lines.append("")
    lines.append("## 阈值与定义")
    lines.append("")
    for key, value in result["thresholds"].items():
        lines.append("- `{}`: {}".format(key, value))
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def analyze(args):
    if args.pred_file:
        data = load_pred_file(args.pred_file, args.pred_name)
    else:
        if not args.checkpoint:
            raise ValueError("需要 --checkpoint 或 --pred_file")
        data = collect_from_checkpoint(args)
    if args.save_arrays:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_arrays)), exist_ok=True)
        torch.save(data, args.save_arrays)
    variants = build_variants(data)
    obs, target, actions = data["obs_xyz"], data["target_xyz"], data["actions"]
    # 场景几何只由 obs + GT 决定，所有方法共用。
    frame = estimate_scene_frame(torch.cat((obs, target), dim=1))
    stats = OrderedDict((name, compute_naturalness_stats(value, target, obs, frame)) for name, value in variants.items())
    primary = next(iter(data["methods"]))

    overall = OrderedDict((name, aggregate_stats(s)) for name, s in stats.items())
    per_action = OrderedDict()
    primary_mpjpe = per_sample_values(stats[primary], ["mpjpe"])["mpjpe"]
    total_err = float(primary_mpjpe.sum().item())
    for action in sorted(set(actions.tolist())):
        index = (actions == action).nonzero(as_tuple=False).reshape(-1)
        code = "A{:03d}".format(int(action) + 1)
        metrics = OrderedDict()
        for name, s in stats.items():
            agg = aggregate_stats(s, index)
            metrics[name] = OrderedDict((key, agg[key]) for key in ACTION_KEYS)
        walk = aggregate_stats(stats["GT"], index)["walking_person_fraction"]
        per_action[code] = OrderedDict(
            [
                ("name", NTU2P_ACTION_NAMES[int(action)]),
                ("count", int(index.numel())),
                ("walking_person_fraction", walk),
                ("error_share", float(primary_mpjpe[index].sum().item()) / total_err if total_err > 0 else float("nan")),
                ("metrics", metrics),
            ]
        )

    thresholds = OrderedDict(
        [
            ("fps", 20),
            ("scene_frame", "up = mean(head - feet_mid) over obs+GT & both persons; per-person robust ridge plane on per-frame lowest foot joint; ground = 5% residual quantile"),
            ("contact_height_m", {str(k): v for k, v in CONTACT_HEIGHT_THRESHOLDS.items()}),
            ("contact_speed_mps", CONTACT_SPEED_THRESHOLD),
            ("smooth_kernel", list(SMOOTH_KERNEL)),
            ("walk_displacement_m", WALK_DISPLACEMENT_THRESHOLD),
            ("slide_root_speed_min_mps", SLIDE_ROOT_SPEED_MIN),
            ("slide_lock_tolerance", SLIDE_LOCK_TOLERANCE),
            ("step_hysteresis_m", STEP_HYSTERESIS),
            ("penetration_depth_m", PENETRATION_DEPTH),
            ("float_height_m", FLOAT_HEIGHT),
            ("horizon_bin_frames", HORIZON_BIN_FRAMES),
        ]
    )
    result = OrderedDict(
        [
            ("source", data["source"]),
            ("num_samples", int(obs.shape[0])),
            ("num_walking_person_windows", overall["GT"]["walking_person_fraction"] * 2 * int(obs.shape[0])),
            ("primary_method", primary),
            ("methods", list(variants)),
            ("thresholds", thresholds),
            ("overall", overall),
            ("per_action", per_action),
        ]
    )
    os.makedirs(args.output_dir, exist_ok=True)
    _write_json(os.path.join(args.output_dir, "naturalness.json"), result)
    write_markdown(os.path.join(args.output_dir, "naturalness.md"), result, variants)

    # 逐样本表：供选例、排序与后续新架构对照。
    scene = to_scene_coords(torch.cat((obs[:, -1:], target), dim=1), frame)
    gt_disp = (scene[:, -1, :, 0, :2] - scene[:, 0, :, 0, :2]).norm(dim=-1).amax(dim=-1)
    sample_keys = ("mpjpe", "mpjpe_body22", "skate_gt_contact_speed", "skate_gt_contact_speed_walk", "bone_rel_err_body", "lr_forward_vel_corr", "leg_swing_amp")
    per_sample = OrderedDict((name, per_sample_values(s, sample_keys)) for name, s in stats.items())
    with open(os.path.join(args.output_dir, "per_sample.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["index", "sample_id", "action_code", "gt_root_disp_max"]
        for name in variants:
            header += ["{}:{}".format(name, key) for key in sample_keys]
        writer.writerow(header)
        for i in range(int(obs.shape[0])):
            meta = data["meta"][i]
            row = [i, meta.get("sample_id", ""), meta.get("action_code", "A{:03d}".format(int(actions[i]) + 1)), "{:.4f}".format(float(gt_disp[i]))]
            for name in variants:
                row += ["{:.5f}".format(float(per_sample[name][key][i])) for key in sample_keys]
            writer.writerow(row)
    print(open(os.path.join(args.output_dir, "naturalness.md")).read())
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pred_file", default=None)
    parser.add_argument("--pred_name", default="model")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_arrays", default=None, help="保存 obs/target/methods/actions/meta，供审查图脚本与复现使用")
    parser.add_argument("--manifest_path", default="results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json")
    parser.add_argument("--train_data_path", default="dataset/ntu120/smplx/conditioned/xsub.train.h5")
    parser.add_argument("--test_data_path", default="dataset/ntu120/smplx/conditioned/xsub.test.h5")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


if __name__ == "__main__":
    analyze(build_arg_parser().parse_args())
