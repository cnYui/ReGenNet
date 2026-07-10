import argparse
import csv
import json
import os
from collections import OrderedDict
from datetime import datetime

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch


COLORS = OrderedDict(
    [
        ("observed", "#2F6BFF"),
        ("generated", "#FF7F0E"),
        ("real", "#2CA02C"),
    ]
)


def _utc_now():
    return datetime.utcnow().isoformat() + "Z"


def _json_ready(value):
    if isinstance(value, OrderedDict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _write_json(path, value):
    with open(path, "w") as f:
        json.dump(_json_ready(value), f, indent=2, sort_keys=False, ensure_ascii=False)


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _prepare_save_dir(args):
    if os.path.exists(args.save_dir):
        has_files = len(os.listdir(args.save_dir)) > 0
        if has_files and not args.overwrite:
            raise FileExistsError("save_dir 已存在，使用 --overwrite: {}".format(args.save_dir))
        if has_files and args.overwrite:
            for root, _, files in os.walk(args.save_dir):
                for filename in files:
                    os.remove(os.path.join(root, filename))
    else:
        os.makedirs(args.save_dir)
    for name in ("videos", "frames", "arrays"):
        os.makedirs(os.path.join(args.save_dir, name), exist_ok=True)


def _load_edges(path, body_only):
    data = np.load(path, allow_pickle=True)
    if "kintree_table" not in data.files:
        raise ValueError("{} 缺少 kintree_table".format(path))
    kintree = data["kintree_table"]
    max_joint = 22 if bool(body_only) else 55
    edges = []
    for child in range(1, min(int(kintree.shape[1]), max_joint)):
        parent = int(kintree[0, child])
        if 0 <= parent < max_joint:
            edges.append((parent, child))
    if not edges:
        raise ValueError("没有可绘制的 skeleton edges")
    return edges


def _resolve_metrics_path(source_dir, metrics_filename):
    if metrics_filename is not None:
        path = (
            metrics_filename
            if os.path.isabs(metrics_filename)
            else os.path.join(source_dir, metrics_filename)
        )
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    for filename in ("metrics_val.json", "metrics_test.json"):
        path = os.path.join(source_dir, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("未找到 metrics_val.json 或 metrics_test.json: {}".format(source_dir))


def _load_source(source_dir, metrics_filename):
    array_path = os.path.join(source_dir, "arrays", "ntu_label_xyz_samples.pt")
    metrics_path = _resolve_metrics_path(source_dir, metrics_filename)
    if not os.path.exists(array_path):
        raise FileNotFoundError(array_path)
    data = torch.load(array_path, map_location="cpu")
    with open(metrics_path) as f:
        metrics = json.load(f)
    required = ("obs_xyz", "target_xyz", "pred_xyz", "copy_last_xyz", "actions", "meta")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("数组文件缺少字段: {}".format(missing))
    return data, metrics, metrics_path


def _ensure_xyz(name, value, seq_len):
    if value.ndim != 5:
        raise ValueError("{} 必须是 [N,T,2,55,3]，当前维度数 {}".format(name, value.ndim))
    if tuple(value.shape[1:]) != (seq_len, 2, 55, 3):
        raise ValueError("{} shape 异常: {}".format(name, tuple(value.shape)))
    if not np.isfinite(value).all():
        raise ValueError("{} 存在非有限数值".format(name))


def _display_xyz(value, flip_z_axis):
    display = np.array(value, dtype=np.float32, copy=True)
    if bool(flip_z_axis):
        display[..., 1] = -display[..., 1]
    return display


def _axis_limits(*motions):
    data = np.concatenate([motion.reshape(-1, 3) for motion in motions], axis=0)
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins)) * 0.58
    if radius < 1e-6:
        radius = 1.0
    return center, radius


def _draw_person(ax, pose, edges, color, linewidth, alpha):
    for parent, child in edges:
        points = pose[[parent, child]]
        ax.plot(
            points[:, 0],
            points[:, 2],
            points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    root = pose[0]
    ax.scatter([root[0]], [root[2]], [root[1]], color=color, s=14, alpha=alpha)


def _draw_two_person(ax, frame_xyz, edges, color, linewidth, alpha):
    _draw_person(ax, frame_xyz[0], edges, color, linewidth, alpha)
    _draw_person(ax, frame_xyz[1], edges, color, linewidth, alpha)


def _setup_axis(ax, center, radius, title):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-70)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title(title, fontsize=9)
    handles = [
        Line2D([0], [0], color=COLORS["observed"], lw=3, label="Input obs20"),
        Line2D([0], [0], color=COLORS["generated"], lw=3, label="Generated future40"),
        Line2D([0], [0], color=COLORS["real"], lw=3, label="Real future40"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92)


def _draw_frame(ax, frame_idx, obs_xyz, pred_xyz, target_xyz, edges, center, radius, title_prefix):
    total = int(obs_xyz.shape[0] + target_xyz.shape[0])
    _setup_axis(ax, center, radius, "{} | frame {}/{}".format(title_prefix, frame_idx + 1, total))
    if frame_idx < obs_xyz.shape[0]:
        _draw_two_person(ax, obs_xyz[frame_idx], edges, COLORS["observed"], 2.5, 0.96)
    else:
        future_idx = frame_idx - obs_xyz.shape[0]
        _draw_two_person(ax, pred_xyz[future_idx], edges, COLORS["generated"], 2.3, 0.9)
        _draw_two_person(ax, target_xyz[future_idx], edges, COLORS["real"], 2.1, 0.9)


def _render_video(video_path, frame_path, obs_xyz, pred_xyz, target_xyz, edges, title_prefix, fps, dpi):
    center, radius = _axis_limits(obs_xyz, pred_xyz, target_xyz)
    total = int(obs_xyz.shape[0] + target_xyz.shape[0])
    fig = plt.figure(figsize=(6.4, 6.4), dpi=int(dpi))
    ax = fig.add_subplot(111, projection="3d")
    writer = imageio.get_writer(video_path, fps=int(fps))
    try:
        for frame_idx in range(total):
            ax.clear()
            _draw_frame(ax, frame_idx, obs_xyz, pred_xyz, target_xyz, edges, center, radius, title_prefix)
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            if frame_idx == 0:
                imageio.imwrite(frame_path, image)
            writer.append_data(image)
    finally:
        writer.close()
        plt.close(fig)


def _case_metrics(pred_xyz, target_xyz, copy_xyz):
    pred_diff = pred_xyz - target_xyz
    copy_diff = copy_xyz - target_xyz
    return {
        "model_xyz_mse": float(np.mean(pred_diff * pred_diff)),
        "model_xyz_mae": float(np.mean(np.abs(pred_diff))),
        "copy_last_xyz_mse": float(np.mean(copy_diff * copy_diff)),
        "copy_last_xyz_mae": float(np.mean(np.abs(copy_diff))),
    }


def run_visualization(args):
    _prepare_save_dir(args)
    data, metrics, metrics_path = _load_source(args.source_dir, args.metrics_filename)
    obs = data["obs_xyz"].numpy().astype(np.float32)
    target = data["target_xyz"].numpy().astype(np.float32)
    pred = data["pred_xyz"].numpy().astype(np.float32)
    copy_last = data["copy_last_xyz"].numpy().astype(np.float32)
    _ensure_xyz("obs_xyz", obs, 20)
    _ensure_xyz("target_xyz", target, 40)
    _ensure_xyz("pred_xyz", pred, 40)
    edges = _load_edges(args.smplx_model, args.body_only)
    count = min(int(args.num_videos), int(obs.shape[0]))
    rows = []
    for index in range(count):
        meta = data["meta"][index]
        action_code = meta.get("action_code", "AUNK")
        base = "case{:04d}_{}".format(index, action_code)
        npz_path = os.path.join(args.save_dir, "arrays", base + ".npz")
        video_path = os.path.join(args.save_dir, "videos", base + ".mp4")
        frame_path = os.path.join(args.save_dir, "frames", base + "_first.png")
        case_metrics = _case_metrics(pred[index], target[index], copy_last[index])
        np.savez_compressed(
            npz_path,
            obs_xyz=obs[index],
            pred_xyz=pred[index],
            target_xyz=target[index],
            copy_last_xyz=copy_last[index],
            meta=json.dumps(meta, ensure_ascii=False),
        )
        title = "{} {} mse={:.4f} mae={:.4f}".format(
            base,
            meta.get("sample_id", ""),
            case_metrics["model_xyz_mse"],
            case_metrics["model_xyz_mae"],
        )
        obs_display = _display_xyz(obs[index], args.flip_z_axis)
        pred_display = _display_xyz(pred[index], args.flip_z_axis)
        target_display = _display_xyz(target[index], args.flip_z_axis)
        _render_video(
            video_path,
            frame_path,
            obs_display,
            pred_display,
            target_display,
            edges,
            title,
            args.fps,
            args.dpi,
        )
        rows.append(
            OrderedDict(
                [
                    ("index", index),
                    ("sample_id", meta.get("sample_id")),
                    ("action_code", action_code),
                    ("model_xyz_mse", case_metrics["model_xyz_mse"]),
                    ("model_xyz_mae", case_metrics["model_xyz_mae"]),
                    ("copy_last_xyz_mse", case_metrics["copy_last_xyz_mse"]),
                    ("copy_last_xyz_mae", case_metrics["copy_last_xyz_mae"]),
                    ("video_path", video_path),
                    ("frame_path", frame_path),
                    ("array_path", npz_path),
                    ("video_size", os.path.getsize(video_path)),
                    ("frame_size", os.path.getsize(frame_path)),
                ]
            )
        )

    _write_json(
        os.path.join(args.save_dir, "run_config.json"),
        OrderedDict(
            [
                ("created_at", _utc_now()),
                ("source_dir", args.source_dir),
                ("metrics_path", metrics_path),
                ("save_dir", args.save_dir),
                ("num_videos", count),
                ("fps", args.fps),
                ("dpi", args.dpi),
                ("colors", COLORS),
                ("visualization_boundary", "true two-person xyz skeleton [T,2,55,3]; blue obs, orange generated, green real"),
                ("flip_z_axis", bool(args.flip_z_axis)),
                ("metrics", metrics),
                ("body_only", bool(args.body_only)),
            ]
        ),
    )
    _write_json(os.path.join(args.save_dir, "selection.json"), rows)
    _write_csv(os.path.join(args.save_dir, "selection.csv"), rows)
    with open(os.path.join(args.save_dir, "summary.md"), "w") as f:
        f.write("# NTU Two-Person XYZ Tricolor Skeleton Videos\n\n")
        f.write("source_dir: `{}`\n\n".format(args.source_dir))
        f.write("颜色: 蓝色 input obs20，橙色 generated future40，绿色 real future40。\n\n")
        f.write("边界: 这是直接 xyz skeleton 双人可视化，不再使用单人 rot6d 转换。\n\n")
        f.write("| index | action | model_mse | copy_mse | model_mae | copy_mae | video |\n")
        f.write("|---:|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| {index} | {action_code} | {model_xyz_mse:.6f} | {copy_last_xyz_mse:.6f} | {model_xyz_mae:.6f} | {copy_last_xyz_mae:.6f} | `{video_path}` |\n".format(
                    **row
                )
            )
    print("Two-person xyz visualization finished. save_dir={}".format(args.save_dir))
    print("videos={}".format(count))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--metrics_filename", default=None)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--num_videos", type=int, default=8)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--smplx_model", default="body_models/smplx/SMPLX_NEUTRAL.npz")
    parser.add_argument("--body_only", action="store_true", default=True)
    parser.add_argument("--full_body", dest="body_only", action="store_false")
    parser.add_argument("--flip_z_axis", action="store_true", default=True)
    parser.add_argument("--no_flip_z_axis", dest="flip_z_axis", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    run_visualization(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
