"""为 NTU 双人预测生成逐案例静态"审查图"，用看图判断腿是否在迈步、脚是否滑行、肢体是否变形。

输入 .pt（与 eval/analyze_ntu2p_naturalness.py --save_arrays 相同结构）：
obs_xyz [N,10,2,55,3]、target_xyz [N,50,2,55,3]、methods {name: [N,50,2,55,3]}（或 pred_xyz）、actions、meta。

每张图：
- 上半：GT 与各方法分行，固定时间点（默认第 10/20/30/40/50/60 帧，第 10 帧为观测末帧）列排；
  正交投影、固定相机（绕竖直轴旋转到观测末帧 A→B 连线为屏幕水平方向）、每案例固定坐标范围、地面网格；
  两人不同色系，左侧肢体深色、右侧浅色；地面上的小点是从观测末帧到当前帧的踝关节"足迹"，
  真实迈步表现为成簇的离散足迹，滑行表现为连续拖痕。
- 下半：俯视踝关节轨迹（颜色随时间渐变，实心点 = 该运动自身的接触帧）；
  "步行者"（GT 位移最大的人）左右踝高度、左右踝水平速度、左右踝前向分离（步幅信号），以及双人 root 速度。
可选 --video：复用 sample/visualize_ntu_label_xyz_tricolor.py 的三色视频（其默认 flip_z_axis）。
"""

import argparse
import json
import os
import random
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import torch

from utils.ntu2p_naturalness import (
    CONTACT_HEIGHT_THRESHOLDS,
    CONTACT_SPEED_THRESHOLD,
    L_ANKLE,
    PELVIS,
    R_ANKLE,
    SMPLX_BODY_JOINTS,
    SMPLX_PARENTS,
    compute_naturalness_stats,
    estimate_scene_frame,
    per_sample_values,
    smooth_time,
    to_scene_coords,
)

try:
    from eval.analyze_ntu2p_naturalness import NTU2P_ACTION_NAMES
except ImportError:  # 仅用于标题，缺失时退化为动作编号
    NTU2P_ACTION_NAMES = None

FPS = 20.0
LEFT_JOINTS = {1, 4, 7, 10, 13, 16, 18, 20}
RIGHT_JOINTS = {2, 5, 8, 11, 14, 17, 19, 21}
PERSON_COLORS = (
    {"left": "#1B3F8B", "right": "#7FB3E6", "center": "#2F6BFF", "cmap": "Blues"},
    {"left": "#9C2A1F", "right": "#F2A091", "center": "#E0452B", "cmap": "Reds"},
)
METHOD_LINE_STYLES = ("--", ":", "-.")
METHOD_COLORS = ("#FF7F0E", "#8C564B", "#9467BD")


def _load_arrays(path):
    data = torch.load(path, map_location="cpu")
    if "methods" in data:
        methods = OrderedDict((str(k), v.float()) for k, v in data["methods"].items())
    else:
        methods = OrderedDict([("model", data["pred_xyz"].float())])
    count = int(data["obs_xyz"].shape[0])
    meta = data.get("meta") or [{"sample_id": "idx{:04d}".format(i)} for i in range(count)]
    actions = torch.as_tensor(data["actions"]).reshape(-1).long()
    return data["obs_xyz"].float(), data["target_xyz"].float(), methods, actions, meta


def _action_label(action):
    code = "A{:03d}".format(int(action) + 1)
    if NTU2P_ACTION_NAMES is None:
        return code
    return "{} {}".format(code, NTU2P_ACTION_NAMES[int(action)])


def select_cases(obs, target, frame, num_random, num_walk, seed, indices):
    cases = []
    if indices:
        cases.extend(("idx", int(i)) for i in indices)
    count = int(obs.shape[0])
    if num_random > 0:
        picks = sorted(random.Random(int(seed)).sample(range(count), min(int(num_random), count)))
        cases.extend(("rand", i) for i in picks)
    if num_walk > 0:
        scene = to_scene_coords(torch.cat((obs[:, -1:], target), dim=1), frame)
        disp = (scene[:, -1, :, PELVIS, :2] - scene[:, 0, :, PELVIS, :2]).norm(dim=-1).amax(dim=-1)
        order = torch.argsort(disp, descending=True)[: int(num_walk)].tolist()
        cases.extend(("walk", int(i)) for i in order)
    return cases


def _display_coords(full_xyz, frame, index):
    """场景旋转 + 公共地面：3D 展示需要两人在同一坐标系，地面高度取两人地面在各自足位处的均值。"""
    sub = {"basis": frame["basis"][index : index + 1], "plane": frame["plane"][index : index + 1], "offset": frame["offset"][index : index + 1]}
    rotated = torch.einsum("btpjc,bkc->btpjk", full_xyz.double(), sub["basis"])
    per_person = to_scene_coords(full_xyz, sub)
    ground = (rotated[..., 2] - per_person[..., 2])[..., [7, 8, 10, 11]].mean()
    rotated[..., 2] = rotated[..., 2] - ground
    return rotated[0].numpy()  # [T,2,55,3]


def _self_contact(scene_full, obs_len):
    """该运动自身的接触帧（高度 + 平滑速度），用于俯视图实心点；[T_future,2,2]（L/R 踝）。"""
    smoothed = smooth_time(scene_full)
    velocity = (smoothed[:, obs_len:, ..., :2] - smoothed[:, obs_len - 1 : -1, ..., :2]) * FPS
    ankles = [L_ANKLE, R_ANKLE]
    speed = velocity[..., ankles, :].norm(dim=-1)[0]
    height = smoothed[:, obs_len:, :, ankles, 2][0]
    low = height < CONTACT_HEIGHT_THRESHOLDS[L_ANKLE]
    return (low & (speed < CONTACT_SPEED_THRESHOLD)).numpy(), speed.numpy(), height.numpy(), smoothed[0].numpy()


# 俯角：正交投影 screen_y = z*cos + depth*sin，给地面网格一点纵深又不压扁人体。
VIEW_ELEVATION_DEG = 15.0


def _project(points):
    """正交投影到屏幕：x 保持（A→B 连线方向），y 为高度与深度的混合；points [...,3] -> [...,2]。"""
    elev = np.radians(VIEW_ELEVATION_DEG)
    return np.stack((points[..., 0], points[..., 2] * np.cos(elev) + points[..., 1] * np.sin(elev)), axis=-1)


def _draw_skeleton(ax, pose, person):
    colors = PERSON_COLORS[person]
    screen = _project(pose)
    # 右侧先画、左侧后画，左侧深色线条压在上面便于区分左右腿。
    order = sorted(range(1, SMPLX_BODY_JOINTS), key=lambda j: 0 if j in RIGHT_JOINTS else 1)
    for child in order:
        parent = SMPLX_PARENTS[child]
        side = "left" if child in LEFT_JOINTS else "right" if child in RIGHT_JOINTS else "center"
        seg = screen[[parent, child]]
        ax.plot(seg[:, 0], seg[:, 1], color=colors[side], linewidth=2.0 if side != "right" else 1.8, solid_capstyle="round", zorder=3)


def _draw_ground(ax, limits, step=0.5):
    (x0, x1), (y0, y1) = limits[0], limits[1]
    for x in np.arange(np.ceil(x0 / step) * step, x1 + 1e-6, step):
        seg = _project(np.array([[x, y0, 0.0], [x, y1, 0.0]]))
        ax.plot(seg[:, 0], seg[:, 1], color="#CCCCCC", linewidth=0.6, zorder=1)
    for y in np.arange(np.ceil(y0 / step) * step, y1 + 1e-6, step):
        seg = _project(np.array([[x0, y, 0.0], [x1, y, 0.0]]))
        ax.plot(seg[:, 0], seg[:, 1], color="#CCCCCC", linewidth=0.6, zorder=1)


def _case_limits(motions):
    """x 轴已对齐 A→B 连线（屏幕水平方向），y 为视线深度方向；深度范围只需容纳人体，不必与 x 等宽。"""
    stacked = np.concatenate([m[..., :SMPLX_BODY_JOINTS, :].reshape(-1, 3) for m in motions], axis=0)
    lo, hi = stacked.min(0), stacked.max(0)
    x_half = max(float(hi[0] - lo[0]) / 2.0 + 0.2, 0.9)
    y_half = max(float(hi[1] - lo[1]) / 2.0 + 0.2, 0.6)
    x_mid, y_mid = (lo[:2] + hi[:2]) / 2.0
    top = max(float(hi[2]) + 0.1, 1.9)
    return ((x_mid - x_half, x_mid + x_half), (y_mid - y_half, y_mid + y_half), (-0.05, top))


def _align_to_pair_axis(motions, obs_len):
    """绕竖直轴旋转，使观测末帧 GT 的 A→B 水平方向成为 +x，这样所有子图从同一侧面观察两人。"""
    gt = next(iter(motions.values()))
    ab = gt[obs_len - 1, 1, PELVIS, :2] - gt[obs_len - 1, 0, PELVIS, :2]
    theta = float(np.arctan2(ab[1], ab[0]))
    cos, sin = np.cos(-theta), np.sin(-theta)
    rot = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return OrderedDict((name, value @ rot.T) for name, value in motions.items())


def render_case(path, title, rows, obs_len, frames, frame, index, metrics_text, dpi):
    """rows: OrderedDict[name -> full 60 帧相机坐标 xyz [60,2,55,3] (torch)]，第一项必须是 GT。"""
    names = list(rows)
    display = OrderedDict((name, _display_coords(value.unsqueeze(0), frame, index)) for name, value in rows.items())
    display = _align_to_pair_axis(display, obs_len)
    sub = {"basis": frame["basis"][index : index + 1], "plane": frame["plane"][index : index + 1], "offset": frame["offset"][index : index + 1]}
    scene = OrderedDict((name, to_scene_coords(value.unsqueeze(0), sub)) for name, value in rows.items())
    contact = OrderedDict((name, _self_contact(value, obs_len)) for name, value in scene.items())
    limits = _case_limits(list(display.values()))
    gt_scene = scene[names[0]][0].numpy()
    disp = np.linalg.norm(gt_scene[-1, :, PELVIS, :2] - gt_scene[obs_len - 1, :, PELVIS, :2], axis=-1)
    walker = int(np.argmax(disp))

    n_rows = len(names)
    n_cols = len(frames)
    corners = _project(np.array([[limits[0][i], limits[1][j], limits[2][k]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]))
    screen_lo, screen_hi = corners.min(0), corners.max(0)
    panel_w = 2.7
    panel_h = float(np.clip(panel_w * (screen_hi[1] - screen_lo[1]) / (screen_hi[0] - screen_lo[0]), 1.2, 3.2))
    fig = plt.figure(figsize=(panel_w * n_cols, panel_h * n_rows + 6.6), dpi=dpi)
    outer = gridspec.GridSpec(3, 1, height_ratios=[panel_h * n_rows, 3.3, 3.1], hspace=0.2, figure=fig)
    top = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer[0], wspace=0.02, hspace=0.02)
    for r, name in enumerate(names):
        motion = display[name]
        for c, frame_no in enumerate(frames):
            t = int(frame_no) - 1
            ax = fig.add_subplot(top[r, c])
            ax.set_xlim(screen_lo[0], screen_hi[0])
            ax.set_ylim(screen_lo[1], screen_hi[1])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#DDDDDD")
            _draw_ground(ax, limits)
            for person in range(2):
                colors = PERSON_COLORS[person]
                for joint, side in ((L_ANKLE, "left"), (R_ANKLE, "right")):
                    trail = motion[obs_len - 1 : t + 1, person, joint].copy()
                    trail[:, 2] = 0.0
                    trail = _project(trail)
                    ax.scatter(trail[:, 0], trail[:, 1], color=colors[side], s=4, alpha=0.8, zorder=2, edgecolors="none")
                _draw_skeleton(ax, motion[t, person], person)
            if r == 0:
                ax.set_title("frame {}{}".format(frame_no, " (obs end)" if t == obs_len - 1 else ""), fontsize=9)
            if c == 0:
                ax.text(0.02, 0.92, name, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    # ---- 俯视轨迹
    mid = gridspec.GridSpecFromSubplotSpec(1, n_rows, subplot_spec=outer[1], wspace=0.15)
    all_xy = np.concatenate([d[obs_len - 1 :, :, [PELVIS, L_ANKLE, R_ANKLE], :2].reshape(-1, 2) for d in display.values()], axis=0)
    xy_center = (all_xy.min(0) + all_xy.max(0)) / 2.0
    xy_half = max(float((all_xy.max(0) - all_xy.min(0)).max()) / 2.0 + 0.15, 0.4)
    for r, name in enumerate(names):
        ax = fig.add_subplot(mid[0, r])
        motion = display[name]
        mask = contact[name][0]
        steps = np.arange(motion.shape[0] - obs_len)
        for person in range(2):
            cmap = plt.get_cmap(PERSON_COLORS[person]["cmap"])
            root = motion[obs_len - 1 :, person, PELVIS, :2]
            ax.plot(root[:, 0], root[:, 1], color="#999999", linewidth=1.0)
            for side, joint, marker in ((0, L_ANKLE, "o"), (1, R_ANKLE, "^")):
                track = motion[obs_len:, person, joint, :2]
                colors = cmap(0.35 + 0.65 * steps / max(len(steps) - 1, 1))
                filled = mask[:, person, side]
                ax.plot(track[:, 0], track[:, 1], color=cmap(0.6), linewidth=0.6, alpha=0.6)
                ax.scatter(track[filled, 0], track[filled, 1], c=colors[filled], marker=marker, s=16, edgecolors="none")
                ax.scatter(track[~filled, 0], track[~filled, 1], facecolors="none", edgecolors=colors[~filled], marker=marker, s=12, linewidths=0.6)
        ax.set_xlim(xy_center[0] - xy_half, xy_center[0] + xy_half)
        ax.set_ylim(xy_center[1] - xy_half, xy_center[1] + xy_half)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        contact_rate = float(mask.mean())
        ax.set_title("{} top view: ankles (o=L, ^=R), filled=contact {:.0%}".format(name, contact_rate), fontsize=8)

    # ---- 时间曲线（步行者）
    bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], wspace=0.3)
    time = np.arange(1, motion.shape[0] - obs_len + 1)
    ax_h = fig.add_subplot(bottom[0, 0])
    ax_v = fig.add_subplot(bottom[0, 1])
    ax_s = fig.add_subplot(bottom[0, 2])
    ax_r = fig.add_subplot(bottom[0, 3])
    gt_dir = gt_scene[-1, walker, PELVIS, :2] - gt_scene[obs_len - 1, walker, PELVIS, :2]
    gt_dir = gt_dir / max(np.linalg.norm(gt_dir), 1e-6)
    for k, name in enumerate(names):
        style = "-" if k == 0 else METHOD_LINE_STYLES[(k - 1) % len(METHOD_LINE_STYLES)]
        width = 1.6 if k == 0 else 1.2
        _, speed, height, smoothed = contact[name]
        for side, color in ((0, PERSON_COLORS[walker]["left"]), (1, PERSON_COLORS[walker]["right"])):
            ax_h.plot(time, height[:, walker, side], linestyle=style, color=color, linewidth=width)
            ax_v.plot(time, speed[:, walker, side], linestyle=style, color=color, linewidth=width)
        rel = smoothed[obs_len:, walker, [L_ANKLE, R_ANKLE], :2] - smoothed[obs_len:, walker, PELVIS : PELVIS + 1, :2]
        separation = (rel[:, 0] - rel[:, 1]) @ gt_dir
        line_color = "#222222" if k == 0 else METHOD_COLORS[(k - 1) % len(METHOD_COLORS)]
        ax_s.plot(time, separation, linestyle=style, color=line_color, linewidth=width, label=name)
        root_v = np.linalg.norm(np.diff(smoothed[obs_len - 1 :, :, PELVIS, :2], axis=0), axis=-1) * FPS
        for person in range(2):
            ax_r.plot(time, root_v[:, person], linestyle=style, color=PERSON_COLORS[person]["center"], linewidth=width)
    ax_h.axhline(CONTACT_HEIGHT_THRESHOLDS[L_ANKLE], color="#AAAAAA", linewidth=0.8, linestyle=":")
    ax_v.axhline(CONTACT_SPEED_THRESHOLD, color="#AAAAAA", linewidth=0.8, linestyle=":")
    ax_h.set_title("walker P{} ankle height (m)  dark=L light=R".format("AB"[walker]), fontsize=8)
    ax_v.set_title("walker P{} ankle horiz. speed (m/s)".format("AB"[walker]), fontsize=8)
    ax_s.set_title("walker P{} L-R ankle fwd separation (m)".format("AB"[walker]), fontsize=8)
    ax_r.set_title("root horiz. speed (m/s)  blue=A red=B", fontsize=8)
    ax_s.axhline(0.0, color="#AAAAAA", linewidth=0.8)
    style_handles = [Line2D([0], [0], color="#444444", linestyle="-" if k == 0 else METHOD_LINE_STYLES[(k - 1) % len(METHOD_LINE_STYLES)], label=name) for k, name in enumerate(names)]
    ax_r.legend(handles=style_handles, fontsize=7, loc="upper right")
    for ax in (ax_h, ax_v, ax_s, ax_r):
        ax.tick_params(labelsize=7)
        ax.set_xlabel("future frame", fontsize=7)
    legend = "A=blue, B=red; dark=left limbs, light=right limbs; ground dots = ankle footprints since obs end (clusters=steps, streaks=sliding)"
    fig.suptitle(title + "\n" + metrics_text + "\n" + legend, fontsize=10)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_video(path, obs, pred, target, title, dpi, obs_len, pred_len):
    from sample.visualize_ntu_label_xyz_tricolor import _display_xyz, _load_edges, _render_video

    edges = _load_edges("body_models/smplx/SMPLX_NEUTRAL.npz", True)
    labels = OrderedDict([("obs", "Input obs{}".format(obs_len)), ("generated", "Generated future{}".format(pred_len)), ("real", "Real future{}".format(pred_len))])
    frame_path = path.replace(".mp4", "_first.png")
    _render_video(path, frame_path, _display_xyz(obs, True), _display_xyz(pred, True), _display_xyz(target, True), edges, title, 20, dpi, labels)


def main(args):
    obs, target, methods, actions, meta = _load_arrays(args.arrays)
    show = [m.strip() for m in args.methods.split(",")] if args.methods else list(methods)
    for name in show:
        if name not in methods:
            raise ValueError("方法 {} 不在数组文件中，可选 {}".format(name, list(methods)))
    frames = [int(f) for f in args.frames.split(",")]
    obs_len = int(obs.shape[1])
    pred_len = int(target.shape[1])
    frame = estimate_scene_frame(torch.cat((obs, target), dim=1))
    indices = [int(i) for i in args.indices.split(",")] if args.indices else []
    cases = select_cases(obs, target, frame, args.num_random, args.num_walk, args.seed, indices)
    per_sample = OrderedDict()
    for name, value in [("GT", target)] + [(n, methods[n]) for n in show]:
        stats = compute_naturalness_stats(value, target, obs, frame)
        per_sample[name] = per_sample_values(stats, ("mpjpe", "skate_gt_contact_speed", "skate_gt_contact_speed_walk", "bone_rel_err_body", "leg_swing_amp", "lr_forward_vel_corr"))
    os.makedirs(args.output_dir, exist_ok=True)
    scene = to_scene_coords(torch.cat((obs[:, -1:], target), dim=1), frame)
    gt_disp = (scene[:, -1, :, PELVIS, :2] - scene[:, 0, :, PELVIS, :2]).norm(dim=-1)
    records = []
    for tag, index in cases:
        sample_id = meta[index].get("sample_id", "idx{:04d}".format(index))
        rows = OrderedDict([("GT", torch.cat((obs[index], target[index]), dim=0))])
        for name in show:
            rows[name] = torch.cat((obs[index], methods[name][index]), dim=0)
        parts = ["GT disp A/B {:.2f}/{:.2f} m".format(float(gt_disp[index, 0]), float(gt_disp[index, 1]))]
        for name in show:
            p = per_sample[name]
            parts.append("{}: mpjpe {:.3f}, skate@GTcontact {:.2f} m/s, bone {:.1%}".format(
                name, float(p["mpjpe"][index]), float(p["skate_gt_contact_speed"][index]), float(p["bone_rel_err_body"][index])))
        parts.append("GT skate@contact {:.2f} m/s".format(float(per_sample["GT"]["skate_gt_contact_speed"][index])))
        title = "[{}] case {:03d} {} {}".format(tag, index, sample_id, _action_label(actions[index]))
        filename = "{}_{:03d}_{}.png".format(tag, index, sample_id)
        path = os.path.join(args.output_dir, filename)
        render_case(path, title, rows, obs_len, frames, frame, index, " | ".join(parts), args.dpi)
        record = OrderedDict([("tag", tag), ("index", index), ("sample_id", sample_id), ("action", _action_label(actions[index])), ("png", path)])
        if args.video:
            video_path = path.replace(".png", "_{}.mp4".format(show[0]))
            render_video(video_path, obs[index].numpy(), methods[show[0]][index].numpy(), target[index].numpy(), title, args.video_dpi, obs_len, pred_len)
            record["video"] = video_path
        records.append(record)
        print(path)
    with open(os.path.join(args.output_dir, "cases.json"), "w") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", default=None, help="逗号分隔；默认数组文件中的全部方法，首个为主方法（用于视频）")
    parser.add_argument("--frames", default="10,20,30,40,50,60", help="60 帧窗口内的 1-based 帧号，第 10 帧是观测末帧")
    parser.add_argument("--num_random", type=int, default=12)
    parser.add_argument("--num_walk", type=int, default=6, help="按 GT future root 水平位移从大到小选取的案例数")
    parser.add_argument("--indices", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_dpi", type=int, default=100)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
