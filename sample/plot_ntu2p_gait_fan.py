"""起步者多样本的左右踝前后分离叠图（Track B 审查图 H 的补充）。

每个起步者案例画 s(t) = (左踝 − 右踝)·f（f 为该人 GT 位移的水平方向，与 gait_phase_stats 同口径）：
GT 黑线、v2 蓝线、mode F 的 K 个样本灰线、mode R 的 K 个样本橙色细线；圆点标出各曲线的先迈脚事件帧
（自身第一个 |s(t) − s_last| > 0.1 m 的帧）。用来直接看"各样本之间先迈脚与起步时机是否有区别"。
输入为 eval/eval_ntu2p_resdiff.py --export_review 写出的 review_samples.pt 与审查窗口 JSON。
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.ntu2p_canonical import estimate_up, horizontal
from utils.ntu2p_naturalness import GAIT_LEAD_DEVIATION, L_ANKLE, PELVIS, R_ANKLE
from utils.ntu2p_probabilistic_metrics import person_groups


COLORS = {"GT": "#111111", "v2": "#1F5FD0", "F": "#8C8C8C", "R": "#F28C28"}


def _separation(xyz, forward):
    """[...,T,55,3] 单人 -> [...,T]。"""
    return ((xyz[..., L_ANKLE, :] - xyz[..., R_ANKLE, :]) * forward).sum(-1)


def _event(curve, last):
    hit = np.nonzero(np.abs(curve - last) > GAIT_LEAD_DEVIATION)[0]
    return None if hit.size == 0 else int(hit[0])


def _walker(obs, target, action):
    """起步者中 GT 位移最大的人；没有起步者时取位移最大的人。"""
    groups = person_groups(obs, target, action)
    up = estimate_up(obs.double())
    distance = horizontal(target[:, -1, :, PELVIS].double() - obs[:, -1, :, PELVIS].double(), up).norm(dim=-1)[0]
    starting = groups["starting"][0]
    score = torch.where(starting, distance, distance - 1e3) if bool(starting.any()) else distance
    return int(torch.argmax(score).item())


def _setting_keys(samples, requested):
    if requested:
        return [key.strip() for key in requested.split(",") if key.strip()]
    keys = []
    for mode in ("F", "R"):
        first = next((key for key in samples if key.startswith(mode + ":")), None)
        if first is not None:
            keys.append(first)
    return keys


def draw_case(ax, obs, target, v2, per_setting, action, title):
    person = _walker(obs.unsqueeze(0), target.unsqueeze(0), action.view(1))
    up = estimate_up(obs.unsqueeze(0).double())
    forward = horizontal(target[-1, person, PELVIS].double().unsqueeze(0) - obs[-1, person, PELVIS].double().unsqueeze(0), up)
    forward = (forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-8))[0]
    obs_curve = _separation(obs[:, person].double(), forward).numpy()
    last = float(obs_curve[-1])
    t_obs = np.arange(-obs.shape[0] + 1, 1)
    t_future = np.arange(1, target.shape[0] + 1)
    ax.plot(t_obs, obs_curve, color=COLORS["GT"], linewidth=1.2, linestyle=":")
    for key, stacked in per_setting.items():
        mode = key.split(":")[0]
        curves = _separation(stacked[:, :, person].double(), forward).numpy()  # [K,T]
        for curve in curves:
            ax.plot(t_future, curve, color=COLORS[mode], linewidth=0.6 if mode == "R" else 0.8, alpha=0.55)
            event = _event(curve, last)
            if event is not None:
                ax.scatter([t_future[event]], [curve[event]], color=COLORS[mode], s=8, zorder=3)
    for name, value, width in (("v2", v2, 1.6), ("GT", target, 2.0)):
        curve = _separation(value[:, person].double(), forward).numpy()
        ax.plot(t_future, curve, color=COLORS[name], linewidth=width, label=name)
        event = _event(curve, last)
        if event is not None:
            ax.scatter([t_future[event]], [curve[event]], color=COLORS[name], s=30, marker="D", zorder=4)
    ax.axhline(last + GAIT_LEAD_DEVIATION, color="#CCCCCC", linewidth=0.6, linestyle="--")
    ax.axhline(last - GAIT_LEAD_DEVIATION, color="#CCCCCC", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="#DDDDDD", linewidth=0.8)
    ax.set_title("{} (P{})".format(title, "AB"[person]), fontsize=8)
    ax.set_xlabel("frame (0 = obs end)", fontsize=7)
    ax.set_ylabel("L−R ankle fwd sep (m)", fontsize=7)
    ax.tick_params(labelsize=7)


def main(args):
    data = torch.load(args.samples, map_location="cpu")
    with open(args.cases) as handle:
        cases = json.load(handle)["cases"]
    indices = [int(i) for i in data["indices"]]
    keys = _setting_keys(data["samples"], args.settings)
    starters = [case for case in cases if case["category"] == "starting" and int(case["index"]) in indices]
    if not starters:
        raise ValueError("审查窗口中没有起步者案例")
    os.makedirs(args.output_dir, exist_ok=True)
    columns = min(3, len(starters))
    rows = int(math.ceil(len(starters) / float(columns)))
    mosaic, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.6 * rows), squeeze=False)
    paths = []
    for number, case in enumerate(starters):
        pos = indices.index(int(case["index"]))
        per_setting = {key: data["samples"][key][pos].float() for key in keys}
        title = "case {:03d} {}".format(int(case["index"]), case.get("sample_id", ""))
        obs, target, v2 = data["obs_xyz"][pos].float(), data["target_xyz"][pos].float(), data["v2"][pos].float()
        action = torch.as_tensor(data["actions"][pos]).long()
        figure, ax = plt.subplots(figsize=(6.0, 4.0))
        draw_case(ax, obs, target, v2, per_setting, action, title)
        ax.legend(fontsize=7, loc="upper left", title="gray=F, orange=R: " + ", ".join(keys), title_fontsize=6)
        path = os.path.join(args.output_dir, "gait_fan_{:03d}_{}.png".format(int(case["index"]), case.get("sample_id", "")))
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
        draw_case(axes[number // columns][number % columns], obs, target, v2, per_setting, action, title)
    for number in range(len(starters), rows * columns):
        axes[number // columns][number % columns].axis("off")
    mosaic.suptitle("starters: s(t) fan — black GT, blue v2, gray mode F, orange mode R ({}); dots = lead-foot event".format(", ".join(keys)), fontsize=9)
    mosaic_path = os.path.join(args.output_dir, "gait_fan_mosaic.png")
    mosaic.savefig(mosaic_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(mosaic)
    print(json.dumps({"cases": paths, "mosaic": mosaic_path}, ensure_ascii=False))
    return paths + [mosaic_path]


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="eval_ntu2p_resdiff.py --export_review 写出的 review_samples.pt")
    parser.add_argument("--cases", required=True, help="审查窗口 JSON（--write_review_cases 的输出）")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--settings", default=None, help="逗号分隔的 mode:τ（缺省为文件中首个 F 与首个 R）")
    parser.add_argument("--dpi", type=int, default=90)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
