"""汇总同 seed 候选的完整轨迹，固定全部推人曲线与受试者簇 bootstrap。

仅用于 H-val 诊断与可复用审查产物，不改变预登记采纳判据，也不读取 test。
"""

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
import torch

from utils.ntu2p_canonical import estimate_up, horizontal
from utils.ntu2p_contact_graph import hand_body_distances
from utils.ntu2p_naturalness import NTU2P_FPS, smooth_time


def load_candidates(directory, seed):
    paths = sorted(Path(directory).glob("*_s{}_val_arrays.pt".format(seed)))
    if not paths:
        raise ValueError("没有完整的 val 候选数组")
    first = torch.load(str(paths[0]), map_location="cpu")
    methods = OrderedDict([("FD2", first["methods"]["prior"])])
    for path in paths:
        data = torch.load(str(path), map_location="cpu")
        if data.get("source", {}).get("split") != "val":
            raise ValueError("只接受 val 数组")
        for key in ("obs_xyz", "target_xyz", "actions"):
            if not torch.equal(data[key], first[key]):
                raise ValueError("候选 {} 数据或顺序不一致".format(path))
        if data["meta"] != first["meta"] or data["source"]["manifest_path"] != first["source"]["manifest_path"]:
            raise ValueError("候选 manifest/meta 不一致")
        if not torch.allclose(data["methods"]["prior"], methods["FD2"], atol=1e-5, rtol=0):
            raise ValueError("候选的原 FD2 reference 不一致")
        name = path.name.split("_s")[0]
        methods[name] = data["methods"]["model"]
    first["methods"] = methods
    first["source"]["candidate_arrays"] = [str(path) for path in paths]
    return first


def paired_subject_bootstrap(data, replicates, seed):
    target = data["target_xyz"].double()
    errors = OrderedDict((name, (value.double() - target).norm(dim=-1).mean(dim=(1, 2, 3)).numpy())
                         for name, value in data["methods"].items())
    subjects = [re.search(r"P(\d+)", row["sample_id"]).group(1) for row in data["meta"]]
    unique = sorted(set(subjects))
    clusters = [np.array([i for i, subject in enumerate(subjects) if subject == person]) for person in unique]
    rng = np.random.RandomState(seed)
    # 受试者是留出划分的单位，重复同人的序列不能被当成独立受试者。
    draws = [np.concatenate([clusters[j] for j in rng.randint(len(clusters), size=len(clusters))]) for _ in range(replicates)]
    result = dict(scope="H-val 受试者簇配对bootstrap，仅诊断，不替代三seed复现", subjects=len(unique),
                  windows=len(subjects), replicates=replicates, seed=seed, candidates={})
    reference = errors["FD2"]
    for name, error in errors.items():
        if name == "FD2":
            continue
        changes = np.array([error[ids].mean() / reference[ids].mean() - 1.0 for ids in draws])
        result["candidates"][name] = dict(mpjpe=float(error.mean()),
                                          delta_mpjpe_percent=float(100 * (error.mean() / reference.mean() - 1.0)),
                                          cluster_bootstrap_95_percent=(100 * np.quantile(changes, [.025, .975])).tolist())
    return result


def plot_contacts(data, destination):
    font_path = findfont(FontProperties(family="Noto Sans CJK JP"), fallback_to_default=False)
    font = FontProperties(fname=font_path)
    obs, target = data["obs_xyz"].double(), data["target_xyz"].double()
    pushing = torch.nonzero(data["actions"].reshape(-1) == 2, as_tuple=False).flatten().tolist()
    if not pushing:
        return
    methods = OrderedDict([("GT", target)] + [(name, value.double()) for name, value in data["methods"].items()])
    palette = plt.get_cmap("tab10")
    colors = dict((name, palette(i)) for i, name in enumerate(methods))
    colors.update(GT="#111111", FD2="#888888")
    up = estimate_up(obs)
    axis = horizontal(obs[:, -1, 1, 0] - obs[:, -1, 0, 0], up)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    time = np.arange(target.shape[1]) / NTU2P_FPS
    rows = (len(pushing) + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(19, 3.5 * rows), dpi=120, squeeze=False)
    for row, index in enumerate(pushing):
        dist_ax, vel_ax = axes[row // 2, (row % 2) * 2:(row % 2) * 2 + 2]
        for name, value in methods.items():
            full = torch.cat((obs[index:index + 1], value[index:index + 1]), dim=1)
            distance = smooth_time(hand_body_distances(full)[:, :, 0].amin(dim=(-1, -2)))[0, obs.shape[1]:]
            roots = smooth_time(full[:, :, 1, 0])
            velocity = ((roots[:, 1:] - roots[:, :-1]) * axis[index:index + 1, None]).sum(-1) * NTU2P_FPS
            dist_ax.plot(time, distance.numpy(), color=colors[name], label=name, lw=1.5 if name == "GT" else 1.0)
            vel_ax.plot(time, velocity[0, obs.shape[1] - 1:].numpy(), color=colors[name], label=name, lw=1.5 if name == "GT" else 1.0)
        dist_ax.axhline(.2, color="#999999", ls=":", lw=.8)
        vel_ax.axhline(.1, color="#999999", ls=":", lw=.8)
        sid = data["meta"][index]["sample_id"]
        dist_ax.set_title("{:03d} {} 手→身体距离".format(index, sid), fontproperties=font, fontsize=9)
        vel_ax.set_title("B 沿观测 A→B 方向速度", fontproperties=font, fontsize=9)
        dist_ax.set_ylabel("m")
        vel_ax.set_ylabel("m/s")
        for ax in (dist_ax, vel_ax):
            ax.grid(alpha=.2)
            ax.set_xlabel("未来时间 / s", fontproperties=font)
            ax.tick_params(labelsize=8)
    for empty in range(len(pushing), rows * 2):
        for ax in axes[empty // 2, (empty % 2) * 2:(empty % 2) * 2 + 2]:
            ax.set_visible(False)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle("全部推人中心窗；六段掌腕代理；图关系不保证真实物理因果", fontproperties=font, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(str(destination))
    plt.close(fig)


def main(args):
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap次数必须正数")
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("输出目录已存在，须使用新目录保留历史")
    data = load_candidates(args.results_dir, args.model_seed)
    output.mkdir(parents=True)
    torch.save(data, str(output / "review_arrays.pt"))
    result = paired_subject_bootstrap(data, args.bootstrap_replicates, args.seed)
    (output / "subject_bootstrap.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    plot_contacts(data, output / "all_pushing_contact_retreat.png")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--bootstrap_replicates", type=int, default=3000)
    main(parser.parse_args())
