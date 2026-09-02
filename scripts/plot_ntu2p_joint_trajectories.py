"""画随机案例中人物 A 的手腕/脚踝相对 root 的轨迹：GT vs control vs s2_5。"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

CTRL = "results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000_visualization_random/source/arrays/ntu_label_xyz_samples.pt"
NEW = "results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s2_5_dct1_root01_s0_5000_visualization_random/source/arrays/ntu_label_xyz_samples.pt"
OUT = sys.argv[1]

JOINTS = [("L wrist", 20), ("R wrist", 21), ("L ankle", 7), ("R ankle", 8)]
AX = ["x", "y", "z"]


def local(x):  # x: [T,2,55,3] -> person A local pose [T,55,3]
    return x[:, 0] - x[:, 0, 0:1]


def main():
    c = torch.load(CTRL)
    n = torch.load(NEW)
    assert c["meta"] == n["meta"], "随机选例不一致"
    cases = [0, 3, 5]  # A010, A008, A008
    fig, axes = plt.subplots(len(cases) * 2, 4, figsize=(20, 4.2 * len(cases) * 2))
    for ci, case in enumerate(cases):
        obs = local(c["obs_xyz"][case])
        gt = local(c["target_xyz"][case])
        pc = local(c["pred_xyz"][case])
        pn = local(n["pred_xyz"][case])
        t_obs = list(range(-10, 0))
        t_fut = list(range(0, 50))
        for ji, (jname, j) in enumerate(JOINTS):
            for ai, coord in enumerate([0, 2]):  # x 与 z（前后/左右），y 竖直变化小
                ax = axes[ci * 2 + ai, ji]
                ax.plot(t_obs, obs[:, j, coord], color="#2F6BFF", lw=2, label="obs")
                ax.plot(t_fut, gt[:, j, coord], color="#2CA02C", lw=2, label="GT")
                ax.plot(t_fut, pc[:, j, coord], color="#999999", lw=1.6, ls="--", label="control")
                ax.plot(t_fut, pn[:, j, coord], color="#FF7F0E", lw=2, label="s2_5")
                ax.axvline(0, color="k", lw=0.5)
                ax.set_title("case{} {} | {} local {}".format(case, c["meta"][case].get("action_label", c["actions"][case].item()) if isinstance(c["meta"][case], dict) else "", jname, AX[coord]), fontsize=9)
                if ci == 0 and ai == 0 and ji == 0:
                    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=80)
    print("saved", OUT)


if __name__ == "__main__":
    main()
