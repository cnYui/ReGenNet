"""腿部步态相位敏感损失（NTU2P v2 的 GL / GH 变体）。

诊断（A6 s0 EMA 终点，150 个 train batch，s2_5 配方）：
- 腿在共享主干梯度中只占约 3%（手指约 83%），其中约 60% 来自 local_velocity，而它对腿的梯度 97% 落在 >2 Hz 的
  拟合抖动带；dct_mid 幅度项与相位无关，它在腿上产生的 1.2–2 Hz 摆动与 GT 的余弦约为 0，误差是"腿不动"的 1.6 倍；
- NTU 步态在 ≤1 Hz（50 帧 DCT 的 k1–5，步频中位 0.51 Hz），观测已决定步态相位（只看腿的 MLP 在已在走的人上
  f1–20 相关 0.87–0.92），A6 却连前 0.5 s 都延续不了。
因此给腿换成有符号、罚相位的监督：局部位置（全频段）+ ≤2 Hz 低通轨迹的帧差速度；并把腿移出 dct_mid 幅度项。
dct_low 幅度项仍覆盖腿：起步者（步行人的 70%）的步态能量在 ≤1 Hz，需要它防止腿退化成均值。

定义与 scratchpad 分析 A（grad_share_parts.py 的 LEG_CAND=lpvel）逐式相同，copy-last 归一化常量可直接对照。
"""

from collections import OrderedDict

import torch

from utils.ntu2p_naturalness import BODY_PARTS, SMPLX_NUM_JOINTS
from utils.ntu_smplx_2p_xyz import dct_band_energies, dct_matrix, local_pose


LEG_JOINTS = tuple(BODY_PARTS["legs"])
NON_LEG_JOINTS = tuple(joint for joint in range(SMPLX_NUM_JOINTS) if joint not in LEG_JOINTS)
# k = 0..10，即 ≤2 Hz（50 帧窗口的第 k 个 DCT 基频率为 k × 0.2 Hz）。
DEFAULT_LEG_GAIT_LOWPASS_K = 11


def lowpass_time(value, num_coeffs):
    """沿时间（dim=1）投影到前 num_coeffs 个正交 DCT 基上。"""
    basis = dct_matrix(int(value.shape[1]), device=value.device, dtype=value.dtype)[: int(num_coeffs)]
    return torch.einsum("kt,ks,bs...->bt...", basis, basis, value)


def leg_local(value):
    """[B,T,2,55,3] -> 腿关节相对当帧 pelvis 的位置 [B,T,2,8,3]（坐标系与输入相同）。"""
    return local_pose(value)[:, :, :, list(LEG_JOINTS)]


def leg_gait_terms(pred, target, lowpass_k=DEFAULT_LEG_GAIT_LOWPASS_K):
    """返回 OrderedDict(leg_pos, leg_lpvel)。

    帧差只在 future 内部取、不含观测末帧：模型首帧恒等于观测末帧（ramp 首帧为 0），含它会引入不可消除的误差。
    """
    mse = torch.nn.functional.mse_loss
    pred_leg, target_leg = leg_local(pred), leg_local(target)
    pred_lp, target_lp = lowpass_time(pred_leg, lowpass_k), lowpass_time(target_leg, lowpass_k)
    return OrderedDict(
        [
            ("leg_pos", mse(pred_leg, target_leg)),
            ("leg_lpvel", mse(pred_lp[:, 1:] - pred_lp[:, :-1], target_lp[:, 1:] - target_lp[:, :-1])),
        ]
    )


def dct_mid_amplitude_masked(pred, target, joints=NON_LEG_JOINTS, eps=1e-6):
    """与 `_raw_terms` 的 dct_mid_amplitude 同式，只在 joints 上取均值；joints 取全部 55 个关节时两者相同。"""
    index = torch.as_tensor(joints, dtype=torch.long, device=pred.device)
    pred_energy = dct_band_energies(pred)["mid"].index_select(2, index)
    target_energy = dct_band_energies(target)["mid"].index_select(2, index)
    return torch.nn.functional.mse_loss(torch.sqrt(pred_energy + eps), torch.sqrt(target_energy + eps))


def gait_loss_terms(pred, target, args):
    """按开关返回 OrderedDict[name -> 原始值]；名称即 copy-last 归一化常量的键。开关全关时为空。"""
    terms = OrderedDict()
    if float(getattr(args, "leg_gait_loss_weight", 0.0)) > 0:
        terms.update(leg_gait_terms(pred, target, int(getattr(args, "leg_gait_lowpass_k", DEFAULT_LEG_GAIT_LOWPASS_K))))
    if getattr(args, "dct_mid_exclude_legs", False):
        terms["dct_mid_nonleg"] = dct_mid_amplitude_masked(pred, target, eps=float(getattr(args, "amplitude_eps", 1e-6)))
    return terms


def gait_loss_weights(args):
    """与 gait_loss_terms 同键的权重：两个腿部项共用 --leg_gait_loss_weight；dct_mid_nonleg 继承 dct_mid 的权重。"""
    weights = OrderedDict()
    if float(getattr(args, "leg_gait_loss_weight", 0.0)) > 0:
        weights["leg_pos"] = float(args.leg_gait_loss_weight)
        weights["leg_lpvel"] = float(args.leg_gait_loss_weight)
    if getattr(args, "dct_mid_exclude_legs", False):
        weights["dct_mid_nonleg"] = float(args.dct_mid_amplitude_loss_weight)
    return weights


__all__ = [
    "DEFAULT_LEG_GAIT_LOWPASS_K",
    "LEG_JOINTS",
    "NON_LEG_JOINTS",
    "dct_mid_amplitude_masked",
    "gait_loss_terms",
    "gait_loss_weights",
    "leg_gait_terms",
    "leg_local",
    "lowpass_time",
]
