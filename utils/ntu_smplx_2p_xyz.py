import math
from collections import OrderedDict

import torch

from model.rotation2xyz import Rotation2xyz_x


NTU_SMPLX_JOINTS_WITH_TRANS = 56
NTU_SMPLX_BODY_JOINTS = 55
NTU_2P_ROTVEC_FEATS = 6
NTU_NUM_PERSONS = 2
XYZ_COORD_DIM = 3
NTU_LEFT_WRIST = 20
NTU_RIGHT_WRIST = 21
NTU_DEFAULT_CONTACT_THRESHOLD = 0.15
NTU_INTERACTION_JOINT_PAIRS = (
    (0, NTU_LEFT_WRIST, 1, NTU_RIGHT_WRIST),
    (0, NTU_RIGHT_WRIST, 1, NTU_LEFT_WRIST),
    (0, NTU_LEFT_WRIST, 1, NTU_LEFT_WRIST),
    (0, NTU_RIGHT_WRIST, 1, NTU_RIGHT_WRIST),
)

NTU_XYZ_METRIC_KEYS = (
    "xyz_mse",
    "xyz_mae",
    "mpjpe",
    "first_step_error",
    "velocity_error",
    "acceleration_error",
    "root_translation_error",
    "local_pose_error",
    "short_xyz_mse",
    "mid_xyz_mse",
    "long_xyz_mse",
    "final_frame_error",
    "relative_root_distance_error",
    "relative_root_velocity_error",
    "inter_person_distance_consistency",
    "key_joint_relation_error",
    "contact_error",
    "contact_mask_ratio",
)


def check_ntu_motion(name, value, seq_len=None):
    if value.dim() != 4:
        raise ValueError("{} 必须是 [B,56,6,T]，当前维度数为 {}".format(name, value.dim()))
    if tuple(value.shape[1:3]) != (NTU_SMPLX_JOINTS_WITH_TRANS, NTU_2P_ROTVEC_FEATS):
        raise ValueError("{} 后三维前两项必须是 [56,6]，当前为 {}".format(name, tuple(value.shape[1:3])))
    if seq_len is not None and int(value.shape[-1]) != int(seq_len):
        raise ValueError("{} 时间长度必须是 {}，当前为 {}".format(name, int(seq_len), int(value.shape[-1])))
    if not torch.isfinite(value).all():
        raise ValueError("{} 存在非有限数值".format(name))


def check_ntu_xyz(name, value, seq_len=None, num_persons=NTU_NUM_PERSONS):
    if value.dim() != 5:
        raise ValueError("{} 必须是 [B,T,{},55,3]，当前维度数为 {}".format(name, int(num_persons), value.dim()))
    expected_tail = (int(num_persons), NTU_SMPLX_BODY_JOINTS, XYZ_COORD_DIM)
    if tuple(value.shape[2:]) != expected_tail:
        raise ValueError("{} 后三维必须是 {}，当前为 {}".format(name, expected_tail, tuple(value.shape[2:])))
    if seq_len is not None and int(value.shape[1]) != int(seq_len):
        raise ValueError("{} 时间长度必须是 {}，当前为 {}".format(name, int(seq_len), int(value.shape[1])))
    if not torch.isfinite(value).all():
        raise ValueError("{} 存在非有限数值".format(name))


def ntu_rotvec_2p_to_xyz(motion, device=None, converter=None):
    motion = torch.as_tensor(motion)
    check_ntu_motion("motion", motion)
    if device is None:
        device = motion.device
    device = torch.device(device)
    motion = motion.to(device=device, dtype=torch.float32)
    if converter is None:
        converter = Rotation2xyz_x(device=device, dataset="ntu120_2p")
    mask = torch.ones((motion.shape[0], motion.shape[-1]), dtype=torch.bool, device=device)
    xyz_cat = converter(
        motion,
        mask=mask,
        pose_rep="rotvec",
        translation=True,
        glob=True,
        jointstype="smplx",
        vertstrans=True,
        num_person=NTU_NUM_PERSONS,
    )
    if tuple(xyz_cat.shape[1:3]) != (NTU_SMPLX_BODY_JOINTS, NTU_NUM_PERSONS * XYZ_COORD_DIM):
        raise ValueError("xyz_cat 必须是 [B,55,6,T]，当前为 {}".format(tuple(xyz_cat.shape)))
    xyz = torch.stack((xyz_cat[:, :, 0:3], xyz_cat[:, :, 3:6]), dim=2)
    xyz = xyz.permute(0, 4, 2, 1, 3).contiguous()
    check_ntu_xyz("xyz", xyz, seq_len=motion.shape[-1])
    return xyz


def copy_last_xyz(obs_xyz, pred_len):
    check_ntu_xyz("obs_xyz", obs_xyz)
    return obs_xyz[:, -1:].expand(-1, int(pred_len), -1, -1, -1).contiguous()


def horizon_slices(seq_len):
    seq_len = int(seq_len)
    if seq_len < 3:
        raise ValueError("seq_len 必须 >= 3")
    short_end = seq_len // 3
    mid_end = (seq_len * 2) // 3
    return slice(0, short_end), slice(short_end, mid_end), slice(mid_end, seq_len)


def _root_distance(value):
    return torch.norm(value[:, :, 0, 0] - value[:, :, 1, 0], dim=-1)


def _prepend_last_obs(pred_xyz, target_xyz, obs_xyz):
    pred_full = torch.cat((obs_xyz[:, -1:], pred_xyz), dim=1)
    target_full = torch.cat((obs_xyz[:, -1:], target_xyz), dim=1)
    return pred_full, target_full


def root_positions(value):
    return value[:, :, :, 0]


def local_pose(value):
    return value - root_positions(value).unsqueeze(3)


def velocity_with_last_obs(value, obs_xyz):
    full = torch.cat((obs_xyz[:, -1:], value), dim=1)
    return full[:, 1:] - full[:, :-1]


def acceleration_with_last_obs(value, obs_xyz):
    velocity = velocity_with_last_obs(value, obs_xyz)
    return velocity[:, 1:] - velocity[:, :-1]


def relative_root(value):
    return value[:, :, 0, 0] - value[:, :, 1, 0]


def relative_root_velocity(value):
    rel = relative_root(value)
    return rel[:, 1:] - rel[:, :-1]


def interaction_pair_distances(value, joint_pairs=NTU_INTERACTION_JOINT_PAIRS):
    distances = []
    for person_a, joint_a, person_b, joint_b in joint_pairs:
        if int(person_a) >= int(value.shape[2]) or int(person_b) >= int(value.shape[2]):
            raise ValueError("person index 超出范围")
        if int(joint_a) >= int(value.shape[3]) or int(joint_b) >= int(value.shape[3]):
            raise ValueError("joint index 超出范围")
        diff = value[:, :, int(person_a), int(joint_a)] - value[:, :, int(person_b), int(joint_b)]
        distances.append(torch.norm(diff, dim=-1))
    if not distances:
        raise ValueError("joint_pairs 不能为空")
    return torch.stack(distances, dim=-1)


def masked_contact_mse(pred_dist, target_dist, threshold=NTU_DEFAULT_CONTACT_THRESHOLD):
    mask = (target_dist < float(threshold)).float()
    denom = mask.sum().clamp_min(1.0)
    loss = (((pred_dist - target_dist) ** 2) * mask).sum() / denom
    return loss, mask.mean()


def masked_contact_l1(pred_dist, target_dist, threshold=NTU_DEFAULT_CONTACT_THRESHOLD):
    mask = (target_dist < float(threshold)).float()
    denom = mask.sum().clamp_min(1.0)
    error = (torch.abs(pred_dist - target_dist) * mask).sum() / denom
    return error, mask.mean()


def _to_float(value):
    return float(value.detach().cpu().item())


def compute_ntu_xyz_metrics(pred_xyz, target_xyz, obs_xyz):
    check_ntu_xyz("pred_xyz", pred_xyz)
    check_ntu_xyz("target_xyz", target_xyz, seq_len=pred_xyz.shape[1])
    check_ntu_xyz("obs_xyz", obs_xyz)
    if tuple(pred_xyz.shape) != tuple(target_xyz.shape):
        raise ValueError("pred_xyz/target_xyz shape 必须一致")

    diff = pred_xyz - target_xyz
    dist = torch.norm(diff, dim=-1)
    pred_full, target_full = _prepend_last_obs(pred_xyz, target_xyz, obs_xyz)
    pred_vel = pred_full[:, 1:] - pred_full[:, :-1]
    target_vel = target_full[:, 1:] - target_full[:, :-1]
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    target_acc = target_vel[:, 1:] - target_vel[:, :-1]
    pred_root_dist = _root_distance(pred_xyz)
    target_root_dist = _root_distance(target_xyz)
    pred_full_root_dist = _root_distance(pred_full)
    target_full_root_dist = _root_distance(target_full)
    inter_delta = torch.abs(
        (pred_full_root_dist[:, 1:] - pred_full_root_dist[:, :-1])
        - (target_full_root_dist[:, 1:] - target_full_root_dist[:, :-1])
    )
    pred_rel_vel = relative_root_velocity(pred_full)
    target_rel_vel = relative_root_velocity(target_full)
    short_slice, mid_slice, long_slice = horizon_slices(pred_xyz.shape[1])
    pred_pair_dist = interaction_pair_distances(pred_xyz)
    target_pair_dist = interaction_pair_distances(target_xyz)
    contact_error, contact_ratio = masked_contact_l1(pred_pair_dist, target_pair_dist)

    metrics = OrderedDict()
    metrics["xyz_mse"] = _to_float((diff * diff).mean())
    metrics["xyz_mae"] = _to_float(torch.abs(diff).mean())
    metrics["mpjpe"] = _to_float(dist.mean())
    metrics["first_step_error"] = _to_float(torch.norm(pred_xyz[:, 0] - obs_xyz[:, -1], dim=-1).mean())
    metrics["velocity_error"] = _to_float(torch.norm(pred_vel - target_vel, dim=-1).mean())
    metrics["acceleration_error"] = _to_float(torch.norm(pred_acc - target_acc, dim=-1).mean())
    metrics["root_translation_error"] = _to_float(dist[:, :, :, 0].mean())
    metrics["local_pose_error"] = _to_float(torch.norm(local_pose(pred_xyz) - local_pose(target_xyz), dim=-1).mean())
    metrics["short_xyz_mse"] = _to_float((diff[:, short_slice] * diff[:, short_slice]).mean())
    metrics["mid_xyz_mse"] = _to_float((diff[:, mid_slice] * diff[:, mid_slice]).mean())
    metrics["long_xyz_mse"] = _to_float((diff[:, long_slice] * diff[:, long_slice]).mean())
    metrics["final_frame_error"] = _to_float(torch.norm(pred_xyz[:, -1] - target_xyz[:, -1], dim=-1).mean())
    metrics["relative_root_distance_error"] = _to_float(torch.abs(pred_root_dist - target_root_dist).mean())
    metrics["relative_root_velocity_error"] = _to_float(torch.norm(pred_rel_vel - target_rel_vel, dim=-1).mean())
    metrics["inter_person_distance_consistency"] = _to_float(inter_delta.mean())
    metrics["key_joint_relation_error"] = _to_float(torch.abs(pred_pair_dist - target_pair_dist).mean())
    metrics["contact_error"] = _to_float(contact_error)
    metrics["contact_mask_ratio"] = _to_float(contact_ratio)

    if tuple(metrics.keys()) != NTU_XYZ_METRIC_KEYS:
        raise AssertionError("NTU xyz metrics key 不稳定")
    for key, value in metrics.items():
        if not torch.isfinite(torch.tensor(float(value))):
            raise ValueError("{} 指标为非有限数值: {}".format(key, value))
    return metrics


# 约为 val GT 局部姿态帧间位移中位数（0.00107 m）的 10%，固定为常量以便跨 run 比较。
NTU_ARTICULATION_FROZEN_EPSILON = 0.001

NTU_ARTICULATION_METRIC_KEYS = (
    "articulation_energy",
    "articulation_energy_target",
    "root_energy",
    "root_energy_target",
    "local_pose_temporal_std",
    "local_pose_temporal_std_target",
    "frozen_ratio",
    "frozen_ratio_target",
    "root_mse",
    "local_mse",
    "dct_low_energy",
    "dct_low_energy_target",
    "dct_mid_energy",
    "dct_mid_energy_target",
    "dct_high_energy",
    "dct_high_energy_target",
)

# T=50 @ 20 FPS 时 DCT 系数 k 对应 k*0.2 Hz：low=k1-5(<=1Hz) 慢速形变，mid=k6-10(1.2-2Hz) 步态/手势，high=k>=11 多为拟合抖动。
NTU_DCT_BAND_EDGES = (1, 6, 11)


def local_pose_velocity(value):
    pose = local_pose(value)
    return pose[:, 1:] - pose[:, :-1]


def dct_matrix(seq_len, device=None, dtype=torch.float32):
    """正交 DCT-II 矩阵 [k, t]。"""
    seq_len = int(seq_len)
    k = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    t = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(0)
    matrix = torch.cos(math.pi * (t + 0.5) * k / seq_len) * math.sqrt(2.0 / seq_len)
    matrix[0] = matrix[0] / math.sqrt(2.0)
    return matrix


def local_pose_dct(value):
    """局部姿态沿时间轴的 DCT 系数 [B, K, P, J, 3]。"""
    pose = local_pose(value)
    matrix = dct_matrix(pose.shape[1], device=pose.device, dtype=pose.dtype)
    return torch.einsum("kt,btpjc->bkpjc", matrix, pose)


def dct_band_energies(value, edges=NTU_DCT_BAND_EDGES):
    """帧差能量被 ω² 加权、由抖动主导；位置域 DCT 分频带能量 [B,P,J,3] 才能区分真实摆动与噪声。"""
    coeff = local_pose_dct(value)
    energy = coeff * coeff
    low_start, mid_start, high_start = (int(edge) for edge in edges)
    return OrderedDict(
        [
            ("low", energy[:, low_start:mid_start].sum(dim=1)),
            ("mid", energy[:, mid_start:high_start].sum(dim=1)),
            ("high", energy[:, high_start:].sum(dim=1)),
        ]
    )


def compute_ntu_articulation_metrics(pred_xyz, target_xyz, frozen_epsilon=NTU_ARTICULATION_FROZEN_EPSILON):
    """摆动幅度类指标：`xyz_mse/xyz_mae/mpjpe` 对'关节是否真的在动'不敏感，需单独报告。"""
    check_ntu_xyz("pred_xyz", pred_xyz)
    check_ntu_xyz("target_xyz", target_xyz, seq_len=pred_xyz.shape[1])
    if tuple(pred_xyz.shape) != tuple(target_xyz.shape):
        raise ValueError("pred_xyz/target_xyz shape 必须一致")

    pred_lv = local_pose_velocity(pred_xyz)
    target_lv = local_pose_velocity(target_xyz)
    pred_rv = root_positions(pred_xyz)[:, 1:] - root_positions(pred_xyz)[:, :-1]
    target_rv = root_positions(target_xyz)[:, 1:] - root_positions(target_xyz)[:, :-1]

    metrics = OrderedDict()
    metrics["articulation_energy"] = _to_float((pred_lv * pred_lv).mean())
    metrics["articulation_energy_target"] = _to_float((target_lv * target_lv).mean())
    metrics["root_energy"] = _to_float((pred_rv * pred_rv).mean())
    metrics["root_energy_target"] = _to_float((target_rv * target_rv).mean())
    metrics["local_pose_temporal_std"] = _to_float(local_pose(pred_xyz).std(dim=1).mean())
    metrics["local_pose_temporal_std_target"] = _to_float(local_pose(target_xyz).std(dim=1).mean())
    metrics["frozen_ratio"] = _to_float((torch.norm(pred_lv, dim=-1) < float(frozen_epsilon)).float().mean())
    metrics["frozen_ratio_target"] = _to_float((torch.norm(target_lv, dim=-1) < float(frozen_epsilon)).float().mean())
    root_diff = root_positions(pred_xyz) - root_positions(target_xyz)
    local_diff = local_pose(pred_xyz) - local_pose(target_xyz)
    metrics["root_mse"] = _to_float((root_diff * root_diff).mean())
    metrics["local_mse"] = _to_float((local_diff * local_diff).mean())
    pred_bands = dct_band_energies(pred_xyz)
    target_bands = dct_band_energies(target_xyz)
    for band in ("low", "mid", "high"):
        metrics["dct_{}_energy".format(band)] = _to_float(pred_bands[band].mean())
        metrics["dct_{}_energy_target".format(band)] = _to_float(target_bands[band].mean())

    if tuple(metrics.keys()) != NTU_ARTICULATION_METRIC_KEYS:
        raise AssertionError("NTU articulation metrics key 不稳定")
    for key, value in metrics.items():
        if not torch.isfinite(torch.tensor(float(value))):
            raise ValueError("{} 指标为非有限数值: {}".format(key, value))
    return metrics


def articulation_ratios(aggregated):
    """由按样本数加权平均后的摆动指标计算相对 GT 的比值。"""
    ratios = OrderedDict()
    for key in (
        "articulation_energy",
        "root_energy",
        "local_pose_temporal_std",
        "dct_low_energy",
        "dct_mid_energy",
        "dct_high_energy",
    ):
        denominator = float(aggregated[key + "_target"])
        ratios[key + "_ratio_to_target"] = float(aggregated[key]) / denominator if denominator > 0 else float("nan")
    return ratios
