from collections import OrderedDict

import torch

from model.rotation2xyz import Rotation2xyz_x
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix


NTU_2P_NUM_JOINTS_WITH_TRANS = 56
NTU_2P_BODY_JOINTS = 55
NTU_2P_RAW_FEATS = 6
NTU_2P_ROT6D_FEATS = 12
NTU_2P_SINGLE_ROT6D_FEATS = 6
NTU_2P_NUM_PERSONS = 2
XYZ_COORD_DIM = 3
PERSON_A_ROT6D_SLICE = slice(0, 6)
PERSON_B_ROT6D_SLICE = slice(6, 12)
PERSON_A_TRANS_SLICE = slice(0, 3)
PERSON_B_TRANS_SLICE = slice(6, 9)
TRANSLATION_SLOT = 55
ROOT_JOINT = 0
NTU_2P_REPRESENTATION = "two_person_rot6d"
NTU_2P_PERSON_ORDER = "person_a_then_person_b_assumed"

NTU_2P_INTERACTION_LOSS_KEYS = (
    "joint_mse",
    "orient_mse",
    "trans_mse",
    "inter_loss",
)


def _shape_text(value):
    return tuple(value.shape)


def _ensure_tensor(name, value):
    if not torch.is_tensor(value):
        raise ValueError("{} 必须是 torch.Tensor，当前为 {}".format(name, type(value).__name__))
    return value


def _ensure_float_tensor(name, value):
    _ensure_tensor(name, value)
    if not value.dtype.is_floating_point:
        raise ValueError("{} dtype 必须是浮点类型，当前为 {}".format(name, value.dtype))
    return value


def _ensure_finite(name, value):
    if not torch.isfinite(value).all():
        raise ValueError("{} 存在 NaN 或 Inf".format(name))


def _check_device_match(name, value, ref):
    if value.device != ref.device:
        raise ValueError("{} device={} 必须等于 {}".format(name, value.device, ref.device))


def check_raw_ntu_2p_motion(value, seq_len=None):
    _ensure_float_tensor("raw_ntu_2p_motion", value)
    if value.dim() != 4:
        raise ValueError("raw NTU 双人 motion 必须是 [B,56,6,T]，当前维度数为 {}".format(value.dim()))
    expected = (NTU_2P_NUM_JOINTS_WITH_TRANS, NTU_2P_RAW_FEATS)
    if tuple(value.shape[1:3]) != expected:
        raise ValueError("raw NTU 双人 motion 中间维度必须是 {}，当前为 {}".format(expected, _shape_text(value)[1:3]))
    if seq_len is not None and int(value.shape[-1]) != int(seq_len):
        raise ValueError("raw NTU 双人 motion 时间长度必须是 {}，当前为 {}".format(int(seq_len), int(value.shape[-1])))
    _ensure_finite("raw_ntu_2p_motion", value)
    return value


def check_ntu_2p_rot6d(value, seq_len=None):
    _ensure_float_tensor("ntu_2p_rot6d", value)
    if value.dim() != 4:
        raise ValueError("canonical NTU 双人 rot6d 必须是 [B,56,12,T]，当前维度数为 {}".format(value.dim()))
    expected = (NTU_2P_NUM_JOINTS_WITH_TRANS, NTU_2P_ROT6D_FEATS)
    if tuple(value.shape[1:3]) != expected:
        raise ValueError("canonical NTU 双人 rot6d 中间维度必须是 {}，当前为 {}".format(expected, _shape_text(value)[1:3]))
    if seq_len is not None and int(value.shape[-1]) != int(seq_len):
        raise ValueError("canonical NTU 双人 rot6d 时间长度必须是 {}，当前为 {}".format(int(seq_len), int(value.shape[-1])))
    _ensure_finite("ntu_2p_rot6d", value)
    return value


def check_single_person_rot6d(value, seq_len=None, name="single_person_rot6d"):
    _ensure_float_tensor(name, value)
    if value.dim() != 4:
        raise ValueError("{} 必须是 [B,56,6,T]，当前维度数为 {}".format(name, value.dim()))
    expected = (NTU_2P_NUM_JOINTS_WITH_TRANS, NTU_2P_SINGLE_ROT6D_FEATS)
    if tuple(value.shape[1:3]) != expected:
        raise ValueError("{} 中间维度必须是 {}，当前为 {}".format(name, expected, _shape_text(value)[1:3]))
    if seq_len is not None and int(value.shape[-1]) != int(seq_len):
        raise ValueError("{} 时间长度必须是 {}，当前为 {}".format(name, int(seq_len), int(value.shape[-1])))
    _ensure_finite(name, value)
    return value


def _axis_angle_person_to_rot6d(axis_angle):
    if axis_angle.shape[1] != NTU_2P_BODY_JOINTS or axis_angle.shape[2] != 3:
        raise ValueError("axis_angle person pose 必须是 [B,55,3,T]，当前为 {}".format(_shape_text(axis_angle)))
    pose_btjc = axis_angle.permute(0, 3, 1, 2).contiguous()
    rot_mats = axis_angle_to_matrix(pose_btjc)
    rot6d = matrix_to_rotation_6d(rot_mats)
    return rot6d.permute(0, 2, 3, 1).contiguous()


def raw_ntu_2p_to_rot6d(value):
    value = check_raw_ntu_2p_motion(value)
    batch_size, _, _, seq_len = value.shape
    result = torch.zeros(
        batch_size,
        NTU_2P_NUM_JOINTS_WITH_TRANS,
        NTU_2P_ROT6D_FEATS,
        seq_len,
        dtype=value.dtype,
        device=value.device,
    )

    person_a_pose = value[:, :NTU_2P_BODY_JOINTS, 0:3, :]
    person_b_pose = value[:, :NTU_2P_BODY_JOINTS, 3:6, :]
    result[:, :NTU_2P_BODY_JOINTS, PERSON_A_ROT6D_SLICE, :] = _axis_angle_person_to_rot6d(person_a_pose)
    result[:, :NTU_2P_BODY_JOINTS, PERSON_B_ROT6D_SLICE, :] = _axis_angle_person_to_rot6d(person_b_pose)
    result[:, TRANSLATION_SLOT, PERSON_A_TRANS_SLICE, :] = value[:, TRANSLATION_SLOT, 0:3, :]
    result[:, TRANSLATION_SLOT, PERSON_B_TRANS_SLICE, :] = value[:, TRANSLATION_SLOT, 3:6, :]
    return check_ntu_2p_rot6d(result, seq_len=seq_len)


def split_ntu_2p_rot6d(value):
    value = check_ntu_2p_rot6d(value)
    batch_size, _, _, seq_len = value.shape
    person_a = torch.zeros(
        batch_size,
        NTU_2P_NUM_JOINTS_WITH_TRANS,
        NTU_2P_SINGLE_ROT6D_FEATS,
        seq_len,
        dtype=value.dtype,
        device=value.device,
    )
    person_b = torch.zeros_like(person_a)
    person_a[:, :NTU_2P_BODY_JOINTS, :, :] = value[:, :NTU_2P_BODY_JOINTS, PERSON_A_ROT6D_SLICE, :]
    person_b[:, :NTU_2P_BODY_JOINTS, :, :] = value[:, :NTU_2P_BODY_JOINTS, PERSON_B_ROT6D_SLICE, :]
    person_a[:, TRANSLATION_SLOT, 0:3, :] = value[:, TRANSLATION_SLOT, PERSON_A_TRANS_SLICE, :]
    person_b[:, TRANSLATION_SLOT, 0:3, :] = value[:, TRANSLATION_SLOT, PERSON_B_TRANS_SLICE, :]
    return (
        check_single_person_rot6d(person_a, seq_len=seq_len, name="person_a_rot6d"),
        check_single_person_rot6d(person_b, seq_len=seq_len, name="person_b_rot6d"),
    )


def join_ntu_2p_rot6d(person_a, person_b):
    person_a = check_single_person_rot6d(person_a, name="person_a_rot6d")
    person_b = check_single_person_rot6d(person_b, seq_len=person_a.shape[-1], name="person_b_rot6d")
    _check_device_match("person_b_rot6d", person_b, person_a)
    if person_b.dtype != person_a.dtype:
        raise ValueError("person_a/person_b dtype 必须一致")

    batch_size, _, _, seq_len = person_a.shape
    result = torch.zeros(
        batch_size,
        NTU_2P_NUM_JOINTS_WITH_TRANS,
        NTU_2P_ROT6D_FEATS,
        seq_len,
        dtype=person_a.dtype,
        device=person_a.device,
    )
    result[:, :NTU_2P_BODY_JOINTS, PERSON_A_ROT6D_SLICE, :] = person_a[:, :NTU_2P_BODY_JOINTS, :, :]
    result[:, :NTU_2P_BODY_JOINTS, PERSON_B_ROT6D_SLICE, :] = person_b[:, :NTU_2P_BODY_JOINTS, :, :]
    result[:, TRANSLATION_SLOT, PERSON_A_TRANS_SLICE, :] = person_a[:, TRANSLATION_SLOT, 0:3, :]
    result[:, TRANSLATION_SLOT, PERSON_B_TRANS_SLICE, :] = person_b[:, TRANSLATION_SLOT, 0:3, :]
    return check_ntu_2p_rot6d(result, seq_len=seq_len)


def check_ntu_2p_xyz(value, seq_len=None):
    _ensure_float_tensor("ntu_2p_xyz", value)
    if value.dim() != 5:
        raise ValueError("NTU 双人 xyz 必须是 [B,T,2,55,3]，当前维度数为 {}".format(value.dim()))
    expected = (NTU_2P_NUM_PERSONS, NTU_2P_BODY_JOINTS, XYZ_COORD_DIM)
    if tuple(value.shape[2:]) != expected:
        raise ValueError("NTU 双人 xyz 后三维必须是 {}，当前为 {}".format(expected, _shape_text(value)[2:]))
    if seq_len is not None and int(value.shape[1]) != int(seq_len):
        raise ValueError("NTU 双人 xyz 时间长度必须是 {}，当前为 {}".format(int(seq_len), int(value.shape[1])))
    _ensure_finite("ntu_2p_xyz", value)
    return value


def ntu_2p_rot6d_to_xyz(value, converter=None):
    value = check_ntu_2p_rot6d(value)
    if converter is None:
        converter = Rotation2xyz_x(device=value.device, dataset="ntu120_2p")
    mask = torch.ones((value.shape[0], value.shape[-1]), dtype=torch.bool, device=value.device)
    xyz_cat = converter(
        value,
        mask=mask,
        pose_rep="rot6d",
        translation=True,
        glob=True,
        jointstype="smplx",
        vertstrans=True,
        num_person=NTU_2P_NUM_PERSONS,
    )
    expected = (NTU_2P_BODY_JOINTS, NTU_2P_NUM_PERSONS * XYZ_COORD_DIM)
    if tuple(xyz_cat.shape[1:3]) != expected:
        raise ValueError("Rotation2xyz_x 输出必须是 [B,55,6,T]，当前为 {}".format(_shape_text(xyz_cat)))
    xyz = torch.stack((xyz_cat[:, :, 0:3], xyz_cat[:, :, 3:6]), dim=2)
    xyz = xyz.permute(0, 4, 2, 1, 3).contiguous()
    return check_ntu_2p_xyz(xyz, seq_len=value.shape[-1])


def root_rotation_matrices(value):
    value = check_ntu_2p_rot6d(value)
    root_a = value[:, ROOT_JOINT, PERSON_A_ROT6D_SLICE, :].permute(0, 2, 1).contiguous()
    root_b = value[:, ROOT_JOINT, PERSON_B_ROT6D_SLICE, :].permute(0, 2, 1).contiguous()
    rot_a = rotation_6d_to_matrix(root_a)
    rot_b = rotation_6d_to_matrix(root_b)
    _ensure_finite("root_rotation_a", rot_a)
    _ensure_finite("root_rotation_b", rot_b)
    return rot_a, rot_b


def root_translations(value):
    value = check_ntu_2p_rot6d(value)
    trans_a = value[:, TRANSLATION_SLOT, PERSON_A_TRANS_SLICE, :].permute(0, 2, 1).contiguous()
    trans_b = value[:, TRANSLATION_SLOT, PERSON_B_TRANS_SLICE, :].permute(0, 2, 1).contiguous()
    _ensure_finite("root_translation_a", trans_a)
    _ensure_finite("root_translation_b", trans_b)
    return trans_a, trans_b


def interaction_targets(value, converter=None):
    value = check_ntu_2p_rot6d(value)
    xyz = ntu_2p_rot6d_to_xyz(value, converter=converter)
    rot_a, rot_b = root_rotation_matrices(value)
    trans_a, trans_b = root_translations(value)
    targets = OrderedDict()
    targets["joint"] = xyz[:, :, 0] - xyz[:, :, 1]
    targets["orient"] = torch.matmul(rot_a.transpose(-1, -2), rot_b)
    targets["trans"] = trans_a - trans_b
    for key, tensor in targets.items():
        _ensure_finite("interaction_{}".format(key), tensor)
    return targets


def _mse(pred, target):
    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError("MSE 输入 shape 不一致: {} vs {}".format(_shape_text(pred), _shape_text(target)))
    return ((pred - target) ** 2).mean()


def interaction_loss(pred, target, converter=None):
    pred = check_ntu_2p_rot6d(pred)
    target = check_ntu_2p_rot6d(target, seq_len=pred.shape[-1])
    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError("pred/target shape 必须一致: {} vs {}".format(_shape_text(pred), _shape_text(target)))

    pred_targets = interaction_targets(pred, converter=converter)
    real_targets = interaction_targets(target, converter=converter)
    terms = OrderedDict()
    terms["joint_mse"] = _mse(pred_targets["joint"], real_targets["joint"])
    terms["orient_mse"] = _mse(pred_targets["orient"], real_targets["orient"])
    terms["trans_mse"] = _mse(pred_targets["trans"], real_targets["trans"])
    terms["inter_loss"] = terms["joint_mse"] + terms["orient_mse"] + terms["trans_mse"]
    if tuple(terms.keys()) != NTU_2P_INTERACTION_LOSS_KEYS:
        raise AssertionError("interaction loss key 不稳定")
    for key, tensor in terms.items():
        _ensure_finite(key, tensor)
    return terms
