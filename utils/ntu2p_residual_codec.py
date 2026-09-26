"""Track B 残差编解码：v2 输出 P 的残差在个人朝向系中的最小二乘斜坡 DCT 系数（root + 21 关节，K=12）。

设计：docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md 第 3.2–3.5 节。

- 坐标系：场景规范系（canonical_frame，camera_x 回退，与 v2 GH 一致）再转到每人的个人朝向系（individual_frames，
  三行依次为前、左、上，与 GH 腿部流同一个系），"前/后""左/右脚"对所有人含义一致；残差是向量，只旋转不平移。
- 通道：每人 66 维 = pelvis 残差 3 维 + 关节 1..21 相对 pelvis 残差的局部残差 63 维；下颌/眼随头、手指随手腕刚性平移，
  手部朝向沿用 v2，由骨架投影的 Procrustes 保持刚体模板。
- 时间基：B[k,t] = ramp(t)·φ_k(t)，k=0..11（0.2k Hz，最高 2.2 Hz，覆盖 1.2–2 Hz 步频）；ramp 与 v2 相同，首帧恒为 0，
  所以解码残差首帧严格为 0，首帧误差只剩投影的浮点误差。
- 编码 c = (BBᵀ)⁻¹B r、解码 r = Bᵀc 互逆（BBᵀ 近单位阵、条件数小），float64 计算后回到输入 dtype。
- 解码后经 SkeletonProjector（与 v2 A4 同一个投影器，骨长取观测末帧）：骨长、刚体手由结构保证；
  root 系数为 0 时 pelvis 残差严格为 0，而投影 root 取预测 pelvis，因此输出 pelvis 与 v2 逐位相同。
"""

from collections import OrderedDict

import torch
from torch import nn

from model.forecasting_ntu2p_intermixer import individual_frames, relative_geometry, rotate_rows, rotate_rows_transposed
from model.forecasting_ntu2p_residual_xyz import build_residual_ramp
from utils.ntu2p_canonical import apply_linear, canonical_frame, estimate_up, horizontal, to_canonical
from utils.ntu_smplx_2p_xyz import dct_matrix


OBS_LEN, PRED_LEN = 10, 50
NUM_COEFFS = 12  # k=0..11，第 k 个基 0.2k Hz，≤2.2 Hz
GEN_JOINTS = tuple(range(1, 22))  # 21 个身体关节（不含 pelvis）
ROOT_CHANNELS = 3
CHANNELS = 66  # root 3 + 21×3 局部
AB_FALLBACK = "camera_x"  # 与 v2 GH 的 canonical_ab_fallback 一致
RAMP_MODE, RAMP_SATURATE_FRAMES = "saturate", 5
FOLLOWERS = ((15, (22, 23, 24)), (20, tuple(range(25, 40))), (21, tuple(range(40, 55))))
GLITCH_STEP_M = 0.25  # m/帧（5 m/s），预登记物理阈值
V2_WALK_DISPLACEMENT = 0.5  # 与 utils.ntu2p_naturalness.WALK_DISPLACEMENT_THRESHOLD 相同
STD_FLOOR_FRAC = 0.05

NUM_BODY = 1 + len(GEN_JOINTS)
NUM_JOINTS = 55


def _follower_index():
    """55 个关节各取哪一个身体关节（0..21）的残差：身体关节取自身，随动关节取源关节。"""
    index = list(range(NUM_BODY))
    source = {}
    for src, dst in FOLLOWERS:
        for joint in dst:
            source[joint] = src
    index += [source[joint] for joint in range(NUM_BODY, NUM_JOINTS)]
    return tuple(index)


FOLLOWER_INDEX = _follower_index()


def ramp_dct_basis(pred_len=PRED_LEN, num_coeffs=NUM_COEFFS, ramp_mode=RAMP_MODE, saturate_frames=RAMP_SATURATE_FRAMES):
    """[K,T] float64 斜坡 DCT 基；首帧整列为 0。"""
    ramp = build_residual_ramp(pred_len, ramp_mode, saturate_frames).reshape(int(pred_len)).double()
    basis = dct_matrix(int(pred_len), dtype=torch.float64)[: int(num_coeffs)] * ramp.unsqueeze(0)
    if not bool((basis[:, 0] == 0).all()):
        raise AssertionError("斜坡 DCT 基首帧必须严格为 0")
    return basis


def glitch_mask(window_xyz, threshold=GLITCH_STEP_M):
    """[B,60,2,55,3] -> bool [B]：任一人 pelvis 帧间位移超过阈值（多为 A/B 身份交换或拟合跳变）。"""
    pelvis = window_xyz[:, :, :, 0]
    step = (pelvis[:, 1:] - pelvis[:, :-1]).norm(dim=-1)
    return (step > float(threshold)).flatten(1).any(dim=1)


def codec_frames(obs_xyz):
    """场景规范系 + 个人朝向系；person [B,2,3,3] 三行依次为前、左、上（场景规范系坐标）。"""
    scene = canonical_frame(obs_xyz, ab_fallback=AB_FALLBACK)
    person = individual_frames(to_canonical(obs_xyz, scene))
    return OrderedDict([("scene", scene), ("person", person)])


def repeat_frames(frames, repeats):
    """每个样本的坐标系重复 repeats 次（与 repeat_interleave 展开的 B×K 样本对齐）。"""
    scene = OrderedDict((key, value.repeat_interleave(int(repeats), dim=0)) for key, value in frames["scene"].items())
    return OrderedDict([("scene", scene), ("person", frames["person"].repeat_interleave(int(repeats), dim=0))])


def _person_rotation(frames):
    return frames["person"].unsqueeze(1).unsqueeze(3)  # [B,1,2,1,3,3]


def to_person(vec_cam, frames):
    """[B,T,2,J,3] 相机系向量 -> 个人朝向系（只旋转）。"""
    return rotate_rows(apply_linear(vec_cam, frames["scene"]["rotation"]), _person_rotation(frames))


def to_camera(vec_person, frames):
    """to_person 的逆。"""
    return apply_linear(rotate_rows_transposed(vec_person, _person_rotation(frames)), frames["scene"]["rotation"].transpose(-1, -2))


def _body_channels(vec_person):
    """[...,J≥22,3] 个人系向量 -> [...,66]：pelvis 3 维 + 关节 1..21 相对 pelvis 的 63 维。"""
    root = vec_person[..., :1, :]
    local = vec_person[..., 1:NUM_BODY, :] - root
    return torch.cat((root, local), dim=-2).flatten(-2)


def obs_features(obs_xyz, frames):
    """[B,10,2,55,3] -> [B,2,10,66]：个人系下 [pelvis(t) − pelvis(last)，关节 1..21 − pelvis(t)]。"""
    relative = obs_xyz[:, :, :, :NUM_BODY] - obs_xyz[:, -1:, :, :1]
    return _body_channels(to_person(relative, frames)).transpose(1, 2).contiguous()


def rel_geometry(obs_xyz, frames):
    """[B,2,6] 旋转不变的双人相对几何（与 v2 InterMixer 同一函数）。"""
    return relative_geometry(to_canonical(obs_xyz[:, -1], frames["scene"]))


def v2_walk(obs_xyz, base_xyz, threshold=V2_WALK_DISPLACEMENT):
    """bool [B,2]：底座预测 f50 的 pelvis 相对观测末帧的水平位移 ≥ 0.5 m（bootstrap 分层用，只看观测与底座输出）。"""
    displacement = horizontal(base_xyz[:, -1, :, 0] - obs_xyz[:, -1, :, 0], estimate_up(obs_xyz))
    return displacement.norm(dim=-1) >= float(threshold)


class ResidualCodec(nn.Module):
    """系数 [B,2,K,66] <-> 个人系逐帧残差；只有非持久 buffer，不进 checkpoint。"""

    def __init__(self, num_coeffs=NUM_COEFFS, pred_len=PRED_LEN):
        super(ResidualCodec, self).__init__()
        self.num_coeffs = int(num_coeffs)
        self.pred_len = int(pred_len)
        basis = ramp_dct_basis(self.pred_len, self.num_coeffs)
        # torch 1.7 无 linalg.solve；BBᵀ 为 K×K 良态矩阵，直接求逆。
        encoder = torch.inverse(basis @ basis.t()) @ basis
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("encoder", encoder, persistent=False)
        self.register_buffer("follower_index", torch.tensor(FOLLOWER_INDEX, dtype=torch.long), persistent=False)

    def config(self):
        return OrderedDict(
            [
                ("num_coeffs", self.num_coeffs),
                ("pred_len", self.pred_len),
                ("channels", CHANNELS),
                ("ramp_mode", RAMP_MODE),
                ("ramp_saturate_frames", RAMP_SATURATE_FRAMES),
                ("joints", list(GEN_JOINTS)),
                ("followers", [[src, list(dst)] for src, dst in FOLLOWERS]),
                ("ab_fallback", AB_FALLBACK),
                ("frame", "canonical_frame -> individual_frames（前、左、上）"),
            ]
        )

    def residual_to_channels(self, res_person):
        """[B,T,2,J≥22,3] -> [B,T,2,66]。"""
        return _body_channels(res_person)

    def encode_channels(self, channels):
        """[B,T,2,66] -> [B,2,K,66]（最小二乘）。"""
        coeffs = torch.einsum("kt,btpc->bpkc", self.encoder, channels.double())
        return coeffs.to(channels.dtype)

    def decode_channels(self, coeffs):
        """[...,2,K,66] -> [...,T,2,66]。"""
        channels = torch.einsum("kt,...pkc->...tpc", self.basis, coeffs.double())
        return channels.to(coeffs.dtype)

    def channels_to_residual(self, channels):
        """[...,T,2,66] -> [...,T,2,55,3]：pelvis = root，关节 1..21 = root + 局部，随动关节逐位复制源关节（手指只平移）。"""
        per_joint = channels.reshape(channels.shape[:-1] + (NUM_BODY, 3))
        root = per_joint[..., :1, :]
        body = torch.cat((root, per_joint[..., 1:, :] + root), dim=-2)
        return body.index_select(-2, self.follower_index)

    def target_coeffs(self, obs_xyz, base_xyz, target_xyz, frames):
        """GT − P 的系数（米）。"""
        return self.encode_channels(self.residual_to_channels(to_person(target_xyz - base_xyz, frames)))

    def draft_coeffs(self, obs_xyz, base_xyz, frames):
        """P − obs_last 的系数（米），作为条件：告诉生成器底座打算怎么动。"""
        return self.encode_channels(self.residual_to_channels(to_person(base_xyz - obs_xyz[:, -1:], frames)))

    def residual_camera(self, coeffs_m, frames):
        """系数（米）-> 相机系逐帧残差 [B,T,2,55,3]（投影前）。"""
        return to_camera(self.channels_to_residual(self.decode_channels(coeffs_m)), frames)

    def apply(self, obs_xyz, base_xyz, coeffs_m, frames, projector):
        """y = Proj(P + R_sceneᵀ R_personᵀ r̂, obs_last)。"""
        return projector(base_xyz + self.residual_camera(coeffs_m, frames), obs_xyz[:, -1])


def condition_features(codec, obs_xyz, base_xyz, frames=None):
    """残差库构建与评估共用：保证训练与评估的条件特征完全同一口径。"""
    frames = codec_frames(obs_xyz) if frames is None else frames
    return OrderedDict(
        [
            ("draft", codec.draft_coeffs(obs_xyz, base_xyz, frames)),
            ("obs_feats", obs_features(obs_xyz, frames)),
            ("rel_geom", rel_geometry(obs_xyz, frames)),
            ("v2_walk", v2_walk(obs_xyz, base_xyz)),
            ("frames", frames),
        ]
    )


__all__ = [
    "AB_FALLBACK",
    "CHANNELS",
    "FOLLOWERS",
    "GLITCH_STEP_M",
    "NUM_COEFFS",
    "ROOT_CHANNELS",
    "ResidualCodec",
    "codec_frames",
    "condition_features",
    "glitch_mask",
    "obs_features",
    "ramp_dct_basis",
    "rel_geometry",
    "repeat_frames",
    "to_camera",
    "to_person",
    "v2_walk",
]
