"""NTU 双人 xyz 的可逆场景规范化与训练用几何增广。

数据调查结论（详见 docs/ai/context/20260925-120854-ntu2p-xyz-seq-cache-canonical-augment-result.md）：
- 缓存 xyz 处于相机系：y 向下、z 指向场景内；重力"上"方向约为 -y，但带随拍摄 setup 变化的俯仰
  （train 逐序列 neck-pelvis 均值相对 -y 倾角中位数 12°，各 setup 均值 0°-37°），不能直接当作坐标轴；
- 沿"上"方向的地面高度随 setup 变化（setup 均值 -2.04 到 -1.07 m，全体标准差 0.26 m），跨样本不一致；
- 双人左右位置各半、A->B 连线多与相机视线垂直，A/B 顺序不携带左右语义。

规范化以观测末帧为参考，全部参数只取自观测，可在测试时使用并精确反变换：
- 原点：双人 pelvis 中点（默认连竖直分量一起减去，因为地面/相机高度跨样本不一致）；
- +Y：重力"上"方向，默认逐样本由观测估计，离数据集常量过远时回退常量；
- +X：A->B pelvis 连线的水平分量（双人距离过近时回退 A 的朝向）；+Z = X × Y。
选 A->B 连线而非 A 的骨盆朝向：pelvis 位置是拟合中最可靠的量，朝向依赖髋/肩且扭身时有歧义；
对齐连线后双人布局只剩距离一个自由度，而 A 朝向 B 在 train 中也只占 85%（60° 内）。

所有 3x3 变换都用逐元素乘加实现：torch 1.7 在 Ampere GPU 上默认让 matmul/einsum 走 TF32，
3x3 旋转往返误差可达 2e-3 m，会破坏"规范化系与原系指标一致"。

增广必须在原始相机系、规范化之前做：观测 up 估计的回退常量是相机系向量，在规范化系里没有意义。
yaw 与平移增广会被规范化精确消去，只在不做规范化的训练中有用；A/B 交换与镜像在两种情况下都有效。
"""

import math
from collections import OrderedDict

import torch
from smplx.joint_names import JOINT_NAMES


PELVIS = 0
L_HIP = 1
R_HIP = 2
NECK = 12
L_SHOULDER = 16
R_SHOULDER = 17
NUM_SMPLX_JOINTS = 55

# train split 1758 条序列、全部帧与双人的 unit(neck - pelvis) 均值；逐样本估计失败时的回退值。
NTU2P_DATASET_UP = (-0.0029876, -0.9770715, -0.2128901)
# 各 setup 平均 up 相对数据集常量最多偏 25°，超过 30° 的逐样本估计视为姿态异常（弯腰/拟合错误）。
DEFAULT_UP_MAX_DEVIATION_DEG = 30.0
# train 中双人水平距离 p1 为 0.25 m；低于 0.1 m 时连线方向由噪声主导。
DEFAULT_MIN_AB_DISTANCE = 0.1

UP_MODES = ("obs", "dataset")
YAW_REFS = ("ab_line", "a_facing")
VERTICAL_ORIGINS = ("pelvis", "keep")
AB_FALLBACKS = ("a_facing", "camera_x")


def _smplx_mirror_permutation():
    names = list(JOINT_NAMES[:NUM_SMPLX_JOINTS])

    def counterpart(name):
        if name.startswith("left_"):
            return "right_" + name[len("left_") :]
        if name.startswith("right_"):
            return "left_" + name[len("right_") :]
        return name

    perm = tuple(names.index(counterpart(name)) for name in names)
    if any(perm[perm[index]] != index for index in range(len(perm))):
        raise AssertionError("SMPL-X 左右置换必须是对合")
    return perm


# 由 SMPL-X 官方关节名 left_*/right_* 配对得到；躯干/头/下颌映射到自身。
SMPLX_MIRROR_PERMUTATION = _smplx_mirror_permutation()


def _unit(value, eps=1e-8):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


def _view_per_sample(value, ndim):
    """把 [B,...] 的逐样本量扩成能与 [B,d1,...,d_{ndim-2},3] 广播的形状。"""
    batch_size = int(value.shape[0])
    return value.reshape((batch_size,) + (1,) * (ndim - 2) + tuple(value.shape[1:]))


def apply_linear(value, matrix):
    """逐样本 3x3 线性变换 x -> M x（行向量形式 x M^T），逐元素乘加以避开 TF32。"""
    matrix = _view_per_sample(matrix, value.dim())
    return (value.unsqueeze(-2) * matrix).sum(dim=-1)


def _flag(mask, ndim):
    """[B] bool -> [B,1,...,1]，与 ndim 维张量广播。"""
    return mask.view((int(mask.shape[0]),) + (1,) * (int(ndim) - 1))


def horizontal(value, up):
    up = _view_per_sample(up, value.dim())
    return value - (value * up).sum(dim=-1, keepdim=True) * up


def _camera_x(up):
    axis = torch.zeros_like(up)
    axis[:, 0] = 1.0
    return horizontal(axis, up)


def dataset_up(batch_size, device=None, dtype=torch.float32):
    return torch.tensor(NTU2P_DATASET_UP, dtype=dtype, device=device).unsqueeze(0).expand(int(batch_size), 3)


def estimate_up(obs_xyz, mode="obs", max_deviation_deg=DEFAULT_UP_MAX_DEVIATION_DEG):
    """[B,T,2,55,3] -> [B,3] 单位"上"向量；obs 模式对 yaw、平移、A/B 交换与竖直面镜像等变。"""
    if mode not in UP_MODES:
        raise ValueError("up mode 必须是 {}，当前为 {}".format(UP_MODES, mode))
    const = _unit(dataset_up(obs_xyz.shape[0], obs_xyz.device, obs_xyz.dtype))
    if mode == "dataset":
        return const
    spine = _unit(obs_xyz[:, :, :, NECK] - obs_xyz[:, :, :, PELVIS])
    up = _unit(spine.mean(dim=(1, 2)))
    far = (up * const).sum(dim=-1, keepdim=True) < math.cos(math.radians(float(max_deviation_deg)))
    return torch.where(far, const, up)


def facing_direction(person_xyz, up):
    """[B,55,3] 单人一帧 -> [B,3] 水平朝向；左减右叉乘上得到前方（SMPL-X 体坐标 x=左, y=上, z=前）。"""
    across = (person_xyz[:, L_HIP] - person_xyz[:, R_HIP]) + (person_xyz[:, L_SHOULDER] - person_xyz[:, R_SHOULDER])
    return horizontal(torch.cross(across, up, dim=-1), up)


def canonical_frame(
    obs_xyz,
    up_mode="obs",
    yaw_ref="ab_line",
    vertical_origin="pelvis",
    min_ab_distance=DEFAULT_MIN_AB_DISTANCE,
    ab_fallback="a_facing",
):
    """由观测求刚体变换 x_can = R (x - t)。

    ab_fallback：双人水平距离低于 min_ab_distance（A->B 方向不可观测）时的 yaw 参考。
    - a_facing：A 的朝向（默认，保持已训练模型的行为）；
    - camera_x：该方向的先验众数，即水平化的相机 x 轴（数据中 A->B 连线多与视线垂直），符号取观测偏移在其上的
      投影，使 A 仍落在 -X 一侧。双人重合多来自跟踪/身份混淆，此时 A 的朝向也不可靠，而 A 朝向回退给出的
      "A->B 沿 A 前方"布局在训练中几乎不存在（train 仅 3 条），val 上一条此类样本的 mse 因此被放大 3–5 倍。

    返回 rotation [B,3,3]（三行依次为规范系 X/Y/Z 轴在原系中的方向）、translation [B,3]、up [B,3]。
    """
    if yaw_ref not in YAW_REFS:
        raise ValueError("yaw_ref 必须是 {}，当前为 {}".format(YAW_REFS, yaw_ref))
    if vertical_origin not in VERTICAL_ORIGINS:
        raise ValueError("vertical_origin 必须是 {}，当前为 {}".format(VERTICAL_ORIGINS, vertical_origin))
    if ab_fallback not in AB_FALLBACKS:
        raise ValueError("ab_fallback 必须是 {}，当前为 {}".format(AB_FALLBACKS, ab_fallback))
    up = estimate_up(obs_xyz, up_mode)
    last = obs_xyz[:, -1]
    pelvis = last[:, :, PELVIS]
    center = pelvis.mean(dim=1)
    if vertical_origin == "keep":
        center = horizontal(center, up)
    ref = facing_direction(last[:, 0], up)
    if yaw_ref == "ab_line":
        ab = horizontal(pelvis[:, 1] - pelvis[:, 0], up)
        if ab_fallback == "camera_x":
            camera = _camera_x(up)
            sign = torch.where((ab * camera).sum(dim=-1, keepdim=True) < 0, -torch.ones_like(ab[:, :1]), torch.ones_like(ab[:, :1]))
            ref = camera * sign
        ref = torch.where(ab.norm(dim=-1, keepdim=True) >= float(min_ab_distance), ab, ref)
    # 连线与朝向都退化（如躺倒）时用相机 x 轴兜底，保证旋转矩阵始终有定义。
    ref = torch.where(ref.norm(dim=-1, keepdim=True) > 1e-6, ref, _camera_x(up))
    axis_x = _unit(ref)
    axis_z = torch.cross(axis_x, up, dim=-1)
    rotation = torch.stack((axis_x, up, axis_z), dim=1)
    return OrderedDict([("rotation", rotation), ("translation", center), ("up", up)])


def to_canonical(value, frame):
    """[B,...,3] 原系 -> 规范系。"""
    return apply_linear(value - _view_per_sample(frame["translation"], value.dim()), frame["rotation"])


def from_canonical(value, frame):
    """[B,...,3] 规范系 -> 原系；预测回到原系后再算指标即与未规范化的协议完全一致。"""
    translation = _view_per_sample(frame["translation"], value.dim())
    return apply_linear(value, frame["rotation"].transpose(-1, -2)) + translation


def canonicalize(obs_xyz, target_xyz=None, **kwargs):
    frame = canonical_frame(obs_xyz, **kwargs)
    target = None if target_xyz is None else to_canonical(target_xyz, frame)
    return to_canonical(obs_xyz, frame), target, frame


def rotation_about_axis(axis, angle):
    """Rodrigues：axis [B,3] 单位向量，angle [B] 弧度 -> [B,3,3]。"""
    cos = torch.cos(angle).view(-1, 1, 1)
    sin = torch.sin(angle).view(-1, 1, 1)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(x)
    cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).view(-1, 3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device).unsqueeze(0)
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)
    return eye * cos + sin * cross + (1.0 - cos) * outer


def reflection_across_plane(normal):
    """过原点、法向为 normal [B,3] 的平面反射 -> [B,3,3]。"""
    eye = torch.eye(3, dtype=normal.dtype, device=normal.device).unsqueeze(0)
    return eye - 2.0 * normal.unsqueeze(-1) * normal.unsqueeze(-2)


def mirror_joint_labels(value):
    """[...,55,3] 左右关节互换；与空间反射配合才得到合法的镜像人体。"""
    index = torch.as_tensor(SMPLX_MIRROR_PERMUTATION, device=value.device)
    return value.index_select(-2, index)


def swap_persons(value):
    """[B,T,2,55,3] 交换 A/B。"""
    return value.flip(2)


class NTU2PSceneAugment(object):
    """训练用整场景几何增广：A/B 交换、竖直面镜像（反射 + SMPL-X 左右关节置换）、绕竖直轴 yaw、水平平移。

    镜像与 yaw 都绕观测末帧双人 pelvis 中点所在的竖直轴进行，人物留在原处；竖直轴用与规范化相同的
    `estimate_up`，因此规范化能精确消去 yaw/平移，并把镜像映成规范系 z 轴反射。每次调用固定消耗
    同样多的随机数（与开关无关），不同配置之间的随机流可对齐。
    """

    def __init__(self, yaw_range_deg=180.0, swap_prob=0.5, mirror_prob=0.5, translate_std=0.0, up_mode="obs"):
        self.yaw_range = math.radians(float(yaw_range_deg))
        self.swap_prob = float(swap_prob)
        self.mirror_prob = float(mirror_prob)
        self.translate_std = float(translate_std)
        self.up_mode = up_mode

    def draw(self, batch_size, generator=None):
        uniform = torch.rand(int(batch_size), 3, generator=generator)
        normal = torch.randn(int(batch_size), 2, generator=generator)
        return OrderedDict(
            [
                ("yaw", (2.0 * uniform[:, 0] - 1.0) * self.yaw_range),
                ("swap", uniform[:, 1] < self.swap_prob),
                ("mirror", uniform[:, 2] < self.mirror_prob),
                ("shift", normal * self.translate_std),
            ]
        )

    def apply(self, obs_xyz, target_xyz, params):
        device, dtype = obs_xyz.device, obs_xyz.dtype
        # 打包成一次 host->device 拷贝。
        packed = torch.cat(
            (
                params["yaw"].view(-1, 1),
                params["swap"].view(-1, 1).to(params["yaw"].dtype),
                params["mirror"].view(-1, 1).to(params["yaw"].dtype),
                params["shift"].view(-1, 2),
            ),
            dim=1,
        ).to(device=device, dtype=dtype, non_blocking=True)
        yaw, swap, mirror, shift = packed[:, 0], packed[:, 1] > 0.5, packed[:, 2] > 0.5, packed[:, 3:5]

        up = estimate_up(obs_xyz, self.up_mode)
        center = obs_xyz[:, -1, :, PELVIS].mean(dim=1)
        normal = _unit(_camera_x(up))
        side = torch.cross(up, normal, dim=-1)
        eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        reflect = torch.where(mirror.view(-1, 1, 1), reflection_across_plane(normal), eye)
        linear = rotation_about_axis(up, yaw)
        # 先反射再旋转：L = R · M，逐元素实现避开 TF32。
        linear = (linear.unsqueeze(-1) * reflect.unsqueeze(-3)).sum(dim=-2)
        offset = center + shift[:, 0:1] * normal + shift[:, 1:2] * side

        outputs = []
        for value in (obs_xyz, target_xyz):
            ndim = value.dim()
            value = torch.where(_flag(swap, ndim), swap_persons(value), value)
            value = apply_linear(value - _view_per_sample(center, ndim), linear) + _view_per_sample(offset, ndim)
            outputs.append(torch.where(_flag(mirror, ndim), mirror_joint_labels(value), value))
        return outputs[0], outputs[1]

    def __call__(self, obs_xyz, target_xyz, generator=None):
        return self.apply(obs_xyz, target_xyz, self.draw(obs_xyz.shape[0], generator))
