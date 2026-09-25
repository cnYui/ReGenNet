"""A4 KinProj：可微骨架投影 + 刚体手；A5 FootLock：脚接触掩码与脚滑损失。

A4 的出发点：数据 xyz 由固定体型（betas=0）的中性 SMPL-X FK 得到，骨长逐帧恒定（标准差 5e-8 m），
而直接回归 xyz 的模型身体骨长相对误差约 5%（p95 20%），前臂、脚掌会伸缩。投影把任意 55 关节点云
一次前向地映射到同一副骨架上（无迭代优化），去掉伸缩；R2 的优化式 IK 已表明这在 L2 上近似中性。

投影规则（自上而下，按拓扑层级批处理）；骨长与模板默认取自参考帧（观测末帧，理由见 SkeletonProjector）：
- root 取预测 pelvis；
- 单子节点：子关节 = 已投影父关节 + 骨长 × 单位(预测子关节 − 已投影父关节)。这是预测点到"以父关节为
  球心、骨长为半径"球面的最近点，切向误差不沿链累积；
- 四肢（髋-膝-踝、肩-肘-腕）用解析两骨 IK：末端（踝/腕）在可达时精确保持预测位置，肘/膝在两球交线圆上
  取离预测最近的点。逐骨贪心放置会把每段的径向（骨长）误差沿链累加到腕，再平移到 15 个手指上；
  val 上贪心 mpjpe +0.47%、xyz_mse +1.53%，两骨 IK 为 +0.04%、+0.46%（优化式 IK 为 +0.006%、+0.13%）；
- 多子节点（pelvis、spine3、head）：以已投影父关节为定点，对子关节做加权 Procrustes 求全局旋转 G_p，
  子关节 = 父关节 + G_p (模板中子关节相对父关节的偏移)；
- 手：手指相对手腕几乎不动（固定为观测末帧只损失 2.7 mm），以参考帧 15 个手指关节相对手腕的构型为刚体
  模板，Procrustes 对齐预测手指点云后整体放置。

Procrustes 用 Horn 四元数法（4×4 对称阵的主特征向量，重复平方幂迭代）而非 torch.svd：torch 1.7 的批量
SVD 在 CUDA 上逐矩阵调用 MAGMA、在 CPU 上逐矩阵调用 LAPACK（4000 个 3×3 约 19 ms），而 Horn 法只有
逐元素运算与 4×4 批量乘，且天然给出 SO(3) 内的全局最优（含需要反射修正的情形）。反向传播不对特征分解
求导，而用最优性条件的闭式解（见 `_ProcrustesRotation.backward`），数值上只在输入秩 ≤ 1 时退化。

所有投影运算在 float64 下进行：每个样本数据量很小，而 torch 1.7 在 Ampere 上 float32 matmul 默认走 TF32。

A5：GT 接触掩码直接复用 `utils/ntu2p_naturalness.py`（训练目标与自然度评估同一口径）；脚滑损失的水平方向
由调用方传入的 up 决定（通常为 `utils.ntu2p_canonical.estimate_up(obs)`）。
"""

import os

import numpy as np
import torch
import torch.nn as nn

from utils.config import SMPLX_MODEL_PATH
from utils.ntu2p_naturalness import (
    CONTACT_HEIGHT_THRESHOLDS,
    FOOT_JOINTS,
    NTU2P_FPS,
    SMPLX_NUM_JOINTS,
    SMPLX_PARENTS,
    estimate_scene_frame,
    gt_contact_mask,
    smooth_time,
)


SMPLX_NEUTRAL_NPZ = os.path.join(SMPLX_MODEL_PATH, "SMPLX_NEUTRAL.npz")
L_WRIST, R_WRIST = 20, 21
HAND_WRISTS = (L_WRIST, R_WRIST)
HAND_FINGERS = (tuple(range(25, 40)), tuple(range(40, 55)))

# 单子节点方向归一化的下限（m）：预测子关节离父关节不足 0.1 mm 时方向无定义，改用参考骨架中的方向，
# 骨长仍成立、梯度有界；身体最短骨约 1.7 cm，正常输入不会触发。
DEFAULT_DIRECTION_EPS = 1e-4
# Horn 幂迭代的重复平方次数（等价于 2^20 次幂迭代）：val 上模型输出的 Procrustes 矩阵收敛比最大 0.99928
# （手部点云近平面、头部模板近共面），10 次平方时 0.3% 的手部旋转与 SVD 解相差 >1e-6（最大 0.3），
# 18 次起与 SVD 全部一致到 1e-13。
DEFAULT_HORN_SQUARINGS = 20
# 两骨 IK 的可达区间收缩比例：肢体完全伸直或完全折叠时肘/膝圆半径为 0，sqrt 的导数发散；
# 收缩 1e-4 后最小半径约 5 mm（腿），对应梯度放大约 35 倍。val GT 无触发（身体恒等误差 ≤ 2e-6 m）。
LIMB_REACH_MARGIN = 1e-4
# 四肢：(根, 中间, 末端)；根由 pelvis 组或锁骨单链先放好。
LIMB_CHAINS = ((1, 4, 7), (2, 5, 8), (16, 18, 20), (17, 19, 21))
# 反向中 (tr S · I − S) 的正则：相对项使正常输入下的梯度偏差 < 1e-5（gradcheck 可过），绝对项（m²，
# 对应 0.1 mm 尺度）只在全部子关节塌缩到父关节时起作用，保证梯度有限。
BACKWARD_REL_EPS = 1e-6
BACKWARD_ABS_EPS = 1e-8

# 软接触权重的温度（m）：脚高每偏离阈值 1 cm，权重变化约 e 倍。
DEFAULT_SOFT_CONTACT_TEMPERATURE = 0.01


# ---------------------------------------------------------------- SMPL-X 静息骨架


def _round_to_tf32(array):
    """float32 -> TF32（10 位尾数，就近舍入、平局远离零），结果仍存为 float32。"""
    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + np.uint64(0x1000)) & ~np.uint64(0x1FFF)
    return bits.astype(np.uint32).view(np.float32)


def smplx_rest_joints(npz_path=SMPLX_NEUTRAL_NPZ, emulate_tf32=True):
    """betas=0、expression=0 的中性 SMPL-X 静息关节 [55,3]（float64）与 parents。

    全部 xyz（旧窗口缓存、新序列缓存、在线训练 FK）都在 Ampere GPU 上由 torch 1.7 生成，
    `J_regressor @ v_template` 的 einsum 默认走 TF32：精确 float64 静息骨长与数据最多差 2.6e-4 m，
    先把两个输入舍入到 TF32 再相乘，与 train/val/test 缓存的骨长最大差 2.1e-6 m（平均 4e-7 m）。
    """
    data = np.load(npz_path, allow_pickle=True)
    regressor = data["J_regressor"].astype(np.float32)
    template = data["v_template"].astype(np.float32)
    if emulate_tf32:
        regressor, template = _round_to_tf32(regressor), _round_to_tf32(template)
    joints = regressor.astype(np.float64) @ template.astype(np.float64)
    parents = data["kintree_table"][0].astype(np.int64)
    parents[0] = -1
    parents = tuple(int(p) for p in parents[:SMPLX_NUM_JOINTS])
    if parents != tuple(SMPLX_PARENTS):
        raise ValueError("SMPL-X kintree 与 utils.ntu2p_naturalness.SMPLX_PARENTS 不一致")
    return torch.from_numpy(joints[:SMPLX_NUM_JOINTS].copy()), parents


# ---------------------------------------------------------------- Procrustes（SO(3) 内加权最优旋转）


def _skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(vector.shape[:-1] + (3, 3))


def _vee_antisym(matrix):
    """vee(X − Xᵀ)，满足 <X, [h]×> = h · vee(X − Xᵀ)。"""
    return torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )


def _solve_sym3(matrix, rhs):
    """对称 3×3 线性方程组的伴随矩阵闭式解，全部逐元素，避免批量 LU 在 CUDA 上的逐矩阵开销。"""
    a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    d, e, f = matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2]
    adj = torch.stack(
        (
            d * f - e * e, c * e - b * f, b * e - c * d,
            c * e - b * f, a * f - c * c, b * c - a * e,
            b * e - c * d, b * c - a * e, a * d - b * b,
        ),
        dim=-1,
    ).reshape(matrix.shape)
    det = a * adj[..., 0, 0] + b * adj[..., 1, 0] + c * adj[..., 2, 0]
    return (adj * rhs.unsqueeze(-2)).sum(-1) / det.unsqueeze(-1)


def _quaternion_to_matrix(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _horn_rotation(cross, squarings=DEFAULT_HORN_SQUARINGS):
    """cross = Σ w d rᵀ [...,3,3] -> 使 tr(Rᵀ cross) 最大的 R ∈ SO(3)（把模板 r 转到目标 d）。"""
    s = cross.transpose(-1, -2)  # Horn 记号 S_ab = Σ r_a d_b
    sxx, sxy, sxz = s[..., 0, 0], s[..., 0, 1], s[..., 0, 2]
    syx, syy, syz = s[..., 1, 0], s[..., 1, 1], s[..., 1, 2]
    szx, szy, szz = s[..., 2, 0], s[..., 2, 1], s[..., 2, 2]
    n = torch.stack(
        (
            sxx + syy + szz, syz - szy, szx - sxz, sxy - syx,
            syz - szy, sxx - syy - szz, sxy + syx, szx + sxz,
            szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy,
            sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz,
        ),
        dim=-1,
    ).reshape(cross.shape[:-2] + (4, 4))
    # N 的特征值绝对值不超过 σ1+σ2+σ3 ≤ √3‖cross‖_F，平移后半正定，重复平方即收敛到主特征向量的外积。
    # 半正定阵的迹不小于最大特征值，用迹归一化防溢出，比逐矩阵求范数便宜；归一化后最大特征值 ≥ 1/4，
    # 连续平方 4 次最小降到 4^-16，远离 float64 下溢，所以每 4 次归一化一次即可。
    shift = torch.sqrt(3.0 * (cross * cross).sum(dim=(-1, -2))) + 1e-12
    power = n + shift[..., None, None] * torch.eye(4, dtype=n.dtype, device=n.device)
    for step in range(int(squarings)):
        if step % 4 == 0:
            power = power / power.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        power = torch.matmul(power, power)
    column = power.diagonal(dim1=-2, dim2=-1).argmax(dim=-1)
    quat = torch.gather(power, -1, column[..., None, None].expand(power.shape[:-1] + (1,))).squeeze(-1)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    return _quaternion_to_matrix(quat)


class _ProcrustesRotation(torch.autograd.Function):
    """R = argmax_{R∈SO(3)} tr(Rᵀ M)，反向用最优性条件的闭式导数。

    最优时 S = Rᵀ M 对称。对 dR = R [ω]× 求微分得 (tr S · I − S) ω = vee(Rᵀ dM − dMᵀ R)，
    因此 dL/dM = R [h]×，h = (tr S · I − S)⁻¹ vee(Rᵀ G − Gᵀ R)，G = dL/dR。
    (tr S · I − S) 的特征值为 s_j + s_k（S 的特征值两两之和），在全局最优处半正定，只在 M 秩 ≤ 1
    （全部子关节共线或塌缩）或"需反射修正且两个最小奇异值相等"（最优旋转本身不唯一）时奇异。
    这比直接对 torch.svd 求导稳定：后者含 1/(σ_i² − σ_j²)，奇异值相等时即为 inf/nan。
    """

    @staticmethod
    def forward(ctx, cross, squarings):
        rotation = _horn_rotation(cross, squarings)
        ctx.save_for_backward(cross, rotation)
        return rotation

    @staticmethod
    def backward(ctx, grad_rotation):
        cross, rotation = ctx.saved_tensors
        sym = torch.matmul(rotation.transpose(-1, -2), cross)
        sym = 0.5 * (sym + sym.transpose(-1, -2))
        trace = sym.diagonal(dim1=-2, dim2=-1).sum(-1)
        eye = torch.eye(3, dtype=sym.dtype, device=sym.device)
        reg = BACKWARD_REL_EPS * trace.abs() + BACKWARD_ABS_EPS
        system = (trace + reg)[..., None, None] * eye - sym
        rhs = _vee_antisym(torch.matmul(rotation.transpose(-1, -2), grad_rotation))
        h = _solve_sym3(system, rhs)
        return torch.matmul(rotation, _skew(h)), None


def procrustes_rotation(cross, squarings=DEFAULT_HORN_SQUARINGS):
    """cross [...,3,3] = Σ_k w_k d_k r_kᵀ -> R [...,3,3]，使 Σ_k w_k |R r_k − d_k|² 最小。

    内部固定用 float64：重复平方对舍入敏感，且 float32 matmul 在 Ampere 上默认走 TF32。
    """
    return _ProcrustesRotation.apply(cross.to(torch.float64), int(squarings)).to(cross.dtype)


def _outer_sum(target, template):
    """Σ_k target_k template_kᵀ：逐元素乘加，[...,K,3] × [...,K,3] -> [...,3,3]（支持广播）。"""
    return (target.unsqueeze(-1) * template.unsqueeze(-2)).sum(dim=-3)


def _rotate(rotation, points):
    """R [...,3,3] 作用于点集 [...,K,3] -> [...,K,3]。"""
    return (rotation.unsqueeze(-3) * points.unsqueeze(-2)).sum(dim=-1)


# ---------------------------------------------------------------- A4 骨架投影


def _unit_or(vector, fallback, eps):
    """单位化；长度不足 eps（方向无定义）时改用 fallback 方向，保证输出骨长仍然成立。"""
    norm = vector.norm(dim=-1, keepdim=True)
    fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(norm > eps, vector / norm.clamp_min(eps), fallback)


def _two_bone(root, mid_pred, end_pred, upper, lower, ref_mid, ref_end, eps=1e-6):
    """解析两骨 IK：root 已定，返回 (mid, end)，|mid−root|=upper、|end−mid|=lower。

    end 取预测末端在可达球壳上的最近点；mid 在两球交线圆上、朝预测 mid 的垂直分量方向（圆上离预测
    mid 最近的点）。ref_mid/ref_end 为参考骨架中 root→mid、root→end 的向量，只在预测退化（末端与根重合、
    预测 mid 恰在轴上）时提供方向。
    """
    unit = _unit_or(end_pred - root, ref_end, eps)
    dist = ((end_pred - root) * unit).sum(dim=-1, keepdim=True)
    low = (upper - lower).abs() * (1.0 + LIMB_REACH_MARGIN) + eps
    reach = torch.min(torch.max(dist, low), (upper + lower) * (1.0 - LIMB_REACH_MARGIN))
    along = (upper * upper - lower * lower + reach * reach) / (2.0 * reach)
    radius = torch.sqrt((upper * upper - along * along).clamp_min(0.0))

    def perpendicular(vector):
        return vector - (vector * unit).sum(dim=-1, keepdim=True) * unit

    pole = _unit_or(perpendicular(mid_pred - root), perpendicular(ref_mid), eps)
    return root + along * unit + radius * pole, root + reach * unit


def _hierarchy_steps(parents, excluded, limbs=LIMB_CHAINS):
    """按深度分层：同层单子节点合并为一步；多子节点父关节各一步（Procrustes）；四肢在根关节所在层之后各合并为一步。"""
    body = [j for j in range(len(parents)) if j not in excluded]
    children = {j: [c for c in body if parents[c] == j] for j in body}
    depth = {}
    for joint in body:
        depth[joint] = 0 if parents[joint] < 0 else depth[parents[joint]] + 1
    for root, mid, end in limbs:
        if parents[mid] != root or parents[end] != mid or children[mid] != [end]:
            raise ValueError("四肢链 {} 与 kintree 不一致".format((root, mid, end)))
    limb_joints = {j for _, mid, end in limbs for j in (mid, end)}
    steps = []
    for level in range(1, max(depth.values()) + 1):
        joints = [j for j in body if depth[j] == level and j not in limb_joints]
        singles = tuple(j for j in joints if len(children[parents[j]]) == 1)
        if singles:
            steps.append(("single", tuple(parents[j] for j in singles), singles))
        for parent in sorted({parents[j] for j in joints if len(children[parents[j]]) > 1}):
            steps.append(("group", parent, tuple(children[parent])))
        chains = [chain for chain in limbs if depth[chain[0]] == level]
        if chains:
            steps.append(("limb", tuple(c[0] for c in chains), tuple(c[1] for c in chains) + tuple(c[2] for c in chains)))
    return tuple(steps)


SKELETON_SOURCES = ("ref", "rest")


class SkeletonProjector(nn.Module):
    """把 [B,T,P,55,3] 投影到固定骨架（身体骨长恒定）+ 刚体手；可微，对旋转、平移等变（ref 模式下对反射也等变）。

    forward(xyz, ref)：ref [B,P,55,3] 为同一人的一帧合法姿态（通常是观测末帧），提供手部模板；
    skeleton="ref"（默认）时骨长与多子节点模板也取自 ref，"rest" 时取自 SMPL-X 静息骨架 J_rest。
    默认用 ref 的原因：
    - 镜像增广（A7）后左右骨长互换，而 SMPL-X 静息骨架左右不对称（大腿/上臂差 1.7 cm）：按 J_rest 投影
      镜像后的 GT，身体关节平均偏 7.7 mm、最大 4.6 cm；ref 随样本一起镜像，没有这个问题；
    - 数据骨架由 GPU 上的 TF32 FK 得到，与 J_rest 仍差 ≤ 2.6e-6 m，两骨 IK 在肢体接近伸直时会放大这一差异；
      取 ref 后首帧（等于观测末帧时）与合法输入都精确不变，也不依赖缓存是在哪种设备上生成的。
    J_rest 与 parents 作为 buffer 保留，用于 "rest" 模式和骨长核对。buffers 不进 state_dict：它们完全由
    body model 文件决定，旧 checkpoint 加载不受影响。

    已投影关节按层级拼接成一个张量、用预存的列索引取父关节，算子数与层级数而非关节数成正比
    （逐关节切片在反向时每个关节都要分配一次整张量的零梯度）。

    detach_parents=True 时前向不变，但每个关节的梯度只回到它自己（及同组兄弟）的预测：精确梯度会把
    全部后代的残差汇总到祖先（val 上 pelvis 梯度约为自由 mse 的 5.8 倍、手臂 7 倍，与自由 mse 梯度的余弦
    0.37），相当于额外加重 root 项；detach 后整体 1.16 倍、余弦 0.69。
    """

    def __init__(
        self,
        skeleton="ref",
        rest_joints=None,
        child_weights=None,
        direction_eps=DEFAULT_DIRECTION_EPS,
        horn_squarings=DEFAULT_HORN_SQUARINGS,
        detach_parents=False,
    ):
        super(SkeletonProjector, self).__init__()
        if skeleton not in SKELETON_SOURCES:
            raise ValueError("skeleton 必须是 {}，当前为 {}".format(SKELETON_SOURCES, skeleton))
        smplx_joints, parents = smplx_rest_joints()
        rest = smplx_joints if rest_joints is None else torch.as_tensor(rest_joints, dtype=torch.float64)
        if tuple(rest.shape) != (SMPLX_NUM_JOINTS, 3):
            raise ValueError("rest_joints 必须是 [55,3]，当前为 {}".format(tuple(rest.shape)))
        for wrist, fingers in zip(HAND_WRISTS, HAND_FINGERS):
            for finger in fingers:
                ancestor = parents[finger]
                while ancestor not in (wrist, -1):
                    ancestor = parents[ancestor]
                if ancestor != wrist:
                    raise ValueError("手指关节 {} 不在手腕 {} 的子树中".format(finger, wrist))
        weights = torch.ones(SMPLX_NUM_JOINTS, dtype=torch.float64)
        if child_weights is not None:
            weights = torch.as_tensor(child_weights, dtype=torch.float64).reshape(SMPLX_NUM_JOINTS)
        self.skeleton = skeleton
        self.direction_eps = float(direction_eps)
        self.horn_squarings = int(horn_squarings)
        self.detach_parents = bool(detach_parents)
        parent_index = torch.tensor([max(p, 0) for p in parents], dtype=torch.long)
        self._buffer("J_rest", rest)
        self._buffer("parents", torch.tensor(parents, dtype=torch.long))
        self._buffer("parent_index", parent_index)
        self._buffer("rest_offset", rest - rest[parent_index])
        self._buffer("child_weight", weights.unsqueeze(-1))

        finger_list = [j for group in HAND_FINGERS for j in group]
        self._steps = _hierarchy_steps(parents, set(finger_list))
        column = {0: 0}
        self._plan = []
        for index, (kind, parent_spec, children) in enumerate(self._steps):
            prefix = "step{}_".format(index)
            parent_joints = list(parent_spec) if kind != "group" else [parent_spec]
            self._buffer(prefix + "child", torch.tensor(children, dtype=torch.long))
            self._buffer(prefix + "parent", torch.tensor([column[p] for p in parent_joints], dtype=torch.long))
            for joint in children:
                column[joint] = len(column)
            self._plan.append((kind, prefix))
        self._buffer("wrist_column", torch.tensor([column[w] for w in HAND_WRISTS], dtype=torch.long))
        self._buffer("wrist_index", torch.tensor(HAND_WRISTS, dtype=torch.long))
        self._buffer("finger_index", torch.tensor(finger_list, dtype=torch.long))
        for joint in finger_list:
            column[joint] = len(column)
        self._buffer("joint_order", torch.tensor([column[j] for j in range(SMPLX_NUM_JOINTS)], dtype=torch.long))

    def _buffer(self, name, value):
        self.register_buffer(name, value, persistent=False)

    def _parent(self, value):
        return value.detach() if self.detach_parents else value

    def forward(self, xyz, ref):
        if xyz.dim() != 5 or tuple(xyz.shape[-2:]) != (SMPLX_NUM_JOINTS, 3):
            raise ValueError("xyz 必须是 [B,T,P,55,3]，当前为 {}".format(tuple(xyz.shape)))
        if tuple(ref.shape) != (xyz.shape[0],) + tuple(xyz.shape[2:]):
            raise ValueError("ref 必须是 [B,P,55,3]，当前为 {}".format(tuple(ref.shape)))
        x = xyz.to(torch.float64)
        ref = ref.to(torch.float64).unsqueeze(1)  # [B,1,P,55,3]，在时间维广播
        ref_offset = ref - ref.index_select(-2, self.parent_index)
        # 骨架来源：每个关节相对父关节的偏移（单子节点只用其长度，多子节点与四肢兜底方向用其向量）。
        # 整个模型被 .float()/.half() 时浮点 buffer 会随之降精度，这里统一回到 float64。
        offset = ref_offset if self.skeleton == "ref" else self.rest_offset.to(torch.float64)
        length = offset.norm(dim=-1, keepdim=True)
        child_weight = self.child_weight.to(torch.float64)
        eps = self.direction_eps

        placed = x[..., :1, :]
        for kind, prefix in self._plan:
            child = getattr(self, prefix + "child")
            parent = self._parent(placed.index_select(-2, getattr(self, prefix + "parent")))
            pred = x.index_select(-2, child)
            child_offset = offset.index_select(-2, child)
            if kind == "limb":
                # child 为 [mid..., end...]，parent 为各链根。
                count = parent.shape[-2]
                child_length = length.index_select(-2, child)
                mid, end = _two_bone(
                    parent,
                    pred[..., :count, :],
                    pred[..., count:, :],
                    child_length[..., :count, :],
                    child_length[..., count:, :],
                    child_offset[..., :count, :],
                    child_offset[..., :count, :] + child_offset[..., count:, :],
                )
                placed = torch.cat((placed, mid, end), dim=-2)
                continue
            target = pred - parent
            if kind == "single":
                new = parent + length.index_select(-2, child) * _unit_or(target, child_offset, eps)
            else:
                cross = _outer_sum(target * child_weight.index_select(-2, child), child_offset)
                new = parent + _rotate(procrustes_rotation(cross, self.horn_squarings), child_offset)
            placed = torch.cat((placed, new), dim=-2)

        finger_shape = (len(HAND_WRISTS), -1, 3)
        # 手部模板 [B,1,P,2,15,3]：参考帧手指相对本手手腕的位置（两种骨架来源下都取 ref）。
        template = ref.index_select(-2, self.finger_index).reshape(ref.shape[:-2] + finger_shape)
        template = template - ref.index_select(-2, self.wrist_index).unsqueeze(-2)
        wrist = self._parent(placed.index_select(-2, self.wrist_column)).unsqueeze(-2)  # [B,T,P,2,1,3]
        target = x.index_select(-2, self.finger_index).reshape(x.shape[:-2] + finger_shape) - wrist
        weight = child_weight.index_select(-2, self.finger_index).reshape(finger_shape[:-1] + (1,))
        cross = _outer_sum(target * weight, template)
        fingers = wrist + _rotate(procrustes_rotation(cross, self.horn_squarings), template)
        placed = torch.cat((placed, fingers.flatten(-3, -2)), dim=-2)
        return placed.index_select(-2, self.joint_order).to(xyz.dtype)


def bone_lengths(xyz, parents=SMPLX_PARENTS):
    """[...,55,3] -> [...,54] 各骨（按子关节 1..54）长度。"""
    children = list(range(1, len(parents)))
    return (xyz[..., children, :] - xyz[..., [parents[c] for c in children], :]).norm(dim=-1)


# ---------------------------------------------------------------- A5 脚接触


def _unit_up(up, ndim):
    up = up / up.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return up.reshape((up.shape[0],) + (1,) * (ndim - 2) + (3,))


def _height(points, up):
    return (points * _unit_up(up, points.dim())).sum(dim=-1)


def _horizontal(vectors, up):
    unit = _unit_up(up, vectors.dim())
    return vectors - (vectors * unit).sum(dim=-1, keepdim=True) * unit


def _feet_with_obs(pred_xyz, obs_xyz, feet, smooth):
    """[B,To+T,2,F,3]：观测 + 预测的脚关节轨迹，按需做与 R1 相同的时间平滑。"""
    feet = list(feet)
    full = torch.cat((obs_xyz[:, :, :, feet].to(pred_xyz.dtype), pred_xyz[:, :, :, feet]), dim=1)
    return smooth_time(full) if smooth else full


def foot_horizontal_velocity(pred_xyz, obs_xyz, up, feet=FOOT_JOINTS, smooth=True):
    """[B,T,2,F,3]：脚关节水平速度（m/s），首帧速度相对观测末帧。

    默认先做 R1 的 5 点二项平滑：GT 含约 5 mm/帧的拟合抖动，不平滑时 GT 自身在接触帧上的速度平方
    （val 0.041）与模型（0.045）几乎相同，损失失去区分度；平滑后为 0.0073 对 0.040。
    """
    obs_len = int(obs_xyz.shape[1])
    full = _feet_with_obs(pred_xyz, obs_xyz, feet, smooth)
    return _horizontal(full[:, obs_len:] - full[:, obs_len - 1 : -1], up) * NTU2P_FPS


def observed_ground(obs_xyz, up, feet=FOOT_JOINTS):
    """[B,2]：每人观测期内最低脚关节沿 up 的高度，作为不依赖 GT 的地面零点（只供自洽版使用）。"""
    return _height(obs_xyz[..., list(feet), :], up).amin(dim=3).amin(dim=1)


def foot_contact_mask(target_xyz, obs_xyz, frame=None):
    """[B,T,2,4] float：GT 接触帧，直接复用 R1 的 `gt_contact_mask`（与自然度评估同一口径）。

    脚关节为左右踝（7、8）与左右 foot（10、11，SMPL-X 的脚掌/脚尖关节）：后跟着地时踝先停，蹬离前
    foot 关节最后离地，两者合起来覆盖整个站立相。判据：相对本人地面高度低于阈值（踝 10 cm、foot 5 cm）
    且平滑后水平速度 < 0.2 m/s，阈值由 R1 在 val GT 上校准。
    地面与竖直方向用 R1 的 `estimate_scene_frame`（obs + GT future 拟合每人地面平面）：训练时 GT 可用，
    而只用观测估计会随行走漂移——`estimate_up(obs)` 与 R1 竖直方向差中位 3.3°、p90 5.8°，步行人
    f41–50 最低脚高的 p90 偏离达 0.21 m（R1 平面 0.054 m），与 R1 掩码在步行子集上 IoU 仅 0.72。
    """
    with torch.no_grad():
        obs = obs_xyz.double()
        target = target_xyz.double()
        if frame is None:
            frame = estimate_scene_frame(torch.cat((obs, target), dim=1))
        return gt_contact_mask(target, obs, frame).to(target_xyz.dtype)


def foot_skate_loss(pred_xyz, target_xyz, obs_xyz, up, mask=None, smooth=True):
    """GT 接触帧上预测脚（平滑后）水平速度平方的均值，单位 (m/s)²。

    与 L2 目标同向（GT 在这些帧上本来不动），但把"何时踩实"的相位显式交给模型。copy-last 恒约为 0，
    不能作为归一化尺度。val：GT 0.0073、模型 0.040、base 0.014、纯滑行参考 0.019；步行子集 GT 0.0135、
    模型 0.204。
    """
    if mask is None:
        mask = foot_contact_mask(target_xyz, obs_xyz)
    speed_sq = foot_horizontal_velocity(pred_xyz, obs_xyz, up, FOOT_JOINTS, smooth).pow(2).sum(dim=-1)
    mask = mask.to(speed_sq.dtype)
    return (mask * speed_sq).sum() / mask.sum().clamp_min(1.0)


def self_skate_loss(
    pred_xyz,
    obs_xyz,
    up,
    feet=FOOT_JOINTS,
    height_thresholds=None,
    temperature=DEFAULT_SOFT_CONTACT_TEMPERATURE,
    smooth=True,
    detach_weight=True,
):
    """不依赖 GT 的自洽脚滑：以预测自身脚高（相对观测地面）的软接触权重 sigmoid((阈值 − 高度)/温度)
    加权水平速度平方。

    权重默认 detach：否则最省力的降损方式是把脚抬离地面，而不是让着地的脚停住。
    不建议作训练损失：val 上 GT 反而远高于模型（平滑后 0.22 对 0.072），因为只凭高度分不开"贴地的
    摆动脚"与"滑行的脚"——GT 的损失 70% 来自速度 > 1 m/s 的低位帧（其中 39% 在非步行窗口，如踢、垫步），
    而模型的脚几乎不动。
    """
    feet = list(feet)
    obs_len = int(obs_xyz.shape[1])
    full = _feet_with_obs(pred_xyz, obs_xyz, feet, smooth)
    ground = observed_ground(obs_xyz.to(pred_xyz.dtype), up, feet)
    height = _height(full[:, obs_len:], up) - ground[:, None, :, None]
    values = [CONTACT_HEIGHT_THRESHOLDS[j] for j in feet] if height_thresholds is None else list(height_thresholds)
    thresholds = torch.tensor(values, dtype=height.dtype, device=height.device)
    weight = torch.sigmoid((thresholds - height) / float(temperature))
    if detach_weight:
        weight = weight.detach()
    speed_sq = (_horizontal(full[:, obs_len:] - full[:, obs_len - 1 : -1], up) * NTU2P_FPS).pow(2).sum(dim=-1)
    return (weight * speed_sq).sum() / weight.sum().clamp_min(1e-6)
