"""NTU 双人 xyz 预测的"自然度"指标：脚底滑行、骨长一致性、步态交替、分部位/分 horizon 误差、交互几何。

L2 类指标（xyz_mse/mpjpe）对"腿在走还是在滑"不敏感：一条以 root 速度匀速平移、脚不离地的轨迹，
其 L2 可以低于真实迈步但相位对不上的预测。本模块给出与相位无关、能直接对应视频观感的指标。

坐标约定（已在 val GT 上核实）：
- 数据是 NTU Kinect 相机坐标，竖直轴近似 -y，但相机有 0~15 度俯仰，且不同 setup 的相机高度不同
  （GT 最低脚点的 y 跨样本分布在 0.86~2.2 m），所以地面不能用全局常数。
- SMPL-X 用中性体型拟合，不同人的脚相对真实地面会浮起或下沉（同一样本两人地面差中位数约 7 cm），
  因此地面按"每样本每人"估计：竖直方向取 head - 双脚中点 的时间/双人平均方向，再对每人每帧最低
  脚点做带岭约束的鲁棒平面拟合，地面偏移取残差 5% 分位。
- 场景几何（竖直方向、地面）只由 obs + GT future 估计，再同样施加到每个待评估方法上，
  这样所有方法共用同一"地面真值"，不会因为预测本身的误差而移动地面。
"""

import math
from collections import OrderedDict

import torch

from utils.ntu2p_canonical import estimate_up, horizontal


NTU2P_FPS = 20.0

# SMPL-X 55 关节父节点，已与 body_models/smplx/SMPLX_NEUTRAL.npz 的 kintree_table 逐项核对。
SMPLX_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
    20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
)
SMPLX_NUM_JOINTS = 55
SMPLX_BODY_JOINTS = 22

PELVIS, L_HIP, R_HIP = 0, 1, 2
L_ANKLE, R_ANKLE, L_FOOT, R_FOOT = 7, 8, 10, 11
NECK, HEAD = 12, 15
FOOT_JOINTS = (L_ANKLE, R_ANKLE, L_FOOT, R_FOOT)

BODY_PARTS = OrderedDict(
    [
        ("legs", (1, 2, 4, 5, 7, 8, 10, 11)),
        ("torso", (0, 3, 6, 9)),
        ("arms", (13, 14, 16, 17, 18, 19, 20, 21)),
        ("head", (12, 15, 22, 23, 24)),
        ("fingers", tuple(range(25, 55))),
    ]
)

# 骨骼分组：以子关节归属划分，仅用于骨长误差分解。
BONE_GROUPS = OrderedDict(
    [
        ("legs", (1, 2, 4, 5, 7, 8, 10, 11)),
        ("spine", (3, 6, 9, 12, 15)),
        ("arms", (13, 14, 16, 17, 18, 19, 20, 21)),
        ("fingers", tuple(range(25, 55))),
    ]
)

# ---- 阈值：全部由 val GT（obs10 + future50，198 条 x 2 人）分布校准，理由写在旁边 ----
# 接触高度：GT 踝关节相对地面高度静止模态 4~8 cm，脚尖（SMPL-X foot 关节）0~2 cm；各加约 3~4 cm 余量。
CONTACT_HEIGHT_THRESHOLDS = OrderedDict([(L_ANKLE, 0.10), (R_ANKLE, 0.10), (L_FOOT, 0.05), (R_FOOT, 0.05)])
# 接触速度：GT 站立脚平滑后水平速度中位数约 0.07 m/s（拟合抖动），步行窗口摆动相 > 1 m/s，
# 两峰之间在 0.2~0.3 m/s 出现低谷；0.2 m/s = 1 cm/帧，约为 GT 步行 root 中位速度 0.41 m/s 的一半。
CONTACT_SPEED_THRESHOLD = 0.20
# 步行窗口：GT future50 内 root 水平净位移 >= 0.5 m（2.5 s 平均 0.2 m/s）；GT 位移分布在 0.05 m 以下
# 与 0.6~1.5 m 两处成团，0.5 m 落在低谷。
WALK_DISPLACEMENT_THRESHOLD = 0.50
# 滑行帧：root 在动（>0.2 m/s）且脚水平速度与 root 速度矢量差 < 30% root 速度，即脚"跟着骨盆平移"。
SLIDE_ROOT_SPEED_MIN = 0.20
SLIDE_LOCK_TOLERANCE = 0.30
# 步数计数的迟滞：左右踝前向分离需越过 ±3 cm 才计一次换脚，抑制拟合抖动造成的伪过零。
STEP_HYSTERESIS = 0.03
# 地面穿透/悬空：最低脚点低于地面 3 cm 视为穿透；双脚最低点高于 10 cm 视为悬空（GT 中跳跃极少）。
PENETRATION_DEPTH = 0.03
FLOAT_HEIGHT = 0.10
# 速度类指标前的时间平滑：5 点二项核（sigma≈1 帧）。GT 含 Kinect/SMPL-X 拟合抖动（约 5 mm/帧），
# 不平滑时 GT 自身的"滑步"会被抖动主导；对所有方法施加同一平滑以保持公平。
SMOOTH_KERNEL = (1.0, 4.0, 6.0, 4.0, 1.0)
# 分 horizon：每 10 帧（0.5 s）一段。
HORIZON_BIN_FRAMES = 10
# 步态相位（gait_phase_stats）：观测末 5 个帧差的 pelvis 平均水平速度超过 0.2 m/s 视为"已在走"，
# 其余步行人为"起步"（val 上 29 / 70 人）；GT 左右踝前后分离首次偏离观测末帧 0.1 m 的帧定义"先迈哪只脚"。
GAIT_MOVING_FRAMES = 5
GAIT_MOVING_SPEED = 0.20
GAIT_LEAD_DEVIATION = 0.10
# 前 0.5 s 的相位由观测决定（可预测），之后逐段变成多模态；f31-50 只作长程护栏。
GAIT_BINS = ((1, 10), (11, 20), (21, 30), (31, 50))
# 预测或 GT 在 bin 内几乎不动时相关无定义，不计入分母：预测侧如 copy-last；GT 侧是少数样本的整段重复帧
# （val 的 f01_10 有 1 人、f11_20 有 2 人），按 r=0 计入会把所有方法的均值同样拉向 0。
GAIT_CORR_MIN_NORM = 1e-6


def _check_xyz(name, value):
    if value.dim() != 5 or tuple(value.shape[2:]) != (2, SMPLX_NUM_JOINTS, 3):
        raise ValueError("{} 必须是 [B,T,2,55,3]，当前为 {}".format(name, tuple(value.shape)))
    if not torch.isfinite(value).all():
        raise ValueError("{} 存在非有限数值".format(name))


def _normalize(value, eps=1e-8):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


def smooth_time(value, kernel=SMOOTH_KERNEL):
    """沿 dim=1（时间）做归一化卷积平滑，边界复制填充。"""
    weights = torch.tensor(kernel, dtype=value.dtype, device=value.device)
    weights = weights / weights.sum()
    pad = (len(kernel) - 1) // 2
    moved = value.transpose(1, -1)
    shape = moved.shape
    flat = moved.reshape(-1, 1, shape[-1])
    flat = torch.nn.functional.pad(flat, (pad, pad), mode="replicate")
    out = torch.nn.functional.conv1d(flat, weights.view(1, 1, -1))
    return out.reshape(shape).transpose(1, -1).contiguous()


# ---------------------------------------------------------------- 场景几何


def estimate_scene_frame(reference_xyz, ridge=1.0, iterations=3):
    """由参考运动（obs + GT future）估计每样本竖直方向与每人地面平面。

    返回 dict：basis [B,3,3]（行向量 e1,e2,up，右手系），plane [B,2,3]（h = a*u + b*v + c 的 a,b,c），
    offset [B,2]（残差 5% 分位，作为地面零点）。
    """
    _check_xyz("reference_xyz", reference_xyz)
    ref = reference_xyz.double()
    batch = ref.shape[0]
    feet_mid = ref[..., list(FOOT_JOINTS), :].mean(dim=-2)
    # 两人面对面时前倾方向相反，时间+双人平均能抵消大部分躯干前倾偏差。
    up = _normalize((ref[..., HEAD, :] - feet_mid).mean(dim=(1, 2)))
    helper = torch.zeros_like(up)
    helper[:, 0] = 1.0
    e1 = _normalize(helper - (helper * up).sum(-1, keepdim=True) * up)
    e2 = torch.cross(up, e1, dim=-1)
    basis = torch.stack((e1, e2, up), dim=1)

    coords = torch.einsum("btpjc,bkc->btpjk", ref, basis)
    feet = coords[..., list(FOOT_JOINTS), :]
    low_index = feet[..., 2].argmin(dim=-1)
    lowest = torch.gather(feet, 3, low_index[..., None, None].expand(-1, -1, -1, 1, 3)).squeeze(3)
    points = lowest.permute(0, 2, 1, 3)  # [B,2,T,3]
    design = torch.cat((points[..., :2], torch.ones_like(points[..., :1])), dim=-1)
    height = points[..., 2]
    weight = torch.ones_like(height)
    reg = torch.diag(torch.tensor([ridge, ridge, 0.0], dtype=ref.dtype, device=ref.device))
    for _ in range(int(iterations)):
        lhs = torch.einsum("bptc,bpt,bptd->bpcd", design, weight, design) + reg
        rhs = torch.einsum("bptc,bpt,bpt->bpc", design, weight, height)
        plane = torch.matmul(torch.inverse(lhs), rhs.unsqueeze(-1)).squeeze(-1)
        residual = height - (design * plane.unsqueeze(2)).sum(-1)
        # 双脚离地的帧（跳、踢）会把平面抬高，只保留残差低分位附近的点。
        cutoff = torch.quantile(residual, 0.2, dim=-1, keepdim=True) + 0.05
        weight = (residual < cutoff).to(ref.dtype) + 1e-3
    offset = torch.quantile(residual, 0.05, dim=-1)
    return {"basis": basis, "plane": plane, "offset": offset, "batch_size": batch}


def to_scene_coords(xyz, frame):
    """相机坐标 -> 场景坐标 (u, v, h)，h 为相对本人地面的高度。xyz [B,T,2,J,3]。"""
    coords = torch.einsum("btpjc,bkc->btpjk", xyz.double(), frame["basis"])
    plane = frame["plane"]
    ground = (coords[..., :2] * plane[:, None, :, None, :2]).sum(-1) + plane[:, None, :, None, 2]
    ground = ground + frame["offset"][:, None, :, None]
    return torch.cat((coords[..., :2], (coords[..., 2] - ground).unsqueeze(-1)), dim=-1)


def _smoothed_future_scene(future_xyz, obs_xyz, frame):
    """拼接 obs 再平滑，避免 future 首帧边界效应；返回 future 段场景坐标与逐帧水平速度（m/s）。"""
    full = to_scene_coords(torch.cat((obs_xyz, future_xyz), dim=1), frame)
    smoothed = smooth_time(full)
    obs_len = int(obs_xyz.shape[1])
    future = smoothed[:, obs_len:]
    velocity = (smoothed[:, obs_len:, ..., :2] - smoothed[:, obs_len - 1 : -1, ..., :2]) * NTU2P_FPS
    return future, velocity


# ---------------------------------------------------------------- 统计容器

def _stat(num, den):
    return (num.double(), den.double())


def _masked_mean_stat(value, mask, reduce_dims):
    mask = mask.double()
    return _stat((value.double() * mask).sum(dim=reduce_dims), mask.sum(dim=reduce_dims))


def aggregate_stats(stats, index=None):
    """per-sample (num, den) -> 标量 sum(num)/sum(den)；index 用于按动作等子集聚合。"""
    result = OrderedDict()
    for key, (num, den) in stats.items():
        if index is not None:
            num = num[index]
            den = den[index]
        total = float(den.sum().item())
        result[key] = float(num.sum().item()) / total if total > 0 else float("nan")
        result[key + "__count"] = total
    return result


# ---------------------------------------------------------------- 各类指标


def walk_mask(target_xyz, obs_xyz, frame, threshold=WALK_DISPLACEMENT_THRESHOLD):
    """[B,2]：GT future 内 root 水平净位移（相对 obs 末帧）超过阈值的人。"""
    scene = to_scene_coords(torch.cat((obs_xyz[:, -1:], target_xyz), dim=1), frame)
    root = scene[:, :, :, PELVIS, :2]
    return (root[:, -1] - root[:, 0]).norm(dim=-1) >= float(threshold)


def gt_contact_mask(target_xyz, obs_xyz, frame):
    """[B,T,2,4]：GT 中脚部关节低于高度阈值且平滑水平速度低于速度阈值的帧。"""
    scene, velocity = _smoothed_future_scene(target_xyz, obs_xyz, frame)
    feet = list(FOOT_JOINTS)
    thresholds = torch.tensor([CONTACT_HEIGHT_THRESHOLDS[j] for j in feet], dtype=scene.dtype, device=scene.device)
    low = scene[..., feet, 2] < thresholds
    slow = velocity[..., feet, :].norm(dim=-1) < CONTACT_SPEED_THRESHOLD
    return low & slow


def foot_skating_stats(pred_xyz, obs_xyz, frame, contact, walking, bin_frames=HORIZON_BIN_FRAMES):
    """两类滑步：
    - gt_contact：GT 判定站定的帧上，方法的脚水平速度（GT 自身 < 速度阈值，是参考下限）与违例比例；
    - self：方法自身脚高度低于阈值的帧上（不看 GT），按 GMD 式高度权重 2-2^(h/H) 加权的水平速度，
      以及"低位且速度超阈值"的帧占比（文献常用的 skating ratio）。
    全体窗口中大部分是原地动作，滑步被稀释，因此另报 GT 步行窗口子集（后缀 _walk）。
    """
    scene, velocity = _smoothed_future_scene(pred_xyz, obs_xyz, frame)
    feet = list(FOOT_JOINTS)
    speed = velocity[..., feet, :].norm(dim=-1)
    height = scene[..., feet, 2]
    thresholds = torch.tensor([CONTACT_HEIGHT_THRESHOLDS[j] for j in feet], dtype=scene.dtype, device=scene.device)
    dims = (1, 2, 3)
    self_low = height < thresholds
    violation = (speed > CONTACT_SPEED_THRESHOLD).double()
    weight = (2.0 - torch.pow(2.0, height.clamp_min(0.0) / thresholds)).clamp(0.0, 1.0)
    walk = walking[:, None, :, None].expand_as(contact)
    stats = OrderedDict()
    for suffix, subset in (("", torch.ones_like(contact)), ("_walk", walk)):
        gt_contact = contact & subset
        stats["skate_gt_contact_speed" + suffix] = _masked_mean_stat(speed, gt_contact, dims)
        stats["skate_gt_contact_violation" + suffix] = _masked_mean_stat(violation, gt_contact, dims)
        sub_weight = weight * subset.double()
        stats["skate_self_weighted_speed" + suffix] = _stat((speed * sub_weight).sum(dim=dims), sub_weight.sum(dim=dims))
        stats["skate_self_ratio" + suffix] = _masked_mean_stat(violation, self_low & subset, dims)
    seq_len = int(speed.shape[1])
    for start in range(0, seq_len, bin_frames):
        stop = min(start + bin_frames, seq_len)
        mask = torch.zeros_like(contact)
        mask[:, start:stop] = True
        key = "skate_gt_contact_speed_f{:02d}_{:02d}".format(start + 1, stop)
        stats[key] = _masked_mean_stat(speed, contact & mask, dims)
    stats["gt_contact_fraction"] = _stat(contact.double().sum(dim=dims), torch.full_like(contact.double().sum(dim=dims), float(contact[0].numel())))
    return stats


def floor_stats(pred_xyz, frame):
    scene = to_scene_coords(pred_xyz, frame)
    lowest = scene[..., list(FOOT_JOINTS), 2].min(dim=-1).values  # [B,T,2]
    ones = torch.ones_like(lowest)
    dims = (1, 2)
    stats = OrderedDict()
    stats["foot_penetration_ratio"] = _masked_mean_stat((lowest < -PENETRATION_DEPTH).double(), ones, dims)
    stats["foot_float_ratio"] = _masked_mean_stat((lowest > FLOAT_HEIGHT).double(), ones, dims)
    stats["lowest_foot_height"] = _masked_mean_stat(lowest, ones, dims)
    return stats


def bone_lengths(xyz):
    """[B,T,2,54]：按 SMPL-X 父子关系的骨长。"""
    children = list(range(1, SMPLX_NUM_JOINTS))
    parents = [SMPLX_PARENTS[j] for j in children]
    return (xyz[..., children, :] - xyz[..., parents, :]).norm(dim=-1)


def bone_length_stats(pred_xyz, obs_xyz):
    """骨长相对观测末帧的相对偏差；GT 由固定体型 FK 得到，理论上恒为 0。"""
    reference = bone_lengths(obs_xyz[:, -1:].double()).clamp_min(1e-4)
    rel = (bone_lengths(pred_xyz.double()) - reference).abs() / reference  # [B,T,2,54]
    absolute = (bone_lengths(pred_xyz.double()) - bone_lengths(obs_xyz[:, -1:].double())).abs()
    stats = OrderedDict()
    body_bones = [j - 1 for j in range(1, SMPLX_BODY_JOINTS)]
    ones = torch.ones_like(rel[..., 0])
    # 相对误差对短骨（骨盆-脊柱、锁骨约 10 cm）敏感，另报绝对误差（m）。
    stats["bone_abs_err_body"] = _stat(absolute[..., body_bones].mean(-1).sum(dim=(1, 2)), ones.sum(dim=(1, 2)))
    stats["bone_rel_err_body"] = _stat(rel[..., body_bones].mean(-1).sum(dim=(1, 2)), ones.sum(dim=(1, 2)))
    for group, joints in BONE_GROUPS.items():
        bones = [j - 1 for j in joints]
        stats["bone_rel_err_" + group] = _stat(rel[..., bones].mean(-1).sum(dim=(1, 2)), ones.sum(dim=(1, 2)))
    # 每样本每人全时段最大伸缩，反映"肢体被拉长/压短"的最坏情况。
    worst = rel[..., body_bones].amax(dim=(1, 3))  # [B,2]
    stats["bone_rel_err_body_max"] = _stat(worst.sum(-1), torch.full_like(worst.sum(-1), 2.0))
    leg_bones = [j - 1 for j in BONE_GROUPS["legs"]]
    worst_leg = rel[..., leg_bones].amax(dim=(1, 3))
    stats["bone_rel_err_legs_max"] = _stat(worst_leg.sum(-1), torch.full_like(worst_leg.sum(-1), 2.0))
    return stats


def _pearson(a, b, dim=-1, eps=1e-8):
    a = a - a.mean(dim=dim, keepdim=True)
    b = b - b.mean(dim=dim, keepdim=True)
    return (a * b).sum(dim) / (a.norm(dim=dim) * b.norm(dim=dim)).clamp_min(eps)


def _count_alternations(signal, hysteresis):
    """带迟滞的符号交替次数：signal [N,T]，返回 [N]。"""
    counts = torch.zeros(signal.shape[0], dtype=torch.float64, device=signal.device)
    state = torch.zeros(signal.shape[0], dtype=torch.float64, device=signal.device)
    for t in range(signal.shape[1]):
        value = signal[:, t]
        new_state = torch.where(value > hysteresis, torch.ones_like(state), state)
        new_state = torch.where(value < -hysteresis, -torch.ones_like(state), new_state)
        counts = counts + ((state != 0) & (new_state != state)).double()
        state = new_state
    return counts


def gait_stats(pred_xyz, target_xyz, obs_xyz, frame, walking):
    """步行窗口（按 GT 判定，所有方法共用）上的步态/滑行指标，逐人计算后在有效人上平均。

    前向方向取 GT root 在 future 内的水平净位移方向，使各方法投影到同一"应走方向"上。
    """
    scene, velocity = _smoothed_future_scene(pred_xyz, obs_xyz, frame)
    gt_scene = to_scene_coords(torch.cat((obs_xyz[:, -1:], target_xyz), dim=1), frame)
    gt_disp = gt_scene[:, -1, :, PELVIS, :2] - gt_scene[:, 0, :, PELVIS, :2]
    forward = _normalize(gt_disp)  # [B,2,2]

    root_vel = velocity[..., PELVIS, :]  # [B,T,2,2]
    root_speed = root_vel.norm(dim=-1)
    ankle_vel = velocity[..., [L_ANKLE, R_ANKLE], :]  # [B,T,2,2(L/R),2]
    rel_vel = ankle_vel - root_vel.unsqueeze(3)
    ankle_speed = ankle_vel.norm(dim=-1)  # [B,T,2,2]

    stats = OrderedDict()
    mask = walking.double()  # [B,2]

    def add(name, per_person):
        stats[name] = _stat((per_person * mask).sum(-1), mask.sum(-1))

    # 下限 0.1 m：方法自身 root 几乎不动（如 copy-last）时比值退化为 ~0，而不是被除零放大。
    root_path = root_speed.sum(dim=1).clamp_min(0.1)  # [B,2]
    # 脚相对骨盆的运动量 / 骨盆运动量：真实步行约 1（站立相脚静止、摆动相超越骨盆），整体滑移 → 0。
    add("foot_rel_motion_ratio", (rel_vel.norm(dim=-1).sum(dim=1).mean(-1)) / root_path)
    # 脚水平路程 / 骨盆水平路程（不依赖接触判定的位移比）。
    add("foot_to_root_path_ratio", (ankle_vel.norm(dim=-1).sum(dim=1).mean(-1)) / root_path)
    moving = (root_speed > SLIDE_ROOT_SPEED_MIN).double()  # [B,T,2]
    locked = (rel_vel.norm(dim=-1) < SLIDE_LOCK_TOLERANCE * root_speed.unsqueeze(-1)).double().mean(-1)
    # 滑行帧比：root 在动的帧中，脚速度矢量与 root 几乎相同（锁定平移）的比例。
    add("slide_frame_ratio", (locked * moving).sum(1) / moving.sum(1).clamp_min(1.0))
    add("moving_frame_fraction", moving.mean(1))
    # 脚速度 95 分位 / root 平均速度：步行摆动相峰值约为 root 速度的 2~3 倍，滑移接近 1。
    peak = torch.quantile(ankle_speed.permute(0, 2, 3, 1).reshape(ankle_speed.shape[0], 2, -1), 0.95, dim=-1)
    add("ankle_peak_speed", peak)
    add("root_speed_mean", root_speed.mean(1))
    # 站立相占比：踝水平速度低于接触速度阈值的帧比例。真实步行每只脚约一半时间静止；
    # 平滑回归常见的"脚速 = root 速度 + 小正弦"永远到不了 0，占比会显著偏低。
    add("stance_fraction", (ankle_speed < CONTACT_SPEED_THRESHOLD).double().mean(dim=(1, 3)))
    add("ankle_peak_to_root_speed", peak / root_speed.mean(1).clamp_min(1e-3))

    fwd = forward.unsqueeze(1)  # [B,1,2,2]
    left_fwd = (ankle_vel[..., 0, :] * fwd).sum(-1)  # [B,T,2]
    right_fwd = (ankle_vel[..., 1, :] * fwd).sum(-1)
    # 真实步行左右脚前向速度交替（一只站立一只摆动）→ 负相关；整体平移 → 两脚同速 → 正相关。
    add("lr_forward_vel_corr", _pearson(left_fwd.transpose(1, 2), right_fwd.transpose(1, 2)))
    rel_left = (rel_vel[..., 0, :] * fwd).sum(-1)
    rel_right = (rel_vel[..., 1, :] * fwd).sum(-1)
    add("lr_rel_forward_vel_corr", _pearson(rel_left.transpose(1, 2), rel_right.transpose(1, 2)))
    # 相位是否对齐：方法与 GT 的"踝相对骨盆前向速度"逐脚时间相关，1 为同相，0 为无关，负为反相。
    _, gt_velocity = _smoothed_future_scene(target_xyz, obs_xyz, frame)
    gt_rel = gt_velocity[..., [L_ANKLE, R_ANKLE], :] - gt_velocity[..., PELVIS, :].unsqueeze(3)
    gt_rel_fwd = (gt_rel * fwd.unsqueeze(3)).sum(-1)  # [B,T,2,2]
    pred_rel_fwd = (rel_vel * fwd.unsqueeze(3)).sum(-1)
    phase_corr = _pearson(pred_rel_fwd.permute(0, 2, 3, 1), gt_rel_fwd.permute(0, 2, 3, 1)).mean(-1)
    add("rel_fwd_vel_corr_with_gt", phase_corr)

    ankle_rel_pos = scene[..., [L_ANKLE, R_ANKLE], :2] - scene[..., PELVIS : PELVIS + 1, :2]  # [B,T,2,2,2]
    ankle_fwd = (ankle_rel_pos * fwd.unsqueeze(3)).sum(-1)  # [B,T,2,2]
    swing_amp = ankle_fwd.std(dim=1).mean(-1)  # [B,2]
    add("leg_swing_amp", swing_amp)
    add("swing_per_speed", swing_amp / root_speed.mean(1).clamp_min(1e-3))
    separation = (ankle_fwd[..., 0] - ankle_fwd[..., 1]).permute(0, 2, 1).reshape(-1, ankle_fwd.shape[1])
    steps = _count_alternations(separation, STEP_HYSTERESIS).reshape(ankle_fwd.shape[0], 2)
    add("step_count", steps)
    return stats


def gait_phase_stats(pred_xyz, target_xyz, obs_xyz):
    """步行人的"左右踝沿行进方向前后分离" s(t) 与 GT 的逐段相位一致性（有符号、不平滑）。

    gait_stats 的 rel_fwd_vel_corr_with_gt 用平滑后的速度且全程一个相关，分不出"前 0.5 s 可预测的相位是否延续"；
    这里按 10 帧分段逐人算 Pearson 与 RMSE，并按观测内状态分"已在走 / 起步"。
    口径：竖直 = estimate_up(obs)；步行人 = GT future 末帧 pelvis 相对 obs 末帧的水平位移 ≥ 0.5 m；
    行进方向 f = 该位移方向；s(t) = (x_L_ankle(t) - x_R_ankle(t))·f（相机系），t = future 第 1..50 帧。
    返回 OrderedDict[name -> (num[B], den[B])]，与 compute_naturalness_stats 同容器。
    """
    pred, target, obs = pred_xyz.double(), target_xyz.double(), obs_xyz.double()
    up = estimate_up(obs)  # [B,3]
    displacement = horizontal(target[:, -1, :, PELVIS] - obs[:, -1, :, PELVIS], up)  # [B,2,3]
    walking = displacement.norm(dim=-1) >= WALK_DISPLACEMENT_THRESHOLD  # [B,2]
    forward = _normalize(displacement).unsqueeze(1)  # [B,1,2,3]
    step = horizontal(obs[:, 1:, :, PELVIS] - obs[:, :-1, :, PELVIS], up)  # [B,T-1,2,3]
    moving = step.norm(dim=-1)[:, -GAIT_MOVING_FRAMES:].mean(1) * NTU2P_FPS > GAIT_MOVING_SPEED  # [B,2]

    def separation(value):
        return ((value[..., L_ANKLE, :] - value[..., R_ANKLE, :]) * forward).sum(-1)  # [B,T,2]

    s_pred, s_gt = separation(pred), separation(target)
    s_last = ((obs[:, -1, :, L_ANKLE] - obs[:, -1, :, R_ANKLE]) * forward[:, 0]).sum(-1)  # [B,2]
    groups = OrderedDict([("", walking), ("_moving", walking & moving), ("_starting", walking & ~moving)])
    stats = OrderedDict()
    for first, last in GAIT_BINS:
        pred_bin, gt_bin = s_pred[:, first - 1 : last], s_gt[:, first - 1 : last]
        pred_c, gt_c = pred_bin - pred_bin.mean(1, keepdim=True), gt_bin - gt_bin.mean(1, keepdim=True)
        pred_norm, gt_norm = pred_c.norm(dim=1), gt_c.norm(dim=1)
        corr = (pred_c * gt_c).sum(1) / (pred_norm * gt_norm).clamp_min(1e-10)
        defined = (pred_norm >= GAIT_CORR_MIN_NORM) & (gt_norm >= GAIT_CORR_MIN_NORM)
        rmse = ((pred_bin - gt_bin) ** 2).mean(1).sqrt()
        tag = "f{:02d}_{:02d}".format(first, last)
        for suffix, mask in groups.items():
            stats["gait_sep_corr_" + tag + suffix] = _masked_mean_stat(corr, mask & defined, -1)
            stats["gait_sep_rmse_" + tag + suffix] = _masked_mean_stat(rmse, mask, -1)
    deviation = s_gt - s_last.unsqueeze(1)
    hit = deviation.abs() > GAIT_LEAD_DEVIATION
    has_event = hit.any(1)
    # 不用 argmax 取首个命中：torch 1.7 对并列最大值不保证返回第一个。
    frames = torch.arange(hit.shape[1], device=hit.device).view(1, -1, 1).expand_as(hit)
    first_hit = torch.where(hit, frames, torch.full_like(frames, hit.shape[1] - 1)).min(1, keepdim=True).values
    gt_sign = torch.sign(deviation.gather(1, first_hit)[:, 0])
    pred_sign = torch.sign((s_pred - s_last.unsqueeze(1)).gather(1, first_hit)[:, 0])
    for suffix, mask in groups.items():
        stats["gait_lead_foot_acc" + suffix] = _masked_mean_stat((gt_sign == pred_sign).double(), mask & has_event, -1)
    # 每样本平均人数（× 样本数 = 总人数），供核对子集大小。
    ones = torch.ones(walking.shape[0], dtype=torch.float64, device=walking.device)
    stats["gait_walk_count"] = _stat(walking.double().sum(-1), ones)
    stats["gait_moving_count"] = _stat((walking & moving).double().sum(-1), ones)
    return stats


def body_part_error_stats(pred_xyz, target_xyz, bin_frames=HORIZON_BIN_FRAMES):
    """分部位 mpjpe（全局与相对 root 的局部），并按 horizon 每 bin_frames 帧分段。"""
    pred = pred_xyz.double()
    target = target_xyz.double()
    global_err = (pred - target).norm(dim=-1)  # [B,T,2,55]
    local_err = ((pred - pred[..., PELVIS : PELVIS + 1, :]) - (target - target[..., PELVIS : PELVIS + 1, :])).norm(dim=-1)
    seq_len = int(pred.shape[1])
    bins = [(start, min(start + bin_frames, seq_len)) for start in range(0, seq_len, bin_frames)]
    stats = OrderedDict()
    count = torch.full((pred.shape[0],), float(seq_len), dtype=torch.float64, device=pred.device)
    for name, err in (("mpjpe", global_err), ("local_mpjpe", local_err)):
        stats[name] = _stat(err.mean(dim=(2, 3)).sum(1), count)
        # 55 关节里 30 个是手指，整体 mpjpe 被手指主导；身体 22 关节单列。
        stats[name + "_body22"] = _stat(err[..., :SMPLX_BODY_JOINTS].mean(dim=(2, 3)).sum(1), count)
        for part, joints in BODY_PARTS.items():
            part_err = err[..., list(joints)].mean(dim=(2, 3))  # [B,T]
            stats["{}_{}".format(name, part)] = _stat(part_err.sum(1), torch.full_like(part_err[:, 0], float(seq_len)))
            for start, stop in bins:
                key = "{}_{}_f{:02d}_{:02d}".format(name, part, start + 1, stop)
                stats[key] = _stat(part_err[:, start:stop].sum(1), torch.full_like(part_err[:, 0], float(stop - start)))
        for start, stop in bins:
            key = "{}_all_f{:02d}_{:02d}".format(name, start + 1, stop)
            stats[key] = _stat(err[:, start:stop].mean(dim=(2, 3)).sum(1), torch.full_like(err[:, 0, 0, 0], float(stop - start)))
    root_err = global_err[..., PELVIS]  # [B,T,2]
    for start, stop in bins:
        key = "root_err_f{:02d}_{:02d}".format(start + 1, stop)
        stats[key] = _stat(root_err[:, start:stop].mean(-1).sum(1), torch.full_like(root_err[:, 0, 0], float(stop - start)))
    return stats


def _heading(scene):
    """水平前向 [B,T,2,2]：(左髋-右髋) × up 投影到水平面；场景坐标中 up=(0,0,1)，结果为 (dv, -du)。"""
    hip = scene[..., L_HIP, :2] - scene[..., R_HIP, :2]
    return _normalize(torch.stack((hip[..., 1], -hip[..., 0]), dim=-1))


def _angle_deg(a, b):
    cos = (a * b).sum(-1).clamp(-1.0, 1.0)
    return torch.acos(cos) * (180.0 / math.pi)


def interaction_stats(pred_xyz, target_xyz, frame):
    """双人几何：root 距离误差、各自朝向误差、相对朝向（A 前向与 A→B 方向夹角）误差、最近身体关节距离误差。"""
    pred = to_scene_coords(pred_xyz, frame)
    target = to_scene_coords(target_xyz, frame)
    stats = OrderedDict()
    seq_len = float(pred.shape[1])
    count = torch.full((pred.shape[0],), seq_len, dtype=torch.float64, device=pred.device)

    def root_dist(value):
        return (value[:, :, 0, PELVIS, :2] - value[:, :, 1, PELVIS, :2]).norm(dim=-1)

    stats["root_distance_abs_err"] = _stat((root_dist(pred) - root_dist(target)).abs().sum(1), count)
    heading_pred, heading_gt = _heading(pred), _heading(target)
    stats["heading_err_deg"] = _stat(_angle_deg(heading_pred, heading_gt).mean(-1).sum(1), count)

    def facing(value, heading):
        to_other = torch.stack(
            (value[:, :, 1, PELVIS, :2] - value[:, :, 0, PELVIS, :2], value[:, :, 0, PELVIS, :2] - value[:, :, 1, PELVIS, :2]),
            dim=2,
        )
        return _angle_deg(heading, _normalize(to_other))  # [B,T,2]

    stats["facing_angle_abs_err_deg"] = _stat((facing(pred, heading_pred) - facing(target, heading_gt)).abs().mean(-1).sum(1), count)
    body = list(range(SMPLX_BODY_JOINTS))

    def min_dist(value):
        a = value[:, :, 0, body].unsqueeze(3)
        b = value[:, :, 1, body].unsqueeze(2)
        return (a - b).norm(dim=-1).amin(dim=(2, 3))  # [B,T]

    stats["min_interperson_dist_abs_err"] = _stat((min_dist(pred) - min_dist(target)).abs().sum(1), count)
    stats["min_interperson_dist"] = _stat(min_dist(pred).sum(1), count)
    return stats


def jitter_stats(pred_xyz, obs_xyz):
    """身体关节（前 22 个）三阶差分 jerk 均值，m/s^3；不做平滑，用于识别高频抖动。"""
    full = torch.cat((obs_xyz[:, -3:], pred_xyz), dim=1).double()[..., :SMPLX_BODY_JOINTS, :]
    jerk = full[:, 3:] - 3 * full[:, 2:-1] + 3 * full[:, 1:-2] - full[:, :-3]
    magnitude = jerk.norm(dim=-1) * (NTU2P_FPS ** 3)
    count = torch.full((pred_xyz.shape[0],), float(magnitude[0].numel()), dtype=torch.float64, device=pred_xyz.device)
    return OrderedDict([("jerk_body", _stat(magnitude.sum(dim=(1, 2, 3)), count))])


def compute_naturalness_stats(pred_xyz, target_xyz, obs_xyz, frame=None):
    """返回 OrderedDict[name -> (num[B], den[B])]，用 aggregate_stats 做全集或子集聚合。

    target_xyz 始终是 GT：它定义地面、接触帧与步行窗口；评估 GT 自身时传 pred_xyz=target_xyz。
    """
    _check_xyz("pred_xyz", pred_xyz)
    _check_xyz("target_xyz", target_xyz)
    _check_xyz("obs_xyz", obs_xyz)
    if tuple(pred_xyz.shape) != tuple(target_xyz.shape):
        raise ValueError("pred_xyz/target_xyz shape 必须一致")
    obs = obs_xyz.double()
    target = target_xyz.double()
    pred = pred_xyz.double()
    if frame is None:
        frame = estimate_scene_frame(torch.cat((obs, target), dim=1))
    contact = gt_contact_mask(target, obs, frame)
    walking = walk_mask(target, obs, frame)
    stats = OrderedDict()
    stats.update(foot_skating_stats(pred, obs, frame, contact, walking))
    stats.update(floor_stats(pred, frame))
    stats.update(bone_length_stats(pred, obs))
    stats.update(gait_stats(pred, target, obs, frame, walking))
    stats.update(body_part_error_stats(pred, target))
    stats.update(interaction_stats(pred, target, frame))
    stats.update(jitter_stats(pred, obs))
    stats["walking_person_fraction"] = _stat(walking.double().sum(-1), torch.full_like(walking.double().sum(-1), 2.0))
    stats.update(gait_phase_stats(pred, target, obs))
    return stats


def per_sample_values(stats, keys):
    """把 (num, den) 转为逐样本值 [B]（den=0 处为 nan），便于选例与排序。"""
    values = OrderedDict()
    for key in keys:
        num, den = stats[key]
        values[key] = torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(num, float("nan")))
    return values
