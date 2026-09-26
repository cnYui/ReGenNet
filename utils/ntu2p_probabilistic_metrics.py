"""Track B 概率预测指标：K 个样本对 1 个 GT 的能量分数、best-of-K、多样性、校准、先迈脚 Brier、步数分布与手臂幅度。

约定：samples [K,B,T,2,55,3]，target [B,T,2,55,3]，obs [B,10,2,55,3]，全部在相机系；"局部"= 关节 − 同帧 pelvis。
人物分组与 utils/ntu2p_naturalness.gait_phase_stats 同一口径（estimate_up 水平分量、0.5 m、末 5 帧 > 0.2 m/s）。
设计：docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md 第 5.1、5.3 节。

评估是按窗口块流式进行的，所以大部分函数给出逐窗（或逐人）的量，汇总在调用方做；需要全局比值的量
（能量分数、SSR）另给 *_parts 形式的分子分母。
"""

import math
import random
from collections import OrderedDict

import torch

from utils.ntu2p_canonical import apply_linear, estimate_up, horizontal
from utils.ntu2p_naturalness import (
    BODY_PARTS,
    GAIT_LEAD_DEVIATION,
    GAIT_MOVING_FRAMES,
    GAIT_MOVING_SPEED,
    L_ANKLE,
    NTU2P_FPS,
    PELVIS,
    R_ANKLE,
    SMPLX_BODY_JOINTS,
    STEP_HYSTERESIS,
    WALK_DISPLACEMENT_THRESHOLD,
    _count_alternations,
    _smoothed_future_scene,
    estimate_scene_frame,
    to_scene_coords,
    walk_mask,
)
from utils.ntu2p_residual_codec import codec_frames
from utils.ntu_smplx_2p_xyz import compute_ntu_xyz_metrics


ARM_ACTIONS = (0, 2, 3, 4, 6, 8, 11, 12, 13, 14, 15, 17, 18, 23, 24, 25)
STRIKE_ACTIONS = (0, 2, 11, 12, 13)
ARM_AMPLITUDE_MIN = 0.10
STATIC_DISPLACEMENT = 0.10
STATIC_OBS_SPEED = 0.10
INTERPEN_NEAR = 0.05
INTERPEN_GT_MIN = 0.15

LEG_JOINTS = BODY_PARTS["legs"]
ARM_JOINTS = BODY_PARTS["arms"]
WRISTS = (20, 21)
GROUP_KEYS = ("walking", "moving", "starting", "static", "arm_action", "strike")
LEAD_LEFT, LEAD_RIGHT, LEAD_NONE = 1, -1, 0
LEAD_CLASSES = (LEAD_LEFT, LEAD_RIGHT, LEAD_NONE)
APD_PARTS = ("all", "root", "legs_local", "arms_local")
SSR_PARTS = ("root", "legs", "arms")
# 审查图预选（只看 GT）：起步者 6、已在走 2、击打类 4、其余手臂动作 2、随机 4。
REVIEW_QUOTA = (("starting", 6), ("moving", 2), ("strike", 4), ("arm_action", 2), ("random", 4))


def _unit(value, eps=1e-8):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


def _local(value):
    return value - value[..., PELVIS : PELVIS + 1, :]


def _select(value, part):
    """part -> 逐关节量：all=全部关节全局，root=pelvis 全局，legs/arms(_local)=部位局部。"""
    if part == "all":
        return value
    if part == "root":
        return value[..., PELVIS : PELVIS + 1, :]
    if part in ("legs", "legs_local"):
        return _local(value)[..., list(LEG_JOINTS), :]
    if part in ("arms", "arms_local"):
        return _local(value)[..., list(ARM_JOINTS), :]
    raise ValueError("未知部位 {}".format(part))


def _obs_motion(obs_xyz, target_xyz):
    """(GT 水平位移 [B,2,3], 观测末 5 帧平均水平速度 [B,2])，竖直方向 = estimate_up(obs)。"""
    obs, target = obs_xyz.double(), target_xyz.double()
    up = estimate_up(obs)
    displacement = horizontal(target[:, -1, :, PELVIS] - obs[:, -1, :, PELVIS], up)
    step = horizontal(obs[:, 1:, :, PELVIS] - obs[:, :-1, :, PELVIS], up)
    speed = step.norm(dim=-1)[:, -GAIT_MOVING_FRAMES:].mean(1) * NTU2P_FPS
    return displacement, speed


def arm_amplitude(xyz, obs_xyz):
    """[...,B,T,2,55,3] -> [...,B,2]：未来帧左右腕相对 pelvis 的位置相对观测末帧的最大位移（copy-last 为 0）。"""
    wrists = list(WRISTS)
    relative = xyz[..., wrists, :] - xyz[..., PELVIS : PELVIS + 1, :]
    last = (obs_xyz[:, -1:, :, wrists] - obs_xyz[:, -1:, :, PELVIS : PELVIS + 1]).to(relative.dtype)
    return (relative - last).norm(dim=-1).amax(dim=-1).amax(dim=-2)


def person_groups(obs_xyz, target_xyz, actions):
    """OrderedDict[name -> bool [B,2]]，只用观测、GT 与动作标签（所有方法共用同一分组）。"""
    displacement, speed = _obs_motion(obs_xyz, target_xyz)
    distance = displacement.norm(dim=-1)
    walking = distance >= WALK_DISPLACEMENT_THRESHOLD
    obs_moving = speed > GAIT_MOVING_SPEED
    amplitude = arm_amplitude(target_xyz.double(), obs_xyz.double())
    actions = torch.as_tensor(actions).to(obs_xyz.device).long().view(-1, 1)
    in_arm = torch.zeros_like(actions, dtype=torch.bool)
    for action in ARM_ACTIONS:
        in_arm = in_arm | (actions == action)
    in_strike = torch.zeros_like(in_arm)
    for action in STRIKE_ACTIONS:
        in_strike = in_strike | (actions == action)
    big_arm = amplitude >= ARM_AMPLITUDE_MIN
    return OrderedDict(
        [
            ("walking", walking),
            ("moving", walking & obs_moving),
            ("starting", walking & ~obs_moving),
            ("static", (distance < STATIC_DISPLACEMENT) & (speed < STATIC_OBS_SPEED)),
            ("arm_action", in_arm & big_arm),
            ("strike", in_strike & big_arm),
        ]
    )


# ---------------------------------------------------------------- L2 与 best-of-K


def window_mpjpe(pred, target):
    """[...,B,T,2,J,3] -> [...,B]。"""
    return (pred - target).norm(dim=-1).mean(dim=(-3, -2, -1))


def sample_l2(samples, target, obs):
    """single：K 个样本各自的 L2 再对 K 平均；mean_of_k：先对 K 平均再算 L2（带 var/K 项）。"""
    single = OrderedDict()
    for sample in samples:
        for key, value in compute_ntu_xyz_metrics(sample, target, obs).items():
            single[key] = single.get(key, 0.0) + float(value) / float(samples.shape[0])
    return OrderedDict([("single", single), ("mean_of_k", compute_ntu_xyz_metrics(samples.mean(dim=0), target, obs))])


def min_ade_from_per_sample(per_sample, ks=(1, 5, 10, 20)):
    """per_sample [K,B] 窗 mpjpe -> {"min_ade@k": [B]}（取前 k 个样本，A/B 用同一样本）。"""
    count = int(per_sample.shape[0])
    return OrderedDict(("min_ade@{}".format(k), per_sample[: int(k)].min(dim=0).values) for k in ks if int(k) <= count)


def min_ade_curve(samples, target, ks=(1, 5, 10, 20)):
    return min_ade_from_per_sample(window_mpjpe(samples, target), ks)


def final_frame_error(samples, target):
    """[K,B]：末帧全部关节平均距离。"""
    return (samples[..., -1, :, :, :] - target[:, -1]).norm(dim=-1).mean(dim=(-2, -1))


def min_fde(samples, target, k=10):
    return final_frame_error(samples, target)[: int(k)].min(dim=0).values


def person_ade_body22(samples, target):
    """[K,B,2]：每人 body22 的平均关节误差。"""
    body = slice(0, SMPLX_BODY_JOINTS)
    return (samples[..., body, :] - target[..., body, :]).norm(dim=-1).mean(dim=(-3, -1))


def person_min_ade_body22(samples, target, k=10):
    return person_ade_body22(samples, target)[: int(k)].min(dim=0).values


# ---------------------------------------------------------------- 能量分数、多样性、校准


def _masked_ratio(num, den):
    return torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(num, float("nan")))


def energy_score_parts(samples, target, joints=None, local=False, person_mask=None, window_mask=None):
    """逐窗 (分子 [B], 分母 [B])：分子 = 被选人物的逐人 ES 之和，分母 = 被选人数；汇总时按人数加权。

    ES = mean_{t,j}[(1/K)Σ_k d(x_k,y) − (1/(2K(K−1)))Σ_{k≠l} d(x_k,x_l)]，d 为欧氏距离；K=1 时第二项为 0，ES = mpjpe。
    """
    x, y = samples.double(), target.double()
    if local:
        x, y = _local(x), _local(y)
    if joints is not None:
        x, y = x[..., list(joints), :], y[..., list(joints), :]
    count = int(x.shape[0])
    first = (x - y.unsqueeze(0)).norm(dim=-1).mean(dim=0)  # [B,T,2,J]
    second = torch.zeros_like(first)
    if count > 1:
        for k in range(count - 1):
            second = second + (x[k + 1 :] - x[k : k + 1]).norm(dim=-1).sum(dim=0)
        second = second / float(count * (count - 1))
    per_person = (first - second).mean(dim=(1, 3))  # [B,2]
    mask = torch.ones_like(per_person, dtype=torch.bool) if person_mask is None else person_mask.to(per_person.device).bool()
    if window_mask is not None:
        mask = mask & window_mask.to(per_person.device).bool().view(-1, 1)
    mask = mask.double()
    return (per_person * mask).sum(dim=-1), mask.sum(dim=-1)


def energy_score(samples, target, joints=None, local=False, person_mask=None, window_mask=None):
    """-> (人数加权均值, 逐窗值 [B]（无被选人物的窗为 nan）)。"""
    num, den = energy_score_parts(samples, target, joints, local, person_mask, window_mask)
    total = float(den.sum().item())
    mean = float(num.sum().item()) / total if total > 0 else float("nan")
    return mean, _masked_ratio(num, den)


def apd_per_person(samples, part="all"):
    """[B,2]：样本两两之间（k<l）该部位平均关节距离的均值；K=1 时为 0。"""
    x = _select(samples.double(), part)
    count = int(x.shape[0])
    total = torch.zeros(x.shape[1], x.shape[3], dtype=torch.float64, device=x.device)
    if count < 2:
        return total
    for k in range(count - 1):
        total = total + (x[k + 1 :] - x[k : k + 1]).norm(dim=-1).mean(dim=(2, 4)).sum(dim=0)
    return total / float(count * (count - 1) / 2)


def apd(samples, part="all", person_mask=None):
    """-> (均值, 逐窗值 [B])；person_mask 给出时只在被选人物上平均。"""
    per_person = apd_per_person(samples, part)
    mask = torch.ones_like(per_person) if person_mask is None else person_mask.to(per_person.device).double()
    num, den = (per_person * mask).sum(-1), mask.sum(-1)
    total = float(den.sum().item())
    return (float(num.sum().item()) / total if total > 0 else float("nan")), _masked_ratio(num, den)


def allocation_ratio_from(per_person_legs, groups):
    """步行人腿局部 APD / 静止人腿局部 APD：随机性应分给真的在走的人，而不是让站着的人被采样出迈步。"""
    walk, static = groups["walking"].bool(), groups["static"].bool()
    if int(walk.sum()) == 0 or int(static.sum()) == 0:
        return float("nan")
    denominator = float(per_person_legs[static].mean().item())
    return float(per_person_legs[walk].mean().item()) / denominator if denominator > 0 else float("nan")


def allocation_ratio(samples, groups):
    return allocation_ratio_from(apd_per_person(samples, "legs_local"), groups)


def spread_skill_parts(samples, target, part):
    """逐窗 (集合方差之和 [B], 集合均值误差平方和 [B], 元素数 [B])；part ∈ root/legs/arms（后两者为局部）。"""
    x, y = _select(samples.double(), part), _select(target.double(), part)
    count = int(x.shape[0])
    if count < 2:
        raise ValueError("SSR 需要至少 2 个样本")
    var = x.var(dim=0, unbiased=True).flatten(1).sum(dim=1)
    err = (x.mean(dim=0) - y).pow(2).flatten(1).sum(dim=1)
    elements = torch.full_like(var, float(y[0].numel()))
    return var, err, elements


def ssr_from_parts(var, err, elements, num_samples, window_mask=None):
    """SSR = sqrt((K+1)/K) × sqrt(平均集合方差) / sqrt(集合均值的均方误差)；校准时为 1。"""
    if window_mask is not None:
        keep = window_mask.to(var.device).bool()
        var, err, elements = var[keep], err[keep], elements[keep]
    total = float(elements.sum().item())
    if total <= 0 or float(err.sum().item()) <= 0:
        return float("nan")
    spread = math.sqrt(float(var.sum().item()) / total)
    skill = math.sqrt(float(err.sum().item()) / total)
    return math.sqrt((num_samples + 1.0) / num_samples) * spread / skill


def spread_skill(samples, target, part, window_mask=None):
    var, err, elements = spread_skill_parts(samples, target, part)
    return ssr_from_parts(var, err, elements, int(samples.shape[0]), window_mask)


# ---------------------------------------------------------------- 先迈脚、步数、手臂幅度、穿插


def lead_foot_category(xyz, obs_xyz, target_xyz):
    """[...,B,T,2,55,3] -> long [...,B,2]：+1 左脚先、−1 右脚先、0 无。

    s(t) = (左踝 − 右踝)·f，f 为 GT 位移的水平方向（所有方法共用）；取该轨迹自身第一个 |s(t) − s_last| > 0.1 m 的帧，
    按其符号分类。不用 argmax 取首个命中：torch 1.7 对并列最大值不保证返回第一个。
    """
    obs = obs_xyz.double()
    displacement, _ = _obs_motion(obs_xyz, target_xyz)
    forward = _unit(displacement)  # [B,2,3]
    value = xyz.double()
    separation = ((value[..., L_ANKLE, :] - value[..., R_ANKLE, :]) * forward.unsqueeze(-3)).sum(-1)  # [...,B,T,2]
    last = ((obs[:, -1, :, L_ANKLE] - obs[:, -1, :, R_ANKLE]) * forward).sum(-1)  # [B,2]
    deviation = separation - last.unsqueeze(-2)
    hit = deviation.abs() > GAIT_LEAD_DEVIATION
    length = int(hit.shape[-2])
    frames = torch.arange(length, device=hit.device).view(length, 1).expand_as(hit)
    first = torch.where(hit, frames, torch.full_like(frames, length - 1)).min(dim=-2, keepdim=True).values
    sign = torch.sign(deviation.gather(-2, first).squeeze(-2)).long()
    return torch.where(hit.any(dim=-2), sign, torch.zeros_like(sign))


def lead_foot_summary(categories, gt_categories, mask):
    """categories [K,B,2]、gt [B,2]、mask [B,2]（通常为起步者）；只统计 GT 有先迈脚事件的人。

    Brier = Σ_c (p̂_c − 1[c=GT])²（三类：左/右/无），确定性预测正确为 0、错误为 2；熵以 bit 计；
    acc_single = 单样本与 GT 同类的概率。
    """
    valid = mask.bool() & (gt_categories != LEAD_NONE)
    count = int(valid.sum().item())
    result = OrderedDict([("n", count)])
    if count == 0:
        result.update([("brier3", float("nan")), ("entropy_bits", float("nan")), ("acc_single", float("nan"))])
        return result
    brier = torch.zeros(gt_categories.shape, dtype=torch.float64, device=gt_categories.device)
    entropy = torch.zeros_like(brier)
    accuracy = torch.zeros_like(brier)
    for label in LEAD_CLASSES:
        prob = (categories == label).double().mean(dim=0)
        truth = (gt_categories == label).double()
        brier = brier + (prob - truth).pow(2)
        entropy = entropy - torch.where(prob > 0, prob * torch.log2(prob.clamp_min(1e-12)), torch.zeros_like(prob))
        accuracy = accuracy + prob * truth
    result["brier3"] = float(brier[valid].mean().item())
    result["entropy_bits"] = float(entropy[valid].mean().item())
    result["acc_single"] = float(accuracy[valid].mean().item())
    return result


def lead_foot_brier(samples, target, obs, mask):
    return lead_foot_summary(lead_foot_category(samples, obs, target), lead_foot_category(target, obs, target), mask)


def _facing_scene(obs_xyz, frame):
    """观测末帧的个人前向（individual_frames，末 3 帧朝向之和）在场景水平面 (u,v) 上的单位向量 [B,2,2]。"""
    frames = codec_frames(obs_xyz.float())
    forward = apply_linear(frames["person"][:, :, 0], frames["scene"]["rotation"].transpose(-1, -2)).double()  # 相机系
    uv = torch.einsum("bpc,bkc->bpk", forward, frame["basis"][:, :2])
    return _unit(uv)


def person_step_counts(xyz, target_xyz, obs_xyz, frame=None):
    """[B,T,2,55,3] 或 [K,B,...] -> float [...,B,2]：左右踝前后分离带迟滞的换脚次数（与 gait_stats 的 step_count 同口径）。

    步行人（场景系 GT 位移 ≥ 0.5 m）沿 GT 位移方向计数；其余人沿观测末帧朝向计数，用于检查"不该走的人被采样出迈步"。
    """
    obs, target = obs_xyz.double(), target_xyz.double()
    if frame is None:
        frame = estimate_scene_frame(torch.cat((obs, target), dim=1))
    walking = walk_mask(target, obs, frame)
    gt_scene = to_scene_coords(torch.cat((obs[:, -1:], target), dim=1), frame)
    gt_dir = _unit(gt_scene[:, -1, :, PELVIS, :2] - gt_scene[:, 0, :, PELVIS, :2])
    direction = torch.where(walking.unsqueeze(-1), gt_dir, _facing_scene(obs_xyz, frame))  # [B,2,2]
    values = xyz.unsqueeze(0) if xyz.dim() == 5 else xyz

    def count(pred):
        scene, _ = _smoothed_future_scene(pred.double(), obs, frame)
        relative = scene[..., [L_ANKLE, R_ANKLE], :2] - scene[..., PELVIS : PELVIS + 1, :2]  # [B,T,2,2,2]
        along = (relative * direction[:, None, :, None, :]).sum(-1)  # [B,T,2,2]
        separation = (along[..., 0] - along[..., 1]).permute(0, 2, 1).reshape(-1, along.shape[1])
        return _count_alternations(separation, STEP_HYSTERESIS).reshape(pred.shape[0], 2)

    steps = torch.stack([count(pred) for pred in values])
    return steps[0] if xyz.dim() == 5 else steps


def step_crps(sample_steps, gt_steps, mask):
    """样本 CRPS = E|X−y| − ½E|X−X'|（k≠l 无偏），在被选人物上平均；确定性预测时等于 |误差|。"""
    keep = mask.bool()
    if int(keep.sum()) == 0:
        return float("nan")
    x = sample_steps.double()[:, keep]  # [K,N]
    y = gt_steps.double()[keep]
    count = int(x.shape[0])
    first = (x - y.unsqueeze(0)).abs().mean(dim=0)
    second = torch.zeros_like(first)
    if count > 1:
        second = (x.unsqueeze(0) - x.unsqueeze(1)).abs().sum(dim=(0, 1)) / float(count * (count - 1))
    return float((first - 0.5 * second).mean().item())


def step_w1(sample_steps, gt_steps, mask):
    """样本池（K × 被选人物）与 GT 池的 1-Wasserstein 距离；步数为整数，W1 = Σ_n |F_s(n) − F_gt(n)|。"""
    keep = mask.bool()
    if int(keep.sum()) == 0:
        return float("nan")
    pool = sample_steps.double()[:, keep].reshape(-1)
    truth = gt_steps.double()[keep].reshape(-1)
    top = int(max(float(pool.max().item()), float(truth.max().item())))
    total = 0.0
    for n in range(top + 1):
        total += abs(float((pool <= n).double().mean().item()) - float((truth <= n).double().mean().item()))
    return total


def arm_bias_from(sample_amplitude, gt_amplitude, mask):
    """(|E log(A_s/A_GT)|, 比值中位数)；A_GT ≥ 0.1 m 由分组保证，A_s 下限 1e-4 m 防止 log(0)。"""
    keep = mask.bool()
    if int(keep.sum()) == 0:
        return float("nan"), float("nan")
    ratio = sample_amplitude.double()[:, keep].clamp_min(1e-4) / gt_amplitude.double()[keep].unsqueeze(0)
    return abs(float(torch.log(ratio).mean().item())), float(ratio.reshape(-1).median().item())


def arm_amplitude_log_bias(samples, target, obs, mask):
    return arm_bias_from(arm_amplitude(samples.double(), obs.double()), arm_amplitude(target.double(), obs.double()), mask)


def _min_body_distance(xyz):
    """[...,B,T,2,55,3] -> [...,B,T]：两人 body22 关节间最小距离。"""
    body = slice(0, SMPLX_BODY_JOINTS)
    a = xyz[..., 0, body, :].unsqueeze(-2)
    b = xyz[..., 1, body, :].unsqueeze(-3)
    return (a - b).norm(dim=-1).amin(dim=(-2, -1))


def interpenetration_ratio(pred, target):
    """逐窗 [...,B]：预测两人 body22 最小距离 < 0.05 m 而 GT ≥ 0.15 m 的帧比例（GT 本身贴身的帧不算）。"""
    near = _min_body_distance(pred.double()) < INTERPEN_NEAR
    far = _min_body_distance(target.double()) >= INTERPEN_GT_MIN
    return (near & far).double().mean(dim=-1)


# ---------------------------------------------------------------- 判据工具


def envelope_at(xs, ys, x):
    """分段线性插值；超出网格时用该侧端点两点外推，并返回 in_range=False。-> (value, in_range)。"""
    pairs = sorted((float(a), float(b)) for a, b in zip(xs, ys) if not (math.isnan(float(a)) or math.isnan(float(b))))
    if len(pairs) < 2:
        raise ValueError("包络至少需要 2 个网格点")
    x = float(x)
    grid = [p[0] for p in pairs]
    if grid[0] <= x <= grid[-1]:
        for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
            if x0 <= x <= x1:
                if x1 == x0:
                    return y0, True
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0), True
    (x0, y0), (x1, y1) = (pairs[0], pairs[1]) if x < grid[0] else (pairs[-2], pairs[-1])
    slope = 0.0 if x1 == x0 else (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0), False


def window_bootstrap_ci(per_window, n=1000, seed=0, level=0.95):
    """逐窗 bootstrap 置信区间；per_window 为 [N]（忽略 nan）或 (分子 [N], 分母 [N])（比值的和）。"""
    if isinstance(per_window, (tuple, list)):
        num, den = (torch.as_tensor(v).double().reshape(-1) for v in per_window)
    else:
        value = torch.as_tensor(per_window).double().reshape(-1)
        keep = ~torch.isnan(value)
        num, den = value[keep], torch.ones_like(value[keep])
    count = int(num.numel())
    if count == 0:
        return float("nan"), float("nan")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    index = torch.randint(0, count, (int(n), count), generator=generator)
    stats = num[index].sum(dim=1) / den[index].sum(dim=1).clamp_min(1e-12)
    alpha = (1.0 - float(level)) / 2.0
    low, high = torch.quantile(stats, torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64))
    return float(low.item()), float(high.item())


def select_review_cases(obs_xyz, target_xyz, actions, seed=0, sample_ids=None):
    """只按 GT 预选审查窗口（设计 5.5-H），与任何预测无关，便于在看结果前固定。"""
    groups = person_groups(obs_xyz, target_xyz, actions)
    displacement, _ = _obs_motion(obs_xyz, target_xyz)
    distance = displacement.norm(dim=-1)  # [B,2]
    amplitude = arm_amplitude(target_xyz.double(), obs_xyz.double())
    has_event = lead_foot_category(target_xyz, obs_xyz, target_xyz) != LEAD_NONE
    count = int(obs_xyz.shape[0])
    chosen = OrderedDict()

    def ranked(mask, score):
        value = torch.where(mask, score, torch.full_like(score, -1.0)).amax(dim=-1)
        order = torch.argsort(value, descending=True).tolist()
        return [index for index in order if float(value[index]) >= 0 and index not in chosen]

    candidates = OrderedDict(
        [
            ("starting", ranked(groups["starting"] & has_event, distance)),
            ("moving", ranked(groups["moving"], distance)),
            ("strike", ranked(groups["strike"], amplitude)),
            ("arm_action", ranked(groups["arm_action"] & ~groups["strike"], amplitude)),
        ]
    )
    for category, quota in REVIEW_QUOTA:
        if category == "random":
            rest = [index for index in range(count) if index not in chosen]
            picks = random.Random(int(seed)).sample(rest, min(int(quota), len(rest)))
        else:
            picks = [index for index in candidates[category] if index not in chosen][: int(quota)]
        for index in picks:
            chosen[index] = category
    cases = []
    for index, category in chosen.items():
        sample_id = sample_ids[index] if sample_ids is not None else "idx{:04d}".format(index)
        cases.append(OrderedDict([("index", int(index)), ("sample_id", str(sample_id)), ("category", category)]))
    return cases


__all__ = [
    "APD_PARTS",
    "ARM_ACTIONS",
    "GROUP_KEYS",
    "SSR_PARTS",
    "STRIKE_ACTIONS",
    "allocation_ratio",
    "allocation_ratio_from",
    "apd",
    "apd_per_person",
    "arm_amplitude",
    "arm_amplitude_log_bias",
    "arm_bias_from",
    "energy_score",
    "energy_score_parts",
    "envelope_at",
    "final_frame_error",
    "interpenetration_ratio",
    "lead_foot_brier",
    "lead_foot_category",
    "lead_foot_summary",
    "min_ade_curve",
    "min_ade_from_per_sample",
    "min_fde",
    "person_ade_body22",
    "person_groups",
    "person_min_ade_body22",
    "person_step_counts",
    "sample_l2",
    "select_review_cases",
    "spread_skill",
    "spread_skill_parts",
    "ssr_from_parts",
    "step_crps",
    "step_w1",
    "window_bootstrap_ci",
    "window_mpjpe",
]
