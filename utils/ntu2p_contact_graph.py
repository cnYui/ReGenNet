"""NTU2P 软接触图、时间有序接触记忆与双腿相位监督。

掌腕到骨架线段的距离是接触代理，不能解释为皮肤表面距离或物理因果证据。
推理特征只读取传入的轨迹；GT 仅在损失与离线评价中出现。
"""

from collections import OrderedDict

import torch
from torch.nn import functional as F

from utils.ntu2p_canonical import estimate_up, facing_direction, horizontal
from utils.ntu2p_gait_losses import DEFAULT_LEG_GAIT_LOWPASS_K, lowpass_time
from utils.ntu2p_naturalness import NTU2P_FPS, smooth_time


BODY22 = 22
WRISTS = (20, 21)
PALM_BASES = ((25, 31), (40, 46))
TORSO_SEGMENTS = ((0, 3), (3, 6), (6, 9), (9, 12), (9, 16), (9, 17))
CONTACT_THRESHOLDS = (0.15, 0.20, 0.25, 0.30)
RETREAT_SPEED_THRESHOLDS = (0.10, 0.20)
CONTACT_TEMPERATURE = 0.05
PUSHING_ACTION = 2
EVENT_SUSTAIN_FRAMES = 3
EVENT_EARLY_TOLERANCE = 2
EVENT_MAX_DELAY_FRAMES = 12


def _check_xyz(value, name):
    if value.dim() != 5 or value.shape[2] != 2 or value.shape[3] < 47 or value.shape[4] != 3:
        raise ValueError("{} 必须为 [B,T,2,J>=47,3]，需要掌心关节".format(name))
    if value.shape[1] == 0:
        raise ValueError("{} 不能为空轨迹".format(name))


def hand_body_distances(xyz):
    """[B,T,2,55,3] -> [B,T,2,2,6]，人物维为本人的手到另一人躯干。

    每手保留腕与 index1/pinky1 中点两处代理；两点中更近的距离避免手腕偏移漏掉掌心接近。
    线段投影采用截断参数，退化段仍有有限距离与梯度。
    """
    _check_xyz(xyz, "xyz")
    wrist = xyz[..., list(WRISTS), :]
    palm = torch.stack([xyz[..., list(joints), :].mean(dim=-2) for joints in PALM_BASES], dim=-2)
    points = torch.stack((wrist, palm), dim=-2)
    other = xyz.flip(2)
    starts = other[..., [a for a, _ in TORSO_SEGMENTS], :]
    ends = other[..., [b for _, b in TORSO_SEGMENTS], :]
    segment = (ends - starts).unsqueeze(3).unsqueeze(3)
    start = starts.unsqueeze(3).unsqueeze(3)
    offset = points.unsqueeze(-2) - start
    position = ((offset * segment).sum(-1) / segment.square().sum(-1).clamp_min(1e-8)).clamp(0.0, 1.0)
    closest = start + position.unsqueeze(-1) * segment
    return (points.unsqueeze(-2) - closest).norm(dim=-1).min(dim=4).values


def causal_contact_memory(contact):
    """[B,T,...] -> 1-∏_{s<=t}(1-p_s)，在当前帧之前的前缀内累计软事件概率。

    这是独立软事件假设下的特征，不是接触发生的物理概率；持续弱接近也会累积，不能据此锁死自主运动。
    """
    if contact.dim() < 2 or contact.shape[1] == 0:
        raise ValueError("contact 必须含非空时间维 dim=1")
    return 1.0 - torch.cumprod(1.0 - contact.clamp(0.0, 1.0), dim=1)


def _contact_adjacency(contact):
    incidence = contact.new_zeros(len(TORSO_SEGMENTS), BODY22)
    for index, (begin, end) in enumerate(TORSO_SEGMENTS):
        incidence[index, begin] = 1.0
        incidence[index, end] = 1.0
    # 重复端点取最强线段关系，防止 spine3 多次计权使边权超过 1。
    endpoint_weight = (contact.unsqueeze(-1) * incidence).max(dim=-2).values
    hand_rows = contact.new_zeros(len(WRISTS), BODY22)
    hand_rows[0, WRISTS[0]] = 1.0
    hand_rows[1, WRISTS[1]] = 1.0
    directed = torch.einsum("btphj,hi->btpij", endpoint_weight, hand_rows)
    # p 的图从 self 节点指向 other 节点，反向边必须取 other 的接触图再交换节点轴。
    return directed + directed.flip(2).transpose(-1, -2)


def contact_graph_features(xyz, threshold=0.20, temperature=CONTACT_TEMPERATURE):
    """返回距离、软接触、前缀记忆、接近速度与双向跨人邻接矩阵。

    adjacency [B,T,2,22,22] 的行是 self 节点，列是 other 节点；图边仅有掌腕与对方躯干。
    接近速度使用当前与上一帧的距离，首帧为 0，避免特征读取未来帧。
    """
    if float(temperature) <= 0:
        raise ValueError("temperature 必须大于 0")
    distance = hand_body_distances(xyz)
    contact = torch.sigmoid((float(threshold) - distance) / float(temperature))
    approach = torch.cat((torch.zeros_like(distance[:, :1]), (distance[:, :-1] - distance[:, 1:]) * NTU2P_FPS), dim=1)
    return OrderedDict([
        ("distance", distance),
        ("contact", contact),
        ("memory", causal_contact_memory(contact)),
        ("approach_velocity", approach),
        ("adjacency", _contact_adjacency(contact)),
    ])


def contact_distance_loss(pred, target):
    """匹配所有双向手—躯干距离，GT 提供监督而不充当预测条件。"""
    return F.mse_loss(hand_body_distances(pred), hand_body_distances(target))


def observed_forward(obs):
    """[B,T,2,55,3] -> [B,2,3]，以最近 3 帧的个人朝向确定左右踝的前后轴。"""
    _check_xyz(obs, "obs")
    up = estimate_up(obs)
    recent = obs[:, -min(3, obs.shape[1]):]
    expanded_up = up[:, None, None].expand(recent.shape[0], recent.shape[1], 2, 3).reshape(-1, 3)
    facing = facing_direction(recent.reshape(-1, recent.shape[3], 3), expanded_up)
    facing = facing / facing.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    forward = facing.view(recent.shape[0], recent.shape[1], 2, 3).sum(dim=1)
    camera = torch.zeros_like(up)
    camera[:, 0] = 1.0
    fallback = horizontal(camera, up).unsqueeze(1).expand_as(forward)
    forward = torch.where(forward.norm(dim=-1, keepdim=True) >= 1e-6, forward, fallback)
    return forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def bilateral_phase_loss(pred, target, obs):
    """在观测确定的个人前向系匹配低通左右踝前后分离；静止/垫步仍以 GT 轨迹为准。"""
    _check_xyz(pred, "pred")
    _check_xyz(target, "target")
    forward = observed_forward(obs).unsqueeze(1)
    separation_pred = ((pred[..., 7, :] - pred[..., 8, :]) * forward).sum(-1)
    separation_target = ((target[..., 7, :] - target[..., 8, :]) * forward).sum(-1)
    coeffs = min(DEFAULT_LEG_GAIT_LOWPASS_K, int(pred.shape[1]))
    return F.mse_loss(lowpass_time(separation_pred, coeffs), lowpass_time(separation_target, coeffs))


def _first_sustained(mask, obs_len):
    tail = mask[:, obs_len:]
    missing = torch.full((mask.shape[0],), -1, dtype=torch.long, device=mask.device)
    if tail.shape[1] < EVENT_SUSTAIN_FRAMES:
        return missing
    hit = tail.unfold(1, EVENT_SUSTAIN_FRAMES, 1).all(dim=-1)
    index = torch.arange(hit.shape[1], device=mask.device).view(1, -1).expand_as(hit)
    first = torch.where(hit, index, torch.full_like(index, tail.shape[1])).min(dim=1).values
    return torch.where(hit.any(dim=1), first + int(obs_len), missing)


def _event_state(obs, future, threshold, speed):
    """只用于离线推人评价；双向平滑的标签不应进入推理特征。"""
    full = torch.cat((obs, future), dim=1)
    obs_len = int(obs.shape[1])
    distance = smooth_time(hand_body_distances(full).amin(dim=(-1, -2)))[..., 0]
    obs_distance = smooth_time(hand_body_distances(obs).amin(dim=(-1, -2)))[..., 0]
    up = estimate_up(obs)
    axis = horizontal(obs[:, -1, 1, 0] - obs[:, -1, 0, 0], up)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    root = smooth_time(full[:, :, 1, 0])
    position = ((root - root[:, obs_len-1:obs_len]) * axis.unsqueeze(1)).sum(-1)
    velocity = torch.cat((torch.zeros_like(position[:, :1]), (position[:, 1:] - position[:, :-1]) * NTU2P_FPS), dim=1)
    obs_root = smooth_time(obs[:, :, 1, 0])
    obs_speed = ((obs_root[:, 1:] - obs_root[:, :-1]) * axis.unsqueeze(1)).sum(-1) * NTU2P_FPS
    # 观测mask只从独立obs轨迹估计，各方法共享相同有效子集。
    observed_retreat = obs_speed[:, -min(5, obs_speed.shape[1]):].mean(dim=1) if obs_speed.shape[1] else obs_speed.new_zeros(obs.shape[0])
    fresh = ~(obs_distance[:, -min(3, obs_len):] <= float(threshold)).any(dim=1) & (observed_retreat < float(speed))
    contact = _first_sustained(distance <= float(threshold), obs_len)
    # 帧差先持续超速，再检验该段的净位移，保持噪声排除条件独立于事件时刻。
    tail_speed = velocity[:, obs_len:]
    tail_pos = position[:, obs_len:]
    retreat = torch.full_like(contact, -1)
    if tail_speed.shape[1] >= EVENT_SUSTAIN_FRAMES:
        hit = (tail_speed > float(speed)).unfold(1, EVENT_SUSTAIN_FRAMES, 1).all(-1)
        hit = hit & (tail_pos.unfold(1, EVENT_SUSTAIN_FRAMES, 1).max(-1).values >= 0.02)
        index = torch.arange(hit.shape[1], device=hit.device).view(1, -1).expand_as(hit)
        first = torch.where(hit, index, torch.full_like(index, tail_speed.shape[1])).min(1).values
        retreat = torch.where(hit.any(1), first + obs_len, retreat)
    delay = retreat - contact
    reliable = fresh & (contact >= 0) & (retreat >= 0) & (delay >= -EVENT_EARLY_TOLERANCE) & (delay <= EVENT_MAX_DELAY_FRAMES)
    return {"contact": contact, "retreat": retreat, "fresh": fresh, "reliable": reliable}


def _number(value, mask=None):
    if mask is not None:
        value = value[mask]
    return float(value.double().mean().item()) if value.numel() else None


def _event_summary(predicted, reference):
    eligible = reference["reliable"]
    contact_present = predicted["contact"] >= 0
    retreat_present = predicted["retreat"] >= 0
    contact_pair = eligible & contact_present
    retreat_pair = eligible & retreat_present
    premature = eligible & retreat_present & (~contact_present | (predicted["retreat"] < predicted["contact"] - EVENT_EARLY_TOLERANCE))
    contact_delta = (predicted["contact"] - reference["contact"]).double() / NTU2P_FPS
    retreat_delta = (predicted["retreat"] - reference["retreat"]).double() / NTU2P_FPS
    return OrderedDict([
        ("num_samples", int(eligible.numel())),
        ("gt_fresh_window_count", int(reference["fresh"].sum().item())),
        ("gt_contact_then_retreat_count", int(eligible.sum().item())),
        ("pred_contact_count", int(contact_present.sum().item())),
        ("pred_retreat_count", int(retreat_present.sum().item())),
        ("contact_missing_on_gt_subset", int((eligible & ~contact_present).sum().item())),
        ("retreat_missing_on_gt_subset", int((eligible & ~retreat_present).sum().item())),
        ("premature_or_missing_contact_count", int(premature.sum().item())),
        ("contact_time_mae_s", _number(contact_delta.abs(), contact_pair)),
        ("contact_time_pair_count", int(contact_pair.sum().item())),
        ("retreat_time_mae_s", _number(retreat_delta.abs(), retreat_pair)),
        ("retreat_time_bias_s", _number(retreat_delta, retreat_pair)),
        ("retreat_time_pair_count", int(retreat_pair.sum().item())),
    ])


def _distance_metrics(distance, target, index=None, actor_only=False):
    if index is not None:
        distance, target = distance[index], target[index]
    if actor_only:
        distance, target = distance[:, :, :1], target[:, :, :1]
    if not distance.shape[0]:
        return OrderedDict([("num_samples", 0), ("distance_rmse", None), ("distance_mae", None), ("pair_distance_rmse", None)])
    nearest = distance.amin(dim=(-1, -2))
    reference = target.amin(dim=(-1, -2))
    delta = nearest - reference
    result = OrderedDict([
        ("num_samples", int(distance.shape[0])),
        ("distance_rmse", None if not delta.numel() else float(delta.double().square().mean().sqrt().item())),
        ("distance_mae", _number(delta.abs())),
        ("pair_distance_rmse", None if not delta.numel() else float((distance.double() - target.double()).square().mean().sqrt().item())),
    ])
    return result


def contact_metrics(obs, target, variants, actions):
    """JSON可序列化的全体/推人连续距离指标及预固定阈值的事件辅助指标。

    variants 是 name->[B,T,2,55,3]；全体统计双向最近距离，推人只取 A 手→B 躯干。
    事件有效子集只由 GT 确定，缺失事件单列，不以丢掉难例的配对误差代替全部结果。
    """
    with torch.no_grad():
        obs_cpu = obs.detach().to(device="cpu", dtype=torch.float64)
        target_cpu = target.detach().to(device="cpu", dtype=torch.float64)
        action_cpu = actions.detach().view(-1).cpu()
        if int(action_cpu.numel()) != int(target_cpu.shape[0]):
            raise ValueError("actions 数量与 target batch 不一致")
        pushing = action_cpu == PUSHING_ACTION
        gt_distance = hand_body_distances(target_cpu)
        references = OrderedDict()
        for threshold in CONTACT_THRESHOLDS:
            for speed in RETREAT_SPEED_THRESHOLDS:
                key = "distance_{:.2f}_speed_{:.2f}".format(threshold, speed)
                references[key] = _event_state(obs_cpu[pushing], target_cpu[pushing], threshold, speed) if pushing.any() else None
        result = OrderedDict()
        for name, value in variants.items():
            pred_cpu = value.detach().to(device="cpu", dtype=torch.float64)
            if tuple(pred_cpu.shape) != tuple(target_cpu.shape):
                raise ValueError("{} 轨迹与 target shape 不同".format(name))
            distance = hand_body_distances(pred_cpu)
            events = OrderedDict()
            for threshold in CONTACT_THRESHOLDS:
                for speed in RETREAT_SPEED_THRESHOLDS:
                    key = "distance_{:.2f}_speed_{:.2f}".format(threshold, speed)
                    events[key] = _event_summary(_event_state(obs_cpu[pushing], pred_cpu[pushing], threshold, speed), references[key]) if references[key] is not None else {"num_samples": 0, "gt_contact_then_retreat_count": 0}
            result[name] = OrderedDict([
                ("all", _distance_metrics(distance, gt_distance)),
                ("pushing", _distance_metrics(distance, gt_distance, pushing, actor_only=True)),
                ("events", events),
            ])
        return result


__all__ = [
    "TORSO_SEGMENTS", "WRISTS", "PALM_BASES", "CONTACT_THRESHOLDS", "RETREAT_SPEED_THRESHOLDS",
    "hand_body_distances", "causal_contact_memory", "contact_graph_features", "contact_distance_loss",
    "observed_forward", "bilateral_phase_loss", "contact_metrics",
]
