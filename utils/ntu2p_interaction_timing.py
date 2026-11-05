"""全动作双人交互时序：接触前后退量、提前后退比例（PCR）与 GRPO 的 ant 奖励。

定义见 docs/ai/context/20261018-201532-ntu2p-jit-flow-rl-design-and-plan.md 4.2 节。对每个窗口的每个受动者 p
（施动者 q 为另一人）：d(t) 为 q 双手到 p 躯干的最小中心距；obs 末 3 帧 d > 阈值为"未接触起始"，未来首次连续 3 帧
d ≤ 阈值于 τ 为新接触；e(t) 为 p 的 pelvis 沿"q→p"水平方向的位移减去 obs 末速度外推。
接触前后退量 A = max(0, max_{t ≤ τ_eff-3} e(t))，τ_eff 为本轨迹自己的接触时刻，始终未接触时取整段未来——
用户看到的"手没碰到就躲"在 FD2 上主要是"A 的手始终没到、B 照样后退"，只统计有接触的轨迹会漏掉它。
接触后反应量 R = e(min(τ+12, T-1)) - e(τ)，只对有新接触的轨迹定义。
主指标在"GT 有新接触"的受动者上配对：relu(A_方法 - A_GT) 的均值与漏接触数。

骨架中心距只是接触代理；函数只读传入的轨迹，GT 只在作为参照的调用里出现。5 点双向平滑只用于离线评价与奖励，
不能当作推理端的因果特征。
"""

from collections import OrderedDict

import torch

from utils.ntu2p_canonical import estimate_up, horizontal
from utils.ntu2p_contact_graph import hand_body_distances
from utils.ntu2p_naturalness import smooth_time


CONTACT_THRESHOLD = 0.20
SUSTAIN_FRAMES = 3
PRE_CONTACT_MARGIN = 3
REACTION_FRAMES = 12
OBS_VELOCITY_FRAMES = 5


def _first_sustained(mask):
    """[B,T,P] bool -> [B,P] 首个连续 SUSTAIN_FRAMES 帧为真的起点，无则 -1。"""
    num_frames = int(mask.shape[1])
    if num_frames < SUSTAIN_FRAMES:
        return torch.full((mask.shape[0], mask.shape[2]), -1, dtype=torch.long, device=mask.device)
    hit = mask.unfold(1, SUSTAIN_FRAMES, 1).all(dim=-1)  # [B,T-S+1,P]
    index = torch.arange(hit.shape[1], device=mask.device).view(1, -1, 1).expand_as(hit)
    first = torch.where(hit, index, torch.full_like(index, num_frames)).min(dim=1).values
    return torch.where(hit.any(dim=1), first, torch.full_like(first, -1))


def receiver_events(obs_xyz, future_xyz, threshold=CONTACT_THRESHOLD):
    """obs [B,To,2,55,3]、future [B,T,2,55,3] -> 逐（窗口, 受动者）的 [B,2] 张量。

    返回 fresh（obs 末未接触）、event（新接触）、tau（未来帧序号，-1 为无）、antic（A，米；非 fresh 为 0）、
    reaction（R，米；无新接触为 0）。
    """
    obs = obs_xyz.double()
    future = future_xyz.double()
    obs_len, pred_len = int(obs.shape[1]), int(future.shape[1])
    full = torch.cat((obs, future), dim=1)
    # hand_body_distances 的人物维是"手的主人"；flip 后变成"受动者"：对方双手到本人躯干。
    received = smooth_time(hand_body_distances(full).amin(dim=(-1, -2)).flip(2))  # [B,N,2]
    fresh = (received[:, obs_len - SUSTAIN_FRAMES : obs_len] > float(threshold)).all(dim=1)
    tau = _first_sustained(received[:, obs_len:] <= float(threshold))
    event = fresh & (tau >= 0)

    up = estimate_up(obs)
    pelvis = full[:, :, :, 0]  # [B,N,2,3]
    last = pelvis[:, obs_len - 1]
    away = horizontal(last - last.flip(1), up)
    away = away / away.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    position = smooth_time(((pelvis - last.unsqueeze(1)) * away.unsqueeze(1)).sum(dim=-1))  # [B,N,2]
    velocity = (position[:, obs_len - 1] - position[:, obs_len - 1 - OBS_VELOCITY_FRAMES]) / float(OBS_VELOCITY_FRAMES)
    steps = torch.arange(1, pred_len + 1, dtype=position.dtype, device=position.device).view(1, -1, 1)
    excess = position[:, obs_len:] - velocity.unsqueeze(1) * steps  # [B,T,2]

    frame = torch.arange(pred_len, device=excess.device).view(1, -1, 1)
    tau_eff = torch.where(tau >= 0, tau, torch.full_like(tau, pred_len + PRE_CONTACT_MARGIN))
    before = frame <= (tau_eff - PRE_CONTACT_MARGIN).unsqueeze(1)
    neg_inf = torch.full_like(excess, float("-inf"))
    antic = torch.where(before, excess, neg_inf).max(dim=1).values.clamp_min(0.0)
    antic = torch.where(torch.isfinite(antic), antic, torch.zeros_like(antic))
    safe_tau = tau.clamp_min(0)
    after = (safe_tau + REACTION_FRAMES).clamp_max(pred_len - 1)
    reaction = excess.gather(1, after.unsqueeze(1)).squeeze(1) - excess.gather(1, safe_tau.unsqueeze(1)).squeeze(1)
    zero = torch.zeros_like(antic)
    return OrderedDict(
        [
            ("fresh", fresh),
            ("event", event),
            ("tau", tau),
            ("antic", torch.where(fresh, antic, zero)),
            ("reaction", torch.where(event, reaction, zero)),
        ]
    )


def anticipation_excess(obs_xyz, pred_xyz, target_xyz, threshold=CONTACT_THRESHOLD):
    """[B] 逐窗口：在 GT 有新接触的受动者上，样本"接触前（或始终未接触）"的后退量超出 GT 的部分，两名受动者求和。

    供 GRPO 的 ant 奖励；GT 没有新接触时受动者可以合理地离开（如各自走开），不计入。
    """
    pred = receiver_events(obs_xyz, pred_xyz, threshold)
    with torch.no_grad():
        target = receiver_events(obs_xyz, target_xyz, threshold)
    excess = (pred["antic"] - target["antic"]).clamp_min(0.0)
    return torch.where(target["event"], excess, torch.zeros_like(excess)).sum(dim=1)


def _ratio(num, den):
    num, den = float(num), float(den)
    return num / den if den > 0 else None


def interaction_timing_summary(obs_xyz, variants, target_xyz, threshold=CONTACT_THRESHOLD):
    """variants: name -> [B,T,2,55,3]；在 GT 有新接触的受动者上配对，返回 JSON 可序列化的统计。

    excess_antic = relu(A_方法 - A_GT) 的均值（主指标，米）；missed = GT 有新接触而方法没有；
    antic_when_missed = 漏接触时方法的后退量均值（"手没到就躲"）。
    """
    with torch.no_grad():
        gt = receiver_events(obs_xyz, target_xyz, threshold)
        mask = gt["event"]
        count = int(mask.sum().item())
        result = OrderedDict(
            [
                ("threshold", float(threshold)),
                ("gt_event_count", count),
                ("gt_mean_antic", _ratio(gt["antic"][mask].sum(), count)),
                ("gt_mean_reaction", _ratio(gt["reaction"][mask].sum(), count)),
            ]
        )
        for name, value in variants.items():
            events = receiver_events(obs_xyz, value, threshold)
            delta = (events["antic"] - gt["antic"])[mask]
            missed = mask & ~events["event"]
            paired = mask & events["event"]
            num_missed = int(missed.sum().item())
            result[name] = OrderedDict(
                [
                    ("excess_antic", _ratio(delta.clamp_min(0.0).sum(), count)),
                    ("delta_antic", _ratio(delta.sum(), count)),
                    ("missed_count", num_missed),
                    ("antic_when_missed", _ratio(events["antic"][missed].sum(), num_missed)),
                    ("paired_count", int(paired.sum().item())),
                    ("paired_reaction", _ratio(events["reaction"][paired].sum(), int(paired.sum().item()))),
                ]
            )
        return result


INTERACTION_THRESHOLDS = (0.20, 0.25, 0.30)


def interaction_timing_block(obs_xyz, variants, target_xyz, actions=None, strike_actions=(0, 1, 2, 11, 13)):
    """三个阈值 × {全部, 击打类, 推人} 的 interaction_timing_summary，另给主指标（全部动作三阈值 excess_antic 均值）。"""
    subsets = OrderedDict([("all", None)])
    if actions is not None:
        actions = actions.reshape(-1).cpu()
        strike = torch.zeros_like(actions, dtype=torch.bool)
        for action in strike_actions:
            strike |= actions == int(action)
        subsets["strike"] = strike
        subsets["push"] = actions == 2
    block = OrderedDict()
    for subset, index in subsets.items():
        for threshold in INTERACTION_THRESHOLDS:
            if index is None:
                obs, target, chosen = obs_xyz, target_xyz, variants
            else:
                obs, target = obs_xyz[index], target_xyz[index]
                chosen = OrderedDict((name, value[index]) for name, value in variants.items())
            block["{}_{:.2f}".format(subset, threshold)] = interaction_timing_summary(obs, chosen, target, threshold)
    primary = OrderedDict()
    for name in variants:
        values = [block["all_{:.2f}".format(threshold)][name]["excess_antic"] for threshold in INTERACTION_THRESHOLDS]
        primary[name] = None if any(value is None for value in values) else sum(values) / len(values)
    block["primary_excess_antic"] = primary
    return block


__all__ = ["anticipation_excess", "interaction_timing_block", "interaction_timing_summary", "receiver_events"]
