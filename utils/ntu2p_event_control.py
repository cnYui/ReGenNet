"""只替换 B root 的最小接触事件控制器，用于检验事件约束是否可行。

接触来自最终骨架几何的掌腕—躯干中心距，不是皮肤表面或物理因果标签。
保留原腿局部轨迹会改变世界系脚速度，当前模块不能解决脚滑。
"""

from collections import OrderedDict

import torch

from utils.ntu2p_canonical import estimate_up, horizontal
from utils.ntu2p_contact_graph import hand_body_distances
from utils.ntu2p_naturalness import NTU2P_FPS, smooth_time
from utils.ntu_smplx_2p_xyz import dct_matrix


DEFAULT_CONTINUATION_DECAY_SECONDS = 2.5
DEFAULT_SUSTAIN_FRAMES = 3
DEFAULT_RESPONSE_RAMP_FRAMES = 3


def observed_continuation(obs, pred_len, decay_seconds=DEFAULT_CONTINUATION_DECAY_SECONDS):
    """只从 obs 估计 B 的已有水平速度；不允许动作标签开启新退后。"""
    if float(decay_seconds) <= 0:
        raise ValueError("decay_seconds 必须大于 0")
    up = estimate_up(obs)
    roots = smooth_time(obs[:, :, 1, 0])
    velocity = (roots[:, 1:] - roots[:, :-1]) * NTU2P_FPS
    recent = velocity[:, -min(5, velocity.shape[1]):]
    initial = recent.mean(dim=1) if recent.shape[1] else roots.new_zeros(roots.shape[0], 3)
    initial = horizontal(initial, up)
    time = torch.arange(int(pred_len), dtype=obs.dtype, device=obs.device) / NTU2P_FPS
    decay = torch.exp(-time / float(decay_seconds))
    return initial.unsqueeze(1) * decay.view(1, -1, 1)


def _initial_contact(obs, threshold, sustain_frames):
    # 观测末帧和首预测帧相同，不把重复帧计为一次新的接触证据。
    hit = hand_body_distances(obs)[:, :, 0].amin(dim=(-1, -2)) <= float(threshold)
    counter = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
    active = torch.zeros_like(counter, dtype=torch.bool)
    first = torch.full_like(counter, obs.shape[1])
    for time in range(obs.shape[1]):
        counter = torch.where(hit[:, time], counter + 1, torch.zeros_like(counter))
        trigger = (counter >= int(sustain_frames)) & ~active
        first = torch.where(trigger, torch.full_like(first, time), first)
        active = active | trigger
    return counter, active, first - (int(obs.shape[1]) - 1)


def _check_shapes(obs, root_relative, prior_roots, response_velocity, contact_route):
    if obs.dim() != 5 or tuple(obs.shape[2:]) != (2, 55, 3) or obs.shape[1] < 1:
        raise ValueError("obs 必须为 [B,Tobs>=1,2,55,3]")
    if root_relative.dim() != 5 or tuple(root_relative.shape[2:]) != (2, 55, 3) or root_relative.shape[1] < 1:
        raise ValueError("root_relative 必须为 [B,Tpred>=1,2,55,3]")
    batch, time = root_relative.shape[:2]
    if int(obs.shape[0]) != batch or tuple(prior_roots.shape) != (batch, time, 2, 3):
        raise ValueError("prior_roots 必须与 local 对应，为 [B,T,2,3]")
    if tuple(response_velocity.shape) != (batch, time, 3) or contact_route.numel() != batch:
        raise ValueError("response_velocity 必须为 [B,T,3]，contact_route 每样本一个值")
    if contact_route.dtype != torch.bool:
        raise ValueError("contact_route 必须为 bool，避免连续路线混合破坏事件保证")


def _decode(obs, root_relative, prior_roots, response_velocity, contact_route, threshold,
            continuation_decay_seconds, sustain_frames, response_ramp_frames, projector,
            oracle_roots=None):
    _check_shapes(obs, root_relative, prior_roots, response_velocity, contact_route)
    if int(sustain_frames) < 1 or int(response_ramp_frames) < 2:
        raise ValueError("sustain_frames 必须>=1，response_ramp_frames 必须>=2")
    batch, time = root_relative.shape[:2]
    route = contact_route.reshape(batch)
    up = estimate_up(obs)
    reference_root = obs[:, -1, :, 0]
    reference_local = obs[:, -1] - reference_root.unsqueeze(-2)
    local = root_relative - root_relative[..., :1, :]
    if projector is not None:
        local = projector(local, reference_local)
    # 首帧保留输入观测，投影的舍入误差不能影响协议中的严格连续性。
    local = torch.cat((reference_local.unsqueeze(1), local[:, 1:]), dim=1)
    continuation = observed_continuation(obs, time, continuation_decay_seconds)
    command = horizontal(response_velocity, up)
    counter, active, first = _initial_contact(obs, threshold, sustain_frames)
    active_in_obs = active.clone()
    first = torch.where(active, first, torch.full_like(first, time))
    receiver_root = reference_root[:, 1]
    outputs, roots, gates, hits, distances, responses, controls = [], [], [], [], [], [], []
    proposed_roots = prior_roots
    for index in range(time):
        if index:
            # 只替换水平 root；竖直摆动来自先验，并与之后同样使用的局部姿态分开。
            vertical = ((proposed_roots[:, index, 1] - reference_root[:, 1]) * up).sum(-1, keepdim=True) * up
            receiver_root = horizontal(receiver_root - reference_root[:, 1], up) + reference_root[:, 1] + vertical
        actor_root = reference_root[:, 0] if index == 0 else proposed_roots[:, index, 0]
        current_roots = torch.stack((actor_root, receiver_root), dim=1)
        candidate = local[:, index] + current_roots.unsqueeze(-2)
        if index == 0:
            candidate = obs[:, -1]
        # 局部投影后的刚性平移不改变骨长；用于门判断的就是返回的最终接触路线轨迹。
        distance = hand_body_distances(candidate.unsqueeze(1))[:, 0, 0].amin(dim=(-1, -2))
        hit = distance <= float(threshold)
        if index:
            counter = torch.where(hit, counter + 1, torch.zeros_like(counter))
            trigger = (counter >= int(sustain_frames)) & ~active
            first = torch.where(trigger, torch.full_like(first, index), first)
            active = active | trigger
        age = (index - first).to(obs.dtype)
        phase = (age / float(int(response_ramp_frames) - 1)).clamp(0.0, 1.0)
        gate = (phase.square() * (3.0 - 2.0 * phase)) * active.to(obs.dtype) * route.to(obs.dtype)
        velocity = command[:, index]
        if oracle_roots is not None and index + 1 < time:
            # 仅用于明确标为 oracle 的表示上限；GT 下一帧绝不能流入部署控制器。
            desired = horizontal(oracle_roots[:, index + 1, 1] - receiver_root, up) * NTU2P_FPS
            velocity = torch.where(gate.unsqueeze(-1) > 0,
                                   (desired - continuation[:, index]) / gate.unsqueeze(-1).clamp_min(1e-8),
                                   torch.zeros_like(velocity))
        response = gate.unsqueeze(-1) * velocity / NTU2P_FPS
        outputs.append(candidate)
        roots.append(current_roots)
        gates.append(gate)
        hits.append(hit)
        distances.append(distance)
        responses.append(response)
        controls.append(velocity)
        # 当前帧的接触最多影响下一帧；最后一帧控制仍记录但没有输出位移。
        receiver_root = receiver_root + continuation[:, index] / NTU2P_FPS + response
    controlled = torch.stack(outputs, dim=1)
    free = local + proposed_roots.unsqueeze(-2)
    free = torch.cat((obs[:, -1:], free[:, 1:]), dim=1)
    pred = torch.where(route[:, None, None, None, None], controlled, free)
    pred = torch.cat((obs[:, -1:], pred[:, 1:]), dim=1)
    return OrderedDict([
        ("pred", pred), ("controlled", controlled), ("projected_local", local),
        ("roots", pred[..., 0, :]), ("response_gate", torch.stack(gates, 1)),
        ("response_increment", torch.stack(responses, 1)), ("continuation_velocity", continuation),
        ("response_velocity", torch.stack(controls, 1)), ("contact_hit", torch.stack(hits, 1)),
        ("contact_distance", torch.stack(distances, 1)), ("first_sustained_contact_frame", first),
        ("contact_active_in_obs", active_in_obs), ("contact_route", route),
    ])


def decode_event_root(obs, root_relative, prior_roots, response_velocity, contact_route,
                      threshold=0.20, continuation_decay_seconds=DEFAULT_CONTINUATION_DECAY_SECONDS,
                      sustain_frames=DEFAULT_SUSTAIN_FRAMES,
                      response_ramp_frames=DEFAULT_RESPONSE_RAMP_FRAMES, projector=None):
    """从 obs 重建接触路线的 B root，输入/控制均为相机坐标。

    local [B,T,2,55,3]、root [B,T,2,3]、response [B,T,3]（m/s）、route [B] bool。
    projector 若缺省，调用方须提供合法局部骨架；本函数只做刚性平移，不能修复非法骨长。
    自由路线返回局部与 root 的重组；要求原绝对预测逐位不变时使用 event_control_from_prior。
    """
    return _decode(obs, root_relative, prior_roots, response_velocity, contact_route, threshold,
                   continuation_decay_seconds, sustain_frames, response_ramp_frames, projector)


def event_control_from_prior(obs, prior, response_velocity, contact_route, **kwargs):
    """绝对预测的便利入口；自由路线严格返回原 prior，不隐含混合门。"""
    roots = prior[..., 0, :]
    details = decode_event_root(obs, prior - roots.unsqueeze(-2), roots, response_velocity, contact_route, **kwargs)
    route = details["contact_route"]
    details["pred"] = torch.where(route[:, None, None, None, None], details["pred"], prior)
    details["roots"] = details["pred"][..., 0, :]
    return details


def prior_response_velocity(obs, prior, continuation_decay_seconds=DEFAULT_CONTINUATION_DECAY_SECONDS):
    """只把先验平滑水平 root 速度转为候选响应控制；不读取 future GT。"""
    up = estimate_up(obs)
    roots = smooth_time(torch.cat((obs[:, -1:, 1, 0], prior[:, :, 1, 0]), dim=1))
    velocity = (roots[:, 2:] - roots[:, 1:-1]) * NTU2P_FPS
    velocity = torch.cat((velocity, torch.zeros_like(roots[:, :1])), dim=1)
    return horizontal(velocity, up) - observed_continuation(obs, prior.shape[1], continuation_decay_seconds)


def oracle_event_root(obs, target, threshold=0.20, num_coeffs=None,
                      continuation_decay_seconds=DEFAULT_CONTINUATION_DECAY_SECONDS, projector=None):
    """显式使用 GT 局部姿态和下一帧 root 的表示上限，只允许诊断，不能作为预测评分。"""
    roots = target[..., 0, :]
    local = target - roots.unsqueeze(-2)
    zero = target.new_zeros(target.shape[0], target.shape[1], 3)
    route = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
    oracle = _decode(obs, local, roots, zero, route, threshold, continuation_decay_seconds,
                     DEFAULT_SUSTAIN_FRAMES, DEFAULT_RESPONSE_RAMP_FRAMES, projector, oracle_roots=roots)
    if num_coeffs is None:
        return oracle
    basis = dct_matrix(int(target.shape[1]), dtype=target.dtype, device=target.device)[:int(num_coeffs)]
    commands = oracle["response_velocity"]
    if commands.shape[1] > 1:
        # 最后一个控制未用于任何输出位移，延续上一帧避免编码时人为制造终点阶跃。
        commands = torch.cat((commands[:, :-1], commands[:, -2:-1]), dim=1)
    coeffs = torch.einsum("kt,btc->bkc", basis, commands)
    smooth_command = torch.einsum("kt,bkc->btc", basis, coeffs)
    result = decode_event_root(obs, local, roots, smooth_command, route, threshold=threshold,
                               continuation_decay_seconds=continuation_decay_seconds, projector=projector)
    result["oracle_control_coeffs"] = coeffs
    return result


__all__ = ["observed_continuation", "decode_event_root", "event_control_from_prior",
           "prior_response_velocity", "oracle_event_root"]
