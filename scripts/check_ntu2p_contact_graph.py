"""用解析几何与扰动验证软接触图，防止因果前缀和双向跨人索引出错。"""

import json

import torch

from utils.ntu2p_canonical import NTU2P_DATASET_UP, rotation_about_axis
from utils.ntu2p_contact_graph import (
    bilateral_phase_loss,
    causal_contact_memory,
    contact_distance_loss,
    contact_graph_features,
    contact_metrics,
    hand_body_distances,
)


def synthetic_xyz(batch=2, frames=10):
    xyz = torch.zeros(batch, frames, 2, 55, 3, dtype=torch.float64)
    up = torch.tensor(NTU2P_DATASET_UP, dtype=xyz.dtype)
    up = up / up.norm()
    horizontal = torch.tensor([1.0, 0.0, 0.0], dtype=xyz.dtype)
    horizontal = horizontal - (horizontal * up).sum() * up
    horizontal = horizontal / horizontal.norm()
    forward = torch.cross(horizontal, up)
    for person in (0, 1):
        origin = horizontal * (2.0 * person)
        xyz[:, :, person] = origin
        for joint, height in ((3, .2), (6, .4), (9, .6), (12, .8)):
            xyz[:, :, person, joint] = origin + up * height
        for left, right in ((1, 2), (16, 17)):
            height = .6 if left == 16 else 0.0
            xyz[:, :, person, left] = origin + up * height + horizontal * .2
            xyz[:, :, person, right] = origin + up * height - horizontal * .2
        for ankle in (7, 8):
            xyz[:, :, person, ankle] = origin - up * .8
        hand = origin + up * .3 + forward * .1
        for joint in (20, 21, 25, 31, 40, 46):
            xyz[:, :, person, joint] = hand
    return xyz


def main():
    torch.set_num_threads(2)
    xyz = synthetic_xyz()
    distance = hand_body_distances(xyz)
    assert tuple(distance.shape) == (2, 10, 2, 2, 6), "手到对方线段的维度错误"
    # 第 2 段包含 height=.3，手与对方躯干的垂直高度相同，因此距离为 sqrt(2²+.1²)。
    analytic = (4.0 + .01) ** .5
    analytic_error = float((distance[..., 1] - analytic).abs().max())
    assert analytic_error < 1e-10, "解析点—线段距离不符"
    moved = xyz.clone()
    moved[:, :, 0, [20, 21, 25, 31, 40, 46], 0] += 1.0
    decreased = hand_body_distances(moved)[:, :, 0].mean() < distance[:, :, 0].mean()
    assert bool(decreased), "移动 A 的手靠近 B 后距离没有下降"
    features = contact_graph_features(moved)
    adjacency = features["adjacency"]
    assert tuple(adjacency.shape) == (2, 10, 2, 22, 22), "跨人图维度错误"
    symmetry_error = float((adjacency - adjacency.flip(2).transpose(-1, -2)).abs().max())
    assert symmetry_error == 0.0, "A→B 与 B→A 的反向关系索引错误"

    probabilities = torch.tensor([[[.2], [.4], [.5], [.0]]], dtype=torch.float64, requires_grad=True)
    memory = causal_contact_memory(probabilities)
    expected = torch.tensor([[[.2], [.52], [.76], [.76]]], dtype=torch.float64)
    assert torch.allclose(memory, expected), "事件前缀累计值错误"
    perturbed = probabilities.detach().clone()
    perturbed[:, 2:] = 1.0
    prefix_error = float((causal_contact_memory(perturbed)[:, :2] - memory[:, :2]).abs().max())
    assert prefix_error == 0.0, "接触记忆读取了未来帧"
    memory[:, 1].sum().backward()
    assert probabilities.grad[:, 2:].abs().max() == 0, "前缀记忆的梯度泄露到未来帧"

    predicted = moved.detach().clone().requires_grad_(True)
    loss = contact_distance_loss(predicted, xyz)
    loss.backward()
    grad = predicted.grad
    assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0, "距离损失没有有限有效梯度"
    actor_hand_grad = float(grad[:, :, 0, [20, 21, 25, 31, 40, 46]].abs().sum())
    receiver_torso_grad = float(grad[:, :, 1, [0, 3, 6, 9, 12, 16, 17]].abs().sum())
    assert actor_hand_grad > 0 and receiver_torso_grad > 0, "距离损失未同时约束手和对方躯干"
    identical = contact_distance_loss(xyz, xyz)
    assert float(identical) == 0, "同轨迹距离损失应为0"
    degenerate = torch.zeros(1, 5, 2, 55, 3, dtype=torch.float64, requires_grad=True)
    hand_body_distances(degenerate).sum().backward()
    assert bool(torch.isfinite(degenerate.grad).all()), "退化躯干产生非有限梯度"

    up = torch.tensor(NTU2P_DATASET_UP, dtype=xyz.dtype).view(1, 3).expand(2, 3)
    up = up / up.norm(dim=-1, keepdim=True)
    rotation = rotation_about_axis(up, torch.tensor([.7, -.3], dtype=xyz.dtype))
    rotated = (moved.unsqueeze(-2) * rotation[:, None, None, None]).sum(-1)
    translation = torch.tensor([1.7, -.4, 2.0], dtype=xyz.dtype)
    rigid_error = float((hand_body_distances(rotated + translation) - hand_body_distances(moved)).abs().max())
    assert rigid_error < 1e-10, "刚体变换改变了接触距离"

    obs = synthetic_xyz(frames=10)
    target = synthetic_xyz(frames=20)
    phase_pred = target.clone().requires_grad_(True)
    with torch.no_grad():
        phase_pred[:, :, :, 7] += torch.linspace(0.0, .2, 20, dtype=xyz.dtype).view(1, 20, 1, 1) * torch.cross(torch.tensor([[1.0, 0.0, 0.0]], dtype=xyz.dtype), up[:1], dim=-1).view(1, 1, 1, 3)
    phase_loss = bilateral_phase_loss(phase_pred, target, obs)
    phase_loss.backward()
    assert float(phase_loss) > 0 and bool(torch.isfinite(phase_pred.grad).all()) and float(phase_pred.grad.abs().sum()) > 0, "左右分离损失未监督前后相位"
    phase_identity = bilateral_phase_loss(target, target, obs)
    assert float(phase_identity) == 0, "相同腿相位损失应为0"
    report = contact_metrics(obs, target, {"same": target}, torch.tensor([2, 0]))
    assert report["same"]["all"]["distance_rmse"] == 0.0
    assert report["same"]["pushing"]["num_samples"] == 1
    json.dumps(report, allow_nan=False)
    empty = contact_metrics(obs, target, {"same": target}, torch.tensor([0, 0]))
    assert empty["same"]["pushing"]["distance_rmse"] is None
    json.dumps(empty, allow_nan=False)
    print(json.dumps({"解析距离最大误差": analytic_error, "刚体变换距离最大误差": rigid_error,
                      "双向图对称误差": symmetry_error, "未来扰动对前缀记忆影响": prefix_error,
                      "手部梯度总量": actor_hand_grad, "对方躯干梯度总量": receiver_torso_grad,
                      "双腿分离损失": float(phase_loss), "空推人子集及JSON序列化": "通过"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
