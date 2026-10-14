"""最小 root 事件解码器的结构与数据可行性检查，诊断不等于模型优化结果。"""

import argparse
import json
import re
import time
from collections import OrderedDict
from pathlib import Path

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache
from utils.ntu2p_canonical import NTU2P_DATASET_UP, estimate_up, horizontal
from utils.ntu2p_contact_graph import CONTACT_THRESHOLDS, _event_state, contact_metrics, hand_body_distances
from utils.ntu2p_event_control import event_control_from_prior, observed_continuation, oracle_event_root, prior_response_velocity
from utils.ntu2p_kinematic_projection import SkeletonProjector, smplx_rest_joints
from utils.ntu2p_naturalness import SMPLX_PARENTS, aggregate_stats, compute_naturalness_stats, estimate_scene_frame
from utils.ntu_smplx_2p_xyz import compute_ntu_xyz_metrics


DEFAULT_ROOT = "results/forecasting/ntu120_label/ntu2p_graphdiff_research/event_control"
DEFAULT_CACHE = "results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval"
DEFAULT_MANIFEST = "results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json"
DEFAULT_ARRAYS = "results/forecasting/ntu120_label/ntu2p_v2subj_screen/review_fd2/val_arrays_s0.pt"


def _axes(dtype=torch.float64):
    up = torch.tensor(NTU2P_DATASET_UP, dtype=dtype)
    up = up / up.norm()
    side = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    side = side - (side * up).sum() * up
    side = side / side.norm()
    return side, up, torch.cross(side, up, dim=-1)


def _synthetic(projector, hand_shift=0.0, frames=50):
    rest, _ = smplx_rest_joints()
    side, up, forward = _axes()
    body = (rest - rest[:1]) @ torch.stack((side, up, forward))
    base = torch.stack((body, body + side * .75)).unsqueeze(0)
    obs = base.unsqueeze(1).expand(1, 10, 2, 55, 3).clone()
    height = (body[9] * up).sum()
    initial = side * .10 + up * height
    shift = initial - obs[:, :, 0, 20]
    obs[:, :, 0, 20] += shift
    obs[:, :, 0, 25:40] += shift.unsqueeze(-2)
    obs = projector(obs, base)
    prior = obs[:, -1:].expand(1, frames, 2, 55, 3).clone()
    target = side * (torch.linspace(.10, .61, frames, dtype=rest.dtype) + float(hand_shift)).view(1, frames, 1) + up * height
    shift = target - prior[:, :, 0, 20]
    prior[:, :, 0, 20] += shift
    prior[:, :, 0, 25:40] += shift.unsqueeze(-2)
    prior = projector(prior, obs[:, -1])
    prior[:, 0] = obs[:, -1]
    response = side.view(1, 1, 3).expand(1, frames, 3) * .40
    return obs, prior, response


def _bone_error(value, reference):
    parents = torch.tensor([max(p, 0) for p in SMPLX_PARENTS], device=value.device)
    lengths = (value - value[..., parents, :]).norm(dim=-1)[..., 1:]
    expected = (reference - reference[..., parents, :]).norm(dim=-1)[..., 1:]
    return float((lengths - expected[:, None]).abs().max().item())


def _first_response(details):
    hit = details["response_increment"].norm(dim=-1) > 1e-10
    index = torch.arange(hit.shape[1], device=hit.device).view(1, -1).expand_as(hit)
    first = torch.where(hit, index, torch.full_like(index, hit.shape[1])).min(1).values
    return int(first[0].item())


def check_synthetic(projector):
    true_route = torch.tensor([True])
    obs, prior, velocity = _synthetic(projector)
    result = event_control_from_prior(obs, prior, velocity, true_route, projector=projector)
    assert torch.equal(result["pred"][:, :1], obs[:, -1:]), "首预测帧未严格等于观测末帧"
    bone_error = _bone_error(result["pred"], obs[:, -1])
    assert bone_error < 1e-8, "root 重建破坏骨长"
    distance = hand_body_distances(result["pred"])[:, :, 0].amin(dim=(-1, -2))
    final_geometry_error = float((distance - result["contact_distance"]).abs().max())
    assert final_geometry_error < 1e-10, "接触门读取的几何与最终输出不一致"
    first_trigger = int(result["first_sustained_contact_frame"][0])
    first_response = _first_response(result)
    assert first_trigger < 50 and first_response > first_trigger, "新增反应未晚于预测接触事件"
    assert not bool((result["response_increment"][:, :first_trigger+1].abs() > 1e-12).any()), "接触前产生新响应"
    ahead = event_control_from_prior(*_synthetic(projector, hand_shift=.07), true_route, projector=projector)
    behind = event_control_from_prior(*_synthetic(projector, hand_shift=-.07), true_route, projector=projector)
    assert _first_response(ahead) < first_response < _first_response(behind), "手部接近干预没有改变 B 响应时刻"
    no_obs, no_prior, no_vel = _synthetic(projector)
    side, _, _ = _axes()
    no_obs[:, :, 0] -= side * 2.0
    no_prior[:, :, 0] -= side * 2.0
    missing = event_control_from_prior(no_obs, no_prior, no_vel, true_route, projector=projector)
    assert missing["response_increment"].abs().max() == 0, "无接触时新增响应未为0"
    moving_obs = no_obs.clone()
    moving_obs[:, :, 1] += side * (torch.arange(10, dtype=obs.dtype).view(1, 10, 1, 1) * .015)
    moving_prior = no_prior.clone()
    moving_prior[:, :, 1] += side * .135
    continued = event_control_from_prior(moving_obs, moving_prior, no_vel, true_route, projector=projector)
    displacement = float(((continued["roots"][:, -1, 1] - moving_obs[:, -1, 1, 0]) * side).sum())
    assert displacement > .1 and continued["response_increment"].abs().max() == 0, "已有运动被接触门锁死"
    unchanged = event_control_from_prior(obs, prior, velocity, torch.tensor([False]), projector=projector)
    assert torch.equal(unchanged["pred"], prior), "自由路线不是原样输出"
    future_changed = velocity.clone()
    future_changed[:, 35:] *= -3.0
    perturbed = event_control_from_prior(obs, prior, future_changed, true_route, projector=projector)
    assert torch.equal(perturbed["pred"][:, :36], result["pred"][:, :36]), "未来响应控制影响了当前输出"
    parameter = velocity.clone().requires_grad_(True)
    differentiable = event_control_from_prior(obs, prior, parameter, true_route, projector=projector)
    differentiable["pred"][:, -1, 1, 0].sum().backward()
    assert bool(torch.isfinite(parameter.grad).all()) and float(parameter.grad.abs().sum()) > 0, "接触后响应控制没有有限有效梯度"
    # 观测已有连续接触时可响应，且不把 obs_last / future_first 的重复帧增加为新证据。
    close_obs = prior[:, -10:].clone()
    close_prior = close_obs[:, -1:].expand_as(prior).clone()
    existing = event_control_from_prior(close_obs, close_prior, velocity, true_route, projector=projector)
    assert bool(existing["contact_active_in_obs"][0]), "观测中的已有接触未被识别"
    return OrderedDict([
        ("首帧严格连续", True), ("自由路线逐位不变", True), ("骨长最大绝对误差_m", bone_error),
        ("最终几何接触距离误差_m", final_geometry_error),
        ("首个持续接触帧", first_trigger), ("首个非零响应控制帧", first_response),
        ("手提前7cm响应帧", _first_response(ahead)), ("手滞后7cm响应帧", _first_response(behind)),
        ("无接触新增响应", 0.0), ("观测已有运动延续位移_m", displacement),
        ("未来控制无前缀泄露", True), ("响应梯度有限有效", True), ("obs中已有接触可响应", True),
    ])


def _select(value, index):
    return value if index is None else value[index]


def _scores(pred, target, obs, index=None):
    pred, target, obs = [_select(value, index) for value in (pred, target, obs)]
    if pred.shape[0] == 0:
        return {"num_samples": 0}
    score = compute_ntu_xyz_metrics(pred, target, obs)
    up = estimate_up(obs)
    root_delta = horizontal(pred[:, :, 1, 0] - target[:, :, 1, 0], up)
    score["receiver_horizontal_root_rmse"] = float(root_delta.double().square().mean().sqrt())
    score["num_samples"] = int(pred.shape[0])
    return score


def check_train(args, projector):
    cache = NTU2PXYZSeqCache.load(args.cache_dir, "train", manifest_path=args.manifest_path, device="cpu")
    windows, ids, starts = [], [], []
    # 每条序列只抽开始/中心/结束，覆盖和统计仍按独立序列/受试者报告。
    for index, action in enumerate(cache.actions.tolist()):
        if action != 2:
            continue
        sequence = cache.sequence(index)
        last = len(sequence) - 60
        for start in sorted({0, last // 2, last}):
            windows.append(sequence[start:start+60])
            ids.append(cache.sample_ids[index])
            starts.append(start)
    values = torch.stack(windows).double()
    obs, target = values[:, :10], values[:, 10:]
    labels = OrderedDict((str(threshold), _event_state(obs, target, threshold, .10)) for threshold in CONTACT_THRESHOLDS)
    stable = torch.stack([item["reliable"] for item in labels.values()]).all(dim=0)
    stationary = observed_continuation(obs, 50)[:, 0].norm(dim=-1) < .10
    stationary_stable = stable & stationary
    report = {"仅train": True, "sampled_windows": len(values), "pushing_sequences": len(set(ids)),
              "performers": sorted({int(re.search(r"P(\d{3})", item).group(1)) for item in ids}),
              "sampling": "每序列开始/中心/结束三点，重复起点只算一次",
              "threshold_label_coverage": OrderedDict(), "oracle": OrderedDict()}
    for threshold, item in labels.items():
        mask = item["reliable"]
        subset = [ids[i] for i in torch.nonzero(mask, as_tuple=False).view(-1).tolist()]
        report["threshold_label_coverage"][threshold] = {"reliable_windows": int(mask.sum()),
              "unique_sequences": len(set(subset)),
              "unique_performers": len({int(re.search(r"P(\d{3})", name).group(1)) for name in subset}),
              "fresh_windows": int(item["fresh"].sum()), "gt_contact_count": int((item["contact"] >= 0).sum())}
    report["stable_four_thresholds"] = {"windows": int(stable.sum()),
        "unique_sequences": len({ids[i] for i in torch.nonzero(stable, as_tuple=False).view(-1).tolist()}),
        "unique_performers": len({int(re.search(r"P(\d{3})", ids[i]).group(1)) for i in torch.nonzero(stable, as_tuple=False).view(-1).tolist()})}
    report["stationary_stable_contact_route"] = {"windows": int(stationary_stable.sum()),
        "unique_sequences": len({ids[i] for i in torch.nonzero(stationary_stable, as_tuple=False).view(-1).tolist()}),
        "unique_performers": len({int(re.search(r"P(\d{3})", ids[i]).group(1)) for i in torch.nonzero(stationary_stable, as_tuple=False).view(-1).tolist()})}
    for threshold in CONTACT_THRESHOLDS:
        full = oracle_event_root(obs, target, threshold=threshold, projector=projector)
        low = oracle_event_root(obs, target, threshold=threshold, num_coeffs=8, projector=projector)
        report["oracle"][str(threshold)] = {"GT_local_unrestricted_control": _scores(full["pred"], target, obs),
            "GT_local_DCT8_control": _scores(low["pred"], target, obs),
            "stable_subset_unrestricted": _scores(full["pred"], target, obs, stable),
            "stable_subset_DCT8": _scores(low["pred"], target, obs, stable),
            "stationary_stable_DCT8": _scores(low["pred"], target, obs, stationary_stable),
            "oracle_no_predicted_contact_windows": int((full["first_sustained_contact_frame"] >= 50).sum()),
            "stable_subset_no_predicted_contact": int(((full["first_sustained_contact_frame"] >= 50) & stable).sum()),
            "response_command_max_mps": float(full["response_velocity"].norm(dim=-1).max()),
            "stationary_stable_command_p95_mps": float(torch.quantile(full["response_velocity"][stationary_stable].norm(dim=-1).reshape(-1), .95)) if stationary_stable.any() else None,
            "stationary_stable_command_max_mps": float(full["response_velocity"][stationary_stable].norm(dim=-1).max()) if stationary_stable.any() else None}
    records = []
    for i, sample_id in enumerate(ids):
        records.append({"sample_id": sample_id, "start": starts[i], "four_thresholds_reliable": bool(stable[i]),
                        "stationary_stable_contact_route": bool(stationary_stable[i]),
                        "thresholds": {threshold: {key: bool(item[key][i]) if item[key].dtype == torch.bool else int(item[key][i]) for key in item}
                                       for threshold, item in labels.items()}})
    return report, records


def check_fixed_val(args, projector):
    data = torch.load(args.arrays, map_location="cpu")
    obs, target, prior = [value.double() for value in (data["obs_xyz"], data["target_xyz"], data["methods"]["FD2"])]
    pushing = data["actions"] == 2
    obs, target, prior = obs[pushing], target[pushing], prior[pushing]
    ids = [data["meta"][i]["sample_id"] for i in torch.nonzero(pushing, as_tuple=False).view(-1).tolist()]
    route = torch.ones(prior.shape[0], dtype=torch.bool)
    command = prior_response_velocity(obs, prior)
    variants = OrderedDict([("FD2", prior)])
    report = {"type": "固定非学习型root替换诊断，不是新模型优化结果", "uses_future_GT_as_input": False,
              "sample_ids": ids, "all_pushing_forced_contact_route": True, "thresholds": OrderedDict()}
    scene = estimate_scene_frame(torch.cat((obs, target), 1))
    natural_keys = ("slide_frame_ratio", "step_count", "skate_gt_contact_speed_walk", "skate_self_ratio_walk", "jerk_body", "foot_penetration_ratio", "mpjpe_legs")
    base_natural = aggregate_stats(compute_naturalness_stats(prior, target, obs, scene))
    report["baseline"] = {"scores": _scores(prior, target, obs), "naturalness": {key: base_natural[key] for key in natural_keys}}
    for threshold in CONTACT_THRESHOLDS:
        details = event_control_from_prior(obs, prior, command, route, threshold=threshold, projector=projector)
        natural = aggregate_stats(compute_naturalness_stats(details["pred"], target, obs, scene))
        variants["fixed_gate_{:.2f}".format(threshold)] = details["pred"]
        no_contact = details["first_sustained_contact_frame"] >= 50
        output_distance = hand_body_distances(details["pred"])[:, :, 0].amin(dim=(-1, -2))
        report["thresholds"][str(threshold)] = {"scores": _scores(details["pred"], target, obs),
            "naturalness": {key: natural[key] for key in natural_keys},
            "no_predicted_contact_count": int(no_contact.sum()),
            "no_contact_sample_ids": [ids[i] for i in torch.nonzero(no_contact, as_tuple=False).view(-1).tolist()],
            "trigger_frames": details["first_sustained_contact_frame"].tolist(),
            "max_final_distance_consistency_error": float((details["contact_distance"] - output_distance).abs().max()),
            "per_sample_mpjpe": (details["pred"] - target).norm(dim=-1).mean(dim=(1,2,3)).tolist(),
            "per_sample_receiver_root_rmse": (details["pred"][:,:,1,0]-target[:,:,1,0]).square().mean(dim=(1,2)).sqrt().tolist()}
    report["contact_metrics"] = contact_metrics(obs, target, variants, torch.full((len(prior),), 2, dtype=torch.long))
    return report, {"obs_xyz": obs.float(), "target_xyz": target.float(), "actions": torch.full((len(prior),), 2, dtype=torch.long),
                    "meta": [item for i,item in enumerate(data["meta"]) if bool(pushing[i])], "methods": {name:value.float() for name,value in variants.items()}}


def plot_fixed_val(arrays, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    obs = arrays["obs_xyz"].double()
    methods = {"真值": arrays["target_xyz"].double(), "FD2": arrays["methods"]["FD2"].double(),
               "固定事件门0.20": arrays["methods"]["fixed_gate_0.20"].double()}
    up = estimate_up(obs)
    axis = horizontal(obs[:, -1, 1, 0] - obs[:, -1, 0, 0], up)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    fig, axes = plt.subplots(len(obs), 2, figsize=(13, 2.4 * len(obs)))
    for row, metadata in enumerate(arrays["meta"]):
        for name, future in methods.items():
            full = torch.cat((obs[row:row+1], future[row:row+1]), 1)
            distance = hand_body_distances(full)[0, :, 0].amin(dim=(-1, -2))
            retreat = ((full[0, :, 1, 0] - obs[row, -1, 1, 0]) * axis[row]).sum(-1)
            time_axis = (torch.arange(full.shape[1]) - obs.shape[1] + 1) / 20.0
            axes[row, 0].plot(time_axis.numpy(), distance.numpy(), label=name)
            axes[row, 1].plot(time_axis.numpy(), retreat.numpy(), label=name)
        for column in (0, 1):
            axes[row, column].axvline(0, color="black", linewidth=.7)
            axes[row, column].set_title(metadata["sample_id"], fontsize=9)
            axes[row, column].set_xlabel("预测起点后的时间 / 秒", fontproperties=font)
        axes[row, 0].axhline(.20, color="black", linestyle="--", linewidth=.7)
        axes[row, 0].set_ylabel("掌腕—躯干中心距 / 米", fontproperties=font)
        axes[row, 1].set_ylabel("受动者后退位移 / 米", fontproperties=font)
    axes[0, 0].legend(prop=font)
    fig.suptitle("固定非学习事件门：全部12条推人窗口，不是模型优化结果", fontproperties=font)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_ROOT)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE)
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--arrays", default=DEFAULT_ARRAYS)
    parser.add_argument("--structural_only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    projector = SkeletonProjector().double().eval()
    report = {"synthetic": check_synthetic(projector)}
    if not args.structural_only:
        with torch.no_grad():
            start = time.time()
            train, records = check_train(args, projector)
            train["elapsed_seconds"] = time.time() - start
            report["train_feasibility"] = train
            val, arrays = check_fixed_val(args, projector)
            report["fixed_Hval_diagnostic"] = val
            torch.save(arrays, root / "fixed_pushing_val_arrays.pt")
            plot_fixed_val(arrays, root / "all12_fixed_gate_contact_retreat.png")
            (root / "train_weak_label_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "feasibility.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["synthetic"], ensure_ascii=False, indent=2))
    if not args.structural_only:
        print(json.dumps({"train": report["train_feasibility"], "fixed_Hval": {key:value for key,value in report["fixed_Hval_diagnostic"].items() if key != "contact_metrics"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
