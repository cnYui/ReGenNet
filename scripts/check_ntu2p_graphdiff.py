"""验证接触图扩散的结构约束，避免把点预测与随机采样混为同一证据。

用法：OMP_NUM_THREADS=2 python scripts/check_ntu2p_graphdiff.py --device cpu
结果包含包络基条件数与 oracle 前 10 帧误差，不以这些 val 记录选择超参数。
"""

import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = "results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json"
DEFAULT_CACHE = "results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval"
DEFAULT_PRIOR = (
    "save/forecasting/ntu120_label/"
    "ntu2p_v2subj_A6-F-A5f0.05-A4-GH0.5-FD2_s0_10000/ema/model000010000.pt"
)
SECTIONS = ("zero", "codec", "geometry", "diffusion", "gradients", "memory", "graph", "checkpoint", "cuda")
FINGERTIPS = ((20, (27, 30, 33, 36, 39)), (21, (42, 45, 48, 51, 54)))


def record(results, section, label, passed, **details):
    entry = dict(section=section, label=label, passed=bool(passed), details=details)
    results.append(entry)
    print("[{}] {}: {} {}".format("通过" if passed else "失败", section, label, details), flush=True)
    return bool(passed)


def max_abs(left, right):
    return float((left - right).abs().max().item())


def random_tensor(shape, generator, device):
    return torch.randn(tuple(shape), generator=generator).to(device)


def check_zero(results, models, obs, action):
    signatures = []
    with torch.no_grad():
        for name, model in models.items():
            y = model.prepare(obs, action)
            prediction = model.point_details(obs, action, prepared=y)["pred"]
            error = max_abs(prediction, y["draft"])
            record(results, "zero", name + " 零头与 prior 相等", error <= 1e-5, max_abs_m=error)
            record(results, "zero", name + " 首帧严格等于观测末帧", torch.equal(prediction[:, 0], obs[:, -1]))
            signatures.append((name, {key: value.clone() for key, value in model.state_dict().items()}))
            model.train()
            record(results, "zero", name + " train 不解冻 prior",
                   not model.prior.training and not any(p.requires_grad for p in model.prior.parameters()))
            model.eval()
    reference = signatures[0][1]
    for name, state in signatures[1:]:
        equal = state.keys() == reference.keys() and all(torch.equal(state[key], reference[key]) for key in reference)
        record(results, "zero", name + " 与 C1 参数初始化一致", equal)


def check_codec(results, model, generator, device):
    codec = model.codec
    coeffs = random_tensor((4, 2, model.num_coeffs, 66), generator, device) * 0.1
    decoded = codec.decode_channels(coeffs)
    error = max_abs(codec.encode_channels(decoded), coeffs)
    record(results, "codec", "系数空间编解码互逆", error <= 1e-5, max_abs=error)
    record(results, "codec", "任意系数的首帧残差严格为零", bool(decoded[:, 0].eq(0).all()))
    singular = torch.svd(codec.basis.double()).S
    condition = float((singular.max() / singular.min()).item())
    record(results, "codec", "平滑基满秩且数值有限",
           bool(singular.min() > 1e-10) and math.isfinite(condition), condition_number=condition,
           smallest_singular=float(singular.min().item()),
           first_discrete_correction=float((decoded[:, 1] - decoded[:, 0]).norm(dim=-1).mean().item()))


def check_geometry(results, model, obs, action, target, generator, device):
    from utils.ntu2p_naturalness import bone_lengths, gait_phase_stats
    from utils.ntu2p_contact_graph import hand_body_distances

    with torch.no_grad():
        y = model.prepare(obs, action)
        oracle_m = model.codec.target_coeffs(obs, y["draft"], target, y["frames"])
        oracle = model.apply_coeffs(oracle_m / model.sigma, y)["pred"]
        prior_err = (y["draft"] - target).norm(dim=-1)
        oracle_err = (oracle - target).norm(dim=-1)
        record(results, "geometry", "oracle 解码与投影有限", bool(torch.isfinite(oracle).all()),
               prior_mpjpe=float(prior_err.mean().item()), oracle_mpjpe=float(oracle_err.mean().item()),
               prior_first10_mpjpe=float(prior_err[:, :10].mean().item()),
               oracle_first10_mpjpe=float(oracle_err[:, :10].mean().item()),
               prior_frames2to10_mpjpe=float(prior_err[:, 1:10].mean().item()),
               oracle_frames2to10_mpjpe=float(oracle_err[:, 1:10].mean().item()),
               oracle_first5_mpjpe=float(oracle_err[:, :5].mean().item()))
        grouped = {}
        for name, value in (("prior", y["draft"]), ("oracle", oracle)):
            phase = gait_phase_stats(value, target, obs)
            grouped[name] = {}
            for key, (numerator, denominator) in phase.items():
                if key.startswith("gait_sep_corr_f01_10") or key.startswith("gait_sep_corr_f11_20"):
                    total = float(denominator.sum().item())
                    grouped[name][key] = None if total == 0 else float(numerator.sum().item()) / total
            grouped[name]["walking_persons"] = float(phase["gait_walk_count"][0].sum().item())
            grouped[name]["moving_persons"] = float(phase["gait_moving_count"][0].sum().item())
        record(results, "geometry", "记录 oracle 的起步与已行走分组表达上限", True, phase=grouped)
        pushing = action == 2
        if bool(pushing.any()):
            target_distance = hand_body_distances(target[pushing]).amin(dim=(-1, -2))[:, :, 0]
            distances = {}
            for name, value in (("prior", y["draft"]), ("oracle", oracle)):
                distance = hand_body_distances(value[pushing]).amin(dim=(-1, -2))[:, :, 0]
                distances[name] = float((distance - target_distance).square().mean().sqrt().item())
            record(results, "geometry", "记录推人 oracle 的连续接触表达上限", True,
                   pushing_windows=int(pushing.sum().item()), actor_distance_rmse=distances)
        noise = 0.05 * random_tensor(oracle_m.shape, generator, device)
        prediction = model.apply_coeffs(noise / model.sigma, y)["pred"]
        record(results, "geometry", "非零残差投影后首帧严格连续", torch.equal(prediction[:, 0], obs[:, -1]))
        reference = bone_lengths(obs[:, -1:].double()).clamp_min(1e-4)
        relative = ((bone_lengths(prediction.double()) - reference).abs() / reference)[..., :21]
        mean_rel = float(relative.mean().item())
        record(results, "geometry", "身体骨长误差均值小于 1e-5", mean_rel <= 1e-5,
               mean_relative=mean_rel, max_relative=float(relative.max().item()))
        finger_error = 0.0
        for wrist, tips in FINGERTIPS:
            current = (prediction[..., list(tips), :] - prediction[..., wrist:wrist + 1, :]).norm(dim=-1)
            reference_hand = (obs[:, -1:, :, list(tips), :] - obs[:, -1:, :, wrist:wrist + 1, :]).norm(dim=-1)
            finger_error = max(finger_error, max_abs(current, reference_hand))
        record(results, "geometry", "刚体手指尖与腕距离保持", finger_error <= 1e-5, max_abs_m=finger_error)


def check_diffusion(results, model, obs, action, generator, device):
    from diffusion.gaussian_diffusion import ModelMeanType
    from model.forecasting_ntu2p_graphdiff import build_sampling_diffusion, build_training_diffusion

    diffusion = build_training_diffusion()
    record(results, "diffusion", "复用 1000 步 START_X 扩散",
           diffusion.num_timesteps == 1000 and diffusion.model_mean_type == ModelMeanType.START_X,
           terminal_alpha_bar=float(diffusion.alphas_cumprod[-1]))
    x0 = 0.1 * random_tensor((obs.shape[0], 2, model.num_coeffs, 66), generator, device)
    noise = random_tensor(x0.shape, generator, device)
    timesteps = torch.linspace(0, 999, obs.shape[0], device=device).long()
    noisy = diffusion.q_sample(x0, timesteps, noise=noise)
    with torch.no_grad():
        y = model.prepare(obs, action)
        denoised = model.predict_coeffs(noisy, timesteps, y)
    record(results, "diffusion", "前向噪声与系数预测形状正确且有限",
           noisy.shape == x0.shape == denoised.shape and bool(torch.isfinite(denoised).all()))
    for count in (20, 50):
        indices = list(build_sampling_diffusion(count).timestep_map)
        record(results, "diffusion", "NFE{} 包含原始端点时间索引".format(count),
               len(indices) == count and indices[0] == 0 and indices[-1] == 999,
               count=len(indices), start=indices[0], end=indices[-1])
    with torch.no_grad():
        sampled = model.sample_details(obs, action, noise, num_steps=3)["pred"]
        prior = model.prepare(obs, action)["draft"]
    error = max_abs(sampled, prior)
    record(results, "diffusion", "零头 DDIM 与 prior 相等", error <= 1e-5, max_abs_m=error)


def check_gradients(results, model, obs, action, generator, device):
    model.eval()
    y = model.prepare(obs, action)
    x = 0.1 * random_tensor((obs.shape[0], 2, model.num_coeffs, 66), generator, device)
    target = 0.1 * random_tensor(x.shape, generator, device)
    timesteps = torch.full((obs.shape[0],), 500, dtype=torch.long, device=device)
    optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad), lr=0.1)
    optimizer.zero_grad()
    first_loss = (model.predict_coeffs(x, timesteps, y) - target).square().mean()
    first_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss = (model.predict_coeffs(x, timesteps, y) - target).square().mean()
    loss.backward()
    groups = {"整体": [], "图消息": [], "腿相位": [], "观测编码": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        groups["整体"].append(parameter.grad)
        if name.startswith("blocks.") and ".spatial." in name:
            groups["图消息"].append(parameter.grad)
        if name.startswith("leg_net."):
            groups["腿相位"].append(parameter.grad)
        if name.startswith("obs_embed."):
            groups["观测编码"].append(parameter.grad)
    for label, gradients in groups.items():
        norm = math.sqrt(sum(float(value.double().square().sum().item()) for value in gradients))
        finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
        record(results, "gradients", label + " 获得有限非零梯度", finite and norm > 0, gradient_norm=norm)
    record(results, "gradients", "prior 不产生梯度", all(param.grad is None for param in model.prior.parameters()))
    model.zero_grad()


def check_memory(results, generator, device):
    from utils.ntu2p_contact_graph import causal_contact_memory, contact_graph_features

    probabilities = torch.sigmoid(random_tensor((2, 50, 2, 2, 4), generator, device))
    changed = probabilities.clone()
    changed[:, 25:] = 1 - changed[:, 25:]
    original_memory = causal_contact_memory(probabilities)
    changed_memory = causal_contact_memory(changed)
    record(results, "memory", "改变未来接触不改变过去记忆",
           torch.equal(original_memory[:, :25], changed_memory[:, :25]))
    record(results, "memory", "事件记忆形状正确、有限且在概率范围内",
           original_memory.shape == probabilities.shape and bool(torch.isfinite(original_memory).all())
           and bool(((original_memory >= 0) & (original_memory <= 1)).all()))
    geometry = random_tensor((2, 50, 2, 55, 3), generator, device)
    future_changed = geometry.clone()
    future_changed[:, 25:, 1] += 2.0
    first = contact_graph_features(geometry)
    second = contact_graph_features(future_changed)
    record(results, "memory", "完整接触特征生成器不把未来送入过去记忆",
           torch.equal(first["memory"][:, :25], second["memory"][:, :25]))
    record(results, "memory", "接触图邻接输出为双人 22 节点",
           tuple(first["adjacency"].shape) == (2, 50, 2, 22, 22)
           and bool(torch.isfinite(first["adjacency"]).all()))


def check_graph(results, model, generator, device):
    block = model.blocks[0].eval()
    features = random_tensor((2, 8, 2, 22, model.hidden), generator, device)
    zero = torch.zeros(2, 8, 2, 22, 22, device=device)
    edge = zero.clone()
    edge[:, :, 0, 20, 1] = 1.0
    modified = features.clone()
    modified[:, :, 1, 1, 0] += 0.5
    # 固定几何和所有其它路径，差分只检验新增接触边承载的对方信息。
    with torch.no_grad():
        active = block(features, edge, True)[:, :, 0, 20]
        inactive = block(features, zero, True)[:, :, 0, 20]
        changed_active = block(modified, edge, True)[:, :, 0, 20]
        changed_inactive = block(modified, zero, True)[:, :, 0, 20]
    increment = max_abs(changed_active - changed_inactive, active - inactive)
    record(results, "graph", "固定图几何下对方节点改变接触边增量", increment > 1e-6,
           graph_specific_response=increment)
    with torch.no_grad():
        disabled = block(features, edge, False)
        disabled_zero = block(features, zero, False)
    record(results, "graph", "关闭图时接触邻接不影响输出", torch.equal(disabled, disabled_zero))


def check_checkpoint(results, model, obs, action, args):
    from model.forecasting_ntu2p_graphdiff import load_ntu2p_graphdiff_checkpoint

    with tempfile.TemporaryDirectory(prefix="ntu2p-graphdiff-check-") as temporary:
        path = Path(temporary) / "model.pt"
        torch.save(dict(model_type=model.model_type, model_config=model.config(),
                        model_state_dict=model.state_dict(), step=1, seed=args.seed,
                        manifest_path=args.manifest_path), path)
        restored, state = load_ntu2p_graphdiff_checkpoint(str(path), args.device)
        with torch.no_grad():
            expected = model.eval()(obs, action)
            actual = restored(obs, action)
        error = max_abs(expected, actual)
        record(results, "checkpoint", "非零 head 权重重载后预测相同", error <= 1e-7, max_abs_m=error)
        config_equal = all(restored.config()[key] == model.config()[key]
                           for key in ("graph", "phase", "diffusion", "num_coeffs", "smooth_frames"))
        record(results, "checkpoint", "配置开关与训练元数据保存", config_equal and state["step"] == 1)


def check_cuda(results):
    trainer = importlib.import_module("train.train_ntu2p_graphdiff")
    for requested, available in (("cpu", True), ("cuda:0", False)):
        args = argparse.Namespace(device=requested, allow_cpu_for_smoke_test=False)
        with mock.patch("torch.cuda.is_available", return_value=available):
            try:
                trainer._device(args)
            except (ValueError, RuntimeError) as error:
                rejected = True
                message = str(error)
            else:
                rejected = False
                message = "训练入口意外接受设备"
        record(results, "cuda", "正式训练拒绝 {} available={}".format(requested, available),
               rejected, message=message)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior_checkpoint", default=DEFAULT_PRIOR)
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--only", default=",".join(SECTIONS))
    parser.add_argument("--output_json", default="results/forecasting/ntu120_label/ntu2p_graphdiff/structural_checks.json")
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.chdir(REPO_ROOT)
    selected = args.only.split(",")
    if any(name not in SECTIONS for name in selected) or args.batch_size < 2:
        raise ValueError("--only 必须为已定义检查项，batch_size 至少为 2")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 结构检查，但 CUDA 不可用")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, eval_windows
    from model.forecasting_ntu2p_graphdiff import NTU2PContactGraphDiffusion
    from model.forecasting_ntu2p_v2 import load_ntu2p_model_checkpoint

    cache = NTU2PXYZSeqCache.load(args.cache_dir, "val", manifest_path=args.manifest_path, device=device)
    data = eval_windows(cache)
    obs = data["obs_xyz"][:args.batch_size]
    action = data["action"][:args.batch_size]
    target = data["target_xyz"][:args.batch_size]
    prior, _ = load_ntu2p_model_checkpoint(args.prior_checkpoint, device)
    models = {}
    for name, graph, phase, diffusion in (("C1", False, False, False), ("C2", True, False, False),
                                          ("C3", True, True, False), ("C4", True, True, True)):
        torch.manual_seed(args.seed)
        models[name] = NTU2PContactGraphDiffusion(copy.deepcopy(prior), graph=graph, phase=phase,
                                                diffusion=diffusion).to(device).eval()
    model = models["C4"]
    generator = torch.Generator().manual_seed(args.seed + 1)
    results = []
    checks = {
        "zero": lambda: check_zero(results, models, obs, action),
        "codec": lambda: check_codec(results, model, generator, device),
        "geometry": lambda: check_geometry(results, model, obs, action, target, generator, device),
        "diffusion": lambda: check_diffusion(results, model, obs, action, generator, device),
        "gradients": lambda: check_gradients(results, model, obs, action, generator, device),
        "memory": lambda: check_memory(results, generator, device),
        "graph": lambda: check_graph(results, model, generator, device),
        "checkpoint": lambda: check_checkpoint(results, model, obs, action, args),
        "cuda": lambda: check_cuda(results),
    }
    for section in selected:
        try:
            checks[section]()
        except Exception as error:
            record(results, section, "检查执行异常", False, error_type=type(error).__name__, message=str(error))
    payload = dict(passed=all(entry["passed"] for entry in results), device=args.device,
                   prior_checkpoint=args.prior_checkpoint, manifest_path=args.manifest_path,
                   seed=args.seed, batch_size=int(obs.shape[0]), checks=results)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("结构检查结果：{}，文件 {}".format("通过" if payload["passed"] else "失败", output), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
