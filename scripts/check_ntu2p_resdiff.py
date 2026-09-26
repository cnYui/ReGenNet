"""Track B CPU 自检（必须全部通过才进入 GPU 阶段）。

用法：
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONPATH=. nice -n 10 python scripts/check_ntu2p_resdiff.py \
        --scratch_dir <scratchpad>/trackb/check [--only codec,sampling]

10 节：codec、apply_equivalence、denoiser、sampling、metrics、folds、bank_smoke、train_smoke、eval_smoke、driver。
后四节依赖前面节的产物（折 -> 冒烟残差库 -> 冒烟生成器 -> 冒烟评估），用 --only 单独跑时会复用 scratch 中已有的产物。
"""

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import DEFAULT_CACHE_DIR, NTU2PXYZSeqCache, eval_windows
from scripts.run_ntu2p_trackb import MAIN_CONFIG  # 与驱动同一主线（第 2 批后为 FD2）

SECTIONS = ("codec", "apply_equivalence", "denoiser", "sampling", "metrics", "folds", "bank_smoke", "train_smoke", "eval_smoke", "driver")
PYTHON = sys.executable
SUBJVAL_MANIFEST = "results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json"
SUBJVAL_CACHE = "results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval"
SUBJVAL_BASELINE = "save/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_subjval_s0_5000/model000005000.pt"
ORIGINAL_MANIFEST = "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"
ORIGINAL_V2 = "save/forecasting/ntu120_label/ntu2p_v2_{}_s{{}}_10000/ema/model000010000.pt".format(MAIN_CONFIG)
SUBJVAL_V2 = "save/forecasting/ntu120_label/ntu2p_v2subj_{}_s{{}}_10000/ema/model000010000.pt".format(MAIN_CONFIG)
ORIGINAL_A0 = "save/forecasting/ntu120_label/ntu2p_v2_A0_s0_10000/ema/model000010000.pt"
FINGERTIPS = ((20, (27, 30, 33, 36, 39)), (21, (42, 45, 48, 51, 54)))


class Report(object):
    def __init__(self):
        self.results = OrderedDict()
        self.section = None

    def start(self, name):
        self.section = name
        self.results[name] = []
        print("== {}".format(name), flush=True)

    def check(self, label, ok, detail=""):
        ok = bool(ok)
        self.results[self.section].append((label, ok, detail))
        print("  [{}] {} {}".format("PASS" if ok else "FAIL", label, detail), flush=True)
        return ok

    def failed(self):
        return [(section, label, detail) for section, items in self.results.items() for label, ok, detail in items if not ok]


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "2"
    return env


def _run(command, log):
    start = time.time()
    with open(log, "w") as handle:
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT, env=_env())
    return code, time.time() - start


def _windows(manifest, cache_dir, count):
    cache = NTU2PXYZSeqCache.load(cache_dir, "val", manifest_path=manifest, device="cpu")
    windows = eval_windows(cache)
    return OrderedDict((key, value[:count]) for key, value in windows.items())


def _subjval_base(seed=0):
    path = SUBJVAL_V2.format(seed)
    return path if os.path.exists(path) else ORIGINAL_V2.format(seed)


# ---------------------------------------------------------------- 1 codec


def check_codec(report, scratch):
    from utils.ntu2p_canonical import NTU2P_DATASET_UP, apply_linear, rotation_about_axis
    from utils.ntu2p_residual_codec import FOLLOWERS, ResidualCodec, codec_frames, condition_features, to_camera, to_person

    codec = ResidualCodec()
    report.check("basis[:,0] 严格为 0", bool((codec.basis[:, 0] == 0).all()))
    generator = torch.Generator().manual_seed(0)
    coeffs = torch.randn(4, 2, 12, 66, generator=generator)
    roundtrip = (codec.encode_channels(codec.decode_channels(coeffs)) - coeffs).abs().max().item()
    report.check("encode(decode(c)) 与 c 的差 ≤ 1e-5", roundtrip <= 1e-5, "{:.2e}".format(roundtrip))
    residual = codec.channels_to_residual(codec.decode_channels(coeffs))
    exact = all(torch.equal(residual[..., list(dst), :], residual[..., src : src + 1, :].expand(-1, -1, -1, len(dst), -1)) for src, dst in FOLLOWERS)
    report.check("随动关节残差逐位等于源关节", exact)

    data = _windows(SUBJVAL_MANIFEST, SUBJVAL_CACHE, 32)
    obs, target = data["obs_xyz"], data["target_xyz"]
    frames = codec_frames(obs)
    # 残差量级（v2 残差 |r| 通常 < 1 m）；float32 旋转往返的误差与 |v| 成正比。
    vec = 0.3 * torch.randn(obs.shape[0], 50, 2, 55, 3, generator=generator)
    back = (to_camera(to_person(vec, frames), frames) - vec).abs().max().item()
    report.check("to_camera(to_person(v)) 与 v 的差 ≤ 1e-6（|v|≤{:.2f} m）".format(vec.norm(dim=-1).max().item()), back <= 1e-6, "{:.2e}".format(back))

    base = obs[:, -1:] + 0.5 * (target - obs[:, -1:])
    distance = (obs[:, -1, 1, 0] - obs[:, -1, 0, 0])
    up = torch.tensor(NTU2P_DATASET_UP).unsqueeze(0)
    horizontal = (distance - (distance * up).sum(-1, keepdim=True) * up).norm(dim=-1)
    keep = horizontal >= 0.1
    angle = torch.rand(obs.shape[0], generator=generator) * 2 * math.pi - math.pi
    rotation = rotation_about_axis(up.expand(obs.shape[0], 3) / up.norm(), angle)
    shift = torch.randn(obs.shape[0], 3, generator=generator)
    shift = shift - (shift * up).sum(-1, keepdim=True) * up
    center = obs[:, -1, :, 0].mean(dim=1)

    def move(value):
        view = (value.shape[0],) + (1,) * (value.dim() - 2) + (3,)
        return apply_linear(value - center.view(view), rotation) + (center + shift).view(view)

    frames_moved = codec_frames(move(obs))
    target_a = codec.target_coeffs(obs, base, target, frames)[keep]
    target_b = codec.target_coeffs(move(obs), move(base), move(target), frames_moved)[keep]
    draft_a = condition_features(codec, obs, base, frames)["draft"][keep]
    draft_b = condition_features(codec, move(obs), move(base), frames_moved)["draft"][keep]
    diff = max((target_a - target_b).abs().max().item(), (draft_a - draft_b).abs().max().item())
    report.check("整场景绕 up 轴 yaw + 水平平移后 target/draft 系数变化 ≤ 1e-4（{} 窗）".format(int(keep.sum())), diff <= 1e-4, "{:.2e}".format(diff))


# ---------------------------------------------------------------- 2 apply_equivalence


def check_apply_equivalence(report, scratch):
    from model.forecasting_ntu2p_resdiff import NTU2PResDiffDenoiser, build_sampling_diffusion, make_condition, repeat_condition, sample_x0, to_meters
    from model.forecasting_ntu2p_v2 import forward_details, load_ntu2p_model_checkpoint
    from utils.ntu2p_kinematic_projection import SkeletonProjector
    from utils.ntu2p_naturalness import bone_lengths
    from utils.ntu2p_residual_codec import ResidualCodec, codec_frames, condition_features, repeat_frames
    from utils.ntu_smplx_2p_xyz import compute_ntu_xyz_metrics

    model, _ = load_ntu2p_model_checkpoint(ORIGINAL_V2.format(0), "cpu")
    data = _windows(ORIGINAL_MANIFEST, DEFAULT_CACHE_DIR, 16)
    obs, target, action = data["obs_xyz"], data["target_xyz"], data["action"]
    codec, projector = ResidualCodec(), SkeletonProjector()
    with torch.no_grad():
        base = forward_details(model, obs, action)["pred"]
        frames = codec_frames(obs)
        zero = codec.apply(obs, base, torch.zeros(obs.shape[0], 2, 12, 66), frames, projector)
        diff = (zero - base).abs().max().item()
        report.check("零系数输出与 v2 最大差 ≤ 1e-5", diff <= 1e-5, "{:.2e}".format(diff))
        oracle_c = codec.target_coeffs(obs, base, target, frames)
        oracle = codec.apply(obs, base, oracle_c, frames, projector)
        metrics = compute_ntu_xyz_metrics(oracle, target, obs)
        v2_metrics = compute_ntu_xyz_metrics(base, target, obs)
        report.check("oracle 系数 mpjpe ≤ 0.045", metrics["mpjpe"] <= 0.045, "{:.4f}（v2 {:.4f}）".format(metrics["mpjpe"], v2_metrics["mpjpe"]))
        first = (oracle[:, 0] - obs[:, -1]).norm(dim=-1).max().item()
        report.check("oracle 首帧误差 ≤ 1e-6（最大值）", first <= 1e-6, "{:.2e}".format(first))
        rooted = oracle_c.clone()
        rooted[..., :3] = 0.0
        anchored = codec.apply(obs, base, rooted, frames, projector)
        report.check("root 系数为 0 时 pelvis 与 v2 torch.equal", torch.equal(anchored[..., 0, :], base[..., 0, :]))
        sigma = oracle_c.pow(2).mean(dim=(0, 1)).sqrt().clamp_min(1e-4)
        noise = torch.randn(oracle_c.shape, generator=torch.Generator().manual_seed(1)) * 3.0 * sigma
        wild = codec.apply(obs, base, noise, frames, projector)
        reference = bone_lengths(obs[:, -1:].double()).clamp_min(1e-4)
        relative = ((bone_lengths(wild.double()) - reference).abs() / reference)[..., :21]
        report.check("3σ 随机系数 bone_rel_err_body（均值）≤ 1e-5", relative.mean().item() <= 1e-5,
                     "均值 {:.2e}，最大 {:.2e}".format(relative.mean().item(), relative.max().item()))
        worst = 0.0
        for wrist, tips in FINGERTIPS:
            now = (wild[..., list(tips), :] - wild[..., wrist : wrist + 1, :]).norm(dim=-1)
            last = (obs[:, -1:, :, list(tips)] - obs[:, -1:, :, wrist : wrist + 1]).norm(dim=-1)
            worst = max(worst, (now - last).abs().max().item())
        report.check("3σ 随机系数下指尖到手腕距离与观测末帧差 ≤ 1e-5", worst <= 1e-5, "{:.2e}".format(worst))
        # 零初始化生成器端到端：采样 -> 反归一化 -> 解码投影 = v2（零残差），mode R 的 pelvis 逐位等于 v2。
        denoiser = NTU2PResDiffDenoiser(stats={"sigma_target": sigma, "sigma_draft": sigma, "feat_std": torch.ones(66)}).eval()
        feats = condition_features(codec, obs, base, frames)
        repeats = 3
        noise = torch.randn(obs.shape[0] * repeats, 2, 12, 66, generator=torch.Generator().manual_seed(2))
        for mode in ("F", "R"):
            y = repeat_condition(make_condition(denoiser, feats, action, mode), repeats)
            x0 = sample_x0(denoiser, build_sampling_diffusion(10), y, noise)
            out = codec.apply(obs.repeat_interleave(repeats, 0), base.repeat_interleave(repeats, 0), to_meters(denoiser, x0),
                              repeat_frames(frames, repeats), projector)
            diff = (out - base.repeat_interleave(repeats, 0)).abs().max().item()
            report.check("零初始化生成器 mode {} 采样输出与 v2 差 ≤ 1e-5".format(mode), diff <= 1e-5, "{:.2e}".format(diff))
            if mode == "R":
                report.check("零初始化生成器 mode R pelvis 与 v2 torch.equal",
                             torch.equal(out[..., 0, :], base.repeat_interleave(repeats, 0)[..., 0, :]))


# ---------------------------------------------------------------- 3 denoiser


def _random_condition(batch, generator, root_known=None):
    y = OrderedDict(
        [
            ("draft", torch.randn(batch, 2, 12, 66, generator=generator)),
            ("obs_feats", torch.randn(batch, 2, 10, 66, generator=generator)),
            ("rel_geom", torch.randn(batch, 2, 6, generator=generator)),
            ("action", torch.randint(0, 26, (batch,), generator=generator)),
            ("root_known", torch.zeros(batch, dtype=torch.bool) if root_known is None else root_known),
            ("root_value", torch.randn(batch, 2, 12, 3, generator=generator)),
        ]
    )
    return y


def check_denoiser(report, scratch):
    from model.forecasting_ntu2p_resdiff import NTU2PResDiffDenoiser, count_parameters

    torch.manual_seed(0)
    model = NTU2PResDiffDenoiser()
    model.eval()
    generator = torch.Generator().manual_seed(3)
    batch = 6
    known = torch.tensor([False, True, False, True, True, False])
    y = _random_condition(batch, generator, known)
    x = torch.randn(batch, 2, 12, 66, generator=generator)
    t = torch.randint(0, 1000, (batch,), generator=generator)
    out = model(x, t, y)
    report.check("零初始化：未给 root 的样本输出严格为 0", torch.equal(out[~known], torch.zeros_like(out[~known])))
    report.check("零初始化：给 root 的样本局部通道严格为 0", torch.equal(out[known][..., 3:], torch.zeros_like(out[known][..., 3:])))
    report.check("root_known 时输出 root 通道 torch.equal root_value", torch.equal(out[known][..., :3], y["root_value"][known]))
    report.check("输出有限", bool(torch.isfinite(out).all()))
    params = count_parameters(model)
    report.check("参数量在 [4.5M, 5.5M]", 4.5e6 <= params <= 5.5e6, "{}".format(params))
    model.train()
    target = torch.randn(batch, 2, 12, 66, generator=generator)
    loss = (model(x, t, y) - target).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    total = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
    report.check("假损失反传：梯度非零且有限", finite and total > 0, "|g|={:.3e}".format(total))


# ---------------------------------------------------------------- 4 sampling


def check_sampling(report, scratch):
    import diffusion.gaussian_diffusion  # noqa: F401
    import model.cmdm  # noqa: F401
    from model.forecasting_ntu2p_resdiff import NTU2PResDiffDenoiser, build_sampling_diffusion, one_step_mean, sample_x0

    report.check("可 import diffusion.gaussian_diffusion 与 model.cmdm", True)
    sampler = build_sampling_diffusion(10)
    report.check("timestep_map == [0,111,…,999]", list(sampler.timestep_map) == [0, 111, 222, 333, 444, 555, 666, 777, 888, 999],
                 str(list(sampler.timestep_map)))
    torch.manual_seed(0)
    zero_model = NTU2PResDiffDenoiser().eval()
    torch.manual_seed(0)
    model = NTU2PResDiffDenoiser().eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.out.weight, std=0.05)
        torch.nn.init.normal_(model.out.bias, std=0.05)
    generator = torch.Generator().manual_seed(4)
    batch = 4
    y = _random_condition(batch, generator)
    noise = torch.randn(batch, 2, 12, 66, generator=generator)
    first = sample_x0(model, sampler, y, noise)
    second = sample_x0(model, sampler, y, noise)
    report.check("同一份噪声跑两次 DDIM 结果 torch.equal", torch.equal(first, second))
    other = sample_x0(model, sampler, y, torch.randn(batch, 2, 12, 66, generator=generator))
    report.check("不同噪声得到不同样本（K 个样本可区分）", (first - other).abs().max().item() > 1e-3, "{:.3e}".format((first - other).abs().max().item()))
    zero = sample_x0(zero_model, sampler, y, noise)
    report.check("零初始化模型采样结果严格为 0", torch.equal(zero, torch.zeros_like(zero)))
    y_root = OrderedDict(y)
    y_root["root_known"] = torch.ones(batch, dtype=torch.bool)
    y_root["root_value"] = torch.zeros(batch, 2, 12, 3)
    rooted = sample_x0(model, sampler, y_root, noise)
    report.check("mode R 采样 root 通道严格为 0", torch.equal(rooted[..., :3], torch.zeros_like(rooted[..., :3])))
    mean = one_step_mean(model, y, torch.randn(batch, 5, 2, 12, 66, generator=generator))
    report.check("one_step_mean 形状 [B,2,K,C]", tuple(mean.shape) == (batch, 2, 12, 66), str(tuple(mean.shape)))
    check_sampler_dispersion(report)


def check_sampler_dispersion(report):
    """解析高斯 oracle 去噪器下的采样离散度：x0 ~ N(μ, s²) 时 x̂0(x_t) 有闭式解，样本 std 偏小只能来自采样器离散化。

    DDIM（eta=0）步数少时系统性欠散（NFE10 仅 0.78–0.87），τ=1 就不是校准采样；主采样 NFE 下须 ≥ 0.95。
    """
    from model.forecasting_ntu2p_resdiff import DEFAULT_NFE, build_sampling_diffusion, build_training_diffusion, sample_x0
    from scripts.run_ntu2p_trackb import NFE_GRID

    for nfe in NFE_GRID:
        steps = list(build_sampling_diffusion(nfe).timestep_map)
        report.check("NFE{} 采样步含 0 与 999、共 {} 步".format(nfe, nfe), steps[0] == 0 and steps[-1] == 999 and len(steps) == nfe)
    abar = torch.tensor(build_training_diffusion().alphas_cumprod, dtype=torch.float64)

    class GaussianOracle(torch.nn.Module):
        def __init__(self, mu, std):
            super(GaussianOracle, self).__init__()
            self.mu, self.std = mu, std

        def forward(self, x, t, y):
            a = abar[t.long()].to(x.dtype).view(-1, 1, 1, 1)
            return self.mu + a.sqrt() * self.std ** 2 / (a * self.std ** 2 + 1 - a) * (x - a.sqrt() * self.mu)

    noise = torch.randn(400, 2, 12, 66, generator=torch.Generator().manual_seed(7))
    mu = 0.3
    for std in (0.3, 1.0):
        oracle = GaussianOracle(mu, std)
        sample = sample_x0(oracle, build_sampling_diffusion(DEFAULT_NFE), {}, noise)
        coarse = sample_x0(oracle, build_sampling_diffusion(10), {}, noise)
        ratio, bias = float(sample.std()) / std, float(sample.mean()) - mu
        report.check("解析高斯 oracle（条件 std={}）NFE{} 样本 std/真 std ≥ 0.95 且均值无偏".format(std, DEFAULT_NFE),
                     ratio >= 0.95 and abs(bias) <= 0.01, "{:.3f}（NFE10 {:.3f}），均值偏差 {:+.4f}".format(ratio, float(coarse.std()) / std, bias))


# ---------------------------------------------------------------- 5 metrics


def check_metrics(report, scratch):
    from utils.ntu2p_probabilistic_metrics import (
        LEAD_NONE,
        arm_amplitude,
        energy_score,
        envelope_at,
        lead_foot_brier,
        lead_foot_summary,
        min_ade_curve,
        select_review_cases,
        spread_skill,
        step_crps,
        step_w1,
        window_mpjpe,
    )
    from utils.ntu_smplx_2p_xyz import copy_last_xyz

    generator = torch.Generator().manual_seed(5)
    target = torch.randn(3, 50, 2, 55, 3, generator=generator)
    pred = target + 0.1 * torch.randn(3, 50, 2, 55, 3, generator=generator)
    mpjpe = window_mpjpe(pred, target).mean().item()
    same = pred.unsqueeze(0).expand(6, -1, -1, -1, -1, -1).contiguous()
    es_same, _ = energy_score(same, target)
    report.check("K 个相同样本 ES = mpjpe（≤1e-6）", abs(es_same - mpjpe) <= 1e-6, "{:.2e}".format(abs(es_same - mpjpe)))
    es_one, _ = energy_score(pred.unsqueeze(0), target)
    report.check("K=1 时 ES = mpjpe", abs(es_one - mpjpe) <= 1e-6)
    samples = torch.stack([pred, target + 0.2 * torch.randn(3, 50, 2, 55, 3, generator=generator)])
    curve = min_ade_curve(samples, target, ks=(1, 2))
    report.check("minADE@1 = 样本 0 的 mpjpe", torch.allclose(curve["min_ade@1"].double(), window_mpjpe(pred, target).double(), atol=1e-6))
    gt_cat = torch.tensor([[1, -1], [-1, 1], [1, 1]])
    mask = torch.ones(3, 2, dtype=torch.bool)
    right = lead_foot_summary(gt_cat.unsqueeze(0), gt_cat, mask)["brier3"]
    wrong = lead_foot_summary((-gt_cat).unsqueeze(0), gt_cat, mask)["brier3"]
    report.check("确定性先迈脚：正确 Brier=0、错误 Brier=2", right == 0.0 and abs(wrong - 2.0) < 1e-12, "{} / {}".format(right, wrong))
    steps_gt = torch.tensor([[0.0, 2.0], [3.0, 1.0], [2.0, 4.0]])
    steps_pred = torch.tensor([[1.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    crps = step_crps(steps_pred.unsqueeze(0), steps_gt, mask)
    report.check("确定性预测 CRPS = |误差|", abs(crps - (steps_pred - steps_gt).double().abs().mean().item()) < 1e-12, "{:.4f}".format(crps))
    report.check("相同分布 W1 = 0", step_w1(steps_gt.unsqueeze(0).expand(4, -1, -1), steps_gt, mask) == 0.0)
    xs, ys = [0.1, 0.2, 0.4], [1.0, 3.0, 2.0]
    exact = all(abs(envelope_at(xs, ys, x)[0] - y) < 1e-12 and envelope_at(xs, ys, x)[1] for x, y in zip(xs, ys))
    middle = envelope_at(xs, ys, 0.3)
    outside = envelope_at(xs, ys, 0.5)
    report.check("envelope_at 网格点精确、之间线性、网格外外推", exact and abs(middle[0] - 2.5) < 1e-12 and middle[1] and abs(outside[0] - 1.5) < 1e-12
                 and not outside[1], "{} {}".format(middle, outside))
    count, windows = 20, 4000
    mu = torch.randn(windows, 1, 2, 1, 3, generator=generator)
    shape = (windows, 1, 2, 55, 3)
    truth = torch.zeros(shape)
    truth[..., :1, :] = mu + torch.randn(windows, 1, 2, 1, 3, generator=generator)
    ensemble = torch.zeros((count,) + shape)
    ensemble[..., :1, :] = mu.unsqueeze(0) + torch.randn(count, windows, 1, 2, 1, 3, generator=generator)
    ssr = spread_skill(ensemble, truth, "root")
    report.check("校准高斯集合（K=20, N=4000）SSR ∈ 1±0.1", abs(ssr - 1.0) <= 0.1, "{:.3f}".format(ssr))

    data = _windows(SUBJVAL_MANIFEST, SUBJVAL_CACHE, 247)
    obs, gt, action = data["obs_xyz"], data["target_xyz"], data["action"]
    amp = arm_amplitude(copy_last_xyz(obs, 50), obs)
    report.check("copy-last 的 arm_amplitude = 0", torch.equal(amp, torch.zeros_like(amp)))
    self_brier = lead_foot_brier(gt.unsqueeze(0), gt, obs, torch.ones(obs.shape[0], 2, dtype=torch.bool))
    report.check("GT 对自身的先迈脚 Brier = 0", self_brier["n"] > 0 and self_brier["brier3"] == 0.0, str(dict(self_brier)))
    ids = [m["sample_id"] for m in data["meta"]]
    cases = select_review_cases(obs, gt, action, seed=0, sample_ids=ids)
    again = select_review_cases(obs.clone(), gt.clone(), action.clone(), seed=0, sample_ids=ids)
    categories = OrderedDict()
    for case in cases:
        categories[case["category"]] = categories.get(case["category"], 0) + 1
    report.check("select_review_cases 只依赖 GT（签名无预测输入；同 GT 两次结果相同），共 18 例", cases == again and len(cases) == 18,
                 json.dumps(categories, ensure_ascii=False))
    starting_events = sum(1 for case in cases if case["category"] == "starting")
    report.check("起步者选例 6 个且都有 GT 先迈脚事件", starting_events == 6 and LEAD_NONE == 0)


# ---------------------------------------------------------------- 6 folds


def check_folds(report, scratch):
    import argparse as _argparse

    from data_loaders.forecasting.ntu_2p_diffusion import load_ntu_2p_diffusion_manifest
    from scripts.build_ntu2p_crossfit_folds import build
    from scripts.build_ntu2p_subject_holdout_manifest import performer

    output = os.path.join(scratch, "folds_subjval")
    summary = build(_argparse.Namespace(source_manifest=SUBJVAL_MANIFEST, source_cache_dir=SUBJVAL_CACHE, output_dir=output, num_folds=3,
                                        protocol_tag="subjval", skip_cache=False))
    sizes = [summary["folds"][str(f)]["num_val_sequences"] for f in range(3)]
    report.check("subjval 折大小 = [569,572,568]", sizes == [569, 572, 568], str(sizes))
    report.check("P008 在第 0 折", 8 in summary["folds"]["0"]["performers"])
    sets = [set(summary["folds"][str(f)]["performers"]) for f in range(3)]
    source = load_ntu_2p_diffusion_manifest(SUBJVAL_MANIFEST)
    train_performers = {performer(item["sample_id"]) for item in source["splits"]["train"]}
    report.check("三折受试者两两不重叠、并集为 train", not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
                 and set().union(*sets) == train_performers)
    parent = torch.load(os.path.join(SUBJVAL_CACHE, "train_xyz_seq.pt"), map_location="cpu")
    position = {sid: i for i, sid in enumerate(parent["sample_ids"])}
    ok_hash, ok_cache = True, True
    for fold in range(3):
        manifest_path = summary["folds"][str(fold)]["manifest_path"]
        manifest = load_ntu_2p_diffusion_manifest(manifest_path)
        ok_hash = ok_hash and manifest["manifest_hash"] == summary["folds"][str(fold)]["manifest_hash"]
        for split in ("train", "val"):
            cache = NTU2PXYZSeqCache.load(summary["folds"][str(fold)]["cache_dir"], split, manifest_path=manifest_path)
            for index in (0, len(cache) // 2, len(cache) - 1):
                sid = cache.sample_ids[index]
                p = position[sid]
                expected = parent["xyz"][int(parent["offsets"][p]) : int(parent["offsets"][p]) + int(parent["lengths"][p])]
                ok_cache = ok_cache and torch.equal(cache.sequence(index), expected)
    report.check("折 manifest 通过 hash 校验", ok_hash)
    report.check("切片缓存可加载且 xyz 与父缓存切片 torch.equal", ok_cache)
    original = build(_argparse.Namespace(source_manifest=ORIGINAL_MANIFEST, source_cache_dir=DEFAULT_CACHE_DIR,
                                         output_dir=os.path.join(scratch, "folds_original"), num_folds=3, protocol_tag="original", skip_cache=True))
    sizes = [original["folds"][str(f)]["num_val_sequences"] for f in range(3)]
    report.check("原协议折大小 = [587,586,585]", sizes == [587, 586, 585], str(sizes))


# ---------------------------------------------------------------- 7 bank_smoke


def _bank_args(scratch, output, skip):
    import argparse as _argparse

    return _argparse.Namespace(
        parent_manifest=SUBJVAL_MANIFEST, parent_cache_dir=SUBJVAL_CACHE, folds_dir=os.path.join(scratch, "folds_subjval"),
        fold_checkpoints=[ORIGINAL_V2.format(0)] * 3, insample_checkpoint=None, output=output, protocol="subjval", batch_size=64,
        glitch_threshold=0.25, device="cpu", allow_cpu_for_smoke_test=True, max_sequences=6, skip_oof_assert_for_smoke=skip)


def check_bank_smoke(report, scratch):
    from data_loaders.forecasting.ntu2p_residual_bank import NTU2PResidualBank
    from data_loaders.forecasting.ntu_2p_diffusion import load_ntu_2p_diffusion_manifest
    from scripts.build_ntu2p_oof_residual_bank import build

    output = os.path.join(scratch, "smoke_bank.pt")
    command = [PYTHON, "scripts/build_ntu2p_oof_residual_bank.py", "--parent_manifest", SUBJVAL_MANIFEST, "--parent_cache_dir", SUBJVAL_CACHE,
               "--folds_dir", os.path.join(scratch, "folds_subjval"), "--fold_checkpoints"] + [ORIGINAL_V2.format(0)] * 3 + [
               "--output", output, "--device", "cpu", "--allow_cpu_for_smoke_test", "--skip_oof_assert_for_smoke", "--max_sequences", "6",
               "--batch_size", "64"]
    code, seconds = _run(command, os.path.join(scratch, "bank_smoke.log"))
    report.check("CLI 冒烟（6 条序列）返回 0", code == 0, "{:.1f}s".format(seconds))
    bank = NTU2PResidualBank.load(output)
    entries = load_ntu_2p_diffusion_manifest(SUBJVAL_MANIFEST)["splits"]["train"][:6]
    expected = sum(int(item["length"]) - 59 for item in entries)
    report.check("窗口数 = Σ(len−59)", len(bank) == expected, "{} vs {}".format(len(bank), expected))
    shapes = (tuple(bank.target.shape[1:]) == (2, 12, 66) and tuple(bank.draft.shape[1:]) == (2, 12, 66)
              and tuple(bank.obs_feats.shape[1:]) == (2, 10, 66) and tuple(bank.rel_geom.shape[1:]) == (2, 6))
    report.check("字段形状正确", shapes)
    stats_ok = all(bool(torch.isfinite(v).all()) and bool((v > 0).all()) for v in bank.stats.values())
    report.check("统计量有限且为正", stats_ok, "clean={}/{}".format(int(bank.clean.sum()), len(bank)))
    try:
        build(_bank_args(scratch, os.path.join(scratch, "should_not_exist.pt"), skip=False))
        raised = False
    except AssertionError as error:
        raised = "不是该折的 OOF 模型" in str(error)
    report.check("不跳过断言时，冒充的折 checkpoint 触发 OOF 断言", raised)
    try:
        args = _bank_args(scratch, os.path.join(scratch, "should_not_exist.pt"), skip=True)
        args.allow_cpu_for_smoke_test = False
        build(args)
        refused = False
    except ValueError:
        refused = True
    report.check("--skip_oof_assert_for_smoke 不带 --allow_cpu_for_smoke_test 被拒绝", refused)


# ---------------------------------------------------------------- 8 train_smoke


def check_train_smoke(report, scratch):
    from model.forecasting_ntu2p_resdiff import load_ntu2p_resdiff_checkpoint
    from train.train_ntu2p_resdiff import build_arg_parser, run

    bank = os.path.join(scratch, "smoke_bank.pt")
    save_dir = os.path.join(scratch, "gen_smoke")
    shutil.rmtree(save_dir, ignore_errors=True)
    args = build_arg_parser().parse_args(["--bank", bank, "--save_dir", save_dir, "--num_steps", "20", "--batch_size", "8", "--device", "cpu",
                                          "--allow_cpu_for_smoke_test", "--save_interval", "10", "--log_interval", "5", "--warmup_steps", "1",
                                          "--lr", "1e-3", "--seed", "0"])
    torch.set_num_threads(2)
    output = run(args)
    records = [json.loads(line) for line in open(os.path.join(save_dir, "train_log.jsonl"))]
    report.check("20 step 训练 loss 有限", all(math.isfinite(r["loss"]) for r in records), "末步 loss {:.4f}".format(records[-1]["loss"]))
    raw, ema_path = os.path.join(save_dir, "model000000020.pt"), os.path.join(save_dir, "ema", "model000000020.pt")
    report.check("生成原始与 EMA checkpoint", os.path.exists(raw) and os.path.exists(ema_path))
    loaded, state = load_ntu2p_resdiff_checkpoint(ema_path, "cpu")
    generator = torch.Generator().manual_seed(6)
    y = _random_condition(4, generator, torch.tensor([False, True, False, True]))
    x = torch.randn(4, 2, 12, 66, generator=generator)
    t = torch.randint(0, 1000, (4,), generator=generator)
    memory = output["ema"]
    memory.eval()
    with torch.no_grad():
        same = torch.equal(loaded(x, t, y), memory(x, t, y))
    report.check("加载的 EMA 与内存中的 EMA 模型输出 torch.equal", same, "step {} arm {}".format(state["step"], state["arm"]))
    with torch.no_grad():
        nonzero = loaded(x, t, y).abs().max().item()
    report.check("训练后输出不再恒为 0（样本可区分的前提）", nonzero > 0, "{:.3e}".format(nonzero))
    # 应急臂开关：各跑 2 步，确认在线重算底座、解码投影与 *_share 日志可用（默认关闭，不改变主臂）。
    arm_dir = os.path.join(scratch, "gen_smoke_arms")
    shutil.rmtree(arm_dir, ignore_errors=True)
    arm_args = build_arg_parser().parse_args(["--bank", bank, "--save_dir", arm_dir, "--num_steps", "2", "--batch_size", "4", "--device", "cpu",
                                              "--allow_cpu_for_smoke_test", "--save_interval", "2", "--log_interval", "1", "--foot_loss_weight", "0.05",
                                              "--foot_loss_max_t", "1000", "--inter_loss_weight", "0.1", "--parent_cache_dir", SUBJVAL_CACHE,
                                              "--arm", "foot"])
    run(arm_args)
    records = [json.loads(line) for line in open(os.path.join(arm_dir, "train_log.jsonl"))]
    shares = all("foot_share" in r and "inter_share" in r and math.isfinite(r["loss"]) for r in records)
    report.check("应急臂 foot+inter 2 步：loss 有限且记录 *_share", shares,
                 "foot_share {:.3f} inter_share {:.3f}".format(records[-1].get("foot_share", float("nan")), records[-1].get("inter_share", float("nan"))))


# ---------------------------------------------------------------- 9 eval_smoke


REQUIRED_TOP = ("meta", "references", "generator", "bootstrap", "nfe_scan", "diagnostics", "subsets", "review")
REQUIRED_GENERATOR = (
    "single_l2", "mean_of_k_l2", "one_step_mean_l2", "min_ade", "min_fde@10", "person_min_ade_body22@10", "es_joint", "es_joint_clean",
    "es_legs_walk", "es_arms_act", "apd", "allocation_ratio", "ssr_clean", "lead_foot", "steps", "arm_amplitude", "interpenetration_ratio",
    "articulation_single", "naturalness_single", "naturalness_per_slot", "first_step_error_max", "bone_rel_err_body_max", "pelvis_equal_v2", "ci",
)
REQUIRED_BOOTSTRAP = ("single_l2", "es_joint", "min_ade@10", "es_legs_walk", "es_arms_act", "lead_foot", "strata_levels")
REQUIRED_META = ("checkpoint", "base_checkpoint", "bank", "split", "num_windows", "K", "noise_seed", "nfe", "settings", "device")


def _eval_command(scratch, output, review_flag, export_dir):
    # 用原始权重而非 EMA：20 步的 EMA（decay 0.999）几乎仍是零初始化，样本差异太小，检验不出"K 个样本可区分"。
    return [PYTHON, "eval/eval_ntu2p_resdiff.py", "--checkpoint", os.path.join(scratch, "gen_smoke", "model000000020.pt"),
            "--base_checkpoint", _subjval_base(0), "--bank", os.path.join(scratch, "smoke_bank.pt"), "--manifest_path", SUBJVAL_MANIFEST,
            "--cache_dir", SUBJVAL_CACHE, "--split", "val", "--baseline_checkpoint", SUBJVAL_BASELINE, "--a0_checkpoint", ORIGINAL_A0,
            "--ensemble_checkpoints", _subjval_base(0), _subjval_base(1), "--num_samples", "4", "--nfe", "2", "--nfe_scan", "3",
            "--naturalness_samples", "1", "--max_samples", "8", "--batch_windows", "4", "--diagnostics", "--export_review", export_dir,
            "--output", output, "--device", "cpu", "--allow_cpu_for_smoke_test"] + review_flag


def _strip(value):
    value = dict(value)
    value.pop("review", None)
    return value


def check_eval_smoke(report, scratch):
    from sample.render_ntu2p_review_sheet import _load_arrays

    cases = os.path.join(scratch, "smoke_review_cases.json")
    first = os.path.join(scratch, "eval_smoke_a.json")
    second = os.path.join(scratch, "eval_smoke_b.json")
    code, seconds = _run(_eval_command(scratch, first, ["--write_review_cases", cases], os.path.join(scratch, "review_a")),
                         os.path.join(scratch, "eval_smoke_a.log"))
    report.check("评估冒烟（subjval val 前 8 窗、K=4、NFE2）返回 0", code == 0, "{:.1f}s，底座 {}".format(seconds, _subjval_base(0)))
    if code != 0:
        return
    code_b, _ = _run(_eval_command(scratch, second, ["--review_cases", cases], os.path.join(scratch, "review_b")),
                     os.path.join(scratch, "eval_smoke_b.log"))
    a, b = json.load(open(first)), json.load(open(second))
    report.check("同样参数跑两次数值完全相同", code_b == 0 and json.dumps(_strip(a), sort_keys=True) == json.dumps(_strip(b), sort_keys=True))
    missing = [key for key in REQUIRED_TOP if key not in a]
    missing += ["meta." + key for key in REQUIRED_META if key not in a["meta"]]
    missing += ["references." + key for key in ("v2", "copy_last", "independent_base", "a0", "v2_ensemble") if key not in a["references"]]
    for mode in ("F", "R"):
        for tau, block in a["generator"][mode].items():
            missing += ["generator.{}.{}.{}".format(mode, tau, key) for key in REQUIRED_GENERATOR if key not in block]
    for mode in ("F", "R"):
        for scale, block in a["bootstrap"][mode].items():
            missing += ["bootstrap.{}.{}.{}".format(mode, scale, key) for key in REQUIRED_BOOTSTRAP if key not in block]
        report.check("bootstrap {} 含 s=1 自然度".format(mode), a["bootstrap"][mode]["1.0"].get("naturalness_single") is not None)
    report.check("JSON 含第 8 节全部键", not missing, ", ".join(missing[:8]))
    generator = a["generator"]["F"]["1.0"]
    report.check("K 个样本彼此不同（APD > 1 mm）", generator["apd"]["all"] > 1e-3, "{:.3e} m".format(generator["apd"]["all"]))
    structural = all(block["first_step_error_max"] <= 1e-6 and block["bone_rel_err_body_mean"] <= 1e-5 for per in a["generator"].values()
                     for block in per.values())
    report.check("首帧误差 ≤ 1e-6、骨长相对误差（均值）≤ 1e-5（全部设置）", structural,
                 "首帧最大 {:.2e}，骨长均值 {:.2e}".format(generator["first_step_error_max"], generator["bone_rel_err_body_mean"]))
    report.check("mode R pelvis 与 v2 逐位相同（全部 τ）", all(block["pelvis_equal_v2"] is True for block in a["generator"]["R"].values()))
    report.check("bootstrap s=0 即 v2（单样本 mpjpe 相同到 1e-5）",
                 abs(a["bootstrap"]["F"]["0.0"]["single_l2"]["mpjpe"] - a["references"]["v2"]["single_l2"]["mpjpe"]) <= 1e-5)
    files = a["review"]["files"]
    readable = True
    for key, path in files.items():
        if key == "review_samples":
            continue
        obs, target, methods, actions, meta = _load_arrays(path)
        readable = readable and obs.shape[0] == target.shape[0] == len(meta) and all(v.shape == target.shape for v in methods.values())
    report.check("导出数组可被 render 脚本 _load_arrays 读取", readable, ", ".join(sorted(files)))
    render_dir = os.path.join(scratch, "render_smoke")
    code, seconds = _run([PYTHON, "sample/render_ntu2p_review_sheet.py", "--arrays", files["review_arrays"], "--output_dir", render_dir,
                          "--indices", "0", "--num_random", "0", "--num_walk", "0"], os.path.join(scratch, "render_smoke.log"))
    report.check("审查图脚本渲染 1 例", code == 0, "{:.1f}s".format(seconds))
    fan_cases = os.path.join(scratch, "fan_cases.json")
    listed = json.load(open(cases))
    listed["cases"][0]["category"] = "starting"  # 前 8 窗未必有起步者；冒烟只验证作图流程。
    json.dump(listed, open(fan_cases, "w"))
    code, _ = _run([PYTHON, "sample/plot_ntu2p_gait_fan.py", "--samples", files["review_samples"], "--cases", fan_cases, "--output_dir",
                    os.path.join(scratch, "fan_smoke")], os.path.join(scratch, "fan_smoke.log"))
    report.check("起步者踝分离叠图脚本运行", code == 0)


# ---------------------------------------------------------------- 10 driver


def check_driver(report, scratch):
    import scripts.run_ntu2p_trackb as driver

    root = os.path.join(scratch, "driver_root")
    shutil.rmtree(root, ignore_errors=True)
    for protocol, needles in (("subjval", ("build_ntu2p_crossfit_folds", "run_ntu2p_v2_screen", "build_ntu2p_oof_residual_bank",
                                           "train_ntu2p_resdiff", "eval_ntu2p_resdiff", "render_ntu2p_review_sheet", "plot_ntu2p_gait_fan")),
                              ("original", ("build_ntu2p_crossfit_folds", "run_ntu2p_v2_screen", "build_ntu2p_oof_residual_bank",
                                            "train_ntu2p_resdiff"))):
        log = os.path.join(scratch, "driver_dry_{}.log".format(protocol))
        code, _ = _run([PYTHON, "scripts/run_ntu2p_trackb.py", "--protocol", protocol, "--stage", "all", "--dry_run", "--results_root", root,
                        "--arms", "main", "insample"] if protocol == "subjval" else
                       [PYTHON, "scripts/run_ntu2p_trackb.py", "--protocol", protocol, "--stage", "all", "--dry_run", "--results_root", root], log)
        text = open(log).read()
        found = [needle for needle in needles if needle in text]
        report.check("--stage all --dry_run 打印 {} 完整计划".format(protocol), code == 0 and len(found) == len(needles),
                     "缺 {}".format(sorted(set(needles) - set(found))) if len(found) != len(needles) else "")
    code, _ = _run([PYTHON, "scripts/run_ntu2p_trackb.py", "--protocol", "original", "--stage", "eval", "--test", "--results_root", root],
                   os.path.join(scratch, "driver_test_gate.log"))
    report.check("无 adopt 结论时 --test 返回非 0", code != 0, "code={}".format(code))
    code, _ = _run([PYTHON, "scripts/run_ntu2p_trackb.py", "--protocol", "subjval", "--stage", "eval", "--test", "--results_root", root],
                   os.path.join(scratch, "driver_test_subjval.log"))
    report.check("--test 用于 subjval 返回非 0", code != 0)
    original = driver.busy_gpu_processes
    driver.busy_gpu_processes = lambda: ["12345 /usr/bin/python train/train_ntu2p_v2.py --device cuda:0"]
    try:
        driver.main(["--protocol", "subjval", "--stage", "bank", "--results_root", root])
        code = 0
    except SystemExit as error:
        code = error.code
    finally:
        driver.busy_gpu_processes = original
    report.check("模拟 train_ntu2p_v2.py 在跑时 GPU 阶段返回非 0（返回码 3）", code == driver.GPU_BUSY_EXIT, "code={}".format(code))
    real = driver.busy_gpu_processes()
    report.check("真实 pgrep 解析可用（当前检测到 {} 个 train_ntu2p_v2.py 进程）".format(len(real)), isinstance(real, list))
    # 汇总冒烟：把评估冒烟 JSON 复制成 3 个 seed，跑完整的预登记判定流程（数值无意义，只验证流程不出错）。
    smoke = os.path.join(scratch, "eval_smoke_a.json")
    if not os.path.exists(smoke):
        report.check("summary 冒烟需要 eval_smoke 的产物", False, smoke)
        return
    folder = os.path.join(root, "subjval")
    os.makedirs(folder, exist_ok=True)
    data = json.load(open(smoke))
    for tag in ("a", "b"):
        shutil.copyfile(smoke, os.path.join(folder, "repro16_{}.json".format(tag)))
    # 第一次去掉诊断，走完 A–H 与 D/E'/F' 全部判据；第二次保留诊断，冒烟模型的 teacher-forced 误差 ≥1 应触发训练失败。
    for label, payload, expect in (("全部判据", dict(data, diagnostics={}), None), ("训练失败停止条件", data, "training_failure")):
        for seed in (0, 1, 2):
            json.dump(payload, open(os.path.join(folder, "eval_main_s{}_val_nfe{}.json".format(seed, driver.NFE_DEFAULT)), "w"))
        try:
            driver.main(["--protocol", "subjval", "--stage", "summary", "--results_root", root])
            code = 0
        except SystemExit as error:
            code = error.code
        summary = json.load(open(os.path.join(folder, "summary.json")))
        arm = summary["arms"]["main"]
        covered = expect is not None or all(key in arm.get("probabilistic", {}) for key in ("A1", "A4", "B", "C", "E6", "F5", "G", "H"))
        ok = code in (0, None) and covered and (expect is None or summary["conclusion"] == expect)
        report.check("summary 阶段跑通（{}）".format(label), ok, "conclusion={}（冒烟数据，数值无意义）".format(summary.get("conclusion")))
    check_driver_routing(report, driver)
    check_driver_adoption(report, scratch, driver)


def _checks(fails=(), display_fails=(), h=True):
    names = ("A1", "A2", "A3", "A4", "A5", "B", "C", "single_sanity", "E1", "E2", "E3", "E4", "E5", "E6", "F1", "F2", "F3", "F4", "F5", "G")
    probabilistic = OrderedDict((name, (name not in fails, "")) for name in names)
    probabilistic["H"] = (h, "")
    display_names = ("D", "E1'", "E2'", "E3'", "E4'", "E5'", "E6'", "E'_skate", "E'_min_dist", "F'")
    display = OrderedDict((name, (name not in display_fails, "")) for name in display_names)
    return probabilistic, display


def check_driver_routing(report, driver):
    """4.8 分流：E1/E5 不设应急臂；E2/E3/E4 只触发 foot，E6 只触发 inter；展示阻断时不为展示触发应急臂。"""
    cases = (
        ("只有 E5 失败", (), ("E5",), (), False, "reject", []),
        ("E2+E5 失败", (), ("E2", "E5"), (), False, "reject", []),
        ("E2 失败", (), ("E2",), (), False, "contingency_required", ["foot"]),
        ("E4 失败", (), ("E4",), (), False, "contingency_required", ["foot"]),
        ("E6 失败", (), ("E6",), (), False, "contingency_required", ["inter"]),
        ("E3+E6 失败", (), ("E3", "E6"), (), False, "contingency_required", ["foot", "inter"]),
        ("展示 E'_skate 失败", (), (), ("E'_skate",), False, "contingency_required", ["foot"]),
        ("展示 E'_min_dist 失败", (), (), ("E'_min_dist",), False, "contingency_required", ["inter"]),
        ("展示 E5'+E2' 失败（P 全过）", (), (), ("E5'", "E2'"), False, "adopt_probabilistic_only", []),
        ("A 未通过优先", (), ("A1", "E2"), (), False, "reject_no_better_than_bootstrap", []),
        ("应急臂上 P 的 E6 仍失败", (), ("E6",), (), True, "reject", None),
        ("应急臂上只剩展示 E2' 失败", (), (), ("E2'",), True, "adopt_probabilistic_only", None),
        ("全部通过", (), (), (), False, "adopt_full", []),
    )
    wrong = []
    for label, _, fails, display_fails, used, expect, arms in cases:
        probabilistic, display = _checks(fails, display_fails)
        conclusion, routing = driver.conclude(probabilistic, display, 1.0, used)
        if conclusion != expect or (arms is not None and routing["contingency_arms"] != arms):
            wrong.append("{}: {} {}".format(label, conclusion, routing["contingency_arms"]))
    report.check("E 失败按类别分流到应急臂（{} 例）".format(len(cases)), not wrong, "；".join(wrong))
    check_nfe_rule(report, driver)


def check_nfe_rule(report, driver):
    """NFE 规则只看离散度比：2N/N 的 APD_all 或 SSR 均值之比 > 1.03 就看下一档。"""
    low, mid, high = driver.NFE_GRID

    def result(apd, ssr):
        def block(nfe):
            return {"apd": {"all": apd[nfe]}, "ssr_clean": {part: ssr[nfe] for part in ("root", "legs", "arms")}, "es_joint": 1.0,
                    "single_l2": {"mpjpe": 0.2}}
        return {"generator": {"F": {"1.0": block(low)}}, "nfe_scan": {str(n): block(n) for n in (mid, high)}}

    flat = {low: 1.0, mid: 1.0, high: 1.0}
    cases = (
        ("两档都收敛", {low: 1.0, mid: 1.02, high: 1.03}, flat, low),
        ("50 未收敛、100 收敛", {low: 1.0, mid: 1.06, high: 1.08}, flat, mid),
        ("100 仍未收敛", {low: 1.0, mid: 1.06, high: 1.13}, flat, high),
        ("只有 SSR 比超阈", flat, {low: 1.0, mid: 1.05, high: 1.06}, mid),
    )
    wrong = []
    for label, apd, ssr, expect in cases:
        nfe, reason, _ = driver.nfe_rule(OrderedDict((seed, result(apd, ssr)) for seed in (0, 1, 2)))
        if nfe != expect:
            wrong.append("{}: {}（{}）".format(label, nfe, reason))
    report.check("NFE 规则按离散度比选档（{} 例）".format(len(cases)), not wrong, "；".join(wrong))


def _driver_main(driver, argv):
    """运行驱动并捕获输出；返回 (退出码, 输出文本)。"""
    buffer = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buffer):
        try:
            driver.main(argv)
        except SystemExit as error:
            code = error.code
    return code, buffer.getvalue()


def check_driver_adoption(report, scratch, driver):
    """应急臂采纳路径：主臂触发 foot，foot 采纳（τ 与 NFE 不同于主臂）；桩判定，不跑评估。"""
    root = os.path.join(scratch, "driver_contingency")
    shutil.rmtree(root, ignore_errors=True)
    folder = os.path.join(root, "subjval")
    os.makedirs(folder)
    for arm in ("main", "foot", "inter"):
        for seed in (0, 1, 2):
            json.dump({"arm": arm}, open(os.path.join(folder, "eval_{}_s{}_val_nfe{}.json".format(arm, seed, driver.NFE_DEFAULT)), "w"))
    verdict_path = os.path.join(folder, "review_verdict.json")
    main_selected = OrderedDict([("nfe", driver.NFE_DEFAULT), ("tau_P", 1.0), ("tau_D", 1.0)])
    foot_selected = OrderedDict([("nfe", driver.NFE_GRID[1]), ("tau_P", 0.8), ("tau_D", 0.6)])
    json.dump(dict(main_selected, arm="main", **{"pass": True, "notes": "看的是 main 臂"}), open(verdict_path, "w"))

    def fake_decide(evals_by_nfe, repro, verdict, arm):
        structural = OrderedDict([("pass", True), ("reasons", [])])
        if arm == "main":
            return OrderedDict([("selected", main_selected), ("structural", structural), ("conclusion", "contingency_required"),
                                ("contingency_arms", ["foot"])])
        h_pass, h_detail = driver.verdict_h(verdict, arm, foot_selected)
        conclusion = "pending_review" if h_pass is None else ("adopt_full" if h_pass else "reject")
        return OrderedDict([("selected", foot_selected), ("structural", structural), ("conclusion", conclusion), ("h_detail", h_detail)])

    def refuse(*args, **kwargs):
        raise AssertionError("判定路径测试不应执行任何命令")

    real_decide, real_run = driver.decide_arm, driver.run_command
    driver.decide_arm, driver.run_command = fake_decide, refuse
    summary_path = os.path.join(folder, "summary.json")
    try:
        code, _ = _driver_main(driver, ["--protocol", "subjval", "--stage", "summary", "--results_root", root])
        summary = json.load(open(summary_path))
        report.check("绑定 main 的审查结论不用于 foot 的 H（结论 pending_review，审查 foot）",
                     code in (0, None) and summary["conclusion"] == "pending_review" and summary["decision_arm"] == "foot"
                     and summary["adopted_arm"] is None, "{} / {}".format(summary["conclusion"], summary["arms"]["foot"].get("h_detail", "")[:60]))
        json.dump(dict(foot_selected, arm="foot", **{"pass": True, "notes": "foot 审查图"}), open(verdict_path, "w"))
        code, _ = _driver_main(driver, ["--protocol", "subjval", "--stage", "summary", "--results_root", root])
        summary = json.load(open(summary_path))
        frozen = driver.frozen_settings(summary)
        report.check("foot 采纳：adopted_arm=foot，selected 与冻结设置取自 foot",
                     summary["conclusion"] == "adopt_full" and summary["adopted_arm"] == "foot" and frozen == ("F:0.8,R:0.6", driver.NFE_GRID[1]),
                     "{} {} {}".format(summary["conclusion"], summary["adopted_arm"], frozen))
        code, _ = _driver_main(driver, ["--protocol", "subjval", "--stage", "summary", "--arms", "main", "inter", "--results_root", root])
        report.check("summary 拒绝未被触发的应急臂 inter", code == 2, "code={}".format(code))
        opts = argparse.Namespace(protocol="subjval", arms=["inter"], dry_run=False)
        try:
            driver.check_contingency_triggered(opts, driver.Paths(argparse.Namespace(protocol="subjval", results_root=root)))
            code = 0
        except SystemExit as error:
            code = error.code
        report.check("train/eval 拒绝未被触发的应急臂 inter", code == 2, "code={}".format(code))
        code, _ = _driver_main(driver, ["--protocol", "original", "--stage", "train", "--arms", "main", "--dry_run", "--results_root", root])
        report.check("原协议拒绝复训未被采纳的 main", code == 2, "code={}".format(code))
        driver.run_command = real_run
        code, text = _driver_main(driver, ["--protocol", "original", "--stage", "train", "--dry_run", "--results_root", root])
        report.check("原协议 --arms 缺省即采纳的 foot（含 foot 损失权重）",
                     code in (0, None) and "--arm foot" in text and "--foot_loss_weight" in text and "--arm main" not in text)
        code, text = _driver_main(driver, ["--protocol", "original", "--stage", "eval", "--test", "--dry_run", "--results_root", root])
        report.check("原协议 test 只评估 foot、用冻结的 NFE，且不算 v2 3 seed 均值",
                     code in (0, None) and "eval_foot_s0_test_nfe{}".format(driver.NFE_GRID[1]) in text and "--ensemble_checkpoints" not in text
                     and "_main_" not in text, "code={}".format(code))
        driver.run_command = refuse
        code, _ = _driver_main(driver, ["--protocol", "subjval", "--stage", "review", "--arms", "main", "foot", "--allow_shared_gpu",
                                        "--results_root", root])
        report.check("review 一次只接受一个臂", code == 2, "code={}".format(code))
        stopped = json.load(open(summary_path))
        stopped["arms"]["main"] = OrderedDict([("conclusion", "pending_nfe_rerun"),
                                               ("selected", OrderedDict([("nfe", driver.NFE_GRID[1]), ("tau_P", None), ("tau_D", None)]))])
        stopped["decision_arm"] = "main"
        json.dump(stopped, open(summary_path, "w"))
        code, text = _driver_main(driver, ["--protocol", "subjval", "--stage", "review", "--allow_shared_gpu", "--results_root", root])
        report.check("停止状态（τ_P 未定）时 review 跳过而不是崩溃", code in (0, None) and "跳过审查" in text, "code={}".format(code))
        stale = dict(stopped, conclusion="adopt_full", decision_arm="foot", adopted_arm="foot", base_config="A6-F-A5f0.05-A4-GH0.5")
        json.dump(stale, open(summary_path, "w"))
        code, text = _driver_main(driver, ["--protocol", "original", "--stage", "train", "--results_root", root])
        report.check("subjval 选型底座与当前主线不一致时原协议拒绝", code == 2 and "不一致" in text, "code={}".format(code))
    finally:
        driver.decide_arm, driver.run_command = real_decide, real_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch_dir", required=True)
    parser.add_argument("--only", default=None, help="逗号分隔的节名")
    args = parser.parse_args()
    torch.set_num_threads(2)
    os.makedirs(args.scratch_dir, exist_ok=True)
    sections = [s.strip() for s in args.only.split(",")] if args.only else list(SECTIONS)
    unknown = [s for s in sections if s not in SECTIONS]
    if unknown:
        raise ValueError("未知节 {}".format(unknown))
    report = Report()
    for name in sections:
        report.start(name)
        start = time.time()
        try:
            globals()["check_" + name](report, args.scratch_dir)
        except Exception as error:  # 单节异常记为失败，其余节照常跑。
            import traceback

            traceback.print_exc()
            report.check("节 {} 未抛异常".format(name), False, repr(error))
        print("  ({:.1f}s)".format(time.time() - start), flush=True)
    failed = report.failed()
    total = sum(len(items) for items in report.results.values())
    print("\n{} / {} 项通过".format(total - len(failed), total))
    for section, label, detail in failed:
        print("FAIL [{}] {} {}".format(section, label, detail))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
