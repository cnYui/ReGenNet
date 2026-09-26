"""Track B 评估：预登记的全部指标（单样本 / mean-of-K / 单步均值 / best-of-K / ES / 多样性 / 校准 / 步态 / 自然度），
确定性参照与平凡 bootstrap 基线，以及审查图数组导出。

设计：docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md 第 5 节。
- 噪声 [N,K,2,12,66] 由 Generator(noise_seed) 在 CPU 上按 manifest 顺序一次性生成，bootstrap 的均匀数 [N,K] 用
  Generator(noise_seed+1)；所有 (mode, τ)、NFE、bootstrap 变体共用同一份，--max_samples 只取前缀，保证配对比较。
- 按窗口块流式生成（每块 B×K 个样本一次采样、解码、投影），逐窗/逐样本累计指标；只保留前 S 个样本槽（单样本自然度）
  与审查窗口的全部 K 个样本，test 1253 条也不需要把 K 份未来整体放进内存。
- 全部指标在相机系上计算；bootstrap = 部署底座 + s × 从 OOF 干净库按 (动作, v2_walk_A, v2_walk_B) 分层抽取的整场景
  target 系数，解码与投影与生成器完全相同。
- best-of-K（minADE/minFDE）是用 GT 挑样本的上界，不是预测；汇报时不得与 v2 的单样本数字并列写成"超过 v2"。
- 主采样 NFE=50，扫描 10/20/100/200：每档都是完整的指标块（含 APD、SSR），NFE 规则按离散度比判定，见
  docs/ai/context/20260926-172000-ntu2p-trackb-review-fixes-and-nfe-preregistration-amendment.md。
"""

import argparse
import hashlib
import json
import os
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu2p_residual_bank import STRATA_LEVELS, NTU2PResidualBank
from data_loaders.forecasting.ntu2p_xyz_seq_cache import DEFAULT_CACHE_DIR, NTU2PXYZSeqCache, eval_windows
from eval.eval_ntu2p_residual_refiner_xyz import _add, _finalize
from eval.eval_ntu2p_v2 import DEFAULT_BASELINE, DEFAULT_MANIFEST, _write_json, naturalness_block
from model.forecasting_ntu2p_resdiff import (
    DEFAULT_NFE,
    build_sampling_diffusion,
    build_training_diffusion,
    load_ntu2p_resdiff_checkpoint,
    make_condition,
    one_step_mean,
    repeat_condition,
    sample_x0,
    to_meters,
)
from model.forecasting_ntu2p_residual_xyz import load_base_model_from_checkpoint
from model.forecasting_ntu2p_v2 import forward_details, independent_base_forward, load_ntu2p_model_checkpoint
from train.train_ntu2p_v2 import _device
from utils.ntu2p_kinematic_projection import SkeletonProjector
from utils.ntu2p_naturalness import bone_lengths, estimate_scene_frame
from utils.ntu2p_probabilistic_metrics import (
    APD_PARTS,
    ARM_JOINTS,
    GROUP_KEYS,
    LEG_JOINTS,
    SSR_PARTS,
    allocation_ratio_from,
    apd_per_person,
    arm_amplitude,
    arm_bias_from,
    energy_score_parts,
    final_frame_error,
    interpenetration_ratio,
    lead_foot_category,
    lead_foot_summary,
    min_ade_from_per_sample,
    person_ade_body22,
    person_groups,
    person_step_counts,
    select_review_cases,
    spread_skill_parts,
    ssr_from_parts,
    step_crps,
    step_w1,
    window_bootstrap_ci,
    window_mpjpe,
)
from utils.ntu2p_residual_codec import ResidualCodec, codec_frames, condition_features, glitch_mask, repeat_frames
from utils.ntu_smplx_2p_xyz import articulation_ratios, compute_ntu_articulation_metrics, compute_ntu_xyz_metrics, copy_last_xyz


DEFAULT_SETTINGS = "F:1.0,F:0.8,R:1.0,R:0.8,R:0.6,R:0.4"
DEFAULT_BOOT_SCALES = "0,0.25,0.5,0.75,1.0,1.25,1.5"
# 100/200 供 NFE 规则判定离散度是否已收敛；10/20 只报告收缩曲线。
DEFAULT_NFE_SCAN = "10,20,100,200"
MIN_ADE_KS = (1, 5, 10, 20)
TEACHER_FORCED_TS = (50, 250, 500, 750, 999)
ES_KEYS = ("es_joint", "es_joint_clean", "es_legs_walk", "es_arms_act")
L2_KEYS = ("xyz_mse", "xyz_mae", "mpjpe")
ARTIC_RATIO_KEYS = (
    "dct_low_energy_ratio_to_target",
    "dct_mid_energy_ratio_to_target",
    "dct_high_energy_ratio_to_target",
    "articulation_energy_ratio_to_target",
    "local_pose_temporal_std_ratio_to_target",
)
BOOTSTRAP_NATURAL_SCALE = 1.0
CI_SAMPLES = 1000


def parse_settings(text):
    settings = []
    for token in [part.strip() for part in str(text).split(",") if part.strip()]:
        mode, tau = token.split(":")
        if mode not in ("F", "R"):
            raise ValueError("setting 的 mode 必须是 F 或 R：{}".format(token))
        settings.append((mode, float(tau)))
    return settings


def setting_key(mode, tau):
    return "{}:{}".format(mode, float(tau))


def _floats(text):
    return [float(part) for part in str(text).split(",") if part.strip()]


def _ints(text):
    return [int(part) for part in str(text).split(",") if part.strip()]


class Windows(object):
    """评估窗口与所有确定性量（常驻 device）：obs/target/action、部署底座 P、条件特征、分组、GT 侧统计。"""

    def __init__(self, args, device, deploy, codec):
        cache = NTU2PXYZSeqCache.load(args.cache_dir, args.split, manifest_path=args.manifest_path, device="cpu")
        windows = eval_windows(cache)
        total = int(windows["obs_xyz"].shape[0])
        self.num_total = total
        count = total if args.max_samples is None or int(args.max_samples) <= 0 else min(int(args.max_samples), total)
        self.count = count
        self.meta = windows["meta"][:count]
        self.obs = windows["obs_xyz"][:count].to(device)
        self.target = windows["target_xyz"][:count].to(device)
        self.action = windows["action"][:count].to(device)
        self.batch = int(args.batch_windows)
        self.device = device
        parts = OrderedDict((key, []) for key in ("base", "draft", "obs_feats", "rel_geom", "v2_walk", "target_coeffs"))
        with torch.no_grad():
            for begin, end in self.chunks():
                obs, target = self.obs[begin:end], self.target[begin:end]
                base = forward_details(deploy, obs, self.action[begin:end])["pred"]
                frames = codec_frames(obs)
                feats = condition_features(codec, obs, base, frames)
                parts["base"].append(base)
                for key in ("draft", "obs_feats", "rel_geom", "v2_walk"):
                    parts[key].append(feats[key])
                parts["target_coeffs"].append(codec.target_coeffs(obs, base, target, frames))
        for key, value in parts.items():
            setattr(self, key, torch.cat(value, dim=0))
        self.groups = person_groups(self.obs, self.target, self.action)
        self.clean = ~glitch_mask(torch.cat((self.obs, self.target), dim=1))
        self.scene_frames = []
        gt_lead, gt_steps = [], []
        for begin, end in self.chunks():
            obs, target = self.obs[begin:end], self.target[begin:end]
            frame = estimate_scene_frame(torch.cat((obs.double(), target.double()), dim=1))
            self.scene_frames.append(frame)
            gt_lead.append(lead_foot_category(target, obs, target))
            gt_steps.append(person_step_counts(target, target, obs, frame))
        self.gt_lead = torch.cat(gt_lead, dim=0)
        self.gt_steps = torch.cat(gt_steps, dim=0)
        self.gt_arm = arm_amplitude(self.target.double(), self.obs.double())

    def chunks(self):
        for begin in range(0, self.count, self.batch):
            yield begin, min(begin + self.batch, self.count)

    def feats(self, begin, end):
        return OrderedDict((key, getattr(self, key)[begin:end]) for key in ("draft", "obs_feats", "rel_geom"))

    def subsets(self):
        result = OrderedDict([("num_windows", int(self.count)), ("num_clean_windows", int(self.clean.sum().item()))])
        for key in GROUP_KEYS:
            result["num_" + key + "_persons"] = int(self.groups[key].sum().item())
        valid = self.groups["starting"] & (self.gt_lead != 0)
        result["num_starting_with_lead_event"] = int(valid.sum().item())
        return result


class SampleAccumulator(object):
    """一个变体（生成器设置 / bootstrap 尺度 / 参照）的 K 个样本在全部窗口上的逐窗统计。"""

    def __init__(self, num_samples, slots, keep_slots, review_index=None, check_pelvis=False):
        self.num_samples = int(num_samples)
        self.slots = min(int(slots), self.num_samples)
        self.keep_slots = bool(keep_slots)
        self.review_index = OrderedDict((int(i), None) for i in (review_index or []))
        self.check_pelvis = bool(check_pelvis)
        self.single, self.mean_of_k, self.one_step = OrderedDict(), OrderedDict(), OrderedDict()
        self.slot_artic = [OrderedDict() for _ in range(self.slots)]
        self.lists = OrderedDict()
        self.count = 0
        self.first_step_max = 0.0
        self.bone_max = 0.0
        self.bone_sum = 0.0
        self.bone_count = 0
        self.pelvis_equal = True
        self.finite = True
        self.digest = hashlib.sha256()
        self.kept = []
        self.levels = []

    def _push(self, key, value):
        self.lists.setdefault(key, []).append(value.detach().cpu())

    def _cat(self, key, dim=0):
        return torch.cat(self.lists[key], dim=dim)

    def update(self, samples, windows, begin, end, one_step=None, levels=None):
        """samples [K,B,T,2,55,3]（device）；one_step [B,...] 为单步条件均值的解码结果。"""
        obs, target = windows.obs[begin:end], windows.target[begin:end]
        size = end - begin
        groups = OrderedDict((key, value[begin:end]) for key, value in windows.groups.items())
        clean = windows.clean[begin:end]
        self.finite = self.finite and bool(torch.isfinite(samples).all())
        self.digest.update(samples.detach().float().cpu().contiguous().numpy().tobytes())
        for sample in samples:
            _add(self.single, compute_ntu_xyz_metrics(sample, target, obs), size)
        _add(self.mean_of_k, compute_ntu_xyz_metrics(samples.mean(dim=0), target, obs), size)
        if one_step is not None:
            _add(self.one_step, compute_ntu_xyz_metrics(one_step, target, obs), size)
        for slot in range(self.slots):
            _add(self.slot_artic[slot], compute_ntu_articulation_metrics(samples[slot], target), size)
        self._push("per_sample_mpjpe", window_mpjpe(samples, target).double())
        self._push("fde", final_frame_error(samples, target).double())
        self._push("person_ade22", person_ade_body22(samples, target).double())
        es_specs = OrderedDict(
            [
                ("es_joint", dict()),
                ("es_joint_clean", dict(window_mask=clean)),
                ("es_legs_walk", dict(joints=LEG_JOINTS, local=True, person_mask=groups["walking"])),
                ("es_arms_act", dict(joints=ARM_JOINTS, local=True, person_mask=groups["arm_action"])),
            ]
        )
        for key, spec in es_specs.items():
            num, den = energy_score_parts(samples, target, **spec)
            self._push(key + "_num", num)
            self._push(key + "_den", den)
        for part in APD_PARTS:
            self._push("apd_" + part, apd_per_person(samples, part))
        if self.num_samples >= 2:
            for part in SSR_PARTS:
                var, err, elements = spread_skill_parts(samples, target, part)
                self._push("ssr_var_" + part, var)
                self._push("ssr_err_" + part, err)
                self._push("ssr_n_" + part, elements)
        self._push("lead", lead_foot_category(samples, obs, target))
        frame = windows.scene_frames[begin // windows.batch]
        self._push("steps", person_step_counts(samples, target, obs, frame))
        self._push("arm", arm_amplitude(samples.double(), obs.double()))
        self._push("interpen", interpenetration_ratio(samples, target))
        first = (samples[:, :, 0] - obs[:, -1].unsqueeze(0)).norm(dim=-1).max()
        self.first_step_max = max(self.first_step_max, float(first.item()))
        reference = bone_lengths(obs[:, -1:].double()).clamp_min(1e-4)
        relative = (bone_lengths(samples.double()) - reference).abs() / reference
        body = relative[..., :21]
        self.bone_max = max(self.bone_max, float(body.max().item()))
        self.bone_sum += float(body.sum().item())
        self.bone_count += int(body.numel())
        if self.check_pelvis:
            base = windows.base[begin:end].unsqueeze(0).expand_as(samples)
            self.pelvis_equal = self.pelvis_equal and torch.equal(samples[..., 0, :], base[..., 0, :])
        if self.keep_slots and self.slots > 0:
            self.kept.append(samples[: self.slots].detach().cpu())
        for index in self.review_index:
            if begin <= index < end:
                self.review_index[index] = samples[:, index - begin].detach().cpu()
        if levels is not None:
            self.levels.append(levels.cpu())
        self.count += size

    def finalize(self, windows, natural_rows=None):
        count, samples = self.count, self.num_samples
        result = OrderedDict()
        result["single_l2"] = OrderedDict((key, value / float(samples)) for key, value in _finalize(self.single, count).items())
        result["mean_of_k_l2"] = _finalize(self.mean_of_k, count)
        result["one_step_mean_l2"] = _finalize(self.one_step, count) if self.one_step else None
        per_sample = self._cat("per_sample_mpjpe", dim=1)  # [K,N]
        # 另加 k=K：K=3 的 v2 多 seed 样本集要报 minADE@3。
        curves = min_ade_from_per_sample(per_sample, sorted(set(MIN_ADE_KS) | {samples}))
        result["min_ade"] = OrderedDict((key, float(value.mean().item())) for key, value in curves.items())
        result["min_ade@10"] = result["min_ade"].get("min_ade@10")
        fde = self._cat("fde", dim=1)
        result["min_fde@10"] = float(fde[:10].min(dim=0).values.mean().item())
        result["person_min_ade_body22@10"] = float(self._cat("person_ade22", dim=1)[:10].min(dim=0).values.mean().item())
        es = OrderedDict()
        for key in ES_KEYS:
            num, den = self._cat(key + "_num"), self._cat(key + "_den")
            total = float(den.sum().item())
            es[key] = (num, den)
            result[key] = float(num.sum().item()) / total if total > 0 else float("nan")
        result["apd"] = OrderedDict((part, float(self._cat("apd_" + part).mean().item())) for part in APD_PARTS)
        groups = OrderedDict((key, value.cpu()) for key, value in windows.groups.items())
        result["allocation_ratio"] = allocation_ratio_from(self._cat("apd_legs_local"), groups) if samples >= 2 else float("nan")
        clean = windows.clean.cpu()
        if samples >= 2:
            result["ssr_clean"] = OrderedDict(
                (part, ssr_from_parts(self._cat("ssr_var_" + part), self._cat("ssr_err_" + part), self._cat("ssr_n_" + part), samples, clean))
                for part in SSR_PARTS
            )
        else:
            result["ssr_clean"] = OrderedDict((part, float("nan")) for part in SSR_PARTS)
        lead = self._cat("lead", dim=1)
        result["lead_foot"] = lead_foot_summary(lead, windows.gt_lead.cpu(), groups["starting"])
        steps = self._cat("steps", dim=1)
        gt_steps = windows.gt_steps.cpu()
        walking, static = groups["walking"], groups["static"]
        result["steps"] = OrderedDict(
            [
                ("crps", step_crps(steps, gt_steps, walking)),
                ("w1", step_w1(steps, gt_steps, walking)),
                ("mean", float(steps[:, walking].mean().item()) if bool(walking.any()) else float("nan")),
                ("gt_mean", float(gt_steps[walking].mean().item()) if bool(walking.any()) else float("nan")),
                ("static_mean", float(steps[:, static].mean().item()) if bool(static.any()) else float("nan")),
                ("gt_static_mean", float(gt_steps[static].mean().item()) if bool(static.any()) else float("nan")),
            ]
        )
        amplitude = self._cat("arm", dim=1)
        gt_arm = windows.gt_arm.cpu()
        strike_bias, strike_median = arm_bias_from(amplitude, gt_arm, groups["strike"])
        arm_bias, arm_median = arm_bias_from(amplitude, gt_arm, groups["arm_action"])
        result["arm_amplitude"] = OrderedDict(
            [
                ("strike_abs_log_bias", strike_bias),
                ("strike_median_ratio", strike_median),
                ("arm_abs_log_bias", arm_bias),
                ("arm_median_ratio", arm_median),
            ]
        )
        interpen = self._cat("interpen", dim=1)
        slots = max(self.slots, 1)
        result["interpenetration_ratio"] = float(interpen[:slots].mean().item())
        result["interpenetration_ratio_all"] = float(interpen.mean().item())
        per_slot = []
        for totals in self.slot_artic:
            aggregated = _finalize(totals, count)
            aggregated.update(articulation_ratios(aggregated))
            per_slot.append(aggregated)
        result["articulation_single"] = (
            OrderedDict((key, sum(item[key] for item in per_slot) / len(per_slot)) for key in ARTIC_RATIO_KEYS) if per_slot else None
        )
        result["naturalness_single"], result["naturalness_per_slot"] = None, None
        if natural_rows is not None and self.kept:
            kept = torch.cat(self.kept, dim=1)  # [S,N,...]
            per_slot_natural = natural_rows(OrderedDict(("slot{}".format(s), kept[s]) for s in range(kept.shape[0])))
            result["naturalness_per_slot"] = list(per_slot_natural.values())
            keys = list(result["naturalness_per_slot"][0])
            result["naturalness_single"] = OrderedDict(
                (key, sum(item[key] for item in result["naturalness_per_slot"]) / len(result["naturalness_per_slot"])) for key in keys
            )
        result["first_step_error_max"] = float(self.first_step_max)
        # 均值是结构自检口径（与 naturalness 的 bone_rel_err_body 同义）；最大值受 float32 坐标舍入影响（短骨约 1e-5 量级），只作报告。
        result["bone_rel_err_body_mean"] = self.bone_sum / max(self.bone_count, 1)
        result["bone_rel_err_body_max"] = float(self.bone_max)
        result["pelvis_equal_v2"] = bool(self.pelvis_equal) if self.check_pelvis else None
        result["all_finite"] = bool(self.finite)
        result["sample_digest"] = self.digest.hexdigest()
        if self.levels:
            levels = torch.cat(self.levels)
            result["strata_levels"] = OrderedDict((name, int((levels == index).sum().item())) for index, name in enumerate(STRATA_LEVELS))
        ci = OrderedDict()
        ci["single_mpjpe"] = window_bootstrap_ci(per_sample.mean(dim=0), CI_SAMPLES)
        ci["min_ade@10"] = window_bootstrap_ci(curves["min_ade@10"], CI_SAMPLES) if "min_ade@10" in curves else None
        ci["es_joint"] = window_bootstrap_ci(es["es_joint"], CI_SAMPLES)
        ci["es_legs_walk"] = window_bootstrap_ci(es["es_legs_walk"], CI_SAMPLES)
        ci["es_arms_act"] = window_bootstrap_ci(es["es_arms_act"], CI_SAMPLES)
        result["ci"] = ci
        return result


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = _device(args)
        self.model, self.state = load_ntu2p_resdiff_checkpoint(args.checkpoint, self.device)
        self.deploy, self.deploy_state = load_ntu2p_model_checkpoint(args.base_checkpoint, self.device)
        self.deploy.eval()
        self.codec = ResidualCodec(self.model.num_coeffs).to(self.device)
        self.projector = SkeletonProjector().to(self.device)
        self.windows = Windows(args, self.device, self.deploy, self.codec)
        self.num_samples = int(args.num_samples)
        noise = torch.randn(
            (self.windows.num_total, self.num_samples, 2, self.model.num_coeffs, self.model.channels),
            generator=torch.Generator().manual_seed(int(args.noise_seed)),
        )
        uniform = torch.rand((self.windows.num_total, self.num_samples), generator=torch.Generator().manual_seed(int(args.noise_seed) + 1))
        self.noise = noise[: self.windows.count]
        self.uniform = uniform[: self.windows.count]
        self.bank = NTU2PResidualBank.load(args.bank, device="cpu")
        self.pool = self.bank.bootstrap_pool()
        self.settings = parse_settings(args.settings)
        self.review_cases = self._review_cases()
        self.review_index = [case["index"] for case in self.review_cases] if self.review_cases else []
        self.copy_last = copy_last_xyz(self.windows.obs, self.windows.target.shape[1])

    def _review_cases(self):
        args = self.args
        cases = None
        if args.write_review_cases:
            cases = select_review_cases(
                self.windows.obs, self.windows.target, self.windows.action, seed=0, sample_ids=[m["sample_id"] for m in self.windows.meta]
            )
            _write_json(
                args.write_review_cases,
                OrderedDict([("split", args.split), ("manifest_path", args.manifest_path), ("num_windows", self.windows.count), ("cases", cases)]),
            )
        if args.review_cases:
            with open(args.review_cases) as handle:
                cases = json.load(handle)["cases"]
        if cases is None:
            return []
        return [case for case in cases if int(case["index"]) < self.windows.count]

    def _natural_block(self, rows):
        """GT、copy_last、v2 与 rows 同一次调用 naturalness_block：比值类指标（摆幅/GT、RMSE/copy_last）口径一致。"""
        variants = OrderedDict([("GT", self.windows.target.cpu()), ("copy_last", self.copy_last.cpu()), ("v2", self.windows.base.cpu())])
        variants.update(rows)
        return naturalness_block(self.windows.obs.cpu(), self.windows.target.cpu(), variants)

    def _natural_rows(self, rows):
        """单样本自然度：返回各样本槽的行。"""
        block = self._natural_block(rows)
        return OrderedDict((name, block[name]) for name in rows)

    # ---- 生成
    def _decode(self, coeffs_m, begin, end, repeats):
        obs, base = self.windows.obs[begin:end], self.windows.base[begin:end]
        frames = codec_frames(obs)
        if repeats > 1:
            obs, base, frames = obs.repeat_interleave(repeats, 0), base.repeat_interleave(repeats, 0), repeat_frames(frames, repeats)
        return self.codec.apply(obs, base, coeffs_m, frames, self.projector)

    def generate(self, mode, tau, sampler, begin, end):
        size, count = end - begin, self.num_samples
        y = make_condition(self.model, self.windows.feats(begin, end), self.windows.action[begin:end], mode)
        noise = self.noise[begin:end].to(self.device)
        flat = noise.reshape((size * count,) + tuple(noise.shape[2:]))
        x0 = sample_x0(self.model, sampler, repeat_condition(y, count), flat, tau)
        samples = self._decode(to_meters(self.model, x0), begin, end, count)
        samples = samples.reshape((size, count) + tuple(samples.shape[1:])).transpose(0, 1)
        mean_x0 = one_step_mean(self.model, y, noise, tau)
        return samples, self._decode(to_meters(self.model, mean_x0), begin, end, 1)

    def bootstrap(self, mode, scale, begin, end):
        size, count = end - begin, self.num_samples
        ids, levels = self.pool.draw(self.windows.action[begin:end], self.windows.v2_walk[begin:end], self.uniform[begin:end])
        coeffs = self.bank.target[ids.reshape(-1)].to(self.device) * float(scale)
        if mode == "R":
            coeffs[..., :3] = 0.0
        samples = self._decode(coeffs, begin, end, count)
        return samples.reshape((size, count) + tuple(samples.shape[1:])).transpose(0, 1), levels

    def run_generator(self, mode, tau, nfe, keep, review=True):
        sampler = build_sampling_diffusion(nfe)
        accumulator = SampleAccumulator(
            self.num_samples, self.args.naturalness_samples, keep, self.review_index if review else None, check_pelvis=(mode == "R")
        )
        for begin, end in self.windows.chunks():
            samples, mean = self.generate(mode, tau, sampler, begin, end)
            accumulator.update(samples, self.windows, begin, end, one_step=mean)
        return accumulator

    def run_bootstrap(self, mode, scale, keep):
        accumulator = SampleAccumulator(self.num_samples, self.args.naturalness_samples, keep, check_pelvis=(mode == "R"))
        for begin, end in self.windows.chunks():
            samples, levels = self.bootstrap(mode, scale, begin, end)
            accumulator.update(samples, self.windows, begin, end, levels=levels)
        return accumulator

    # ---- 参照
    def references(self):
        args, windows = self.args, self.windows
        outputs = OrderedDict([("v2", windows.base), ("copy_last", self.copy_last)])
        base_model, _ = load_base_model_from_checkpoint(args.baseline_checkpoint, self.device)
        outputs["independent_base"] = self._chunked(lambda obs, action: independent_base_forward(base_model, obs, action))
        if args.a0_checkpoint:
            a0, _ = load_ntu2p_model_checkpoint(args.a0_checkpoint, self.device)
            outputs["a0"] = self._chunked(lambda obs, action: forward_details(a0.eval(), obs, action)["pred"])
        ensemble = []
        for path in args.ensemble_checkpoints or []:
            member, _ = load_ntu2p_model_checkpoint(path, self.device)
            ensemble.append(self._chunked(lambda obs, action: forward_details(member.eval(), obs, action)["pred"]))
        if ensemble:
            outputs["v2_ensemble"] = torch.stack(ensemble).mean(dim=0)
        result = OrderedDict((name, None) for name in ("v2", "copy_last", "independent_base", "a0", "v2_ensemble", "v2_ensemble_samples"))
        for name, value in outputs.items():
            accumulator = SampleAccumulator(1, 1, False)
            for begin, end in windows.chunks():
                accumulator.update(value[begin:end].unsqueeze(0), windows, begin, end)
            result[name] = accumulator.finalize(windows)
        if not args.skip_naturalness:
            block = self._natural_block(OrderedDict((name, value.cpu()) for name, value in outputs.items() if name not in ("v2", "copy_last")))
            for name in outputs:
                result[name]["naturalness_single"] = block[name]
        if len(ensemble) >= 2:
            # v2 多 seed 输出当作 K 个样本：只作诊断（ES、minADE@K）。
            accumulator = SampleAccumulator(len(ensemble), 0, False)
            stacked = torch.stack(ensemble)
            for begin, end in windows.chunks():
                accumulator.update(stacked[:, begin:end], windows, begin, end)
            result["v2_ensemble_samples"] = accumulator.finalize(windows)
        return result

    def _chunked(self, forward):
        parts = []
        for begin, end in self.windows.chunks():
            parts.append(forward(self.windows.obs[begin:end], self.windows.action[begin:end]))
        return torch.cat(parts, dim=0)

    # ---- 诊断
    def teacher_forced(self):
        diffusion = build_training_diffusion()
        windows, model = self.windows, self.model
        result = OrderedDict()
        for t_value in TEACHER_FORCED_TS:
            totals, count, sq_sum, sq_root, elements = OrderedDict(), 0, 0.0, 0.0, 0
            for begin, end in windows.chunks():
                size = end - begin
                x0 = windows.target_coeffs[begin:end] / model.sigma_target
                t = torch.full((size,), int(t_value), dtype=torch.long, device=self.device)
                x_t = diffusion.q_sample(x0, t, self.noise[begin:end, 0].to(self.device))
                y = make_condition(model, windows.feats(begin, end), windows.action[begin:end], "F")
                x0_hat = model(x_t, t, y)
                sq_sum += float((x0_hat - x0).pow(2).sum().item())
                sq_root += float((x0_hat - x0)[..., :3].pow(2).sum().item())
                elements += int(x0.numel())
                decoded = self._decode(to_meters(model, x0_hat), begin, end, 1)
                _add(totals, compute_ntu_xyz_metrics(decoded, windows.target[begin:end], windows.obs[begin:end]), size)
                count += size
            l2 = _finalize(totals, count)
            result[str(t_value)] = OrderedDict(
                [
                    ("norm_mse", sq_sum / elements),
                    ("norm_mse_root", sq_root / (elements * 3.0 / model.channels)),
                    ("mpjpe", l2["mpjpe"]),
                    ("xyz_mse", l2["xyz_mse"]),
                ]
            )
        return result

    # ---- 审查导出
    def export_review(self, generator_accumulators):
        args, windows = self.args, self.windows
        if not self.review_cases:
            raise ValueError("--export_review 需要 --review_cases 或 --write_review_cases")
        indices = [int(case["index"]) for case in self.review_cases]
        review_settings = parse_settings(args.review_settings) if args.review_settings else []
        if not review_settings:
            for mode in ("F", "R"):
                first = next(((m, t) for m, t in self.settings if m == mode), None)
                if first is not None:
                    review_settings.append(first)
        os.makedirs(args.export_review, exist_ok=True)
        obs = windows.obs[indices].cpu()
        target = windows.target[indices].cpu()
        v2 = windows.base[indices].cpu()
        meta = [dict(windows.meta[i], review_category=case["category"]) for i, case in zip(indices, self.review_cases)]
        actions = windows.action[indices].cpu()
        per_setting = OrderedDict()
        for mode, tau in review_settings:
            accumulator = generator_accumulators[setting_key(mode, tau)]
            per_setting[setting_key(mode, tau)] = torch.stack([accumulator.review_index[i] for i in indices])  # [R,K,T,2,55,3]
        methods = OrderedDict([("v2", v2)])
        for mode, tau in review_settings:
            stacked = per_setting[setting_key(mode, tau)]
            for k in range(min(3, stacked.shape[1])):
                methods["{}_k{}".format(mode, k)] = stacked[:, k]
        files = OrderedDict()
        path = os.path.join(args.export_review, "review_arrays.pt")
        source = OrderedDict([("checkpoint", args.checkpoint), ("base_checkpoint", args.base_checkpoint), ("split", args.split),
                              ("settings", [setting_key(m, t) for m, t in review_settings])])
        torch.save(OrderedDict([("obs_xyz", obs), ("target_xyz", target), ("methods", methods), ("actions", actions), ("meta", meta),
                                ("source", source)]), path)
        files["review_arrays"] = path
        # 每个设置另存一份"v2 + 样本均值 + 全部 K 个样本"的数组，可直接交给审查图脚本逐样本渲染。
        for key, stacked in per_setting.items():
            sample_methods = OrderedDict([("v2", v2), ("mean", stacked.mean(dim=1))])
            for k in range(stacked.shape[1]):
                sample_methods["sample{}".format(k)] = stacked[:, k]
            name = "review_arrays_{}{}.pt".format(key.split(":")[0], key.split(":")[1].replace(".", "p"))
            sample_path = os.path.join(args.export_review, name)
            torch.save(OrderedDict([("obs_xyz", obs), ("target_xyz", target), ("methods", sample_methods), ("actions", actions),
                                    ("meta", meta), ("source", dict(source, setting=key))]), sample_path)
            files[key] = sample_path
        samples_path = os.path.join(args.export_review, "review_samples.pt")
        torch.save(OrderedDict([("obs_xyz", obs), ("target_xyz", target), ("v2", v2), ("samples", per_setting), ("indices", indices),
                                ("cases", self.review_cases), ("meta", meta), ("actions", actions), ("source", source)]), samples_path)
        files["review_samples"] = samples_path
        return files

    def run(self):
        args = self.args
        result = OrderedDict()
        result["meta"] = OrderedDict(
            [
                ("checkpoint", args.checkpoint),
                ("checkpoint_step", int(self.state.get("step", -1))),
                ("checkpoint_arm", self.state.get("arm")),
                ("checkpoint_seed", self.state.get("seed")),
                ("checkpoint_bank_config_sha256", self.state.get("bank_config_sha256")),
                ("base_checkpoint", args.base_checkpoint),
                ("base_checkpoint_manifest", self.deploy_state.get("manifest_path")),
                ("baseline_checkpoint", args.baseline_checkpoint),
                ("a0_checkpoint", args.a0_checkpoint),
                ("ensemble_checkpoints", list(args.ensemble_checkpoints or [])),
                ("bank", args.bank),
                ("bank_config_sha256", self.bank.config_sha256()),
                ("manifest_path", args.manifest_path),
                ("cache_dir", args.cache_dir),
                ("split", args.split),
                ("num_windows", int(self.windows.count)),
                ("K", self.num_samples),
                ("noise_seed", int(args.noise_seed)),
                ("nfe", int(args.nfe)),
                ("settings", [setting_key(m, t) for m, t in self.settings]),
                ("boot_scales", _floats(args.boot_scales)),
                ("naturalness_samples", int(args.naturalness_samples)),
                ("device", str(self.device)),
            ]
        )
        result["subsets"] = self.windows.subsets()
        print("references", flush=True)
        result["references"] = self.references()
        natural = None if args.skip_naturalness else self._natural_rows
        sampler_nfe = int(args.nfe)
        result["generator"] = OrderedDict()
        accumulators = OrderedDict()
        for mode, tau in self.settings:
            print("generator {}:{} nfe={}".format(mode, tau, sampler_nfe), flush=True)
            accumulator = self.run_generator(mode, tau, sampler_nfe, keep=not args.skip_naturalness)
            accumulators[setting_key(mode, tau)] = accumulator
            result["generator"].setdefault(mode, OrderedDict())[str(float(tau))] = accumulator.finalize(self.windows, natural)
            accumulator.kept = []
        a0 = result["references"].get("a0")
        if a0 is not None:
            for per_mode in result["generator"].values():
                for block in per_mode.values():
                    block["single_l2_ratio_to_a0"] = OrderedDict((key, block["single_l2"][key] / a0["single_l2"][key]) for key in L2_KEYS)
        result["bootstrap"] = OrderedDict()
        for mode in ("F", "R"):
            for scale in _floats(args.boot_scales):
                print("bootstrap {} s={}".format(mode, scale), flush=True)
                keep = (not args.skip_naturalness) and abs(scale - BOOTSTRAP_NATURAL_SCALE) < 1e-9
                accumulator = self.run_bootstrap(mode, scale, keep)
                result["bootstrap"].setdefault(mode, OrderedDict())[str(float(scale))] = accumulator.finalize(self.windows, natural if keep else None)
        result["nfe_scan"] = OrderedDict()
        for nfe in _ints(args.nfe_scan):
            if nfe == sampler_nfe:
                continue
            print("nfe scan F:1.0 nfe={}".format(nfe), flush=True)
            accumulator = self.run_generator("F", 1.0, nfe, keep=False, review=False)
            result["nfe_scan"][str(nfe)] = accumulator.finalize(self.windows)
        result["diagnostics"] = OrderedDict()
        if args.diagnostics:
            print("diagnostics", flush=True)
            result["diagnostics"]["teacher_forced"] = self.teacher_forced()
            result["diagnostics"]["one_step_mean_l2"] = OrderedDict(
                (setting_key(m, t), result["generator"][m][str(float(t))]["one_step_mean_l2"]) for m, t in self.settings
            )
        result["review"] = OrderedDict([("cases", self.review_cases), ("indices", self.review_index)])
        if args.export_review:
            result["review"]["files"] = self.export_review(accumulators)
        return result


def evaluate(args):
    if args.split == "test" and os.path.exists(args.output):
        # test 只评估一次：已存在的结果绝不覆盖。
        raise FileExistsError("test 结果 {} 已存在，拒绝重跑".format(args.output))
    with torch.no_grad():
        result = Evaluator(args).run()
    _write_json(args.output, result)
    summary = OrderedDict()
    for mode, per_mode in result["generator"].items():
        for tau, block in per_mode.items():
            summary["{}:{}".format(mode, tau)] = OrderedDict(
                [("mpjpe", round(block["single_l2"]["mpjpe"], 5)), ("es_joint", round(block["es_joint"], 5)),
                 ("min_ade@10", None if block["min_ade@10"] is None else round(block["min_ade@10"], 5))]
            )
    print(json.dumps(OrderedDict([("v2_mpjpe", result["references"]["v2"]["single_l2"]["mpjpe"]), ("generator", summary)]), ensure_ascii=False))
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(description="NTU2P Track B 残差扩散评估")
    parser.add_argument("--checkpoint", required=True, help="生成器 EMA 终点")
    parser.add_argument("--base_checkpoint", required=True, help="部署底座（v2 主线，与生成器 seed 配对）")
    parser.add_argument("--bank", required=True, help="OOF 残差库：bootstrap 基线的系数池")
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--baseline_checkpoint", default=DEFAULT_BASELINE, help="独立单人 base（与协议一致）")
    parser.add_argument("--a0_checkpoint", default=None)
    parser.add_argument("--ensemble_checkpoints", nargs="*", default=None,
                        help="v2 多 seed 底座（诊断：输出均值与 K=3 样本集）；输出均值是未立项的旁支，只在 val 上算")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS, help="逗号分隔的 mode:τ")
    parser.add_argument("--nfe", type=int, default=DEFAULT_NFE)
    parser.add_argument("--nfe_scan", default=DEFAULT_NFE_SCAN, help="mode F τ=1 的 NFE 扫描（完整指标块，含 APD/SSR；与 --nfe 相同的档跳过）")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--boot_scales", default=DEFAULT_BOOT_SCALES)
    parser.add_argument("--naturalness_samples", type=int, default=5, help="单样本自然度与摆动指标取前 S 个样本槽")
    parser.add_argument("--skip_naturalness", action="store_true", help="只用于审查数组导出等不需要自然度的场合")
    parser.add_argument("--diagnostics", action="store_true", help="teacher-forced 分 t 误差与单步均值汇总")
    parser.add_argument("--write_review_cases", default=None, help="按 GT 预选审查窗口并写出 JSON")
    parser.add_argument("--review_cases", default=None, help="读取已预选的审查窗口")
    parser.add_argument("--review_settings", default=None, help="导出审查数组用的 mode:τ（缺省为 settings 中首个 F 与首个 R）")
    parser.add_argument("--export_review", default=None, help="审查数组输出目录")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_windows", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=-1, help="只评估按 manifest 顺序的前 N 个窗口（噪声取同一份张量的前缀）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow_cpu_for_smoke_test", action="store_true", help="仅冒烟测试：允许 --device cpu")
    return parser


if __name__ == "__main__":
    evaluate(build_arg_parser().parse_args())
