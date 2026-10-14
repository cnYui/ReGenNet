"""图扩散随机未来的探索性评价，与确定性端点、残差bootstrap明确分开。

固定 NFE20/50、eta0、K10、标准高斯起点；结果不用于 GT 选样或替换点预测数字。
"""

import argparse
import json
from pathlib import Path
import time

import torch

from data_loaders.forecasting.ntu2p_residual_bank import NTU2PResidualBank, BootstrapPool
from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, eval_windows, iter_eval_batches
from eval.eval_ntu2p_v2 import contact_geometry_metadata, naturalness_block
from model.forecasting_ntu2p_graphdiff import load_ntu2p_graphdiff_checkpoint
from utils.ntu2p_contact_graph import contact_metrics
from utils.ntu2p_probabilistic_metrics import energy_score, min_ade_curve, apd
from utils.ntu2p_residual_codec import ResidualCodec, v2_walk
from utils.ntu_smplx_2p_xyz import compute_ntu_xyz_metrics, copy_last_xyz, compute_ntu_articulation_metrics, articulation_ratios
from utils.fixseed import fixseed


PREREGISTERED_SCOPE = dict(
    manifest_path="results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json",
    candidate="C4", checkpoint_step=5000, training_seed=0, num_windows=247,
    K=10, nfe=[20, 50], sample_seed=2718,
)


def output_paths(args):
    """先拒绝历史产物，长时间采样后不应覆盖已有 JSON 或审查数组。"""
    output = Path(args.output)
    export = Path(args.export_arrays) if args.export_arrays else None
    if export is not None and output.resolve() == export.resolve():
        raise ValueError("JSON 与数组导出必须使用不同路径")
    for path in (output, export):
        if path is not None and path.exists():
            raise ValueError("已有采样产物，不能覆盖：{}".format(path))
    return output, export


def bootstrap_codec(bank, device):
    """同一 manifest 也可能有 in-sample 库或不同系数基，不能静默当作预登记 OOF 参照。"""
    codec = ResidualCodec().to(device)
    if bank.config.get("mode") != "oof" or bank.config.get("smoke_skip_oof_assert"):
        raise ValueError("bootstrap 参照必须是未跳过 OOF 检查的真实残差库")
    if bank.config.get("codec_config") != codec.config():
        raise ValueError("bootstrap 库系数基与预登记 K12/saturate 解码器不一致")
    return codec


def sampling_scope(args, state, num_windows, full_window_count):
    """范围外的流水线诊断要显式标记，不能仅凭文件名区分冒烟与正式数字。"""
    actual = dict(manifest_path=args.manifest_path, candidate=state.get("candidate"),
                  checkpoint_step=state["step"], training_seed=state.get("seed"),
                  num_windows=int(num_windows), K=args.num_samples, nfe=list(args.nfe), sample_seed=args.seed)
    reasons = ["{}：实际 {}，预登记 {}".format(key, actual[key], expected)
               for key, expected in PREREGISTERED_SCOPE.items() if actual[key] != expected]
    if int(num_windows) != int(full_window_count):
        reasons.append("未评价完整验证窗口")
    if Path(args.checkpoint).parent.name != "ema":
        reasons.append("checkpoint 不是预登记 EMA 终点")
    return dict(actual=actual, preregistered=PREREGISTERED_SCOPE,
                full_window_count=int(full_window_count), is_smoke=bool(reasons),
                matches_preregistered_scope=not reasons, diagnostic_reasons=reasons,
                label="冒烟或预登记范围外诊断" if reasons else "正式预登记的探索性分布评价")


def run(args):
    output, export = output_paths(args)
    if args.num_samples < 2 or not args.nfe or len(set(args.nfe)) != len(args.nfe) or min(args.nfe) < 2:
        raise ValueError("随机评价至少需要两个样本，NFE 必须是互不重复且至少为 2 的整数")
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise ValueError("正式采样评价必须cuda:0")
    fixseed(args.seed)
    model, state = load_ntu2p_graphdiff_checkpoint(args.checkpoint, args.device)
    if state["manifest_path"] != args.manifest_path:
        raise ValueError("评估manifest与训练不一致")
    # 旧 OOF 库以原 FD2 为中心，不能套在微调 draft 上冒充同一 bootstrap 参照。
    if model.fine_tune_prior:
        raise ValueError("当前随机评价仅支持冻结先验；F 候选须先登记冻结 reference 与残差参照的评价方案")
    cache = NTU2PXYZSeqCache.load(args.cache_dir, "val", args.manifest_path, args.device)
    windows = eval_windows(cache)
    full_window_count = int(windows["obs_xyz"].shape[0])
    if args.max_samples > 0:
        windows = {key: value[:args.max_samples] for key, value in windows.items()}
    scope = sampling_scope(args, state, int(windows["obs_xyz"].shape[0]), full_window_count)
    bank = NTU2PResidualBank.load(args.bank_path)
    if bank.config.get("parent_manifest") != args.manifest_path or bank.config.get("parent_manifest_hash") != cache.manifest_hash:
        raise ValueError("bootstrap bank与评估manifest不一致")
    pool = BootstrapPool(bank)
    old_codec = bootstrap_codec(bank, args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    bootstrap_generator = torch.Generator().manual_seed(args.seed + 71)
    results = dict(checkpoint=args.checkpoint, checkpoint_step=state["step"], manifest_path=args.manifest_path,
                   split="val", sample_seed=args.seed, K=args.num_samples, nfe=list(args.nfe), eta=0,
                   num_windows=scope["actual"]["num_windows"], full_window_count=full_window_count,
                   is_smoke=scope["is_smoke"], evaluation_scope=scope,
                   mode="{}；标准高斯起点，与确定性端点分开".format(scope["label"]),
                   bootstrap_bank=args.bank_path, bootstrap_codec="已存在OOF K12残差转新smoothstep K16基",
                   bootstrap_bank_config_sha256=bank.config_sha256(), bootstrap_bank_mode=bank.config["mode"],
                   bootstrap_bank_codec=bank.config["codec_config"],
                   ES_definition="逐帧逐关节 xyz 边际 Energy Score 的均值，样本间项使用 K(K−1) 无偏估计；不独立验证联合时序分布",
                   contact_geometry=contact_geometry_metadata(),
                   bank_training_scope="OOF真残差；生成器in-sample残差，仅探索性分布评价", variants={})
    target, obs, actions, priors, points, meta = [], [], [], [], [], []
    arrays = {"ddim{}".format(nfe): [] for nfe in args.nfe}
    arrays.update({"bootstrap0.5": [], "bootstrap1": []})
    strata_counts = torch.zeros(3, dtype=torch.long)
    started = time.monotonic()
    with torch.no_grad():
        for batch in iter_eval_batches(windows, args.batch_size):
            observed, truth, action = batch["obs_xyz"], batch["target_xyz"], batch["action"]
            size = observed.shape[0]
            prepared = model.prepare(observed, action)
            point = model.point_details(observed, action, prepared)["pred"]
            noise = torch.randn(size, args.num_samples, 2, model.num_coeffs, 66, device=args.device, generator=generator)
            repeated_obs = observed.repeat_interleave(args.num_samples, 0)
            repeated_action = action.repeat_interleave(args.num_samples, 0)
            flat_noise = noise.reshape(size * args.num_samples, 2, model.num_coeffs, 66)
            for nfe in args.nfe:
                pred = model.sample_details(repeated_obs, repeated_action, flat_noise, num_steps=nfe)["pred"]
                pred = pred.reshape(size, args.num_samples, 50, 2, 55, 3).transpose(0, 1)
                arrays["ddim{}".format(nfe)].append(pred.cpu())
            uniform = torch.rand(size, args.num_samples, generator=bootstrap_generator)
            ids, levels = pool.draw(action.cpu(), v2_walk(observed, prepared["draft"]).cpu(), uniform)
            strata_counts += torch.bincount(levels, minlength=3)
            coeffs = bank.target[ids.flatten()].to(args.device)
            converted = model.codec.encode_channels(old_codec.decode_channels(coeffs))
            repeated_prior = prepared["draft"].repeat_interleave(args.num_samples, 0)
            for scale in (0.5, 1.0):
                coeff = converted * scale
                y = model.prepare(repeated_obs, repeated_action)
                y["draft"] = repeated_prior
                pred = model.apply_coeffs(coeff / model.sigma, y)["pred"]
                arrays["bootstrap{:g}".format(scale)].append(pred.reshape(size, args.num_samples, 50, 2, 55, 3).transpose(0, 1).cpu())
            target.append(truth.cpu())
            obs.append(observed.cpu())
            actions.append(action.cpu())
            priors.append(prepared["draft"].cpu())
            points.append(point.cpu())
            meta.extend(batch["meta"])
            print("采样进度 {}，耗时{:.1f}s".format(sum(v.shape[0] for v in target), time.monotonic() - started), flush=True)
    truth, observed, action = torch.cat(target), torch.cat(obs), torch.cat(actions)
    prior, point = torch.cat(priors), torch.cat(points)
    all_samples = {key: torch.cat(values, dim=1) for key, values in arrays.items()}
    natural = dict(prior=prior, point=point, copy_last=copy_last_xyz(observed, 50), GT=truth)
    for key, samples in all_samples.items():
        es, _ = energy_score(samples, truth)
        metrics = compute_ntu_xyz_metrics(samples[0], truth, observed)
        art = compute_ntu_articulation_metrics(samples[0], truth)
        art.update(articulation_ratios(art))
        results["variants"][key] = dict(single_sample_metrics=metrics, ES=es,
                                       minADE_GT_selected_upper_bound={name: float(v.mean()) for name, v in min_ade_curve(samples, truth, (1, 5, 10)).items()},
                                       APD=apd(samples)[0], articulation=art)
        for index in range(min(3, args.num_samples)):
            natural["{}_sample{}".format(key, index)] = samples[index]
    results["point_metrics"] = compute_ntu_xyz_metrics(point, truth, observed)
    results["prior_metrics"] = compute_ntu_xyz_metrics(prior, truth, observed)
    results["naturalness"] = naturalness_block(observed, truth, natural)
    sampled_contact = dict(point=point, prior=prior)
    # 保留 sample0 的旧入口，另完整报告固定前三样本，不用 GT 或效果选择接触案例。
    sampled_contact.update({key: value[0] for key, value in all_samples.items() if key.startswith("ddim")})
    sampled_contact.update({"{}_sample{}".format(key, index): value[index]
                            for key, value in all_samples.items() for index in range(min(3, args.num_samples))})
    results["contact"] = contact_metrics(observed, truth, sampled_contact, action)
    results["fixed_samples_reviewed"] = list(range(min(3, args.num_samples)))
    results["bootstrap_strata_window_counts"] = dict(zip(("action_walk", "action", "all"), strata_counts.tolist()))
    results["elapsed_s"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    if export is not None:
        methods = dict(point=point, prior=prior)
        methods.update({"{}_sample{}".format(key, index): value[index] for key, value in all_samples.items() for index in range(min(3, args.num_samples))})
        export.parent.mkdir(parents=True, exist_ok=True)
        with export.open("xb") as handle:
            torch.save(dict(obs_xyz=observed, target_xyz=truth, methods=methods, actions=action, meta=meta,
                            source=dict(checkpoint=args.checkpoint, split="val", manifest_path=args.manifest_path,
                                        num_windows=results["num_windows"], K=args.num_samples, nfe=list(args.nfe),
                                        is_smoke=scope["is_smoke"], sample_seed=args.seed)), handle)
    print(json.dumps({key: dict(ES=v["ES"], mpjpe=v["single_sample_metrics"]["mpjpe"]) for key, v in results["variants"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest_path", default="results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json")
    parser.add_argument("--cache_dir", default="results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval")
    parser.add_argument("--bank_path", default="results/forecasting/ntu120_label/ntu2p_trackb/subjval/oof_bank.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--nfe", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--max_samples", type=int, default=-1, help="仅流水线冒烟时限制窗口数，正式评价固定全体")
    parser.add_argument("--export_arrays")
    run(parser.parse_args())
