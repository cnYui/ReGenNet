"""在 CPU 上验证旧 C 兼容性及 F 候选的真实两步训练、参照冻结、EMA 与重载。

旧输出快照必须在修改冻结模式前生成；本检查不产生选型分数，也不占用 CUDA。
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.forecasting_ntu2p_graphdiff import NTU2PContactGraphDiffusion, load_ntu2p_graphdiff_checkpoint
from model.forecasting_ntu2p_v2 import load_ntu2p_model_checkpoint
from train import train_ntu2p_graphdiff as trainer
from utils.ntu2p_residual_codec import ResidualCodec


DEFAULT_MANIFEST = "results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json"
DEFAULT_CACHE = "results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval"
DEFAULT_PRIOR = ("save/forecasting/ntu120_label/"
                 "ntu2p_v2subj_A6-F-A5f0.05-A4-GH0.5-FD2_s0_10000/ema/model000010000.pt")
DEFAULT_BASE = ("save/forecasting/ntu120_label/"
                "ntu2p_independent_single_person_o10_p50_subjval_s0_5000/model000005000.pt")
DEFAULT_SNAPSHOT = "results/forecasting/ntu120_label/ntu2p_graphdiff/finetune_legacy_snapshot_20261014-133324.pt"


def record(checks, label, passed, **details):
    checks.append(dict(label=label, passed=bool(passed), details=details))
    print("[{}] {} {}".format("通过" if passed else "失败", label, details), flush=True)


def state_equal(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def grad_summary(parameters):
    gradients = [param.grad for param in parameters if param.grad is not None]
    norm = math.sqrt(sum(float(gradient.double().square().sum()) for gradient in gradients))
    return bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients), norm


def check_legacy(checks, snapshot):
    obs, action = snapshot["obs"], snapshot["action"]
    for candidate, expected in sorted(snapshot["models"].items()):
        model, _ = load_ntu2p_graphdiff_checkpoint(expected["checkpoint"], "cpu")
        record(checks, candidate + " 默认冻结且无新增参照权重",
               not model.fine_tune_prior and model.reference is None
               and not any(name.startswith("reference.") for name in model.state_dict()))
        with torch.no_grad():
            actual = model(obs, action, return_details=True)
        equal = state_equal(actual, expected["eval"])
        record(checks, candidate + " 旧权重 eval 输出逐位一致", equal,
               max_abs=float((actual["pred"] - expected["eval"]["pred"]).abs().max()))
        model.train()
        torch.manual_seed(8743)
        with torch.no_grad():
            actual = model(obs, action, return_details=True)
        record(checks, candidate + " 旧权重 train 输出及随机流逐位一致",
               state_equal(actual, expected["train"])
               and torch.equal(torch.get_rng_state(), expected["rng_after_train"])
               and not model.prior.training
               and not any(param.requires_grad for param in model.prior.parameters()))


def check_initialization(checks, prior):
    signatures, rng_states = {}, {}
    for name in ("C4", "F1", "F2", "F3"):
        graph, phase, diffusion = trainer.CANDIDATES[name]
        torch.manual_seed(17)
        model = NTU2PContactGraphDiffusion(copy.deepcopy(prior), graph=graph, phase=phase,
                                         diffusion=diffusion, fine_tune_prior=name in trainer.FINE_TUNE_CANDIDATES)
        signatures[name] = {key: value.clone() for key, value in model.state_dict().items()}
        rng_states[name] = torch.get_rng_state()
    for name in ("F2", "F3"):
        record(checks, name + " 与 F1 参数初始化逐位一致", state_equal(signatures[name], signatures["F1"]))
    for name in ("F1", "F2", "F3"):
        head = {key: value for key, value in signatures[name].items() if not key.startswith("reference.")}
        record(checks, name + " 复制参照不消耗额外随机数",
               state_equal(head, signatures["C4"]) and torch.equal(rng_states[name], rng_states["C4"]))


def check_training(checks, args, prior, snapshot, temporary):
    original_build = trainer.build_optimizer
    original_update = trainer._update_ema
    original_prepare = NTU2PContactGraphDiffusion.prepare
    original_target = ResidualCodec.target_coeffs
    initial_states, initial_rng, scales = {}, {}, {}
    obs, action = snapshot["obs"], snapshot["action"]
    with torch.no_grad():
        reference_prediction = prior(obs, action)
    for candidate in ("F1", "F2", "F3"):
        captured = dict(steps=0, detached_targets=[], feature_gradients=[], ema_updates=[])

        def capture_optimizer(model, train_args):
            captured["model"] = model
            captured["reference_before"] = {key: value.clone() for key, value in model.reference.state_dict().items()}
            captured["prior_before"] = {key: value.clone() for key, value in model.prior.state_dict().items()}
            initial_states[candidate] = {key: value.clone() for key, value in model.state_dict().items()}
            initial_rng[candidate] = torch.get_rng_state()
            scales[candidate] = (copy.deepcopy(train_args.loss_scales), copy.deepcopy(train_args.new_loss_scales))
            optimizer = original_build(model, train_args)
            captured["optimizer"] = optimizer
            return optimizer

        def capture_prepare(model, obs_xyz, labels):
            prepared = original_prepare(model, obs_xyz, labels)
            if model.training and torch.is_grad_enabled():
                captured["feature_gradients"].append(all(prepared[key].requires_grad for key in
                                                        ("draft", "draft_local", "velocity", "draft_scene"))
                                                       and not prepared["reference"].requires_grad)
            return prepared

        def capture_target(codec, *values, **kwargs):
            target = original_target(codec, *values, **kwargs)
            captured["detached_targets"].append(not target.requires_grad)
            return target

        def capture_ema(ema, model, decay):
            captured["steps"] += 1
            record(checks, candidate + " step{} 底座 train、参照 eval".format(captured["steps"]),
                   model.training and model.prior.training and not model.reference.training)
            prior_finite, prior_norm = grad_summary(model.prior.parameters())
            record(checks, candidate + " step{} 底座有限非零梯度".format(captured["steps"]),
                   prior_finite and prior_norm > 0, gradient_norm=prior_norm)
            record(checks, candidate + " step{} 参照无梯度".format(captured["steps"]),
                   all(param.grad is None and not param.requires_grad for param in model.reference.parameters()))
            expected = next(ema.prior.parameters()).detach().clone()
            expected.mul_(decay).add_(next(model.prior.parameters()).detach(), alpha=1.0 - decay)
            original_update(ema, model, decay)
            captured["ema"] = ema
            captured["ema_updates"].append(torch.equal(next(ema.prior.parameters()), expected)
                                           and not ema.training and not ema.reference.training
                                           and state_equal(ema.reference.state_dict(), captured["reference_before"]))

        directory = Path(temporary) / candidate
        cli = ["--candidate", candidate, "--prior_checkpoint", args.prior_checkpoint,
               "--manifest_path", args.manifest_path, "--cache_dir", args.cache_dir,
               "--baseline_checkpoint", args.baseline_checkpoint, "--save_dir", str(directory),
               "--device", "cpu", "--allow_cpu_for_smoke_test", "--num_steps", "2",
               "--save_interval", "2", "--log_interval", "1", "--seed", "0"]
        train_args = trainer.build_arg_parser().parse_args(cli)
        with mock.patch.object(trainer, "build_optimizer", side_effect=capture_optimizer), \
                mock.patch.object(trainer, "_update_ema", side_effect=capture_ema), \
                mock.patch.object(NTU2PContactGraphDiffusion, "prepare", new=capture_prepare), \
                mock.patch.object(ResidualCodec, "target_coeffs", new=capture_target):
            trainer.run(train_args)
        model = captured["model"]
        groups = captured["optimizer"].param_groups
        all_ids = [id(param) for group in groups for param in group["params"]]
        expected_ids = {id(param) for param in model.parameters() if param.requires_grad}
        record(checks, candidate + " 双学习率覆盖所有可训练参数且不含参照",
               len(groups) == 2 and groups[0]["lr"] == 3e-4 and abs(groups[1]["lr"] - 3e-5) <= 1e-12
               and len(all_ids) == len(set(all_ids)) and set(all_ids) == expected_ids
               and not set(all_ids) & {id(param) for param in model.reference.parameters()},
               lr=[group["lr"] for group in groups])
        record(checks, candidate + " 真实训练改变底座但不改变原参照",
               not state_equal(model.prior.state_dict(), captured["prior_before"])
               and state_equal(model.reference.state_dict(), captured["reference_before"]))
        record(checks, candidate + " 点轨迹与几何保留梯度、残差目标停止梯度",
               captured["feature_gradients"] and all(captured["feature_gradients"])
               and captured["detached_targets"] and all(captured["detached_targets"]))
        head_finite, head_norm = grad_summary(model.out.parameters())
        record(checks, candidate + " 新头有限非零梯度", head_finite and head_norm > 0, gradient_norm=head_norm)
        if candidate != "F1":
            graph_finite, graph_norm = grad_summary(model.event_embed.parameters())
            leg_finite, leg_norm = grad_summary(model.leg_net.parameters())
            record(checks, candidate + " 图与腿专用流有限非零梯度",
                   graph_finite and leg_finite and min(graph_norm, leg_norm) > 0,
                   graph_gradient_norm=graph_norm, leg_gradient_norm=leg_norm)
        record(checks, candidate + " EMA 平均底座且逐位保留冻结参照", all(captured["ema_updates"]))
        with torch.no_grad():
            current_details = model.eval()(obs, action, return_details=True)
        record(checks, candidate + " prior 指标始终来自原 FD2",
               torch.equal(current_details["prior"], reference_prediction)
               and not torch.equal(model.prior(obs, action), current_details["prior"]))
        for name, expected_model in (("raw", model), ("ema", captured["ema"])):
            path = directory / ("ema" if name == "ema" else "") / "model000000002.pt"
            restored, state = load_ntu2p_graphdiff_checkpoint(str(path), "cpu")
            with torch.no_grad():
                expected = expected_model.eval()(obs, action, return_details=True)
                actual = restored(obs, action, return_details=True)
            record(checks, candidate + " " + name + " 重载逐位一致且保存原参照",
                   state_equal(actual, expected) and restored.fine_tune_prior and state["step"] == 2
                   and state_equal(restored.reference.state_dict(), captured["reference_before"]))
        saved = json.loads((directory / "args.json").read_text())
        record(checks, candidate + " args 保存模式及两个学习率",
               saved["fine_tune_prior"] and saved["forward_calls_per_step"] == 2
               and [group["name"] for group in saved["optimizer_groups"]] == ["corrector", "prior"])
    for name in ("F2", "F3"):
        record(checks, name + " 与 F1 真实训练起点、尺度及全局随机流一致",
               state_equal(initial_states[name], initial_states["F1"])
               and torch.equal(initial_rng[name], initial_rng["F1"]) and scales[name] == scales["F1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior_checkpoint", default=DEFAULT_PRIOR)
    parser.add_argument("--baseline_checkpoint", default=DEFAULT_BASE)
    parser.add_argument("--manifest_path", default=DEFAULT_MANIFEST)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE)
    parser.add_argument("--legacy_snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path(args.output_json)
    if output.exists():
        raise ValueError("结构检查结果不能覆盖历史文件")
    torch.set_num_threads(2)
    checks = []
    snapshot = torch.load(args.legacy_snapshot, map_location="cpu")
    check_legacy(checks, snapshot)
    prior, _ = load_ntu2p_model_checkpoint(args.prior_checkpoint, "cpu")
    check_initialization(checks, prior)
    with tempfile.TemporaryDirectory(prefix="ntu2p-finetune-check-") as temporary:
        check_training(checks, args, prior, snapshot, temporary)
    result = dict(passed=all(check["passed"] for check in checks), device="cpu", checks=checks,
                  prior_checkpoint=args.prior_checkpoint, legacy_snapshot=args.legacy_snapshot,
                  legacy_source_sha256=snapshot["source_sha256"],
                  note="真实训练入口两步集成检查；不作为模型选型或自然度结果")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("端到端结构检查：{}，{}项，{}".format("通过" if result["passed"] else "失败", len(checks), output), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
