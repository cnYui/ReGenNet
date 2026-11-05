"""NTU2P JiT 流匹配（J1–J4）结构检查，CPU 运行。

设计 6 节：编解码互逆与 oracle、零初始化等于 copy-last、首帧/骨长、J4 跨人因果掩码（单层性质）、
梯度有限且各新模块有梯度、checkpoint 重载逐位一致、样本固定 seed 可复现、CUDA 强制。
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict

import torch

from data_loaders.forecasting.ntu2p_xyz_seq_cache import NTU2PXYZSeqCache, eval_windows
from model.forecasting_ntu2p_jitflow import NTU2PJiTFlow, load_ntu2p_jitflow_checkpoint
from utils.ntu2p_kinematic_projection import bone_lengths
from utils.ntu_smplx_2p_xyz import copy_last_xyz, dct_matrix


CONFIGS = OrderedDict(
    [
        ("J1", dict(denoiser="mixer")),
        ("J2", dict(denoiser="mixer", self_cond=True, contact_graph=True)),
        ("J3", dict(denoiser="stg", self_cond=True)),
        ("J4", dict(denoiser="stg", self_cond=True, contact_graph=True, causal_cross=True)),
    ]
)


def _perturb(model, scale=0.02):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * scale)


def main(args):
    torch.manual_seed(0)
    cache = NTU2PXYZSeqCache.load(args.cache_dir, "val", manifest_path=args.manifest_path)
    windows = eval_windows(cache)
    obs, target, action = windows["obs_xyz"][: args.num], windows["target_xyz"][: args.num], windows["action"][: args.num]
    results = OrderedDict()

    def record(name, passed, value=None):
        results[name] = OrderedDict([("passed", bool(passed)), ("value", value)])
        print("{} {} {}".format("PASS" if passed else "FAIL", name, "" if value is None else value), flush=True)

    base = NTU2PJiTFlow(root_scale=0.14, local_scale=0.14).eval()
    prep = base.prepare(obs, action)
    root, local = base.encode_target(prep, target)
    again = base.codec.encode(base.codec.decode(root * 0.14, local * 0.14))
    record("codec_roundtrip", max(float((again[0] / 0.14 - root).abs().max()), float((again[1] / 0.14 - local).abs().max())) < 1e-4)
    oracle = float((base.decode_xyz(prep, root, local)[0] - target).norm(dim=-1).mean())
    # 参照：InterMixer 输出子空间对全部 55 关节逐一自由投影（手指不受刚体约束），即 FD2 表示的上限。
    window = dct_matrix(60, dtype=torch.float64)
    ramp = (torch.arange(50, dtype=torch.float64) / 5.0).clamp(max=1.0)

    def projector(order):
        basis = ramp[:, None] * window[:order, 10:].t()
        return basis @ torch.pinverse(basis)

    last = obs[:, -1:].double()
    residual = target.double() - last
    root_part = residual[:, :, :, :1]
    reference = last + torch.einsum("st,btpjc->bspjc", projector(8), root_part) + torch.einsum("st,btpjc->bspjc", projector(20), residual - root_part)
    reference = float((reference - target.double()).norm(dim=-1).mean())
    record("codec_oracle_mpjpe_le_1.1x_fd2_basis", oracle <= 1.1 * reference, OrderedDict([("ours", oracle), ("fd2_basis", reference)]))

    for name, config in CONFIGS.items():
        torch.manual_seed(0)
        model = NTU2PJiTFlow(root_scale=0.14, local_scale=0.14, **config).eval()
        with torch.no_grad():
            diff = float((model(obs, action) - copy_last_xyz(obs, 50)).abs().max())
        record(name + "_zero_init_equals_copy_last", diff == 0.0, diff)
        _perturb(model)
        with torch.no_grad():
            pred = model(obs, action)
        first = float((pred[:, 0] - obs[:, -1]).abs().max())
        record(name + "_first_frame_error_le_1e-6", first <= 1e-6, first)
        ref = bone_lengths(obs[:, -1:].double())
        rel = ((bone_lengths(pred.double()) - ref) / ref.clamp_min(1e-6)).abs()[..., 1:22]
        # 与评估口径一致：平均相对误差 ≤ 1e-5；最大值受两骨 IK 的可达收缩（1e-4）约束，随机权重下肢体常被拉直。
        record(name + "_body_bone_rel_err_mean_le_1e-5_max_le_1e-3", float(rel.mean()) <= 1e-5 and float(rel.max()) <= 1e-3,
               OrderedDict([("mean", float(rel.mean())), ("max", float(rel.max()))]))

        model.train()
        prep = model.prepare(obs, action)
        x_root, x_local = model.encode_target(prep, target)
        t = torch.rand(obs.shape[0])
        z_root = t.view(-1, 1, 1, 1) * x_root + (1 - t.view(-1, 1, 1, 1)) * torch.randn_like(x_root)
        z_local = t.view(-1, 1, 1, 1, 1) * x_local + (1 - t.view(-1, 1, 1, 1, 1)) * torch.randn_like(x_local)
        sc = model.denoise(prep, z_root, z_local, t) if model.self_cond else None
        if sc is not None:
            sc = (sc[0].detach(), sc[1].detach())
        est = model.denoise(prep, z_root, z_local, t, sc=sc)
        pred, _, _ = model.decode_xyz(prep, *est)
        loss = (est[0] - x_root).pow(2).mean() + (est[1] - x_local).pow(2).mean() + (pred - target).pow(2).mean()
        model.zero_grad()
        loss.backward()
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        record(name + "_grad_finite", finite)
        leg = sum(float(p.grad.abs().sum()) for p in model.leg_net.parameters() if p.grad is not None)
        record(name + "_leg_stream_has_grad", leg > 0, leg)
        if model.contact_graph:
            graph_params = [model.net.rel_proj] if config["denoiser"] == "mixer" else [model.net.graph_recv, model.net.graph_send]
            graph = sum(float(layer.weight.grad.abs().sum()) for layer in graph_params)
            record(name + "_contact_graph_has_grad", graph > 0, graph)

    # J4：一层注意力内，A 在第 j 块之后的 token 不影响 B 在第 < j 块的输出（跨人因果掩码）；同人内仍双向。
    torch.manual_seed(1)
    model = NTU2PJiTFlow(root_scale=0.14, local_scale=0.14, **CONFIGS["J4"]).eval()
    _perturb(model, 0.05)
    net, block = model.net, model.net.blocks[0]
    parts, patches, width = 5, net.num_patches, net.width
    tokens = torch.randn(2, 2, parts * patches, width)
    cond = torch.randn(2, 2, model.cond_dim)
    bias = net._static_bias()
    with torch.no_grad():
        out = block(tokens, cond, bias)
        changed = tokens.clone().view(2, 2, parts, patches, width)
        cut = 7
        changed[:, 0, :, cut:] += torch.randn_like(changed[:, 0, :, cut:])
        out2 = block(changed.view(2, 2, -1, width), cond, bias)
    out, out2 = out.view(2, 2, parts, patches, width), out2.view(2, 2, parts, patches, width)
    leak = float((out2[:, 1, :, :cut] - out[:, 1, :, :cut]).abs().max())
    record("J4_cross_person_causal_one_layer", leak < 1e-5, leak)
    later = float((out2[:, 1, :, cut:] - out[:, 1, :, cut:]).abs().max())
    record("J4_cross_person_sees_past_and_present", later > 1e-4, later)

    # checkpoint 重载逐位一致；固定 seed 的样本可复现。
    model = NTU2PJiTFlow(root_scale=0.14, local_scale=0.14, **CONFIGS["J2"]).eval()
    _perturb(model)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        torch.save({"model_state_dict": model.state_dict(), "model_type": model.model_type, "model_config": model.config()}, path)
        loaded, _ = load_ntu2p_jitflow_checkpoint(path, "cpu")
    with torch.no_grad():
        same = torch.equal(model(obs, action), loaded(obs, action))
        record("checkpoint_reload_bitwise", same)
        for item in (model, loaded):
            item.readout, item.num_samples, item.nfe_steps = "median", 3, 4
            item.reset_sampler(7)
        record("samples_reproducible_with_seed", torch.equal(model(obs, action), loaded(obs, action)))
        ema = copy.deepcopy(model)
        for item in (model, ema):
            item.reset_sampler(7)
        record("deepcopy_for_ema", torch.equal(model(obs, action), ema(obs, action)))

    # GRPO：初始策略 = 参考策略时，重算的对数似然与采样时逐位相同（比值 1）、KL 为 0；对数似然与 Normal 一致。
    from train.train_ntu2p_jitflow_grpo import _kl, _log_prob, _step_stats, rollout
    from model.forecasting_ntu2p_jitflow import _repeat

    torch.manual_seed(3)
    policy = NTU2PJiTFlow(root_scale=0.14, local_scale=0.14, **CONFIGS["J2"]).eval()
    _perturb(policy)
    reference = copy.deepcopy(policy).eval()
    group = _repeat(policy.prepare(obs[:2], action[:2]), 3)
    _, records = rollout(policy, reference, group, 5, 0.7)
    ratio_err, kl_max = 0.0, 0.0
    for item in records:
        _, mean = _step_stats(policy, group, item["z"], item["t"], item["sc"], item["s"], 0.7)
        ratio_err = max(ratio_err, float((_log_prob(item["z_next"], mean, item["sigma"]) - item["logp_old"]).abs().max()))
        kl_max = max(kl_max, float(_kl(mean, item["mean_ref"], item["sigma"]).abs().max()))
    record("grpo_initial_ratio_one_and_kl_zero", ratio_err == 0.0 and kl_max == 0.0, OrderedDict([("logp_diff", ratio_err), ("kl", kl_max)]))
    item = records[1]
    normal = torch.distributions.Normal(torch.zeros(1), torch.tensor([item["sigma"]]))
    flat_next = torch.cat([value.flatten(1) for value in item["z_next"]], dim=1)
    _, mean = _step_stats(policy, group, item["z"], item["t"], item["sc"], item["s"], 0.7)
    flat_mean = torch.cat([value.flatten(1) for value in mean], dim=1)
    exact = normal.log_prob((flat_next - flat_mean).unsqueeze(-1)).squeeze(-1).mean(dim=1) - normal.log_prob(torch.zeros(1)).squeeze()
    record("grpo_log_prob_matches_normal", float((exact - _log_prob(item["z_next"], mean, item["sigma"])).abs().max()) < 1e-4)

    command = [sys.executable, "train/train_ntu2p_jitflow.py", "--device", "cpu", "--num_steps", "1", "--save_dir", os.path.join(tempfile.gettempdir(), "jitflow_cuda_check"),
               "--kin_proj", "--manifest_path", args.manifest_path, "--cache_dir", args.cache_dir]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    record("refuses_cpu_without_smoke_flag", proc.returncode != 0 and "CUDA" in proc.stdout.decode("utf-8", "replace"))

    passed = all(item["passed"] for item in results.values())
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(OrderedDict([("passed", passed), ("checks", results)]), handle, ensure_ascii=False, indent=2)
    print("ALL PASSED" if passed else "SOME CHECKS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", default="results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json")
    parser.add_argument("--cache_dir", default="results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval")
    parser.add_argument("--num", type=int, default=32)
    parser.add_argument("--output", default=None)
    sys.exit(main(parser.parse_args()))
