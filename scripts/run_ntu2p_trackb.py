"""Track B 分阶段驱动：交叉拟合折 -> 折模型 -> OOF 残差库 -> 生成器 -> 评估 -> 汇总判定 -> 审查图；test 拦截。

设计：docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md 第 4–6 节。
- 选型只在受试者留出协议（subjval，H-val 247 条）上做；summary 按 5.4 预登记流程依次判定：结构自检 -> NFE 规则
  -> τ_P 规则 -> τ_D 规则 -> A–G 与 D/E'/F' -> H（读人工审查结论 review_verdict.json，须绑定臂与选定设置）。
- NFE 规则按离散度而不是 ES 判定（ES 在最优点附近是平的，对欠散不敏感），见
  docs/ai/context/20260926-172000-ntu2p-trackb-review-fixes-and-nfe-preregistration-amendment.md。
- 应急臂按 4.8 只由对应类别的 E 失败触发（E2/E3/E4 -> foot，E6 -> inter；E1/E5 不设应急臂）；最终结论取自按固定顺序
  第一个未被拒绝的被触发臂，summary 记录 adopted_arm，原协议只允许复训与 test 这一个臂。
- 原协议（original）只在 subjval 结论为 adopt_full，或 adopt_probabilistic_only 且显式 --confirm_probabilistic_only 时
  才允许 GPU 阶段；test 只评估一次，结果已存在时拒绝重跑，NFE/τ_P/τ_D 从 subjval 的 summary 冻结读取。
- 每个 GPU 阶段开始前检查是否有 train_ntu2p_v2.py 进程（第 1/2 批）在跑，有则退出（返回码 3），除非 --allow_shared_gpu。
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from data_loaders.forecasting.ntu2p_xyz_seq_cache import DEFAULT_CACHE_DIR
from eval.eval_ntu2p_v2 import DEFAULT_BASELINE
from model.forecasting_ntu2p_resdiff import DEFAULT_NFE
from utils.ntu2p_probabilistic_metrics import SSR_PARTS, envelope_at


SAVE_ROOT = "save/forecasting/ntu120_label"
RESULTS_ROOT = "results/forecasting/ntu120_label/ntu2p_trackb"
# 第 2 批采纳手指损失去重 FD2 后的主线（docs/ai/context/20260926-165932-ntu2p-finger-dedup-result.md）；FD2 只改训练损失的关节子集，
# 结构与前向不变。summary 记录它，原协议核对一致，避免在一个底座上选定的设置被冻结给另一个底座。
MAIN_CONFIG = "A6-F-A5f0.05-A4-GH0.5-FD2"
PROTOCOLS = OrderedDict(
    [
        (
            "subjval",
            OrderedDict(
                [
                    ("manifest", "results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json"),
                    ("cache_dir", "results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval"),
                    ("deploy", SAVE_ROOT + "/ntu2p_v2subj_" + MAIN_CONFIG + "_s{seed}_10000/ema/model000010000.pt"),
                    ("baseline", SAVE_ROOT + "/ntu2p_independent_single_person_o10_p50_subjval_s0_5000/model000005000.pt"),
                    ("a0", SAVE_ROOT + "/ntu2p_v2subj_A0_s{seed}_10000/ema/model000010000.pt"),
                    ("split", "val"),
                ]
            ),
        ),
        (
            "original",
            OrderedDict(
                [
                    ("manifest", "results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json"),
                    ("cache_dir", DEFAULT_CACHE_DIR),
                    ("deploy", SAVE_ROOT + "/ntu2p_v2_" + MAIN_CONFIG + "_s{seed}_10000/ema/model000010000.pt"),
                    ("baseline", DEFAULT_BASELINE),
                    ("a0", SAVE_ROOT + "/ntu2p_v2_A0_s{seed}_10000/ema/model000010000.pt"),
                    ("split", "test"),
                ]
            ),
        ),
    ]
)
STAGES = ("folds", "fold_models", "bank", "train", "eval", "review", "summary", "all")
GPU_STAGES = ("fold_models", "bank", "train", "eval", "review")
ARMS = ("main", "insample", "foot", "inter")
# 固定顺序即预登记的优先级：两类都触发时先看 foot，不按结果挑臂。
CONTINGENCY_ARMS = ("foot", "inter")
# 应急臂的预登记权重（设计 4.8）。
ARM_TRAIN_ARGS = OrderedDict(
    [("main", []), ("insample", []), ("foot", ["--foot_loss_weight", "0.05"]), ("inter", ["--inter_loss_weight", "0.1"])]
)
# 4.8 的护栏类别 -> 应急臂：滑行/脚滑/穿地悬空 -> C1 foot，穿插 -> C2 inter；E1（结构）与 E5（jerk/高频）不设应急臂。
E_ROUTES = OrderedDict([("E1", None), ("E2", "foot"), ("E3", "foot"), ("E4", "foot"), ("E5", None), ("E6", "inter")])
# 展示模式另加的两项：GT 站定帧脚速属脚滑；双人最小距离误差属双人几何，C2 的相对关节向量损失直接约束它。
DISPLAY_EXTRA_ROUTES = OrderedDict([("E'_skate", "foot"), ("E'_min_dist", "inter")])
NUM_FOLDS = 3
GPU_BUSY_EXIT = 3
TAU_P_GRID = (1.0, 0.8)
TAU_D_GRID = (1.0, 0.8, 0.6, 0.4)
# NFE 规则（修订后预登记）：依次看 50、100，若步数加倍使 mode F τ=1 的 APD_all 或 SSR 均值（3 seed 均值）增大超过 3%，
# 说明仍有明显离散化收缩，改看下一档；都超过时用 200。
NFE_GRID = (DEFAULT_NFE, 2 * DEFAULT_NFE, 4 * DEFAULT_NFE)
NFE_DEFAULT = NFE_GRID[0]
NFE_DISPERSION_GAIN = 1.03
ADOPT_STATES = ("adopt_full", "adopt_probabilistic_only")
REJECT_STATES = ("reject", "reject_no_better_than_bootstrap")
CONCLUSIONS = ADOPT_STATES + REJECT_STATES + ("contingency_required", "pending_review")
# 预登记流程中途停止的状态（不是采纳结论）：结构自检失败 / 训练失败属实现或训练错误，须修复后重跑；NFE 规则选中更多步时待重评。
STOP_STATES = ("structural_failure", "training_failure", "pending_nfe_rerun")
REPRO_WINDOWS = 16
DIAG_INTERVAL = 5000
# 立即停止条件（设计第 8 节）：终点 teacher-forced 在 t=50 的归一化 MSE ≥ 1.0，即不如直接预测 0。
TRAIN_FAILURE_T, TRAIN_FAILURE_MSE = "50", 1.0
BUSY_SCRIPTS = ("train_ntu2p_v2.py", "eval_ntu2p_v2.py", "run_ntu2p_v2_screen.py")

_print_lock = threading.Lock()


def _say(message):
    with _print_lock:
        print(message, flush=True)


# ---------------------------------------------------------------- 路径


class Paths(object):
    def __init__(self, opts):
        self.opts = opts
        self.protocol = PROTOCOLS[opts.protocol]
        self.dir = os.path.join(opts.results_root, opts.protocol)
        self.folds = os.path.join(self.dir, "folds")

    def fold_manifest(self, fold):
        return os.path.join(self.folds, "manifest_fold{}.json".format(fold))

    def fold_cache(self, fold):
        return os.path.join(self.folds, "cache_fold{}".format(fold))

    def fold_prefix(self, fold):
        return "ntu2p_v2fold{}_{}".format(fold, self.opts.protocol)

    def fold_model(self, fold):
        steps = int(self.opts.fold_steps)
        return os.path.join(self.opts.save_root, "{}_{}_s0_{}".format(self.fold_prefix(fold), MAIN_CONFIG, steps), "ema", "model{:09d}.pt".format(steps))

    def bank(self, arm="main"):
        return os.path.join(self.dir, "insample_bank_s0.pt" if arm == "insample" else "oof_bank.pt")

    def generator_dir(self, arm, seed):
        return os.path.join(self.opts.save_root, "ntu2p_trackb_{}_{}_s{}_{}".format(self.opts.protocol, arm, seed, int(self.opts.gen_steps)))

    def generator(self, arm, seed):
        return os.path.join(self.generator_dir(arm, seed), "ema", "model{:09d}.pt".format(int(self.opts.gen_steps)))

    def deploy(self, seed):
        return self.protocol["deploy"].format(seed=seed)

    def a0(self, seed):
        return self.protocol["a0"].format(seed=seed)

    def eval_json(self, arm, seed, split, nfe):
        return os.path.join(self.dir, "eval_{}_s{}_{}_nfe{}.json".format(arm, seed, split, nfe))

    def diag(self, step):
        return os.path.join(self.dir, "diag_main_s0_step{:09d}.json".format(int(step)))

    def repro(self, tag):
        return os.path.join(self.dir, "repro{}_{}.json".format(REPRO_WINDOWS, tag))

    @property
    def review_cases(self):
        return os.path.join(self.dir, "review_cases.json")

    def review_dir(self, arm):
        return os.path.join(self.dir, "review_{}".format(arm))

    @property
    def verdict(self):
        return os.path.join(self.dir, "review_verdict.json")

    def summary(self, protocol=None):
        return os.path.join(self.opts.results_root, protocol or self.opts.protocol, "summary.json")


# ---------------------------------------------------------------- 运行与安全检查


def busy_gpu_processes():
    """正在运行的第 1/2 批 GPU 进程（train_ntu2p_v2.py，及其驱动与评估：驱动在训练间隙会跑 GPU 评估）。

    只认 python 进程的脚本参数，避免把命令行里恰好带这些字符串的 shell 算进去；本驱动的 fold_models 阶段自己也会
    启动 run_ntu2p_v2_screen.py，但检查发生在启动之前，且各阶段串行，不会误伤。
    """
    try:
        output = subprocess.run(["pgrep", "-af", "ntu2p_v2"], stdout=subprocess.PIPE, universal_newlines=True).stdout
    except OSError:
        return []
    found = []
    own = {os.getpid(), os.getppid()}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 3 or int(parts[0]) in own:
            continue
        if "python" in os.path.basename(parts[1]) and any(os.path.basename(arg) in BUSY_SCRIPTS for arg in parts[2:]):
            found.append(line)
    return found


def check_gpu(opts, stage):
    if opts.dry_run or opts.allow_shared_gpu or stage not in GPU_STAGES:
        return
    busy = busy_gpu_processes()
    if busy:
        _say("GPU 被第 1/2 批占用（{} 个 {} 进程），拒绝启动 {}；确认可共享时加 --allow_shared_gpu".format(len(busy), "/".join(BUSY_SCRIPTS), stage))
        for line in busy[:5]:
            _say("  " + line)
        sys.exit(GPU_BUSY_EXIT)


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = "." + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OMP_NUM_THREADS", "2")
    return env


def run_command(command, log_path, opts):
    text = " ".join(command)
    if opts.dry_run:
        _say("[dry_run] " + text + ("  > " + log_path if log_path else ""))
        return 0
    _say("[run] " + text)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a") as handle:
            code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT, env=_env())
    else:
        code = subprocess.call(command, env=_env())
    if code != 0:
        raise RuntimeError("命令失败（返回码 {}）：{}".format(code, text))
    return code


def run_parallel(jobs, opts, workers):
    """jobs: [(command, log_path)]；并发执行，任一失败则整体失败。"""
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(run_command, command, log, opts) for command, log in jobs]
        for future in futures:
            try:
                future.result()
            except Exception as error:  # 其它任务照常跑完，结束时统一报告。
                failures.append(repr(error))
    if failures:
        raise RuntimeError("；".join(failures))


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _write(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def subjval_decision(opts):
    path = os.path.join(opts.results_root, "subjval", "summary.json")
    if not os.path.exists(path):
        return None, "subjval 的 summary.json 不存在：{}".format(path)
    summary = _load(path)
    conclusion = summary.get("conclusion")
    if conclusion in ADOPT_STATES and summary.get("adopted_arm") not in ARMS:
        return None, "subjval summary 未记录 adopted_arm（旧格式），须重跑 subjval summary"
    if conclusion in ADOPT_STATES and summary.get("base_config") != MAIN_CONFIG:
        return None, "subjval summary 的底座 {} 与当前主线 {} 不一致，须在当前主线上重做 subjval 选型".format(summary.get("base_config"), MAIN_CONFIG)
    if conclusion == "adopt_full":
        return summary, None
    if conclusion == "adopt_probabilistic_only" and opts.confirm_probabilistic_only:
        return summary, None
    if conclusion == "adopt_probabilistic_only":
        return None, "subjval 结论为 adopt_probabilistic_only，原协议复训与 test 须用户确认（--confirm_probabilistic_only）"
    return None, "subjval 结论为 {}，不允许原协议复训与 test".format(conclusion)


def check_original_allowed(opts, stage):
    if opts.protocol != "original" or stage not in GPU_STAGES:
        return None
    summary, reason = subjval_decision(opts)
    if reason is not None:
        if opts.dry_run and not opts.test:
            _say("[dry_run] 注意：{}（正式运行时会被拒绝）".format(reason))
            return None
        _say("拒绝：" + reason)
        sys.exit(2)
    return summary


def resolve_arms(opts, stage):
    """各阶段实际使用的臂。--arms 缺省时：subjval 为 main（review 为 summary 的 decision_arm）；原协议为 subjval 采纳的臂。

    原协议的 train/eval/summary 只允许 subjval 采纳的那一个臂：否则会复训并 test 一个未被采纳的臂，或依次 test 多个臂再挑。
    """
    explicit = None if opts.arms is None else list(opts.arms)
    if opts.protocol == "original":
        if stage not in ("train", "eval", "summary"):
            return explicit or ["main"]
        summary, _ = subjval_decision(opts)
        adopted = summary.get("adopted_arm") if summary else None
        if adopted is None:
            # 没有采纳结论时 GPU 阶段、test 与 summary 另有拦截；这里只给 dry_run 一个可打印的臂。
            return explicit or ["main"]
        if explicit is not None and explicit != [adopted]:
            _say("拒绝：原协议只允许 subjval 采纳的臂 {}，当前 --arms {}".format(adopted, explicit))
            sys.exit(2)
        return [adopted]
    if stage == "review":
        # in-sample 是诊断臂，从不审查；--stage all --arms main insample 时审查 main。
        arms = [arm for arm in (explicit or []) if arm != "insample"]
        if not arms:
            path = os.path.join(opts.results_root, opts.protocol, "summary.json")
            arms = [_load(path).get("decision_arm", "main")] if os.path.exists(path) else ["main"]
        return arms
    return explicit or ["main"]


def triggered_contingency_arms(paths):
    """subjval 主臂判定触发的应急臂（主臂结论为 contingency_required 时才非空）。"""
    if not os.path.exists(paths.summary()):
        return []
    main = _load(paths.summary()).get("arms", {}).get("main", {})
    return list(main.get("contingency_arms", [])) if main.get("conclusion") == "contingency_required" else []


def check_contingency_triggered(opts, paths):
    """subjval 上应急臂只在主臂对应类别的 E 失败触发后才允许训练与评估（4.8：按失败类别各跑一轮，未触发的不跑）。"""
    requested = [arm for arm in opts.arms if arm in CONTINGENCY_ARMS]
    if opts.protocol != "subjval" or not requested:
        return
    triggered = triggered_contingency_arms(paths)
    untriggered = [arm for arm in requested if arm not in triggered]
    if not untriggered:
        return
    reason = "应急臂 {} 未被主臂的 E 失败类别触发（当前触发：{}）".format(untriggered, triggered or "无")
    if opts.dry_run:
        _say("[dry_run] 注意：{}（正式运行时会被拒绝）".format(reason))
        return
    _say("拒绝：" + reason)
    sys.exit(2)


# ---------------------------------------------------------------- 阶段


def stage_folds(opts, paths):
    if os.path.exists(os.path.join(paths.folds, "folds.json")) and not opts.dry_run:
        _say("skip folds（已存在 {}）".format(paths.folds))
        return
    command = [sys.executable, "scripts/build_ntu2p_crossfit_folds.py", "--source_manifest", paths.protocol["manifest"],
               "--source_cache_dir", paths.protocol["cache_dir"], "--output_dir", paths.folds, "--num_folds", str(NUM_FOLDS),
               "--protocol_tag", opts.protocol]
    run_command(command, os.path.join(paths.dir, "logs", "folds.log"), opts)


def stage_fold_models(opts, paths):
    jobs = []
    for fold in range(NUM_FOLDS):
        command = [sys.executable, "scripts/run_ntu2p_v2_screen.py", "--stage", "1", "--configs", MAIN_CONFIG, "--seeds", "0",
                   "--steps", str(int(opts.fold_steps)), "--workers", "1", "--manifest_path", paths.fold_manifest(fold),
                   "--cache_dir", paths.fold_cache(fold), "--baseline_checkpoint", paths.protocol["baseline"],
                   "--run_prefix", paths.fold_prefix(fold), "--summary_dir", os.path.join(paths.folds, "summary_fold{}".format(fold)),
                   "--device", opts.device]
        jobs.append((command, os.path.join(paths.dir, "logs", "fold_model{}.log".format(fold))))
    run_parallel(jobs, opts, NUM_FOLDS)


def stage_bank(opts, paths):
    jobs = []
    if opts.dry_run or not os.path.exists(paths.bank("main")):
        command = [sys.executable, "scripts/build_ntu2p_oof_residual_bank.py", "--parent_manifest", paths.protocol["manifest"],
                   "--parent_cache_dir", paths.protocol["cache_dir"], "--folds_dir", paths.folds,
                   "--fold_checkpoints"] + [paths.fold_model(f) for f in range(NUM_FOLDS)] + [
                   "--output", paths.bank("main"), "--protocol", opts.protocol, "--device", opts.device]
        jobs.append((command, os.path.join(paths.dir, "logs", "oof_bank.log")))
    if "insample" in opts.arms:
        if opts.protocol != "subjval":
            raise ValueError("in-sample 对照只在 subjval 上跑")
        if opts.dry_run or not os.path.exists(paths.bank("insample")):
            command = [sys.executable, "scripts/build_ntu2p_oof_residual_bank.py", "--parent_manifest", paths.protocol["manifest"],
                       "--parent_cache_dir", paths.protocol["cache_dir"], "--folds_dir", paths.folds,
                       "--insample_checkpoint", paths.deploy(0), "--output", paths.bank("insample"), "--protocol", opts.protocol,
                       "--device", opts.device]
            jobs.append((command, os.path.join(paths.dir, "logs", "insample_bank.log")))
    for command, log in jobs:
        run_command(command, log, opts)


def _arm_seeds(arm, opts):
    # in-sample 对照只跑 seed 0（诊断）。
    return [0] if arm == "insample" else list(opts.seeds)


def stage_train(opts, paths):
    check_contingency_triggered(opts, paths)
    jobs = []
    for arm in opts.arms:
        for seed in _arm_seeds(arm, opts):
            if os.path.exists(paths.generator(arm, seed)) and not opts.dry_run:
                _say("skip train {} s{}（EMA 终点已存在）".format(arm, seed))
                continue
            command = [sys.executable, "train/train_ntu2p_resdiff.py", "--bank", paths.bank(arm), "--save_dir", paths.generator_dir(arm, seed),
                       "--protocol", opts.protocol, "--arm", arm, "--seed", str(seed), "--num_steps", str(int(opts.gen_steps)),
                       "--device", opts.device] + ARM_TRAIN_ARGS[arm]
            if arm in ("foot", "inter"):
                command += ["--parent_cache_dir", paths.protocol["cache_dir"]]
            jobs.append((command, os.path.join(paths.dir, "logs", "train_{}_s{}.log".format(arm, seed))))
    run_parallel(jobs, opts, max(1, len(jobs)))


def _eval_command(opts, paths, arm, seed, split, output, settings=None, nfe=None, extra=None, checkpoint=None):
    deploy = paths.deploy(seed)
    command = [sys.executable, "eval/eval_ntu2p_resdiff.py", "--checkpoint", checkpoint or paths.generator(arm, seed), "--base_checkpoint", deploy,
               "--bank", paths.bank("main"), "--manifest_path", paths.protocol["manifest"], "--cache_dir", paths.protocol["cache_dir"],
               "--split", split, "--baseline_checkpoint", paths.protocol["baseline"], "--output", output, "--device", opts.device,
               "--nfe", str(int(nfe or opts.nfe)), "--batch_windows", str(int(opts.eval_batch_windows))]
    if os.path.exists(paths.a0(seed)) or opts.dry_run:
        command += ["--a0_checkpoint", paths.a0(seed)]
    if split == "val":
        # 3 seed 均值是设计第 9 节的旁支（未立项的确定性变体），只在 val 上诊断；test 上不算，保留旁支将来干净的 test。
        command += ["--ensemble_checkpoints"] + [paths.deploy(s) for s in (0, 1, 2)]
    if settings:
        command += ["--settings", settings]
    return command + list(extra or [])


def frozen_settings(summary):
    """被采纳臂在 H-val 上确定的设置（summary 的 selected 取自 adopted_arm）。"""
    if summary.get("adopted_arm") != summary.get("decision_arm"):
        raise ValueError("summary 的 adopted_arm {} 与 decision_arm {} 不一致".format(summary.get("adopted_arm"), summary.get("decision_arm")))
    selected = summary["selected"]
    settings = "F:{}".format(float(selected["tau_P"]))
    if selected.get("tau_D") is not None:
        settings += ",R:{}".format(float(selected["tau_D"]))
    return settings, int(selected["nfe"])


def stage_eval(opts, paths):
    jobs = []
    if opts.protocol == "original":
        if not opts.test:
            _say("original 协议的评估只有 test 一次：需显式 --test（并满足 subjval 采纳结论）")
            return
        summary, reason = subjval_decision(opts)
        if reason is not None:
            _say("拒绝 --test：" + reason)
            sys.exit(2)
        settings, nfe = frozen_settings(summary)
        missing = [paths.generator(arm, seed) for arm in opts.arms for seed in _arm_seeds(arm, opts) if not os.path.exists(paths.generator(arm, seed))]
        if missing and not opts.dry_run:
            _say("拒绝 --test：原协议生成器 EMA 终点不全：{}".format(missing))
            sys.exit(2)
        for arm in opts.arms:
            for seed in _arm_seeds(arm, opts):
                output = paths.eval_json(arm, seed, "test", nfe)
                if os.path.exists(output):
                    _say("拒绝 --test：test 结果已存在（只评估一次）：{}".format(output))
                    sys.exit(2)
                command = _eval_command(opts, paths, arm, seed, "test", output, settings=settings, nfe=nfe, extra=["--nfe_scan", ""])
                jobs.append((command, os.path.join(paths.dir, "logs", "eval_test_{}_s{}.log".format(arm, seed))))
        run_parallel(jobs, opts, opts.eval_workers)
        return
    if opts.test:
        _say("--test 只允许 --protocol original")
        sys.exit(2)
    check_contingency_triggered(opts, paths)
    for arm in opts.arms:
        for seed in _arm_seeds(arm, opts):
            output = paths.eval_json(arm, seed, "val", opts.nfe)
            if os.path.exists(output) and not opts.dry_run:
                _say("skip eval {} s{}（{} 已存在）".format(arm, seed, output))
                continue
            extra = []
            if arm != "insample" and seed == 0:
                # 训练失败停止条件（teacher-forced t=50）对每个被判定的臂都适用。
                extra = ["--diagnostics"]
            if arm == "main" and seed == 0:
                extra += ["--write_review_cases", paths.review_cases]
            command = _eval_command(opts, paths, arm, seed, "val", output, extra=extra)
            jobs.append((command, os.path.join(paths.dir, "logs", "eval_{}_s{}_nfe{}.log".format(arm, seed, opts.nfe))))
    run_parallel(jobs, opts, opts.eval_workers)
    if "main" in opts.arms:
        # 结构自检的复跑：seed 0 前 16 窗跑两次，样本摘要须逐位相同。
        for tag in ("a", "b"):
            output = paths.repro(tag)
            if os.path.exists(output) and not opts.dry_run:
                continue
            command = _eval_command(opts, paths, "main", 0, "val", output, extra=["--max_samples", str(REPRO_WINDOWS), "--nfe_scan", "",
                                                                                 "--skip_naturalness", "--settings", "F:1.0,R:1.0"])
            run_command(command, os.path.join(paths.dir, "logs", "repro.log"), opts)
        # 训练诊断（4.7，不参与选择）：seed 0 中间 EMA checkpoint 的 teacher-forced 分 t 误差。
        for step in range(DIAG_INTERVAL, int(opts.gen_steps), DIAG_INTERVAL):
            checkpoint = os.path.join(paths.generator_dir("main", 0), "ema", "model{:09d}.pt".format(step))
            output = paths.diag(step)
            if (os.path.exists(output) or not os.path.exists(checkpoint)) and not opts.dry_run:
                continue
            command = _eval_command(opts, paths, "main", 0, "val", output, checkpoint=checkpoint, settings="F:1.0",
                                    extra=["--nfe_scan", "", "--skip_naturalness", "--diagnostics", "--boot_scales", "0"])
            run_command(command, os.path.join(paths.dir, "logs", "diag.log"), opts)


VERDICT_KEYS = ("arm", "nfe", "tau_P", "tau_D")


def verdict_binding(arm, selected):
    return OrderedDict([("arm", arm), ("nfe", selected.get("nfe")), ("tau_P", selected.get("tau_P")), ("tau_D", selected.get("tau_D"))])


def stage_review(opts, paths):
    if opts.protocol != "subjval":
        _say("审查图只在 subjval 上做")
        return
    if len(opts.arms) != 1:
        _say("review 一次只审查一个臂（--arms 只能给一个，in-sample 除外），当前 {}".format(opts.arms))
        sys.exit(2)
    arm = opts.arms[0]
    summary_path = paths.summary()
    if os.path.exists(summary_path):
        decision = _load(summary_path).get("arms", {}).get(arm)
        if decision is None:
            _say("跳过审查：summary 中没有臂 {} 的判定".format(arm))
            return
    elif opts.dry_run:
        decision = {"conclusion": "pending_review", "selected": {"tau_P": "<τ_P>", "tau_D": "<τ_D>", "nfe": opts.nfe}}
    else:
        _say("review 需要先运行 summary 以确定 τ_P/τ_D：{}".format(summary_path))
        sys.exit(2)
    selected = decision["selected"]
    if decision["conclusion"] in STOP_STATES or selected.get("tau_P") is None:
        # 停止状态下 τ_P 未定，H 无从判定；审查只会白跑一次 GPU 评估。
        _say("跳过审查：臂 {} 的结论为 {}（τ_P={}），先按 summary 的提示处理".format(arm, decision["conclusion"], selected.get("tau_P")))
        return
    review_dir = paths.review_dir(arm)
    settings = ["F:{}".format(selected["tau_P"])]
    if selected.get("tau_D") is not None:
        settings.append("R:{}".format(selected["tau_D"]))
    settings = ",".join(settings)
    output = os.path.join(review_dir, "eval_review_s0.json")
    command = _eval_command(opts, paths, arm, 0, "val", output, settings=settings, nfe=selected.get("nfe", opts.nfe),
                            extra=["--review_cases", paths.review_cases, "--export_review", review_dir, "--skip_naturalness",
                                   "--nfe_scan", "", "--boot_scales", "1.0", "--review_settings", settings])
    run_command(command, os.path.join(paths.dir, "logs", "review_{}.log".format(arm)), opts)
    arrays = os.path.join(review_dir, "review_arrays.pt")
    count = len(_load(paths.review_cases)["cases"]) if os.path.exists(paths.review_cases) else 18
    indices = ",".join(str(i) for i in range(count))
    render_log = os.path.join(paths.dir, "logs", "review_render_{}.log".format(arm))
    sheet = [sys.executable, "sample/render_ntu2p_review_sheet.py", "--arrays", arrays, "--output_dir", os.path.join(review_dir, "sheets"),
             "--indices", indices, "--num_random", "0", "--num_walk", "0"]
    run_command(sheet, render_log, opts)
    for method in ("F_k0", "R_k0") if selected.get("tau_D") is not None else ("F_k0",):
        video = [sys.executable, "sample/render_ntu2p_review_sheet.py", "--arrays", arrays, "--output_dir",
                 os.path.join(review_dir, "video_" + method), "--indices", indices, "--num_random", "0", "--num_walk", "0",
                 "--methods", method, "--video"]
        run_command(video, render_log, opts)
    fan = [sys.executable, "sample/plot_ntu2p_gait_fan.py", "--samples", os.path.join(review_dir, "review_samples.pt"),
           "--cases", paths.review_cases, "--output_dir", os.path.join(review_dir, "gait_fan"), "--settings", settings]
    run_command(fan, render_log, opts)
    template = os.path.join(review_dir, "review_verdict_template.json")
    if not opts.dry_run:
        _write(template, OrderedDict(list(verdict_binding(arm, selected).items()) + [("pass", None), ("notes", "")]))
    _say("{}审查图目录：{}；看图后把 {} 复制为 {} 并填写 pass（true/false）与 notes；arm/nfe/tau_P/tau_D 须保持不变，"
         "与 summary 当前判定不一致时 H 视为缺失。再运行 --stage summary".format("[dry_run] " if opts.dry_run else "", review_dir, template, paths.verdict))


# ---------------------------------------------------------------- 汇总与预登记判定


def _mean(values):
    values = [float(v) for v in values]
    return statistics.mean(values) if values else float("nan")


def _std(values):
    values = [float(v) for v in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _gen(result, mode, tau):
    return result["generator"][mode][str(float(tau))]


def _boot_curve(result, mode, key_fn):
    points = []
    for scale, block in sorted(result["bootstrap"][mode].items(), key=lambda item: float(item[0])):
        points.append((block["single_l2"]["mpjpe"], key_fn(block)))
    return [p[0] for p in points], [p[1] for p in points]


def structural_checks(evals, repro):
    """5.4 第 1 步：任何判据之前的结构自检；失败即实现错误。"""
    reasons = []
    for seed, result in evals.items():
        for mode, per_mode in result["generator"].items():
            for tau, block in per_mode.items():
                tag = "s{} {}:{}".format(seed, mode, tau)
                if block["first_step_error_max"] > 1e-6:
                    reasons.append("{} first_step_error_max={:.2e} > 1e-6".format(tag, block["first_step_error_max"]))
                if block.get("bone_rel_err_body_mean", 0.0) > 1e-5:
                    reasons.append("{} bone_rel_err_body={:.2e} > 1e-5".format(tag, block["bone_rel_err_body_mean"]))
                if not block["all_finite"]:
                    reasons.append("{} 样本含非有限值".format(tag))
                if mode == "R" and block["pelvis_equal_v2"] is not True:
                    reasons.append("{} mode R 的 pelvis 与 v2 不逐位相同".format(tag))
    if repro is None:
        reasons.append("缺少 seed 0 前 {} 窗的两次复跑结果".format(REPRO_WINDOWS))
    else:
        first, second = repro
        for mode, per_mode in first["generator"].items():
            for tau, block in per_mode.items():
                if block["sample_digest"] != second["generator"][mode][tau]["sample_digest"]:
                    reasons.append("复跑 {}:{} 样本摘要不同（不可复现）".format(mode, tau))
    return not reasons, reasons


def _dispersion(block):
    """(APD_all, root/腿/臂 SSR 均值)。"""
    return block["apd"]["all"], _mean(block["ssr_clean"][part] for part in SSR_PARTS)


def _ratio(high, low):
    # 零离散（零初始化模型）或 K<2 时比值无意义，按"无收缩"处理，不触发加步。
    return high / low if low > 0 and high == high and low == low else 1.0


def nfe_rule(evals_default):
    """mode F τ=1 按离散度选 NFE（修订后的预登记规则）：依次看 N=50、100，2N 相对 N 的 APD_all 或 SSR 均值之比
    （3 seed 均值）≤ 1.03 即用 N，否则看下一档，都超过时用 200。

    不用 ES：ES/CRPS 在最优点附近是平的，离散度比 0.87 -> 0.94 时 CRPS 只改善 0.3%，ES 阈值几乎不会触发。
    返回 (nfe, 原因, 各档比值)；扫描只取自 NFE_DEFAULT 那次评估。
    """
    ratios = OrderedDict()
    for low, high in zip(NFE_GRID[:-1], NFE_GRID[1:]):
        apd, ssr = [], []
        for result in evals_default.values():
            scan = result.get("nfe_scan", {})
            low_block = _gen(result, "F", 1.0) if low == NFE_DEFAULT else scan.get(str(low))
            high_block = scan.get(str(high))
            if low_block is None or high_block is None:
                return low, "缺 NFE{} 扫描，按 NFE{}".format(high if high_block is None else low, low), ratios
            (apd_low, ssr_low), (apd_high, ssr_high) = _dispersion(low_block), _dispersion(high_block)
            apd.append(_ratio(apd_high, apd_low))
            ssr.append(_ratio(ssr_high, ssr_low))
        ratios["{}/{}".format(high, low)] = OrderedDict([("apd_all", _mean(apd)), ("ssr_mean", _mean(ssr))])
        gain = max(_mean(apd), _mean(ssr))
        if gain <= NFE_DISPERSION_GAIN:
            return low, "NFE{}/{} 离散度比 APD {:.3f}、SSR {:.3f} ≤ {}".format(high, low, _mean(apd), _mean(ssr), NFE_DISPERSION_GAIN), ratios
    return NFE_GRID[-1], "NFE{}/{} 离散度比仍 > {}，取最大档".format(NFE_GRID[-1], NFE_GRID[-2], NFE_DISPERSION_GAIN), ratios


def _nfe_table(evals_default):
    """NFE 扫描表：离散度（APD、SSR）与 ES、单样本 mpjpe 并列，NFE 的选择不只看 ES。"""
    header = ["seed", "NFE", "ES_joint", "APD_all", "SSR r/l/a", "单样本 mpjpe"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for seed, result in evals_default.items():
        blocks = OrderedDict(
            sorted([(NFE_DEFAULT, _gen(result, "F", 1.0))] + [(int(k), v) for k, v in result.get("nfe_scan", {}).items()], key=lambda item: item[0])
        )
        for nfe, block in blocks.items():
            lines.append("| {} | {} | {:.4f} | {:.4f} | {} | {:.4f} |".format(
                seed, nfe, block["es_joint"], block["apd"]["all"], "/".join("{:.2f}".format(block["ssr_clean"][p]) for p in SSR_PARTS),
                block["single_l2"]["mpjpe"]))
    return lines


def tau_p_rule(evals):
    per_part = OrderedDict((part, _mean(_gen(r, "F", 1.0)["ssr_clean"][part] for r in evals.values())) for part in ("root", "legs", "arms"))
    median = statistics.median(per_part.values())
    return (0.8 if median > 1.4 else 1.0), per_part, median


def tau_d_rule(evals):
    for tau in TAU_D_GRID:
        ok = True
        for result in evals.values():
            if str(float(tau)) not in result["generator"].get("R", {}):
                ok = False
                break
            gen, v2 = _gen(result, "R", tau)["single_l2"], result["references"]["v2"]["single_l2"]
            ok = ok and gen["mpjpe"] <= 1.05 * v2["mpjpe"] and gen["xyz_mse"] <= 1.10 * v2["xyz_mse"]
        if ok:
            return tau
    return None


def _natural(block):
    return block.get("naturalness_single") or {}


def e_checks(evals, mode, tau, display=False):
    """E1–E6（display 时追加 E' 的两项）；v2 取同 seed 部署底座，比较 3 seed 均值。"""
    def mean_gen(fn):
        return _mean(fn(_gen(r, mode, tau)) for r in evals.values())

    def mean_v2(fn):
        return _mean(fn(r["references"]["v2"]) for r in evals.values())

    def nat(key):
        return lambda block: _natural(block).get(key, float("nan"))

    checks = OrderedDict()
    first = max(_gen(r, mode, tau)["first_step_error_max"] for r in evals.values())
    checks["E1"] = (mean_gen(nat("bone_rel_err_body")) <= 1e-5 and first <= 1e-6,
                    "bone_rel_err_body={:.2e}, first_step_max={:.2e}".format(mean_gen(nat("bone_rel_err_body")), first))
    for name, key, ratio in (("E2", "skate_self_ratio_walk", 1.15), ("E3", "slide_frame_ratio", 1.10)):
        value, ref = mean_gen(nat(key)), mean_v2(nat(key))
        checks[name] = (value <= ratio * ref, "{}={:.4f} vs {:.2f}×v2={:.4f}".format(key, value, ratio, ratio * ref))
    pen, pen_ref = mean_gen(nat("foot_penetration_ratio")), mean_v2(nat("foot_penetration_ratio"))
    flo, flo_ref = mean_gen(nat("foot_float_ratio")), mean_v2(nat("foot_float_ratio"))
    checks["E4"] = (pen <= pen_ref + 0.01 and flo <= flo_ref + 0.01, "穿地 {:.4f}（v2 {:.4f}）悬空 {:.4f}（v2 {:.4f}）".format(pen, pen_ref, flo, flo_ref))
    jerk, jerk_ref = mean_gen(nat("jerk_body")), mean_v2(nat("jerk_body"))
    high = mean_gen(lambda b: b["articulation_single"]["dct_high_energy_ratio_to_target"])
    high_ref = mean_v2(lambda b: b["articulation_single"]["dct_high_energy_ratio_to_target"])
    checks["E5"] = (jerk <= 1.5 * jerk_ref and high <= max(1.2 * high_ref, 0.05),
                    "jerk {:.1f}（1.5×v2 {:.1f}）dct_high {:.4f}（上限 {:.4f}）".format(jerk, 1.5 * jerk_ref, high, max(1.2 * high_ref, 0.05)))
    inter, inter_ref = mean_gen(lambda b: b["interpenetration_ratio"]), mean_v2(lambda b: b["interpenetration_ratio"])
    checks["E6"] = (inter <= inter_ref + 0.01, "穿插帧比 {:.4f}（v2 {:.4f}）".format(inter, inter_ref))
    if display:
        for name, key, ratio in (("E'_skate", "skate_gt_contact_speed_walk", 1.5), ("E'_min_dist", "min_interperson_dist_abs_err", 1.10)):
            value, ref = mean_gen(nat(key)), mean_v2(nat(key))
            checks[name] = (value <= ratio * ref, "{}={:.4f} vs {:.2f}×v2={:.4f}".format(key, value, ratio, ratio * ref))
    return checks


def report_only_naturalness(evals, tau):
    """P 模式只报告的三项：root 被采样时它们必然随精度升高，不作护栏。"""
    parts = []
    for key in ("skate_gt_contact_speed_walk", "min_interperson_dist_abs_err", "root_distance_abs_err"):
        value = _mean(_natural(_gen(r, "F", tau)).get(key, float("nan")) for r in evals.values())
        ref = _mean(_natural(r["references"]["v2"]).get(key, float("nan")) for r in evals.values())
        parts.append("{} {:.4f}（v2 {:.4f}）".format(key, value, ref))
    return "；".join(parts)


def f_checks(evals, mode, tau):
    def mean_gen(fn):
        return _mean(fn(_gen(r, mode, tau)) for r in evals.values())

    def mean_v2(fn):
        return _mean(fn(r["references"]["v2"]) for r in evals.values())

    checks = OrderedDict()
    gt_mean = mean_v2(lambda b: b["steps"]["gt_mean"])
    gap, gap_ref = abs(mean_gen(lambda b: b["steps"]["mean"]) - gt_mean), abs(mean_v2(lambda b: b["steps"]["mean"]) - gt_mean)
    w1, w1_ref = mean_gen(lambda b: b["steps"]["w1"]), mean_v2(lambda b: b["steps"]["w1"])
    checks["F1"] = (gap <= 0.7 * gap_ref and w1 <= 0.8 * w1_ref, "步数差 {:.3f}（v2 {:.3f}）W1 {:.3f}（v2 {:.3f}）".format(gap, gap_ref, w1, w1_ref))
    entropy = mean_gen(lambda b: b["lead_foot"]["entropy_bits"])
    acc, acc_ref = mean_gen(lambda b: b["lead_foot"]["acc_single"]), mean_v2(lambda b: b["lead_foot"]["acc_single"])
    checks["F2"] = (entropy >= 0.5 and acc >= acc_ref - 0.10, "熵 {:.3f} bit，单样本正确率 {:.3f}（v2 {:.3f}）".format(entropy, acc, acc_ref))
    corr = mean_gen(lambda b: _natural(b).get("gait_sep_corr_f01_10", float("nan")))
    corr_ref = mean_v2(lambda b: _natural(b).get("gait_sep_corr_f01_10", float("nan")))
    checks["F3"] = (corr >= corr_ref - 0.05, "gait_sep_corr_f01_10 {:.3f}（v2 {:.3f}）".format(corr, corr_ref))
    bias, bias_ref = mean_gen(lambda b: b["arm_amplitude"]["strike_abs_log_bias"]), mean_v2(lambda b: b["arm_amplitude"]["strike_abs_log_bias"])
    median = mean_gen(lambda b: b["arm_amplitude"]["strike_median_ratio"])
    checks["F4"] = (bias <= 0.7 * bias_ref and median <= 1.3, "击打 |E log| {:.3f}（v2 {:.3f}）中位比 {:.3f}".format(bias, bias_ref, median))
    static, static_ref = mean_gen(lambda b: b["steps"]["static_mean"]), mean_v2(lambda b: b["steps"]["static_mean"])
    allocation = mean_gen(lambda b: b["allocation_ratio"])
    checks["F5"] = (static <= static_ref + 0.2 and allocation >= 3.0, "静止人步数 {:.3f}（v2 {:.3f}）分配比 {:.2f}".format(static, static_ref, allocation))
    return checks


def _num(value):
    return float("nan") if value is None else float(value)


def _envelope_point(result, fn, tau):
    """(生成器指标, bootstrap-F 曲线在生成器单样本 mpjpe 处的插值, 是否在网格内)；指标缺失时为 nan（判据自然不通过）。"""
    xs, ys = _boot_curve(result, "F", lambda block: _num(fn(block)))
    try:
        env, inside = envelope_at(xs, ys, _gen(result, "F", tau)["single_l2"]["mpjpe"])
    except ValueError:
        env, inside = float("nan"), False
    return _num(fn(_gen(result, "F", tau))), env, inside


def a_checks(evals, tau):
    checks = OrderedDict()
    specs = (
        ("A1", lambda b: b["es_joint"], 0.98, True),
        ("A2", lambda b: b["min_ade@10"], 0.97, True),
        ("A3", lambda b: b["es_legs_walk"], 0.95, False),
        ("A5", lambda b: b["es_arms_act"], 0.97, False),
    )
    for name, fn, factor, per_seed in specs:
        points = [_envelope_point(result, fn, tau) for result in evals.values()]
        values, envelopes = [p[0] for p in points], [p[1] for p in points]
        outside = not all(p[2] for p in points)
        ok = all(v <= factor * e for v, e in zip(values, envelopes)) if per_seed else _mean(values) <= factor * _mean(envelopes)
        note = "{} vs {:.2f}×包络 {}".format(["{:.4f}".format(v) for v in values], factor, ["{:.4f}".format(e) for e in envelopes])
        checks[name] = (ok, note + ("（超出网格，外推）" if outside else ""))
    points = [_envelope_point(result, lambda b: b["lead_foot"]["brier3"], tau) for result in evals.values()]
    values, envelopes = [p[0] for p in points], [p[1] for p in points]
    outside = not all(p[2] for p in points)
    v2_brier = [_num(result["references"]["v2"]["lead_foot"]["brier3"]) for result in evals.values()]
    ok = _mean(values) <= _mean(envelopes) - 0.02 and _mean(values) < _mean(v2_brier)
    checks["A4"] = (ok, "Brier {:.4f} vs 包络−0.02 {:.4f}，v2 {:.4f}{}".format(_mean(values), _mean(envelopes) - 0.02, _mean(v2_brier),
                                                                          "（超出网格，外推）" if outside else ""))
    return checks


def b_c_checks(evals, tau):
    checks = OrderedDict()
    one = [_gen(r, "F", tau)["one_step_mean_l2"] for r in evals.values()]
    v2 = [r["references"]["v2"]["single_l2"] for r in evals.values()]
    mean_k = [_gen(r, "F", tau)["mean_of_k_l2"] for r in evals.values()]
    ok = (_mean(o["mpjpe"] for o in one) <= 1.01 * _mean(v["mpjpe"] for v in v2)
          and _mean(o["xyz_mse"] for o in one) <= 1.02 * _mean(v["xyz_mse"] for v in v2)
          and _mean(m["mpjpe"] for m in mean_k) <= 1.04 * _mean(v["mpjpe"] for v in v2))
    checks["B"] = (ok, "单步均值 mpjpe {:.4f} xyz_mse {:.5f}；mean-of-K mpjpe {:.4f}；v2 {:.4f}/{:.5f}".format(
        _mean(o["mpjpe"] for o in one), _mean(o["xyz_mse"] for o in one), _mean(m["mpjpe"] for m in mean_k),
        _mean(v["mpjpe"] for v in v2), _mean(v["xyz_mse"] for v in v2)))
    coverage = [(_num(_gen(r, "F", tau)["min_ade@10"]), r["references"]["v2"]["single_l2"]["mpjpe"]) for r in evals.values()]
    checks["C"] = (all(m <= 0.92 * v for m, v in coverage), "minADE@10（GT 选样上界）/ v2 mpjpe：{}".format(
        ["{:.3f}".format(m / v) for m, v in coverage]))
    sane = True
    for result in evals.values():
        gen = _gen(result, "F", tau)["single_l2"]
        for ref in ("copy_last", "independent_base"):
            sane = sane and all(gen[key] < result["references"][ref]["single_l2"][key] for key in ("xyz_mse", "xyz_mae", "mpjpe"))
    checks["single_sanity"] = (sane, "单样本三项 L2 低于 copy-last 与独立 base（3/3）")
    return checks


def g_check(evals, tau):
    parts = OrderedDict((part, _mean(_gen(r, "F", tau)["ssr_clean"][part] for r in evals.values())) for part in ("root", "legs", "arms"))
    ok = all(0.6 <= value <= 1.4 for value in parts.values())
    return OrderedDict([("G", (ok, "SSR " + ", ".join("{} {:.3f}".format(k, v) for k, v in parts.items())))])


def d_checks(evals, tau_d):
    checks = OrderedDict()
    if tau_d is None:
        checks["D"] = (False, "τ_D 不存在（R 模式所有温度都超出单样本 L2 护栏）")
        return checks
    beats = all(all(_gen(r, "R", tau_d)["single_l2"][k] < r["references"]["copy_last"]["single_l2"][k] for k in ("xyz_mse", "xyz_mae", "mpjpe"))
                for r in evals.values())
    low = _mean(_gen(r, "R", tau_d)["articulation_single"]["dct_low_energy_ratio_to_target"] for r in evals.values())
    mid = _mean(_gen(r, "R", tau_d)["articulation_single"]["dct_mid_energy_ratio_to_target"] for r in evals.values())
    apd_d = _mean(_gen(r, "R", tau_d)["apd"]["legs_local"] for r in evals.values())
    apd_1 = _mean(_gen(r, "R", 1.0)["apd"]["legs_local"] for r in evals.values())
    ok = beats and low >= 0.40 and mid >= 0.10 and apd_d >= 0.5 * apd_1
    checks["D"] = (ok, "τ_D={}；低于 copy-last {}；dct_low {:.3f} dct_mid {:.3f}；腿 APD {:.4f} vs 0.5×τ1 {:.4f}".format(
        tau_d, beats, low, mid, apd_d, 0.5 * apd_1))
    return checks


def _passed(checks, names=None):
    return all(ok for name, (ok, _) in checks.items() if names is None or name in names)


def _same_value(got, expected):
    if expected is None or got is None:
        return got is None and expected is None
    if isinstance(expected, str):
        return got == expected
    try:
        return abs(float(got) - float(expected)) < 1e-9
    except (TypeError, ValueError):
        return False


def verdict_h(verdict, arm, selected):
    """H 只认绑定到本臂与本臂当前选定设置的审查结论：看的不是这组样本的结论不能拿来判定。"""
    if verdict is None:
        return None, "缺 review_verdict.json（待人工审查）"
    expected = verdict_binding(arm, selected)
    got = OrderedDict((key, verdict.get(key)) for key in VERDICT_KEYS)
    if not all(_same_value(got[key], expected[key]) for key in VERDICT_KEYS):
        return None, "review_verdict.json 绑定 {}，与本臂当前判定 {} 不一致，视为缺失".format(json.dumps(got, ensure_ascii=False),
                                                                                      json.dumps(expected, ensure_ascii=False))
    if not isinstance(verdict.get("pass"), bool):
        return None, "review_verdict.json 的 pass 未填写（须为 true/false）"
    return verdict["pass"], verdict.get("notes", "")


def route_e_failures(probabilistic, display, tau_d):
    """按 4.8 分流 E/E' 失败：返回 (P 阻断项, 展示阻断项, 需要的应急臂)。

    阻断项（E1/E5）不设应急臂：P 阻断即不通过；展示阻断时展示模式无法经应急臂修复，展示侧的其它失败也不再触发应急臂。
    """
    p_block, arms = [], set()
    for name, route in E_ROUTES.items():
        if not probabilistic[name][0]:
            if route is None:
                p_block.append(name)
            else:
                arms.add(route)
    d_block, d_arms = [], set()
    if tau_d is not None:
        routes = OrderedDict((name + "'", route) for name, route in E_ROUTES.items())
        routes.update(DISPLAY_EXTRA_ROUTES)
        for name, route in routes.items():
            if name in display and not display[name][0]:
                if route is None:
                    d_block.append(name)
                else:
                    d_arms.add(route)
    if not d_block:
        arms |= d_arms
    return p_block, d_block, [arm for arm in CONTINGENCY_ARMS if arm in arms]


def conclude(probabilistic, display, tau_d, contingency_used):
    """5.6 判定表（含 4.8 分流）；输入为各判据的 (通过, 细节)。返回 (结论, 分流信息)。

    分流信息的 routed_arms 是失败类别对应的应急臂，contingency_arms 只在结论为 contingency_required 时非空（真正触发的臂）。
    """
    conclusion, routing = _conclude(probabilistic, display, tau_d, contingency_used)
    routing["contingency_arms"] = list(routing["routed_arms"]) if conclusion == "contingency_required" else []
    return conclusion, routing


def _conclude(probabilistic, display, tau_d, contingency_used):
    p_block, d_block, needed = route_e_failures(probabilistic, display, tau_d)
    p_routed = [name for name, route in E_ROUTES.items() if route is not None and not probabilistic[name][0]]
    routing = OrderedDict([("p_blocking", p_block), ("display_blocking", d_block), ("routed_arms", needed)])
    a_ok = _passed(probabilistic, ("A1", "A2", "A3", "A4", "A5"))
    others_ok = _passed(probabilistic, ("B", "C", "single_sanity", "F1", "F2", "F3", "F4", "F5", "G"))
    display_ok = tau_d is not None and all(v[0] for v in display.values())
    h_pass = probabilistic["H"][0]

    def by_h(passed):
        if h_pass is None:
            return "pending_review"
        if not h_pass:
            return "reject"
        return "adopt_full" if passed else "adopt_probabilistic_only"

    if not a_ok:
        return "reject_no_better_than_bootstrap", routing
    if p_block:
        return "reject", routing
    if needed:
        if not contingency_used:
            return "contingency_required", routing
        # 应急臂之后 P 已通过、只有展示模式仍失败：按 5.6 第 2 行只作多假设采纳（仍需审查图）。
        return (by_h(False) if not p_routed and others_ok else "reject"), routing
    if not others_ok:
        return "reject", routing
    return by_h(display_ok), routing


def decide_arm(evals_by_nfe, repro, verdict, arm):
    """一个臂（3 seed）的完整预登记判定；NFE 规则只看 NFE_DEFAULT 评估里的扫描，选中更多步时其余判定全部改用该 NFE 的评估。"""
    decision = OrderedDict()
    evals_default = evals_by_nfe[NFE_DEFAULT]
    nfe, nfe_reason, nfe_ratios = nfe_rule(evals_default)
    decision["nfe_dispersion_ratios"] = nfe_ratios
    evals = evals_by_nfe.get(nfe)
    structural_ok, structural = structural_checks(evals_default if evals is None else evals, repro)
    decision["structural"] = OrderedDict([("pass", structural_ok), ("reasons", structural)])
    if not structural_ok:
        decision["selected"] = OrderedDict([("nfe", nfe), ("nfe_reason", nfe_reason), ("tau_P", None), ("tau_D", None)])
        decision["conclusion"] = "structural_failure"
        return decision
    failures = [(seed, r["diagnostics"]["teacher_forced"][TRAIN_FAILURE_T]["norm_mse"]) for seed, r in evals_default.items()
                if r.get("diagnostics", {}).get("teacher_forced")]
    decision["teacher_forced_t50"] = OrderedDict((str(seed), value) for seed, value in failures)
    if any(value >= TRAIN_FAILURE_MSE for _, value in failures):
        decision["selected"] = OrderedDict([("nfe", nfe), ("nfe_reason", nfe_reason), ("tau_P", None), ("tau_D", None)])
        decision["conclusion"] = "training_failure"
        return decision
    if evals is None:
        decision["selected"] = OrderedDict([("nfe", nfe), ("nfe_reason", nfe_reason), ("tau_P", None), ("tau_D", None)])
        decision["conclusion"] = "pending_nfe_rerun"
        return decision
    tau_p, ssr_parts, ssr_median = tau_p_rule(evals)
    tau_d = tau_d_rule(evals)
    decision["selected"] = OrderedDict([("nfe", nfe), ("nfe_reason", nfe_reason), ("tau_P", tau_p), ("tau_P_ssr_tau1", ssr_parts),
                                        ("tau_P_ssr_median", ssr_median), ("tau_D", tau_d)])
    probabilistic = OrderedDict()
    probabilistic.update(a_checks(evals, tau_p))
    probabilistic.update(b_c_checks(evals, tau_p))
    probabilistic.update(e_checks(evals, "F", tau_p))
    probabilistic["E_report_only"] = (True, report_only_naturalness(evals, tau_p))
    probabilistic.update(f_checks(evals, "F", tau_p))
    probabilistic.update(g_check(evals, tau_p))
    display = OrderedDict()
    display.update(d_checks(evals, tau_d))
    if tau_d is not None:
        display.update(("{}'".format(k) if not k.startswith("E'") else k, v) for k, v in e_checks(evals, "R", tau_d, display=True).items())
        f_display = f_checks(evals, "R", tau_d)
        display["F'"] = (f_display["F3"][0] and f_display["F5"][0], "F3: {}；F5: {}".format(f_display["F3"][1], f_display["F5"][1]))
        display["F'_report_only"] = (True, "；".join("{} {}".format(k, v[1]) for k, v in f_display.items() if k in ("F1", "F2", "F4")))
    probabilistic["H"] = verdict_h(verdict, arm, decision["selected"])
    decision["probabilistic"] = OrderedDict((k, OrderedDict([("pass", v[0]), ("detail", v[1])])) for k, v in probabilistic.items())
    decision["display"] = OrderedDict((k, OrderedDict([("pass", v[0]), ("detail", v[1])])) for k, v in display.items())
    conclusion, routing = conclude(probabilistic, display, tau_d, contingency_used=arm in CONTINGENCY_ARMS)
    decision["e_routing"] = routing
    if conclusion == "contingency_required":
        decision["contingency_arms"] = routing["contingency_arms"]
    decision["conclusion"] = conclusion
    return decision


def final_conclusion(arms, triggered):
    """主臂触发应急臂时，按固定顺序逐个看被触发的臂，第一个未被拒绝的结论即最终结论；前面的臂尚未评估时等待，
    不越过它看后面的臂，避免按结果挑臂。返回 (结论, 决定结论的臂)。"""
    main = arms["main"]["conclusion"]
    if main != "contingency_required":
        return main, "main"
    for arm in triggered:
        if arm not in arms:
            return "contingency_required", "main"
        if arms[arm]["conclusion"] not in REJECT_STATES:
            return arms[arm]["conclusion"], arm
    return "reject", "main"


def _eval_set(paths, arm, seeds, split, nfe):
    found = OrderedDict()
    for seed in seeds:
        path = paths.eval_json(arm, seed, split, nfe)
        if os.path.exists(path):
            found[seed] = _load(path)
    return found


def _table(evals, tau_p, tau_d):
    header = ["seed", "v2 mpjpe", "F 单样本 mpjpe", "F mean-of-K", "F 单步均值", "F minADE@10（GT 选样上界）", "F ES_joint",
              "Boot-F s1 ES", "F SSR r/l/a", "Brier F / v2", "步数 F / v2 / GT", "R({}) 单样本 mpjpe".format(tau_d)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for seed, r in evals.items():
        f = _gen(r, "F", tau_p)
        boot = r["bootstrap"]["F"].get("1.0", {})
        cells = [str(seed), "{:.4f}".format(r["references"]["v2"]["single_l2"]["mpjpe"]), "{:.4f}".format(f["single_l2"]["mpjpe"]),
                 "{:.4f}".format(f["mean_of_k_l2"]["mpjpe"]), "{:.4f}".format(f["one_step_mean_l2"]["mpjpe"]),
                 "{:.4f}".format(_num(f["min_ade@10"])), "{:.4f}".format(f["es_joint"]), "{:.4f}".format(boot.get("es_joint", float("nan"))),
                 "/".join("{:.2f}".format(f["ssr_clean"][p]) for p in ("root", "legs", "arms")),
                 "{:.3f} / {:.3f}".format(f["lead_foot"]["brier3"], r["references"]["v2"]["lead_foot"]["brier3"]),
                 "{:.2f} / {:.2f} / {:.2f}".format(f["steps"]["mean"], r["references"]["v2"]["steps"]["mean"], f["steps"]["gt_mean"]),
                 "{:.4f}".format(_gen(r, "R", tau_d)["single_l2"]["mpjpe"]) if tau_d is not None else "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def stage_summary(opts, paths):
    if opts.protocol == "original":
        return summary_test(opts, paths)
    repro = None
    if os.path.exists(paths.repro("a")) and os.path.exists(paths.repro("b")):
        repro = (_load(paths.repro("a")), _load(paths.repro("b")))
    verdict = _load(paths.verdict) if os.path.exists(paths.verdict) else None

    def judge(arm):
        seeds = _arm_seeds(arm, opts)
        by_nfe = OrderedDict()
        for nfe in NFE_GRID:
            found = _eval_set(paths, arm, seeds, "val", nfe)
            by_nfe[nfe] = found if len(found) == len(seeds) else None
        if by_nfe[NFE_DEFAULT] is None:
            _say("skip summary {}：NFE{} 评估不全".format(arm, NFE_DEFAULT))
            return None
        decision = decide_arm(by_nfe, repro, verdict, arm)
        nfe = decision["selected"]["nfe"]
        if decision["conclusion"] == "pending_nfe_rerun":
            decision["rerun_command"] = "python scripts/run_ntu2p_trackb.py --protocol subjval --stage eval --nfe {} --arms {}".format(nfe, arm)
        decision["table"] = _table(by_nfe[nfe], decision["selected"]["tau_P"], decision["selected"]["tau_D"]) if "probabilistic" in decision else []
        if "nfe_dispersion_ratios" in decision:
            decision["nfe_table"] = _nfe_table(by_nfe[NFE_DEFAULT])
        return decision

    arms = OrderedDict()
    main_decision = judge("main")
    if main_decision is None:
        _say("主臂评估不全，无法汇总")
        sys.exit(2)
    arms["main"] = main_decision
    triggered = list(main_decision.get("contingency_arms", [])) if main_decision["conclusion"] == "contingency_required" else []
    untriggered = [arm for arm in opts.arms if arm in CONTINGENCY_ARMS and arm not in triggered]
    if untriggered:
        _say("拒绝：应急臂 {} 未被主臂的 E 失败类别触发（当前触发：{}），不参与汇总".format(untriggered, triggered or "无"))
        sys.exit(2)
    for arm in triggered:
        decision = judge(arm)
        if decision is not None:
            arms[arm] = decision
    conclusion, decision_arm = final_conclusion(arms, triggered)
    adopted_arm = decision_arm if conclusion in ADOPT_STATES else None
    diagnostics = OrderedDict()
    insample = _eval_set(paths, "insample", [0], "val", NFE_DEFAULT)
    main0 = _eval_set(paths, "main", [0], "val", NFE_DEFAULT)
    if insample and main0:
        ssr_in = _mean(_gen(insample[0], "F", 1.0)["ssr_clean"][p] for p in ("root", "legs", "arms"))
        ssr_main = _mean(_gen(main0[0], "F", 1.0)["ssr_clean"][p] for p in ("root", "legs", "arms"))
        diagnostics["H_cf"] = OrderedDict([("ssr_insample", ssr_in), ("ssr_oof", ssr_main), ("supported", ssr_main - ssr_in >= 0.15)])
    summary = OrderedDict(
        [
            ("protocol", opts.protocol),
            ("base_config", MAIN_CONFIG),
            ("conclusion", conclusion),
            ("decision_arm", decision_arm),
            ("adopted_arm", adopted_arm),
            ("contingency_arms", triggered),
            ("selected", arms[decision_arm]["selected"]),
            ("arms", arms),
            ("diagnostics", diagnostics),
            ("review_verdict", verdict),
        ]
    )
    _write(paths.summary(), summary)
    lines = ["# NTU2P Track B 汇总（{}，H-val）".format(opts.protocol), "",
             "结论：**{}**（决定结论的臂 {}，采纳臂 {}，触发的应急臂 {}）".format(conclusion, decision_arm, adopted_arm, triggered or "无"), ""]
    for arm, decision in arms.items():
        lines += ["## 臂 {}：{}".format(arm, decision["conclusion"]), ""]
        selected = decision["selected"]
        median = selected.get("tau_P_ssr_median")
        lines.append("选定：NFE {}（{}），τ_P {}{}，τ_D {}".format(selected.get("nfe"), selected.get("nfe_reason"), selected.get("tau_P"),
                                                            "" if median is None else "（τ=1 SSR 中位数 {:.3f}）".format(median), selected.get("tau_D")))
        if decision.get("teacher_forced_t50"):
            lines.append("终点 teacher-forced t=50 归一化 MSE：{}（≥ {} 判训练失败）".format(
                json.dumps(decision["teacher_forced_t50"], ensure_ascii=False), TRAIN_FAILURE_MSE))
        lines.append("")
        if not decision["structural"]["pass"]:
            lines += ["结构自检失败（实现错误，停止）："] + ["- " + r for r in decision["structural"]["reasons"]]
        for block in ("probabilistic", "display"):
            if block in decision:
                lines += ["", "| {} | 通过 | 细节 |".format("概率模式 P" if block == "probabilistic" else "展示模式"), "|---|---|---|"]
                for name, item in decision[block].items():
                    lines.append("| {} | {} | {} |".format(name, item["pass"], item["detail"]))
        if decision.get("e_routing"):
            lines.append("E 分流：{}".format(json.dumps(decision["e_routing"], ensure_ascii=False)))
        if decision.get("table"):
            lines += ["", "逐 seed（best-of-K 为 GT 选样上界，不与 v2 单样本并列比较）："] + decision["table"]
        if decision.get("nfe_table"):
            lines += ["", "NFE 扫描（mode F τ=1；NFE 规则只看离散度比 {}）：".format(
                json.dumps(decision.get("nfe_dispersion_ratios", {}), ensure_ascii=False))] + decision["nfe_table"]
        lines.append("")
    if diagnostics:
        lines += ["诊断：{}".format(json.dumps(diagnostics, ensure_ascii=False))]
    with open(paths.summary().replace(".json", ".md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    _say("summary: {} -> {}".format(paths.summary(), conclusion))
    return summary


def summary_test(opts, paths):
    """原协议：只汇总唯一一次 test 的数字（设置已冻结），不做采纳判定。"""
    parent, reason = subjval_decision(opts)
    if reason is not None:
        _say("拒绝：" + reason)
        sys.exit(2)
    settings, nfe = frozen_settings(parent)
    rows = OrderedDict()
    for arm in opts.arms:
        evals = _eval_set(paths, arm, _arm_seeds(arm, opts), "test", nfe)
        if not evals:
            continue
        rows[arm] = OrderedDict()
        for setting in settings.split(","):
            mode, tau = setting.split(":")
            blocks = [_gen(r, mode, float(tau)) for r in evals.values()]
            rows[arm][setting] = OrderedDict(
                [
                    ("single_mpjpe", (_mean(b["single_l2"]["mpjpe"] for b in blocks), _std(b["single_l2"]["mpjpe"] for b in blocks))),
                    ("es_joint", (_mean(b["es_joint"] for b in blocks), _std(b["es_joint"] for b in blocks))),
                    ("min_ade@10_gt_selected_upper_bound", (_mean(_num(b["min_ade@10"]) for b in blocks), _std(_num(b["min_ade@10"]) for b in blocks))),
                    ("ssr_clean", OrderedDict((p, _mean(b["ssr_clean"][p] for b in blocks)) for p in ("root", "legs", "arms"))),
                ]
            )
        rows[arm]["v2_single_mpjpe"] = (_mean(r["references"]["v2"]["single_l2"]["mpjpe"] for r in evals.values()),
                                        _std(r["references"]["v2"]["single_l2"]["mpjpe"] for r in evals.values()))
    summary = OrderedDict([("protocol", "original"), ("split", "test"), ("base_config", MAIN_CONFIG), ("adopted_arm", parent["adopted_arm"]),
                           ("frozen_settings", settings),
                           ("nfe", nfe), ("rows", rows)])
    _write(paths.summary(), summary)
    _say("test summary: {}".format(paths.summary()))
    return summary


# ---------------------------------------------------------------- 入口


STAGE_FUNCS = OrderedDict(
    [
        ("folds", stage_folds),
        ("fold_models", stage_fold_models),
        ("bank", stage_bank),
        ("train", stage_train),
        ("eval", stage_eval),
        ("summary", stage_summary),
        ("review", stage_review),
    ]
)


def plan(opts):
    if opts.stage != "all":
        return [opts.stage]
    stages = ["folds", "fold_models", "bank", "train", "eval", "summary"]
    if opts.protocol == "subjval":
        stages.append("review")
    return stages


def build_arg_parser():
    parser = argparse.ArgumentParser(description="NTU2P Track B 分阶段驱动")
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--arms", nargs="*", default=None, choices=ARMS,
                        help="缺省：subjval 为 main（review 为 summary 的 decision_arm）；original 为 subjval 采纳的臂（只允许该臂）")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--gen_steps", type=int, default=20000)
    parser.add_argument("--fold_steps", type=int, default=10000)
    parser.add_argument("--nfe", type=int, default=NFE_DEFAULT)
    parser.add_argument("--eval_workers", type=int, default=3)
    parser.add_argument("--eval_batch_windows", type=int, default=32, help="评估每块窗口数（×K 个样本一起解码投影；显存不足时调小）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_root", default=SAVE_ROOT)
    parser.add_argument("--results_root", default=RESULTS_ROOT)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_shared_gpu", action="store_true")
    parser.add_argument("--test", action="store_true", help="原协议唯一一次 test 评估")
    parser.add_argument("--confirm_probabilistic_only", action="store_true", help="用户已确认：只作多假设生成器采纳时仍做原协议复训与 test")
    return parser


def main(argv=None):
    opts = build_arg_parser().parse_args(argv)
    if opts.test and opts.protocol != "original":
        _say("--test 只允许 --protocol original")
        sys.exit(2)
    if opts.test and opts.stage not in ("eval", "all", "summary"):
        _say("--test 只用于 eval/all/summary 阶段")
        sys.exit(2)
    paths = Paths(opts)
    if opts.test:
        # 先拦截：没有采纳结论时，任何阶段都不应走到 test。
        _, reason = subjval_decision(opts)
        if reason is not None:
            _say("拒绝 --test：" + reason)
            sys.exit(2)
    for stage in plan(opts):
        _say("== stage {} ({})".format(stage, opts.protocol))
        check_original_allowed(opts, stage)
        check_gpu(opts, stage)
        stage_opts = argparse.Namespace(**vars(opts))
        stage_opts.arms = resolve_arms(opts, stage)
        if opts.dry_run and stage in ("summary",):
            _say("[dry_run] 汇总 {} 的评估 JSON 并按预登记流程判定 -> {}".format(stage_opts.arms, paths.summary()))
            continue
        STAGE_FUNCS[stage](stage_opts, paths)
    _say("done")


if __name__ == "__main__":
    main()
