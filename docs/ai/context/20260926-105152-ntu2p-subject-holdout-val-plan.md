# NTU2P 受试者留出选型 val：设计与计划（下一步候选第 1 批）

背景：`docs/ai/context/20260926-100701-ntu2p-v2-mainline-adoption-decision.md` 的下一步候选有三项，按依赖顺序分三批执行：
1. 受试者留出的选型 val（本文）；
2. 手指损失去重；
3. Track B residual diffusion。

后两批都要用第 1 批的 val 选型，所以它排在最前。

## 问题

- 现有 val（manifest_seed0，198 条）是从 xsub train 中按动作分层、按序列随机抽 10% 得到的。它的 42 个受试者（sample_id 的 P 字段）**全部**出现在 train 中；而 test 的 53 个受试者与 train 完全不重叠。
- 后果：v2 相对 A0 的收益在 val 上是 −7.8%，在 test 上只有 −3.85%，val 高估了约 2 倍。
- 同一受试者会以 R001/R002 重复表演同一动作，模型可能靠记住个人风格在 val 上获益，这部分收益到了 test 上不存在。

## 设计

### 1. 新 manifest：按受试者留出

- **受试者池**：原 train + val 共 1956 条序列（都来自 xsub train 的 53 个受试者）。test 保持不变（xsub test，1253 条）。
- **留出集 H**：选一组受试者，他们的全部序列作为新 val（`val`），其余作为新 train（`train`）。
- **选择过程**（预登记，不看任何模型结果）：
  - 候选：排除 P008（335 条，占池 17%，留出它会大幅缩小训练集）以外的全部受试者。
  - 约束：H 的序列总数在池的 12–16% 之间，约 235–315 条。
  - 目标：H 的构成尽量接近 test。评分 = |NTU60 占比 − test 的 0.555| + 0.5 × 动作分布与 test 的 L1 距离 + 0.05 × "池中该动作 ≥ 5 条而 H 中为 0"的动作数。其中 NTU60 指 setup ≤ 17。
  - 用 `random.Random(0)` 做 50000 次随机子集搜索（逐个加入随机排列的受试者，直到落入数量区间），取评分最低者。
- **格式**：沿用原 manifest。`split_config` 记录：
  - `split_unit=performer`、`heldout_performers`；
  - 源 manifest 路径与 hash、搜索 seed、评分。

  用 `manifest_payload_hash` 重算 hash，并断言 train/val/test 的受试者两两不重叠。
- **路径**：
  - manifest：`results/forecasting/ntu120_label/ntu2p_subjval/manifest_subjval_seed0.json`；
  - xyz 缓存：`results/forecasting/ntu120_label/ntu2p_xyz_seq_cache_subjval/`，用 `scripts/build_ntu2p_xyz_seq_cache.py --manifest_path … --save_dir …` 生成。

### 2. 冻结 base 重训（防泄漏）

- A0（旧 refiner）依赖的冻结独立单人 base 是在原 train 上训的，包含 H 的受试者。另外，评估中的 base 指标与摆动 gate 的"相对 base 回退 ≤ 5%"判据也要用到 base。
- 做法：在新 train 上按原配置（`save/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_cuda_retrain_s0_5000/args.json`，5000 step，seed 0）重训一个 base，新协议下的 A0 与评估一律用它。
- v2（InterMixer）不依赖 base。

### 3. 代码改动（新增参数，默认值行为不变）

- `scripts/build_ntu2p_subject_holdout_manifest.py`（新增）：实现第 1 节的选择过程。
- `scripts/run_ntu2p_v2_screen.py`：新增 `--manifest_path`、`--cache_dir`、`--baseline_checkpoint`、`--run_prefix`（默认分别为现有的 MANIFEST、默认缓存、BASELINE、`ntu2p_v2`），并传给训练与评估命令。新协议的 run 目录为 `save/forecasting/ntu120_label/ntu2p_v2subj_<config>_s<seed>_<steps>/`，汇总放在 `results/forecasting/ntu120_label/ntu2p_v2subj_screen/`，与原协议互不覆盖。
- 训练与评估入口已有 `--manifest_path`、`--cache_dir`，评估另有 `--baseline_checkpoint`，不需改动。

### 4. 实验

新协议下训练 A0（旧 refiner，用新 base）与 v2 主线（`A6-F-A5f0.05-A4-GH0.5`），各 3 seed × 10000 step、EMA 终点，评估新 val。

**协议有效性检验**：
- 同时评估这 6 个模型在 test 上的结果。这只用来检验"新 val 的配对 Δ 能否预测 test 的配对 Δ"，不参与任何选型。所评估的都是已经采纳或作对照的配置，结果文档会写明这一点。
- 判据：新 val 上 v2 相对 A0 的配对 Δmpjpe，要比旧 val 的 −7.8% 更接近 test 的 Δ。同时报告 xyz_mse/xyz_mae 的 Δ，以及 seed 标准差。

### 5. 之后的选型规则（若检验通过）

- 第 2、3 批的候选都在新协议下训练，用新 val 选型，参照为新协议下的 v2 主线。
- 被采纳的候选再在原协议（原 train 1758）上训练 3 seed × 10000 step，评估一次 test，得到与主数字可比的数字。
- 主数字仍是原协议下的 v2（test 0.1810）。

## 风险

- H 较小（约 250 条），val 的噪声会变大；以 3 seed 的标准差实测。
- 新 train 少了约 13% 的数据，新协议下的绝对数字会变差，只能在新协议内部比较。
- 本数据的受试者 P 字段只记录一名表演者，所以"受试者不重叠"只针对该字段，与 xsub 协议口径一致。
