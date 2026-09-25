# NTU2P 步态变体 GL / GH：审查跟进

前置：`docs/ai/context/20260925-221217-ntu2p-gait-gl-gh-variants-design-and-implementation.md`（设计、实现与 CPU 验证）。本文修复其审查发现的 4 条问题（均为 low）。该文档按 AGENTS.md 不再修改：其中 §3 的相关定义口径与面板数字、§6 的 GPU 启动命令，以本文为准。

**前提**：Stage 2 驱动（PID 1620895）与其后的 Stage 3 驱动（PID 1627573）都已退出；修改时没有训练或评估进程在跑，GPU 空闲。本次只在 CPU 上验证，没有启动 GPU，没有 commit。

## 1. 审查结论与修复

| # | 位置 | 结论 | 修复 |
|---|---|---|---|
| 1 | `scripts/run_ntu2p_v2_screen.py` "GH vs GL 配对"行 | 成立 | 见 1.1 |
| 2 | 同一行的列 | 成立 | "Δ% vs A0 逐 seed"列改为"—" |
| 3 | 设计文档 §6 的启动命令 | 成立 | 改用环境解释器全路径（第 3 节） |
| 4 | `utils/ntu2p_naturalness.gait_phase_stats` | 成立 | 在看到任何 GL/GH 结果前修改口径（第 2 节） |

### 1.1 配对行把"不确定"写成"两者均未通过"（第 1 条）

- **问题**：配对行原来只读 `candidate_adopt`。GL、GH 都不是候选时一律写"两者均未通过（本轮不再调权重…）"，并写进 JSON 的 `choice`。但预登记规则允许"不确定"（A 与非 L2 护栏都过、只有 L2 越界）的配置补跑一次 w=0.25。按 `choice` 行事的人或脚本会因此丢掉这次补跑。
- **修复**：新增三个函数。
  - `gh_gl_choice`：两者都是候选时按 GH 增量规则取舍；只有一个是候选时取它；都不是候选但有"不确定"的，写明哪个"不确定"并列出补跑配置；只有两者都不是"不确定"时才写"两者均未通过"。
  - `gait_rerun_config`：把最后一个步态 token 的权重换成 0.25，例如 `A6-F-A5f0.05-GL0.5` 变为 `A6-F-A5f0.05-GL0.25`。
  - `gait_uncertain_label`：单个配置行的"不确定"标签。
- **JSON**：`decisions["GH vs GL: …"]` 新增 `reruns` 列表，便于脚本读取。
- **只补跑一次**：预登记只允许补跑一次。所以 w=0.25 的配置本身再判"不确定"时，不再给出补跑。原来的单行标签是常量，对 w=0.25 的配置也会写"允许补跑一次 w=0.25"，现在一并改掉。
- **表注**：汇总表脚注补上这条取舍规则与配对行的列含义。

### 1.2 配对行的列错位（第 2 条）

- 配对行"参照"列是 GL，所以"Δ% vs A0 逐 seed"列不适用，改为"—"。
- "Δ% vs 参照"列保持 GH 对 GL 的 Δmpjpe 均值。
- 逐 seed 值只写在判断理由里，理由文本原来就含这部分。

## 2. 步态相关在 GT 侧无定义（第 4 条）

**复核**：val 步行人（99 人）中，GT 分离量 s(t) 在段内的离均差范数恰为 0 的有：
- f01_10：1 人，S014C001P008R002A010 person 1；
- f11_20：2 人，上面这人加 S030C001P081R001A020 person 1；
- f21_30 与 f31_50：0 人。

原实现只检查预测侧，这些人对所有方法都按 r = 0 计入，把均值同样拉向 0（×98/99、×97/99）。GT 对 GT 时预测侧范数也为 0，所以他们反而被排除，口径不一致。

**为什么现在改**：还没有任何 GL/GH 结果。GPU 未启动；现有评估 JSON 都没有步态键，驱动会用新口径统一补评估，所以不存在新旧口径混用。

**修改**：`defined = (pred_norm >= GAIT_CORR_MIN_NORM) & (gt_norm >= GAIT_CORR_MIN_NORM)`。
- 只影响 `gait_sep_corr_f01_10*`、`gait_sep_corr_f11_20*`（含 moving/starting 分组）；
- RMSE、先迈脚准确率与人数不变。

**重生成的 seed 0 现状面板**（EMA 终点，val 198，括号内为旧口径）：

| 方法 | corr_f01_10 | corr_f11_20 | rmse_f01_10 |
|---|---|---|---|
| copy-last | NaN | NaN | 0.159 |
| A0 | 0.050（0.050） | 0.243（0.239） | 0.158 |
| A6 | 0.129（0.128） | 0.260（0.254） | 0.174 |
| A6-F | 0.164（0.163） | 0.217（0.212） | 0.178 |
| A6-F-A4 | 0.055（0.054） | 0.190（0.186） | 0.203 |
| A6-F-A5f0.05 | −0.016（−0.016） | 0.279（0.273） | 0.161 |
| A6-F-A5f0.2 | −0.004（−0.004） | 0.333（0.327） | 0.160 |
| A6-F-A7 | 0.121（0.120） | 0.312（0.305） | 0.158 |

- **阈值不改**：变化最多 0.007，远小于 Δ 门槛（+0.15 / +0.10）。绝对值门槛 0.25 / 0.35 维持预登记。
- **本轮参照**：R = A6-F-A5f0.05 在 seed 0 上 corr_f11_20 为 0.279。GL/GH 要过判据 A 的 corr_f11_20 一条，须在 3 seed 均值上达到 ≥ 0.35，且 Δ ≥ +0.10。

## 3. GPU 启动命令（替换设计文档 §6）

非交互 shell 中 `python` 不在 PATH（`python: command not found`），照抄原命令时 nohup 立即失败，什么都不会训练。改用环境解释器全路径；驱动会把 `sys.executable` 原样传给子进程：

```bash
cd /home/rpartx3080/CodeSpace/ReGenNet/.claude/worktrees/two-person-action-prediction-model-0c948c
PYTHONPATH=. nohup /home/rpartx3080/.local/micromamba/envs/regennet/bin/python -u scripts/run_ntu2p_v2_screen.py \
  --stage 2 --steps 10000 --configs A6-F A6-F-A5f0.05 A6-F-A5f0.05-GL0.5 A6-F-A5f0.05-GH0.5 --workers 3 \
  >> results/forecasting/ntu120_label/ntu2p_v2_screen/gait_driver.log 2>&1 &
```

- 换底座（A5 审查图被否）时，把 `--configs` 改为 `A6-F A6-F-GL0.5 A6-F-GH0.5`，其余不变。
- 补评估前旧 JSON 备份为 `*.pre_gait.json`。L2 容差 1e-9 依赖 GPU 上同代码、同 batch 的前向逐位可复现。万一某个 run 报 L2 mismatch：新 JSON 已写入，汇总照常生成；先看差值量级（设备或 kernel 差异约 1e-7，口径被改会远大于此），再决定是否采信。

## 4. 验证（CPU，2 线程，nice 10）

- **训练逐位等价**（开关关闭）：改动前代码（`orig/` 备份覆盖的副本）与当前 worktree，用同一 CLI 各训 5 step，配置 A6-F-A5f0.05、A6、A0。state_dict、EMA、optimizer、loss_scales、逐步 train_loss 全部 `torch.equal`（ALL EQUAL）；args 只新增 4 个开关键。
- **评估逐位等价**：A6 s0 EMA 终点，val 198。
  - 导出数组逐位相同；L2、articulation、gate 与全部已有自然度键完全相同；
  - 新增 31 个自然度键（29 个步态键 + 2 个 RMSE 比）与 2 个块级人数；
  - 新口径下 corr_f01_10 0.129、corr_f11_20 0.260、rmse_f01_10 / copy-last 1.089，步行 99 人、moving 29 人。
- **指标实现**：utils 实现与"裁决参考实现 + GT 侧掩码"逐位相同。与原参考实现相比只在 corr_f01_10* / corr_f11_20* 上不同，分母恰好少 1 / 2 人。`compute_naturalness_stats` 中的步态键与单独调用一致。
- **损失与 GH 不变量**（重跑原测试）：全部通过（ALL OK）。
  - leg_pos / leg_lpvel 等于分析 A；
  - 全关节掩码下 dct_mid_nonleg 等于原项；
  - GH 初始输出等于 copy-last，非腿关节逐位不变；
  - 主干腿行梯度为 0，个人系正交，刚体等变误差 3.6e-7。
- **配对行单测**：`gh_gl_choice` 的 9 个分支全部符合规则（都候选、单候选、GL / GH / 两者"不确定"、都不是、w=0.25 再"不确定"）。
- **write_summary 端到端**：用 5 step 冒烟 run，分别用真实 `decide` 和强制"GL 不确定"/"都未通过"跑。检查项：
  - 配对行列数与表头一致，"Δ% vs A0 逐 seed"为"—"；
  - JSON 的 `choice` / `reruns` 与规则一致；
  - GL 行标签写明补跑配置 `A6-F-A5f0.05-GL0.25`。
- **驱动冒烟**：在临时目录用驱动训练 A0、A6-F、A6-F-A5f0.05、A6-F-{GL,GH}0.5、A6-F-A5f0.05-{GL,GH}0.5，seed 0、每个 5 step，然后评估与汇总，全部跑通（exit 0）。
  - raw 与 EMA 终点都能用评估加载函数加载，val 前 16 条输出有限、首帧误差为 0；
  - GH 参数量 3,649,533，GL 2,714,141；常量与分析 A 的相对差为 0；
  - 两个配对行的"Δ% vs A0 逐 seed"都为"—"，JSON 含 `reruns`。
- **补评估路径**：删掉一个冒烟 run 两份 JSON 里的步态键，再篡改其中一份的 mpjpe（+1e-6），然后重跑驱动。
  - 两份都备份为 `*.pre_gait.json`；
  - 未篡改的那份 L2 差为 0，通过；
  - 篡改的那份报 mismatch（1e-6 ≥ 1e-9），该 run 记为失败，汇总照常写出。
- **`--dry_run`**（上面的启动命令）：6 个训练命令、30 个评估命令。其中 A0、A6-F、A6-F-A5f0.05 的 EMA 与 raw 终点为补评估；解释器为环境 python。

验证脚本与日志在会话 scratchpad 的 `gait/fix/` 下。
