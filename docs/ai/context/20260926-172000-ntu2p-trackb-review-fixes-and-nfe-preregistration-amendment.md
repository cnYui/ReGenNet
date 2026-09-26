# NTU2P Track B：审查修复与 NFE 预登记修订（T1-d 之前）

设计：`docs/ai/context/20260926-121543-ntu2p-trackb-residual-generative-design-and-plan.md`；实现：`docs/ai/context/20260926-130308-ntu2p-trackb-implementation-result.md`。
本文在任何 H-val 评估（T1-d）之前写入，**修订设计 5.4 第 2 步（NFE 规则）与 4.8/5.6 的应急臂执行口径**，并把**底座切换到第 2 批采纳的 FD2 主线**（第 5 节）。其余预登记阈值不变。
审查问题 1–4 不影响折模型、残差库和生成器训练（T1-a 到 T1-c）。第 5 节的底座切换发生在 T1-a 之前，GPU 上尚未跑过任何 Track B 阶段。仅用 CPU，没有训练模型，也没有看 test。

## 1. NFE：主采样由 10 步改为 50 步，NFE 规则改按离散度（审查问题 1，medium，成立）

**问题。** DDIM（eta=0）步数少时，离散化误差会系统性收缩样本离散度。复现方法：直接调用实现里的 `build_sampling_diffusion` / `sample_x0`，用解析最优的 x0 去噪器，数据为高斯，x0 ~ N(0.3, s²)，2000 个样本。

| 样本 std / 真 std | NFE10 | NFE20 | NFE50 | NFE100 | NFE200 |
|---|---|---|---|---|---|
| s = 0.1 | 0.506 | 0.715 | 0.879 | 0.937 | 0.967 |
| s = 0.3 | 0.776 | 0.888 | 0.955 | 0.977 | 0.988 |
| s = 1.0 | 0.872 | 0.937 | 0.975 | 0.987 | 0.993 |

- 均值无偏，偏差 ≤ 0.001。
- 审查另测了 eta=1：NFE10 为 0.810，结果更差。
- 用真实训练入口在合成库上训练的模型也会收缩，而且幅度更大：NFE10 → NFE50 时，root std 从 0.733 升到 0.866。
- 原规则要求"NFE20 的 ES_joint ≤ 0.99×NFE10，且 3/3 seed"，但 ES/CRPS 在最优点附近是平的：离散度比从 0.87 升到 0.94，CRPS 只改善 0.3%。所以这条规则按构造几乎不会触发。
- 结果是 τ=1 实际约等于 τ≈0.87。G、A1/A2/A3/A5（与离散度精确的 bootstrap 包络比较）、F2 熵、D 的 APD 比值和 H_cf 的 SSR 差值，都会被系统性地推向欠散一侧。

**修订后的预登记规则（替代设计 5.4 第 2 步）：**

- **主采样 NFE = 50**，常量为 `model/forecasting_ntu2p_resdiff.py:DEFAULT_NFE`。评估默认 `--nfe 50`，并扫描 `--nfe_scan 10,20,100,200`。每一档都是完整的指标块，含 APD、SSR、ES 与单样本 L2。
- **NFE 规则**：在 NFE50 那次评估的扫描里，依次看 N = 50、100，计算 mode F τ=1 下 2N 相对 N 的两个比值（3 seed 均值）：APD_all 之比与 SSR 均值之比（root/腿/臂 SSR 的平均）。
  - 两者都 ≤ 1.03：用 N 步；
  - 任一 > 1.03：看下一档；
  - 两档都超过：用 200 步。
  - 选中 100 或 200 时，summary 给出 `pending_nfe_rerun` 和重评命令，其余判定全部用该 NFE 的评估。
- **ES 不再参与 NFE 选择**，只在 summary 的 NFE 扫描表中与 APD、SSR、单样本 mpjpe 并列报告。
- **τ_P 规则不变**：SSR 中位数 > 1.4 时 τ_P = 0.8，只能降温。不加入 τ>1：那是看到数据后才加的自由度。上面的 NFE 规则已经把采样器造成的欠散压到 oracle 下 ≤ 5%（s ≥ 0.3）。
- **语义**：τ=1 表示"NFE 规则选定步数下的 DDIM"，不等于精确校准。s = 0.1 时 NFE50 仍只有 0.879；规则的比值在 oracle 下为 1.066，会自动加步。

**新增自检**（`scripts/check_ntu2p_resdiff.py` 的 sampling 节）：

- 解析高斯 oracle 在 DEFAULT_NFE 下，条件 std 0.3 与 1.0 两档的样本 std / 真 std ≥ 0.95，且均值无偏。实测分别为 0.955 与 0.976；NFE10 为 0.777 与 0.873，不会通过这项检查。
- NFE 50/100/200 的采样步都包含 0 与 999。
- driver 节加了 NFE 规则的单元测试，共 4 例：两档都收敛、50 未收敛、100 仍未收敛、只有 SSR 比超过阈值。

**冒烟佐证**：冒烟模型在 subjval val 前 8 窗、K=4 上运行，从 NFE10 到 NFE200，ES_joint 只在 0.1180 到 0.1186 之间变化，而 SSR（root）从 0.20 升到 0.25。这说明 ES 对欠散不敏感，SSR/APD 才看得出来。

**代价**：去噪网络只有 4.9M 参数，采样步数 ×5 的开销与解码、投影和指标计算相比可以忽略。GPU 上没有实测，H-val 的评估时间预计仍在同一量级。

## 2. 应急臂采纳路径（审查问题 2，medium，成立，五处全部修复）

- **4.8 分流**（`E_ROUTES`、`route_e_failures`、`conclude`）：

  | 失败项 | 处理 |
  |---|---|
  | E2 / E3 / E4（滑行、脚滑、穿地 / 悬空） | 只触发 foot |
  | E6（穿插） | 只触发 inter |
  | E1（结构）、E5（jerk / 高频） | 不设应急臂，P 侧失败即 `reject` |

  - 展示模式按同一张表分流：E'2–E'4 与 `E'_skate` → foot，E'6 与 `E'_min_dist` → inter。后两项的归类是本次补充的口径：GT 站定帧脚速属于脚滑；双人最小距离误差属于双人几何，由 C2 的相对关节向量损失直接约束。
  - 展示侧出现 E1'/E5' 时，展示模式无法通过应急臂修复，展示侧其它失败也不再触发应急臂。P 全过时走 5.6 第 2 行，只作多假设采纳。
- **只跑被触发的臂**：
  - summary 的 `arms.main.contingency_arms` 记录被触发的臂，只在主臂结论为 `contingency_required` 时非空；
  - subjval 的 train/eval 拒绝未被触发的应急臂（返回码 2），summary 显式请求未被触发的臂时同样拒绝；
  - 被触发的臂自动参与汇总。
- **最终结论不按结果挑臂**：
  - 按固定顺序 foot → inter 逐个看被触发的臂，第一个未被拒绝的结论即最终结论；
  - 顺序靠前的臂还没评估时，结论保持 `contingency_required` 并等待，不越过它去看后面的臂；
  - 两类同时失败时 foot 和 inter 都会跑，这是设计"按失败类别各跑一轮"的直接结果。此时有两次机会，但顺序固定。
- **summary 新增三个字段**：`decision_arm`（决定结论的臂）、`adopted_arm`（仅采纳时非空）、`contingency_arms`。`selected` 取自 `decision_arm`，所以冻结给原协议的 NFE/τ_P/τ_D 是被采纳臂的设置。
  - `frozen_settings` 会核对 adopted_arm 与 decision_arm 一致。
  - 旧格式 summary（没有 adopted_arm）不允许进入原协议。
- **原协议只允许被采纳的臂**：
  - `--arms` 缺省即为 `adopted_arm`；
  - train/eval/summary 显式传其它臂时拒绝（返回码 2）。因此不能先复训、test 一个没被采纳的臂，也不能依次 test 多个臂再挑结果。
- **审查结论绑定臂与设置**：
  - `--stage review` 一次只审查一个臂。缺省为 summary 的 decision_arm，传入的 in-sample 会被忽略。
  - 审查图输出到 `review_<arm>/`，并写出 `review_verdict_template.json`，其中 arm/nfe/tau_P/tau_D 已预填。
  - `review_verdict.json` 必须包含 `arm, nfe, tau_P, tau_D, pass, notes`。summary 核对前四项与该臂当前的判定是否一致，不一致、或 pass 不是布尔值，都把 H 视为缺失（pending_review）。
- **teacher-forced 诊断扩到每个被判定臂**：训练失败停止条件（t=50 归一化 MSE ≥ 1.0）原来只能在主臂 s0 上触发，现在每个被判定臂的 s0 评估都带 `--diagnostics`。

## 3. review 在停止状态下跳过（审查问题 3，low，成立）

summary 为 `pending_nfe_rerun`、`structural_failure` 或 `training_failure` 时，或该臂的 τ_P 为 None 时，review 打印原因后返回 0，不再拼出 `F:None` 让评估脚本崩溃，也不会白跑一次 GPU 评估。`--stage all` 的计划仍会列出 review，由它自己判断是否跳过。

## 4. test 上不算 v2 3 seed 均值（审查问题 4，low，成立）

`--ensemble_checkpoints` 只在 val 上传入。原协议的 test 评估不会产出 `v2_ensemble` / `v2_ensemble_samples`。这样设计第 9 节的旁支（3 seed 输出均值）将来仍能做一次干净的 test。

val 上的 `v2_ensemble` 只是诊断，不能作为旁支立项的依据。

## 5. 底座切换到 FD2 主线（审查之外，据第 2 批结论）

`docs/ai/context/20260926-165932-ntu2p-finger-dedup-result.md` 已把主线更新为 `A6-F-A5f0.05-A4-GH0.5-FD2`，并写明"第 3 批 Track B 以它为底座"。这里落实设计风险 9。

- **只改了驱动的 `MAIN_CONFIG`**。它影响三处路径：
  - subjval 部署底座：`ntu2p_v2subj_…-FD2_s{i}_10000`；
  - 原协议部署底座：`ntu2p_v2_…-FD2_s{i}_10000`；
  - 折模型：`run_ntu2p_v2_screen.py --configs …-FD2`，产物在 `ntu2p_v2fold{f}_{protocol}_…-FD2_s0_10000`。

  上述两组部署底座的 3 个 EMA 终点都已存在。用 screen 驱动做 dry-run，确认折模型命令带 `--loss_joint_subset hand2`，且 save_dir 与本驱动期望的路径一致。
- **FD2 只改训练损失的关节子集**，结构与前向不变，所以编解码、生成器与评估无需改动。自检改用 FD2 checkpoint 后，结构保证不变：零系数与 v2 的差 2.3e-6，oracle mpjpe 0.0383（v2 0.1497），首帧误差 0。
- **summary 记录 `base_config`**。subjval 采纳时若 base_config 与当前主线不一致，原协议拒绝进入：在一个底座上选定的 NFE/τ 不能冻结给另一个底座。
- **折与切片缓存不依赖底座**，已生成的 `results/.../ntu2p_trackb/{subjval,original}/folds/` 继续可用。

## 6. 验证（CPU，2 线程，nice 10）

- **CPU 自检**：`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONPATH=. nice -n 10 python scripts/check_ntu2p_resdiff.py --scratch_dir <scratchpad>/trackb/check_fix3` → **93/93 通过**，墙钟 92 s。底座为 FD2。原有 76 项全部保留，新增 17 项：
  - sampling 节 5 项：NFE50/100/200 采样步含端点 3 项，oracle 离散度 2 项；
  - driver 节 12 项（另有一项：subjval 选型底座与当前主线不一致时，原协议拒绝）：
    - E 分流 13 例；
    - NFE 规则 4 例；
    - 绑定 main 的审查结论不用于 foot 的 H；
    - foot 被采纳时 adopted_arm 与冻结设置都取自 foot，得到 `F:0.8,R:0.6` 与 NFE100；
    - summary 与 train/eval 都拒绝未被触发的 inter；
    - 原协议拒绝 `--arms main`，缺省时即为 foot；
    - 原协议 test 只评估 foot、使用冻结的 NFE，且不带 `--ensemble_checkpoints`；
    - review 只接受一个臂；
    - 停止状态下 review 跳过。

    采纳路径的测试用桩判定，并把 `run_command` 换成抛异常的桩，保证不会真的启动任何命令。
- **默认参数的评估冒烟**：subjval val 前 8 窗，K=4，`--nfe` 与 `--nfe_scan` 都用默认值。输出 meta.nfe = 50，扫描键为 10/20/100/200；把这份 JSON 交给 `nfe_rule`，得到 NFE50（APD 比 0.993，SSR 比 1.012）。
- **ruff**（E9,F63,F7,F82 加全部 F 类，py37）对 11 个 Track B 文件无问题。
- **改动范围**：只改了本批新建的 4 个文件：`model/forecasting_ntu2p_resdiff.py`、`eval/eval_ntu2p_resdiff.py`、`scripts/run_ntu2p_trackb.py`、`scripts/check_ntu2p_resdiff.py`。第 1/2 批用到的文件一律没改。

## 7. GPU 命令（受试者留出协议；生成器 seed i 配底座 v2subj FD2 s_i）

```bash
cd /home/rpartx3080/CodeSpace/ReGenNet/.claude/worktrees/two-person-action-prediction-model-0c948c
PY=/home/rpartx3080/.local/micromamba/envs/regennet/bin/python; export PYTHONPATH=. OMP_NUM_THREADS=2
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage fold_models --device cuda:0                 # T1-a：3 折并行 10000 步
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage bank  --arms main insample --device cuda:0   # T1-b
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage train --arms main insample --device cuda:0   # T1-c：main s0–s2 + insample s0，20000 步
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage eval  --arms main insample --device cuda:0   # T1-d：NFE50，扫描 10/20/100/200
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage summary
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage review --device cuda:0                      # 审查 summary 的 decision_arm；停止状态自动跳过
# 看图后把 results/forecasting/ntu120_label/ntu2p_trackb/subjval/review_<arm>/review_verdict_template.json
# 复制为 results/forecasting/ntu120_label/ntu2p_trackb/subjval/review_verdict.json，填写 pass（true/false）与 notes（其余四项不改）
$PY scripts/run_ntu2p_trackb.py --protocol subjval --stage summary
```

条件分支：

- **`pending_nfe_rerun`**：执行 summary 给出的 `rerun_command`（即 `--stage eval --nfe <100|200> --arms <臂> --device cuda:0`），然后重跑 summary。
- **`contingency_required`**：summary 的 `contingency_arms` 列出被触发的臂。只对这些臂依次执行：
  1. `--stage train --arms <被触发臂> --device cuda:0`；
  2. `--stage eval --arms <被触发臂> --device cuda:0`；
  3. `--stage summary`，被触发的臂会自动汇总；
  4. `--stage review --device cuda:0`，审查的是 decision_arm；
  5. 写好 verdict 后再跑一次 summary。
- **采纳后回到原协议**：若结论为 `adopt_probabilistic_only`，须经用户确认，并在每条命令后加 `--confirm_probabilistic_only`。`--arms` 缺省即为采纳臂，传其它臂会被拒绝。
  1. `$PY scripts/run_ntu2p_trackb.py --protocol original --stage fold_models --device cuda:0`，随后依次 `--stage bank`、`--stage train`；
  2. `--protocol original --stage eval --test --device cuda:0`：唯一一次 test，设置从 subjval summary 冻结读取；
  3. `--protocol original --stage summary`。

## 8. 仍待主会话处理

- 实现文档第 5 节的 GPU 命令已被本文第 7 节取代。主要变化：底座改为 FD2，以及 NFE、review 与应急臂分支。`AGENTS.md` 的入口条目还没有加。
- 第 1/2 批当前已无进程在跑（GPU 255 MiB、0%）。subjval A0 s0–s2 的 EMA 终点已经存在，评估时会自动作为参照。
- 设计文档 3.1、4.2 中写的底座配置名仍是 `A6-F-A5f0.05-A4-GH0.5`，以本文第 5 节为准。
