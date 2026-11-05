# NTU2P 最新主线 vs 初始 baseline：全方位对比与周报（计划）

## 目标

用户要求：把最新训练出来的结果与"一开始做的 baseline"做全方位对比，写周报说明提升在哪里。本轮**只做评估与汇报，不训练新模型、不改模型代码**。

## 口径（先固定，再出数）

- **对象**：NTU120 双人 xyz 预测，`obs10 → future50`，test 1253 条（xsub.test，受试者与 train 不重叠）。InterHuman 线是另一协议，不混入。
- **初始 baseline**：
  - `copy-last`（非学习型 persistence）；
  - 独立单人学习型 baseline（`20260822-102337-…-independent-single-person-baseline-result.md`）。当时的 `cuda_full_s0_5000@3000` 权重已在 20260822 清理；后续所有 refiner/v2 冻结使用的是 `cuda_retrain_s0_5000` 同配置重训权重（test mpjpe 0.2397，原 run 为 0.2442），**同口径对比以后者为准**，原 run 数字只作脚注。
- **最新结果**：NTU2P 当前主线 `A6-F-A5f0.05-A4-GH0.5-FD2`，3 seed × 10000 step，取 `ema/` 终点，报均值 ± 标准差。
- **里程碑链**（每一级都在同一 test 上用同一评估脚本重评，仅作报告，**不用 test 选型**）：
  copy-last → 独立单人 baseline → 首版 residual refiner（inter001, 5000 step）→ 摆动恢复（s2_5, 10000 step）→ + 权重 EMA → + root DCT 头 → v2 InterMixer（GH）→ v2 + FD2（最新）。
- 扩散探索（`L_dm+L_inter`、跨人 attention diffusion、Track B）是否定性结果，数字取自既有文档，不重跑。

## 对比维度

1. 主指标：`xyz_mse / xyz_mae / mpjpe`（均值 ± std，逐 seed 配对 Δ%）。
2. 统计可靠性：对 1253 条样本做配对 bootstrap 的 Δmpjpe 95% CI，以及逐样本胜率。
3. 误差分解：root vs local；分部位（腿/臂/躯干/头）；分 horizon（0.5/1.0/1.5/2.0/2.5 s）；分 26 类动作。
4. 摆动（回归到均值）：`dct_low/mid/high_energy_ratio`、frozen 比例、`passes_full_gate`。
5. 自然度：骨长误差、GT 站定帧脚速、滑行帧比、迈步相位相关、步数、先迈脚正确率。
6. 模型代价：参数量、结构（冻结 5.6M base + 7.1M refiner → 端到端 3.65M）。
7. 视觉证据：同一批 test 案例并排 GT / copy-last / baseline / 最新主线。
8. 诚实边界：列出没有改善或退化的指标（如步数仍低于 GT、出拳幅度偏小、扩散探索未过 gate、交互建模收益不能声称）。

## 做法

- 评估复用 `eval/eval_ntu2p_v2.py --split test`（同一次运行同时给出 model / base / copy-last / GT 的 L2、摆动与自然度）。
- 汇总与图表用一个新脚本 `scripts/summarize_ntu2p_weekly_vs_baseline.py`（读评估 JSON 与导出的数组，输出 markdown 表、bootstrap、分层误差和图），结果放 `results/forecasting/ntu120_label/ntu2p_weekly_20261012/`。
- 审查图复用 `sample/merge_ntu2p_pred_arrays.py` + `sample/render_ntu2p_review_sheet.py`。
- 周报写入 `docs/ai/context/`（带 `<!-- prune:keep -->`，作为长期留存的汇报文档）。

## 验收

- 各里程碑的新评估数字与既有文档（rootdct 0.1845、v2 0.1810、FD2 0.1805、base 0.2397、copy-last 0.2960）一致到约 1e-4；不一致则查明原因再写报告。
- 周报中的每个数字都能追溯到评估 JSON 或既有结果文档；基线口径与脚注写明。
- 不把 L2 提升表述成"交互建模收益"；明确 `inter_loss_weight=0.01` 在归一化下实际等于无 inter 项。
