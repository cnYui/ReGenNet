# AGENTS.md 每日压缩归档（20260926）

- 时间：2026-09-26 03:30:01 +09:00
- 压缩前：54 行，19690 字节
- 压缩后：54 行，19560 字节（本次运行 Bash 不可用，字节数按 UTF-8 编码逐段计算得出）
- 范围：只精简 `## 当前研究入口` 中 3 条；未整条移除任何条目，所有 `docs/ai/context/` 路径与 checkpoint 路径原样保留。

## 被精简的条目

### 1. 摆动恢复 Stage 3 条目

命中判据：已被后续条目明确取代的中间状态（单 seed 20000 step 数字已被 3 seed × 10000 终点取代）；已被后续条目记录完成的"下一步"计划（"可修：root 轨迹 DCT 头、新损失下重测 inter 权重"已在 rootdct 条目中完成，假设未获支持）。

原文：

> - 摆动恢复 Stage 3 已完成（单 seed 20000 step 数字已被下方 3 seed × 10000 终点取代，不再作主数字引用）：s2_5 配置 3 seed @5000 mpjpe 0.1911 ± 0.0040、dct_mid 0.59 ± 0.10，全部过 gate；20000 step 达 mpjpe 0.1789，摆动指标保持。误差随 horizon 分解：root 误差近似线性增长不饱和（2.5 s 时 0.179，为长程偏差主因，可修：root 轨迹 DCT 头、新损失下重测 inter 权重）；局部姿态误差 1.0–1.5 s 后饱和于 0.15–0.19（本质多模态，不可由确定性模型修复，交 Track B residual diffusion 并对 ≥1.5 s 改报合理性指标）；不要用加大 long/final 权重压长程 L2。结果与 horizon 曲线见 `docs/ai/context/20260902-205844-ntu2p-articulation-recovery-stage3-result.md`。

精简后：

> - 摆动恢复 Stage 3 已完成：s2_5 配置 3 seed @5000 mpjpe 0.1911 ± 0.0040、dct_mid 0.59 ± 0.10，全部过 gate；20000 step 摆动指标保持（其单 seed 数字不作主数字引用）。误差随 horizon 分解：root 误差近似线性增长不饱和（2.5 s 时 0.179，为长程偏差主因；"可修"假设经 root 轨迹 DCT 头与 inter 权重重测检验未获支持，见下方 rootdct 条目）；局部姿态误差 1.0–1.5 s 后饱和于 0.15–0.19（本质多模态，不可由确定性模型修复，交 Track B residual diffusion 并对 ≥1.5 s 改报合理性指标）；不要用加大 long/final 权重压长程 L2。结果与 horizon 曲线见 `docs/ai/context/20260902-205844-ntu2p-articulation-recovery-stage3-result.md`。

并入说明：rootdct 条目中已有的"Stage 3 'root 线性增长可修'的假设未获支持"以交叉引用形式写入本条，避免下一个会话把旧"可修"计划当作待办。移除的数值：20000 step mpjpe 0.1789（单 seed，已声明不作主数字，完整保存在上述 Stage 3 结果文档）。

### 2. NTU2P 首次 test 评估条目

命中判据：已被后续条目明确取代的中间状态（主数字已被 EMA / rootdct 取代）；成串数值在引用文档中完整保存。

原文：

> - NTU2P 首次 test 评估（s2_5 × 3 seed × 10000 step 终点，原始权重；主数字已被下方 EMA 结果取代）：val mpjpe 0.1830 ± 0.0016、test（1253）0.1967 ± 0.0015，三项 L2 与摆动 gate 在 val/test 上 3/3；摆动恢复前的 control 在 test 上不过 gate（dct_mid 0.0017）。训练逐位确定，方差只来自 seed。计划、结果与 test horizon 曲线见 `docs/ai/context/20260924-113322-ntu2p-final-10k-3seed-test-eval-plan.md`、`docs/ai/context/20260924-130524-ntu2p-final-10k-3seed-test-result.md`。

精简后：

> - NTU2P 首次 test 评估（s2_5 × 3 seed × 10000 step 终点，原始权重；主数字已被下方 EMA 结果取代）：test（1253）0.1967 ± 0.0015，三项 L2 与摆动 gate 在 val/test 上 3/3；摆动恢复前的 control 在 test 上不过 gate（dct_mid 0.0017）。训练逐位确定，方差只来自 seed。计划、结果与 test horizon 曲线见 `docs/ai/context/20260924-113322-ntu2p-final-10k-3seed-test-eval-plan.md`、`docs/ai/context/20260924-130524-ntu2p-final-10k-3seed-test-result.md`。

移除的数值：原始权重 val mpjpe 0.1830 ± 0.0016（保存在 `20260924-130524-ntu2p-final-10k-3seed-test-result.md`）。

### 3. NTU2P 权重 EMA 结论条目

命中判据：已被后续条目取代的主数字（条目自述"主数字已被下方 rootdct 取代"）的成串相对比较数值，引用文档中完整保存。

原文：

> - NTU2P 权重 EMA 结论（主数字已被下方 rootdct 取代，EMA 协议保留）：学习率余弦尾部衰减 × 权重 EMA 的 2×2 对比已完成，按预登记规则（只看 val）采纳 const-EMA，即恒定学习率 + `--ema_decay 0.999`、报告 `ema/` 终点。终点抖动 J 从 0.0031 降到 0.0004（−87%）；val mpjpe 0.1751 ± 0.0014；test（仅评估选中变体）mpjpe **0.1882 ± 0.0005**（相对原始权重 −4.3%、base −21.5%、copy-last −36.4%），xyz_mse/xyz_mae 同步改善，摆动 gate 3/3，dct_mid 不降反升。余弦尾部衰减单独使用未达标（J 仅 −33%），叠加 EMA 也不如只用 EMA，不采纳，开关保留、默认关闭。两个开关关闭时及开启 EMA 时原始权重均与已有 run 逐位相同（`torch.equal`）。EMA 对照 checkpoint 为 `save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_ema_s2_5_dct1_root01_s{0,1,2}_10000/ema/model000010000.pt`。计划、实现、审查跟进与结果见 `docs/ai/context/20260924-154319-ntu2p-lr-cosine-ema-plan.md`、`docs/ai/context/20260924-155201-ntu2p-lr-cosine-ema-implementation-result.md`、`docs/ai/context/20260924-155510-ntu2p-lr-ema-driver-review-followup.md`、`docs/ai/context/20260924-190149-ntu2p-lr-cosine-ema-result.md`（其结论 6 的措辞更正见 `docs/ai/context/20260924-190412-ntu2p-lr-ema-result-review-followup.md`）。

精简后：仅删去"（相对原始权重 −4.3%、base −21.5%、copy-last −36.4%）"，其余不变：

> ……test（仅评估选中变体）mpjpe **0.1882 ± 0.0005**，xyz_mse/xyz_mae 同步改善，摆动 gate 3/3，dct_mid 不降反升。……

移除的数值保存在 `20260924-190149-ntu2p-lr-cosine-ema-result.md`。

## 本次刻意保留的内容

- "当前推进中的 NTU 双人 ReGenNet 风格 diffusion"与"当前新的主要比较 gate 是 L_dm + L_inter…"：措辞看似过时，但承载 diffusion 协议（60/10/50、rot6d、条件与目标）和旧 diffusion 未过 gate 的负结果与根因，后续 Track B 仍会用到。
- 显式跨人 attention diffusion、inter_loss_weight 消融、regression-to-mean 诊断、Stage 1 条目：均为负结果或反直觉事实（梯度消失、GT 帧差能量 2/3 为抖动、L2 指标对摆动不敏感），保留。
- Stage 2 条目末尾的 Stage 3 驱动脚本与汇总路径：是 `scripts/run_ntu2p_articulation_stage3.py` 的唯一入口引用，保留。
- EMA 条目的 val 0.1751 ± 0.0014、test 0.1882 ± 0.0005、J −87%、余弦衰减不采纳、逐位相同及 EMA 对照 checkpoint：仍生效的协议与后续条目比较基准。
- 训练步数预算、rootdct 主数字与 checkpoint、v2 待用户确认条目（未完成事项，含步态护栏未过的偏离说明）、用户确认参数（mpjpe 回退上限 5%、inter 0.01）：全部原样保留。
- `## 文档入口`：各条均为长期流程文档的唯一入口引用，无重复，未改动。
