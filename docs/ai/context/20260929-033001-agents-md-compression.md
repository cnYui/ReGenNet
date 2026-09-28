# AGENTS.md 每日压缩归档（20260929）

- 时间：2026-09-29 03:30（+09:00）
- 压缩前：55 行、22479 字节
- 压缩后：55 行、22234 字节

## 精简的条目

### 1. `## 当前研究入口` 中的"补充分解"条目

命中判据：已被后续条目明确取代的"下一步"计划。括号内列出的调整设计清单是 Stage 0 之前的计划，已由后续 Stage 0/1/2/3 条目与 Track B 条目记录其执行结果（采纳或否定）；完整清单仍保存在所引用的 design 文档中。该条的诊断数值、评估要求、用户确认参数与两个 `docs/ai/context/` 路径全部保留。

原文：

> - 补充分解：学习型模型的 L2 收益主要来自 root（root MSE 相对 copy-last −44%，local −24%）；训练中 velocity/acceleration 项数值上仅占位置 MSE 的 0.6%，实际不起作用；模型输出是幅度约 1/3 的缓慢单调形变（时间 std 比值 0.34）而无振荡（帧差能量比值 0.0055），评估必须同时报告帧差能量口径。摆动恢复的调整设计（评估 gate 加入摆动指标、loss 尺度归一化、局部姿态速度损失、相位无关摆动能量匹配损失、饱和 ramp、正弦位置编码、双头输出、延长 step、Track B residual diffusion 的 mean-of-K 评估协议）与分阶段训练计划见 `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-design.md` 与 `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-training-plan.md`；用户已确认：mpjpe 相对 base 回退上限 5%、本轮 `inter_loss_weight=0.01`。

精简后：

> - 补充分解：学习型模型的 L2 收益主要来自 root（root MSE 相对 copy-last −44%，local −24%）；训练中 velocity/acceleration 项数值上仅占位置 MSE 的 0.6%，实际不起作用；模型输出是幅度约 1/3 的缓慢单调形变（时间 std 比值 0.34）而无振荡（帧差能量比值 0.0055），评估必须同时报告帧差能量口径。摆动恢复的调整设计与分阶段训练计划见 `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-design.md` 与 `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-training-plan.md`；用户已确认：mpjpe 相对 base 回退上限 5%、本轮 `inter_loss_weight=0.01`。

## 事实并入

无。

## 本次刻意保留的内容

- "NTU2P 首次 test 评估"条目：主数字已被 EMA 取代，但含负结果（摆动恢复前 control 在 test 上不过 gate）与"训练逐位确定、方差只来自 seed"的事实。
- "NTU2P 权重 EMA 结论"与"rootdct"条目：主数字已被取代，但承载仍生效的 EMA 协议、余弦衰减不采纳、inter 权重 1/3/10 不采纳、root 线性增长"可修"假设未获支持等负结果。
- "v2 端到端架构"条目：主数字已由 FD2 更新，但含手指占 loss 81%、腿骨被拉长近 2 倍等根因诊断和步态护栏未过的偏离记录。
- 回归到均值诊断条目：其机制解释在 Stage 2 被修正，但仍是定位问题的依据，且修正已在 Stage 2 条目中说明，保留两者以免误读。
- Stage 0 条目中的旧 gate 定义（能量比值、frozen）：Stage 1 已改为 DCT 分频带 gate，但 L2 三项与 mpjpe 回退 5% 部分仍生效，改写有改变含义的风险。
- Stage 2 条目末尾的 Stage 3 驱动与汇总路径：为脚本与结果路径，不是单纯计划。
- "补充分解"条目中"评估必须同时报告帧差能量口径"：Stage 1 已改为 DCT 口径，但改写会新增表述，保留原文。
- `## 文档入口` 全部条目：均为仍在使用的维护流程与索引，删除会使所引用文档失去清理保护。
