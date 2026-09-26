# AGENTS.md 每日压缩归档（20260927）

- 时间：2026-09-27 03:30（+09:00）
- 压缩前：55 行，22931 字节
- 压缩后：55 行，22702 字节（`wc -lc AGENTS.md`）

本次只精简 `## 当前研究入口` 中 2 条条目里已被后续条目取代的"下一步"句子，未整条移除任何条目，所有 `docs/ai/context/` 路径均保留在入口中。

## 精简的条目

### 1. residual refinement `inter_loss_weight` 消融条目

命中判据：已被后续条目明确取代的"下一步"计划。"下一阶段默认保留 `inter_loss_weight=0`"已被摆动恢复条目中用户确认的"本轮 `inter_loss_weight=0.01`"及 rootdct 条目的"`inter_loss_weight` 保持 0.01"取代；"先迁移到 residual diffusion"已由 Track B 条目记录完成（不采纳）。保留该句会误导下一会话把默认值设为 0。

原文：

> - NTU 双人 residual refinement 的 `inter_loss_weight=0.01/0.05/0.1` 消融已完成。`0.01` 的最终 5000 step 最好，但没有在所有中间 checkpoint 稳定超过 `lambda=0`；`0.05/0.1` 中后期主指标退化。下一阶段默认保留 `inter_loss_weight=0`，先迁移到 residual diffusion；完整结果见 `docs/ai/context/20260822-222701-ntu2p-residual-inter-loss-ablation-result.md`。

精简后：

> - NTU 双人 residual refinement 的 `inter_loss_weight=0.01/0.05/0.1` 消融已完成。`0.01` 的最终 5000 step 最好，但没有在所有中间 checkpoint 稳定超过 `lambda=0`；`0.05/0.1` 中后期主指标退化。完整结果见 `docs/ai/context/20260822-222701-ntu2p-residual-inter-loss-ablation-result.md`。

### 2. NTU2P v2 端到端架构条目（末句）

命中判据：已被后续条目明确取代的"下一步"计划。三项候选均已在下一条"NTU2P 当前主线与主数字（v2 + 手指损失去重 FD2）"中记录完成：手指损失去重（FD2 采纳）、受试者留出 val（选型协议已改）、Track B residual diffusion（按预登记规则不采纳）。

被删除的原文（条目其余内容不变）：

> 采纳决定见 `docs/ai/context/20260926-100701-ntu2p-v2-mainline-adoption-decision.md`；下一步候选：Track B residual diffusion（以 v2 终点为确定性底座）、手指损失去重、从 train 留出受试者作选型 val。

精简后：

> 采纳决定见 `docs/ai/context/20260926-100701-ntu2p-v2-mainline-adoption-decision.md`。

## 事实并入

无。未向其它条目并入新内容。

## 本次刻意保留的内容

- 旧 diffusion（"当前推进中的 NTU 双人 ReGenNet 风格 diffusion"、`L_dm + L_inter` 结果、显式跨人 attention diffusion）：虽已不是主线，但包含协议定义、负结果（未超过 `copy-last`）与深度排查结论，属于负结果与坑。
- 回归到均值诊断、Stage 0–3 摆动恢复条目：含 gate 定义、指标口径（DCT 分频带）、反直觉机制（鞍点、能量型损失梯度消失）及"不要用加大 long/final 权重压长程 L2"等禁令。
- 首次 test 评估、EMA、rootdct 条目：主数字虽被取代（条目内已注明），但保留 EMA 协议、inter 权重负结果与"root 线性增长可修"假设未获支持等结论。
- 训练步数预算、CUDA 强制、受试者留出 val 协议、`pgrep -f` 死锁坑、Track B 不采纳结果：仍生效的决策与坑。
- v2 条目的其余内容：含自然度诊断（手指占 loss 81%、骨长个别帧拉长）、步态护栏偏离与 val 高估说明，属于反直觉事实。
- `## 文档入口` 全部条目：为长期记忆与维护流程文档提供免清理引用。
