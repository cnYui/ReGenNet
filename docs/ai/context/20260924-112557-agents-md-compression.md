# AGENTS.md 每日压缩归档（20260924）

- 时间：2026-09-24 11:25:57 +09:00
- 压缩前：50 行、14462 字节
- 压缩后：50 行、14418 字节（`wc -lc AGENTS.md`）

## 被精简的条目

### 1. `## 当前研究入口` 中 NTU 双人 baseline residual-refinement 条目

命中判据：已被后续条目明确取代的"下一步"计划（紧随其后的 `inter_loss_weight=0.01/0.05/0.1` 消融条目已记录该扫描完成及结论）。

原文：

> - NTU 双人 baseline residual-refinement 已完成 CUDA `5000 step`：冻结 independent single-person xyz baseline，新增双流 temporal self-attention、双向 cross-person attention 和零初始化 residual head。1000/2000/3000/4000/5000 五个 val checkpoint 均同时超过 inherited baseline 与 `copy-last` 的 paired `xyz_mse/xyz_mae/mpjpe`，首帧误差保持 0。结果见 `docs/ai/context/20260822-214340-ntu2p-baseline-residual-refinement-training-result.md`；下一步固定其他条件扫描 `inter_loss_weight=0.01/0.05/0.1`，计划见 `docs/ai/context/20260822-214340-ntu2p-residual-inter-loss-ablation-plan.md`。

精简后：

> - NTU 双人 baseline residual-refinement 已完成 CUDA `5000 step`：冻结 independent single-person xyz baseline，新增双流 temporal self-attention、双向 cross-person attention 和零初始化 residual head。1000/2000/3000/4000/5000 五个 val checkpoint 均同时超过 inherited baseline 与 `copy-last` 的 paired `xyz_mse/xyz_mae/mpjpe`，首帧误差保持 0。结果见 `docs/ai/context/20260822-214340-ntu2p-baseline-residual-refinement-training-result.md`；`inter_loss_weight` 消融计划见 `docs/ai/context/20260822-214340-ntu2p-residual-inter-loss-ablation-plan.md`。

两条文档路径均保留在入口中。

## 事实并入

无。扫描取值 `0.01/0.05/0.1` 已在下一条消融结果条目中完整出现。

## 本次刻意保留的内容

- 旧 `L_dm + L_inter` diffusion、显式跨人 attention diffusion 未超过 `copy-last` 的条目：负结果，防止重复尝试或误称 attention 有提升。
- 旧 diffusion 未过 gate 的深度排查条目：记录失败原因（采样与训练目标不对齐、首帧连续性、`L_inter` 尺度、rot6d 误差），属坑。
- `inter_loss_weight` 消融条目中"下一阶段默认保留 `inter_loss_weight=0`，先迁移到 residual diffusion"：虽与后续用户确认的本轮 `0.01` 不同，但属决策历史，residual diffusion（Track B）仍未完成，保留。
- 回归到均值诊断与"补充分解"条目中的成串数值：属反直觉事实（L2 指标对摆动缺失不敏感、velocity/acceleration 项实际不起作用），且含用户确认参数（mpjpe 回退上限 5%、`inter_loss_weight=0.01`）。
- Stage 0 条目中 `passes_full_gate` 的"能量比值 ≥ 0.10"口径：Stage 1 已改为 DCT 分频带 gate，但两者关系需对照理解，有疑问即保留。
- Stage 1–3 条目：含梯度消失、零初始化鞍点、root 误差不饱和/局部误差饱和、不要用加大 long/final 权重压长程 L2 等坑，以及当前最佳 checkpoint 路径与关键数值。
- 训练步数预算条目：现行决策。
- "当前推进中的 NTU 双人 ReGenNet 风格 diffusion"条目：协议定义仍被后续条目引用。
- `## 文档入口` 全部条目：均为仍生效的流程/设施入口，且路径决定对应文档免于清理。
