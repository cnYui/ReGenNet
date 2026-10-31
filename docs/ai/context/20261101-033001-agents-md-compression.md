# AGENTS.md 每日压缩归档（20261101）

- 时间：2026-11-01 03:30:01 +09:00
- 压缩前：57 行、23920 字节；压缩后：57 行、23650 字节（`wc -lc AGENTS.md`）。
- 本次精简 2 条（均为原位精简，没有整条移除，所引用的 `docs/ai/context/` 路径全部保留在入口中）。

## 被精简的条目

### 1. 「补充分解」条（`## 当前研究入口`）

命中判据：与其它条目重复；被后续条目明确取代的中间口径（评估"帧差能量口径"已被摆动恢复 Stage 1 条改为 DCT 分频带口径；"残差在时间上严格线性/慢速单调形变"已在 Stage 2 条记录）。

原文：

```
- 补充分解：学习型模型的 L2 收益主要来自 root（root MSE 相对 copy-last −44%，local −24%）；训练中 velocity/acceleration 项数值上仅占位置 MSE 的 0.6%，实际不起作用；模型输出是幅度约 1/3 的缓慢单调形变（时间 std 比值 0.34）而无振荡（帧差能量比值 0.0055），评估必须同时报告帧差能量口径。摆动恢复的调整设计与分阶段训练计划见 `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-design.md` 与 `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-training-plan.md`；用户已确认：mpjpe 相对 base 回退上限 5%、本轮 `inter_loss_weight=0.01`。
```

精简后：

```
- 补充分解：学习型模型的 L2 收益主要来自 root（root MSE 相对 copy-last −44%，local −24%）；训练中 velocity/acceleration 项数值上仅占位置 MSE 的 0.6%，实际不起作用。摆动恢复的调整设计与分阶段训练计划见 `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-design.md` 与 `docs/ai/context/20260902-145142-ntu2p-articulation-recovery-training-plan.md`；用户已确认：mpjpe 相对 base 回退上限 5%、本轮 `inter_loss_weight=0.01`。
```

移除的片段：`模型输出是幅度约 1/3 的缓慢单调形变（时间 std 比值 0.34）而无振荡（帧差能量比值 0.0055），评估必须同时报告帧差能量口径。`（"帧差能量比值 0.0055" 的同一诊断，其中局部摆动能量仅为 GT 的 0.55% 已保留在上一条；成串数值保存在 `20260902-144108-ntu2p-residual-refiner-regression-to-mean-quantitative-result.md`。）

### 2. 「NTU2P 权重 EMA 结论」条（`## 当前研究入口`）

命中判据：引用文档中已完整保存的过程细节（实现等价性验证）；结论（开关保留、默认关闭）与 EMA 协议、checkpoint 路径、全部文档路径均保留。

原文：

```
- NTU2P 权重 EMA 结论（主数字已被下方 rootdct 取代，EMA 协议保留）：学习率余弦尾部衰减 × 权重 EMA 的 2×2 对比已完成，按预登记规则（只看 val）采纳 const-EMA，即恒定学习率 + `--ema_decay 0.999`、报告 `ema/` 终点。终点抖动 J 从 0.0031 降到 0.0004（−87%）；val mpjpe 0.1751 ± 0.0014；test（仅评估选中变体）mpjpe **0.1882 ± 0.0005**，xyz_mse/xyz_mae 同步改善，摆动 gate 3/3，dct_mid 不降反升。余弦尾部衰减单独使用未达标（J 仅 −33%），叠加 EMA 也不如只用 EMA，不采纳，开关保留、默认关闭。两个开关关闭时及开启 EMA 时原始权重均与已有 run 逐位相同（`torch.equal`）。EMA 对照 checkpoint 为 `save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_ema_s2_5_dct1_root01_s{0,1,2}_10000/ema/model000010000.pt`。计划、实现、审查跟进与结果见 `docs/ai/context/20260924-154319-ntu2p-lr-cosine-ema-plan.md`、`docs/ai/context/20260924-155201-ntu2p-lr-cosine-ema-implementation-result.md`、`docs/ai/context/20260924-155510-ntu2p-lr-ema-driver-review-followup.md`、`docs/ai/context/20260924-190149-ntu2p-lr-cosine-ema-result.md`（其结论 6 的措辞更正见 `docs/ai/context/20260924-190412-ntu2p-lr-ema-result-review-followup.md`）。
```

精简后：

```
- NTU2P 权重 EMA 结论（主数字已被下方 rootdct 取代，EMA 协议保留）：学习率余弦尾部衰减 × 权重 EMA 的 2×2 对比已完成，按预登记规则（只看 val）采纳 const-EMA，即恒定学习率 + `--ema_decay 0.999`、报告 `ema/` 终点。终点抖动 J 从 0.0031 降到 0.0004（−87%）；val mpjpe 0.1751 ± 0.0014；test（仅评估选中变体）mpjpe **0.1882 ± 0.0005**，xyz_mse/xyz_mae 同步改善，摆动 gate 3/3，dct_mid 不降反升。余弦尾部衰减单独使用未达标（J 仅 −33%），叠加 EMA 也不如只用 EMA，不采纳，开关保留、默认关闭。EMA 对照 checkpoint 为 `save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_ema_s2_5_dct1_root01_s{0,1,2}_10000/ema/model000010000.pt`。计划、实现、审查跟进与结果见 `docs/ai/context/20260924-154319-ntu2p-lr-cosine-ema-plan.md`、`docs/ai/context/20260924-155201-ntu2p-lr-cosine-ema-implementation-result.md`、`docs/ai/context/20260924-155510-ntu2p-lr-ema-driver-review-followup.md`、`docs/ai/context/20260924-190149-ntu2p-lr-cosine-ema-result.md`（其结论 6 的措辞更正见 `docs/ai/context/20260924-190412-ntu2p-lr-ema-result-review-followup.md`）。
```

移除的片段：`两个开关关闭时及开启 EMA 时原始权重均与已有 run 逐位相同（`torch.equal`）。`（实现等价性验证的过程细节，已完整保存在 `20260924-155201-ntu2p-lr-cosine-ema-implementation-result.md`。）

## 事实并入

无。本次没有把任何事实并入其它条目。

## 本次刻意保留的内容

- `## 稳定约束`、`## 解释边界`、标题与引用块：按范围要求原样保留。
- 「NTU2P 首次 test 评估」条：虽标注"主数字已被下方 EMA 取代"，但含 test 集规模（1253）、"摆动恢复前 control 在 test 上不过 gate（dct_mid 0.0017）"的负结果对照与"训练逐位确定、方差只来自 seed"，保留。
- rootdct 条与 v2 条：主数字已被取代，但仍是后续各条比较的对照基线，且含 inter 权重未采纳、"不能据此声称交互建模收益"、手指占 loss 81% 的根因诊断、滑行帧比护栏未过的偏离记录、可复用管线/工具等负结果与坑，保留。
- 早期 diffusion / attention / 独立单人 baseline / `L_dm-only` gate / 深度排查等条：虽属历史阶段，但含未超过 `copy-last` 的负结果、"不能声称 attention 带来性能提升"、CUDA 显式设备强制与现行比较 gate，保留。
- 「当前最佳 residual refiner 的架构图…」条：措辞已过时，但含可复用 xyz 导出脚本入口，且整条移除会使其引用文档失去入口保护，保留。
- `inter_loss_weight` 消融、回归到均值诊断（第 30 条）、Stage 0–3、步数预算、EMA/rootdct/v2/FD2 主线、Track B 负结果、接触图/双腿/DCT 扩散本轮结论及其审查入口：均为现行协议、gate、checkpoint 路径、用户确认参数或负结果，保留。
- `## 文档入口` 全部条目：每条均引用仍需入口保护的文档（含维护脚本设计与首次运行记录），保留。
