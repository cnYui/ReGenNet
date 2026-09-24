# AGENTS.md 每日压缩归档（20260925）

- 时间：2026-09-25 03:30:01 +09:00
- 压缩前：52 行、16702 字节
- 压缩后：52 行、16552 字节（运行环境不允许执行 `wc -lc`，按删除字节数推算：删掉 47 + 56 + 47 = 150 字节）

本次只精简了 `## 当前研究入口` 的 2 条，没有整条删除；两条原有的 `docs/ai/context/` 路径都保留了。

## 精简的条目

### 1. 摆动恢复 Stage 3 结果

命中的判据：条目自己已标明"单 seed 20000 step 数字已被下方 3 seed × 10000 终点取代，不再作主数字引用"，其中的相对百分比和摆动指标数值是一串已被取代的过程细节，引用文档 `docs/ai/context/20260902-205844-ntu2p-articulation-recovery-stage3-result.md` 里有完整记录。保留了 mpjpe 0.1789 和"摆动指标保持"这个结论。

原文：

> - 摆动恢复 Stage 3 已完成（单 seed 20000 step 数字已被下方 3 seed × 10000 终点取代，不再作主数字引用）：s2_5 配置 3 seed @5000 mpjpe 0.1911 ± 0.0040、dct_mid 0.59 ± 0.10，全部过 gate；20000 step 达 mpjpe 0.1789（相对 base −21.0%、copy-last −35.6%），摆动指标保持（low 0.595 / mid 0.513 / high 0.0098，frozen 0.123）。误差随 horizon 分解：root 误差近似线性增长不饱和（2.5 s 时 0.179，为长程偏差主因，可修：root 轨迹 DCT 头、新损失下重测 inter 权重）；局部姿态误差 1.0–1.5 s 后饱和于 0.15–0.19（本质多模态，不可由确定性模型修复，交 Track B residual diffusion 并对 ≥1.5 s 改报合理性指标）；不要用加大 long/final 权重压长程 L2。结果与 horizon 曲线见 `docs/ai/context/20260902-205844-ntu2p-articulation-recovery-stage3-result.md`。

精简后：

> - 摆动恢复 Stage 3 已完成（单 seed 20000 step 数字已被下方 3 seed × 10000 终点取代，不再作主数字引用）：s2_5 配置 3 seed @5000 mpjpe 0.1911 ± 0.0040、dct_mid 0.59 ± 0.10，全部过 gate；20000 step 达 mpjpe 0.1789，摆动指标保持。误差随 horizon 分解：root 误差近似线性增长不饱和（2.5 s 时 0.179，为长程偏差主因，可修：root 轨迹 DCT 头、新损失下重测 inter 权重）；局部姿态误差 1.0–1.5 s 后饱和于 0.15–0.19（本质多模态，不可由确定性模型修复，交 Track B residual diffusion 并对 ≥1.5 s 改报合理性指标）；不要用加大 long/final 权重压长程 L2。结果与 horizon 曲线见 `docs/ai/context/20260902-205844-ntu2p-articulation-recovery-stage3-result.md`。

### 2. NTU2P 首次 test 评估（原始权重）

命中的判据：条目已标明"主数字已被下方 EMA 结果取代"。原始权重相对 base/copy-last 的百分比属于已被取代的派生数值，引用的结果文档里有完整记录。保留了原始权重的 val/test mpjpe 绝对值，因为 EMA 条目里"替换 0.1830/0.1967"要用到它们。

原文：

> - NTU2P 首次 test 评估（s2_5 × 3 seed × 10000 step 终点，原始权重；主数字已被下方 EMA 结果取代）：val mpjpe 0.1830 ± 0.0016、test（1253）0.1967 ± 0.0015（相对 base −18.0%、copy-last −33.6%），三项 L2 与摆动 gate 在 val/test 上 3/3；摆动恢复前的 control 在 test 上不过 gate（dct_mid 0.0017）。训练逐位确定，方差只来自 seed。计划、结果与 test horizon 曲线见 `docs/ai/context/20260924-113322-ntu2p-final-10k-3seed-test-eval-plan.md`、`docs/ai/context/20260924-130524-ntu2p-final-10k-3seed-test-result.md`。

精简后：

> - NTU2P 首次 test 评估（s2_5 × 3 seed × 10000 step 终点，原始权重；主数字已被下方 EMA 结果取代）：val mpjpe 0.1830 ± 0.0016、test（1253）0.1967 ± 0.0015，三项 L2 与摆动 gate 在 val/test 上 3/3；摆动恢复前的 control 在 test 上不过 gate（dct_mid 0.0017）。训练逐位确定，方差只来自 seed。计划、结果与 test horizon 曲线见 `docs/ai/context/20260924-113322-ntu2p-final-10k-3seed-test-eval-plan.md`、`docs/ai/context/20260924-130524-ntu2p-final-10k-3seed-test-result.md`。

## 并入其它条目的事实

无。

## 本次刻意保留的内容

- **首次 test 评估条目整条保留**：虽然主数字已被取代，但它记录了负结果（control 在 test 上不过 gate）、"训练逐位确定，方差只来自 seed"这一事实，以及两份文档的保护路径。
- **Stage 3 条目里的 horizon 分解**：root 误差线性增长、局部误差会饱和、不要加大 long/final 权重，这些是仍然有效的坑和下一步方向。
- **Stage 0 条目里旧的 `passes_full_gate` 定义（能量比值 ≥ 0.10）**：Stage 1 已把 gate 改成 DCT 口径，但 Stage 0 条目还记录了训练开关、默认值下逐位等价这类仍然有效的实现事实。只删旧 gate 容易误改含义，所以保留。
- **residual refinement 条目里的"inter_loss_weight 消融计划"路径，以及消融条目里的"默认保留 inter_loss_weight=0，先迁移到 residual diffusion"**：Track B residual diffusion 还没完成。和之后用户确认的本轮 `inter_loss_weight=0.01` 放在一起看，能说清楚参数是怎么演变的。
- **旧 diffusion、显式跨人 attention diffusion 未超过 copy-last 的负结果**，以及 `L_dm-only` 不作 gate、必须用 `--device cuda:0`：都是现行协议或负结果。
- **回归到均值诊断及其定量数值**：说明 L2 指标对视觉质量不敏感，是反直觉事实。
- **`## 文档入口` 全部保留**：都是长期有效的索引，也保护着相关文档不被清理。
