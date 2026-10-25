# AGENTS.md 每日压缩归档（20261026）

- 时间：2026-10-26 03:30:01 +09:00
- 压缩前：57 行、23939 字节
- 压缩后：57 行、23920 字节（字节数按被删字符 UTF-8 长度计算：`0.1967 ± 0.0015，` 共 19 字节；本次运行中 Bash 被禁用，未能用 `wc -lc` 复核）

## 被精简的条目

### 「NTU2P 首次 test 评估」条（`## 当前研究入口`）

命中判据：已被后续条目明确取代的中间状态——该条自带"主数字已被下方 EMA 结果取代"，其 test mpjpe 0.1967 ± 0.0015 已被 EMA 条的 0.1882 ± 0.0005、rootdct 条的 0.1845 ± 0.0001 及 v2+FD2 的 0.1805 ± 0.0005 依次取代。

原文：

```
- NTU2P 首次 test 评估（s2_5 × 3 seed × 10000 step 终点，原始权重；主数字已被下方 EMA 结果取代）：test（1253）0.1967 ± 0.0015，三项 L2 与摆动 gate 在 val/test 上 3/3；摆动恢复前的 control 在 test 上不过 gate（dct_mid 0.0017）。训练逐位确定，方差只来自 seed。计划、结果与 test horizon 曲线见 `docs/ai/context/20260924-113322-ntu2p-final-10k-3seed-test-eval-plan.md`、`docs/ai/context/20260924-130524-ntu2p-final-10k-3seed-test-result.md`。
```

精简后：

```
- NTU2P 首次 test 评估（s2_5 × 3 seed × 10000 step 终点，原始权重；主数字已被下方 EMA 结果取代）：test（1253）三项 L2 与摆动 gate 在 val/test 上 3/3；摆动恢复前的 control 在 test 上不过 gate（dct_mid 0.0017）。训练逐位确定，方差只来自 seed。计划、结果与 test horizon 曲线见 `docs/ai/context/20260924-113322-ntu2p-final-10k-3seed-test-eval-plan.md`、`docs/ai/context/20260924-130524-ntu2p-final-10k-3seed-test-result.md`。
```

并入其它条目的事实：无。保留了该条原有的两个 `docs/ai/context/` 路径。

## 本次刻意保留的内容

- 第 22–28 行（L_dm-only 不再是 gate、修订协议、attention diffusion 未超过 copy-last、residual refinement 与 `inter_loss_weight` 消融）：虽属较早阶段，但含负结果与"不能声称 attention 带来提升"、"0.01 未在所有中间 checkpoint 稳定超过 λ=0"等仍生效的措辞边界，保留。
- 第 29–31 行（架构图/导出脚本/视频、回归到均值诊断、补充分解）：诊断与"三项 L2 指标对视觉质量不敏感"是反直觉事实，且后续摆动恢复条目依赖其数值，保留。
- 第 32–35 行（摆动恢复 Stage 0–3）：gate 定义、指标口径、机制修正与"不要加大 long/final 权重"的坑仍生效，保留。
- 第 38、39 行（EMA、rootdct）：虽标注主数字已被取代，但 EMA 协议、余弦尾部衰减/inter 权重的负结果、checkpoint 路径仍有效，保留。
- 第 16–17 行（接触图/扩散本轮负结果）与第 40–41 行（v2、v2+FD2 主线）：现行主线与最新负结果，保留。
- `## 文档入口` 全部条目：每条均含受保护的文档路径，整条移除会使被引用文档失去免清理保护，保留。
