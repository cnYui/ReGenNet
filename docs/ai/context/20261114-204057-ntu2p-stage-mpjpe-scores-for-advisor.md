<!-- prune:keep -->

# NTU2P 各阶段 test mpjpe 汇总（供向导师汇报）

用户 2026-11-14：读 `docs/ai/context` 上下文，了解历代模型试验阶段（阶段 1、阶段 2 和现在的模型），整理 mpjpe 分数。起因是导师 gu 在 Slack 上问"各个方法的平均测试 MPJPE 是多少"，并列了四项：最原始的 ReGenNet、对比 1、对比 2、你的方法。导师没有定义"对比 1/2"；本文按下面的阶段划分整理，划分经用户 2026-11-14 确认。

本文只汇总已有结果，没有新训练，也没有新评价。

## 口径

- 任务：NTU120 双人 xyz 预测，观测 10 帧 → 预测 50 帧（20 fps，约 2.5 s），条件为 26 类动作标签，55 关节。
- **只比较 test**：xsub.test 1253 条，受试者与训练集不重叠，指标实现同为 `eval/eval_ntu2p_v2.py` 的 paired mpjpe，单位米，越低越好。
- 早期 val（198 条）的受试者全部在 train 中，会高估收益（v2 在原 val 上 −7.8%，test 上 −3.85%），不能和 test 混比。
- 阶段 1 的"摆动恢复"及之后各行，以及阶段 2 与现在，均为 3 seed × 10000 step、EMA 终点的均值。

## 阶段划分

| 阶段 | 仓库 | 内容 |
|---|---|---|
| 起点 | ReGenNet | ReGenNet 风格 rot6d 扩散，条件为 obs10 + 动作标签，训练目标 `L_dm + L_inter` |
| 阶段 1 | ReGenNet | 确定性 xyz 回归：独立单人 baseline → 残差精修 → 摆动恢复 → EMA → root DCT 头 → v2 InterMixer → FD2 |
| 阶段 2 | NTU2PForecast（2026-11-03 迁出） | 扩散完整预测 BR:ode0（11-04 ~ 11-07 曾带落脚规划层，已按用户要求永久删除） |
| 现在 | NTU2PForecast | BR:ode0 + 训练得到的相位流形步态模块 PGait v4b（2026-11-07 并入） |

## test mpjpe 演进

| 阶段 | 方法 | test mpjpe | vs copy-last | 说明 |
|---|---|---:|---:|---|
| 参照 | copy-last（不学习） | .29604 | — | |
| 起点 | ReGenNet 风格扩散 `L_dm + L_inter` | **.53066** | +79.3% | 1 seed、5000 step，val 选 step 4000、DDIM50；三项主指标都未超过 copy-last |
| 阶段 1 | 独立单人 baseline | .23974 | −19.0% | 后续各行实际冻结的 base；初次 run 为 .24416，权重已清理，单 seed |
| 阶段 1 | + 残差精修 | .22683 | −23.4% | 1 seed、5000 step |
| 阶段 1 | + 摆动恢复 | .19670 ± .00150 | −33.6% | 3 seed |
| 阶段 1 | + 权重 EMA | .18816 ± .00046 | −36.4% | |
| 阶段 1 | + root DCT 头 | .18455 ± .00011 | −37.7% | |
| 阶段 1 | v2 InterMixer | .18098 ± .00042 | −38.9% | 端到端取代"冻结 base + refiner" |
| 阶段 1 末 | **v2 + FD2** | **.18052 ± .00045** | −39.0% | 11-03 前的主线 |
| 阶段 2 | **扩散 BR:ode0** | **.17906 ± .00061** | −39.5% | |
| 阶段 2 | BR:ode0 + 落脚规划层（已删除） | .17971 ± .00057 | −39.3% | 11-07 起永久不用 |
| 现在 | **BR:ode0 + PGait v4b** | **.17989 ± .0005** | −39.2% | 当前主线 |

- 现在相对 BR:ode0 高 0.46%（逐 seed +0.52 / +0.42 / +0.45%），仍比 FD2 低 0.35%。
- **后三行 mpjpe 几乎持平。** 大幅下降发生在更早：.5307 → .2397 → .1805。

## 阶段 2 与现在的价值在脚步，不在 mpjpe

原协议 test，3 seed 均值：

| 方法 | 滑行帧比 | 步数 | 穿地 | jerk |
|---|---:|---:|---:|---:|
| FD2 | .203 | 1.46 | .041 | 20.7 |
| BR:ode0 | .194 | 1.38 | .038 | 15.9 |
| BR:ode0 + 落脚规划层（已删除） | .030 | 1.72 | .015 | 21.8 |
| **BR:ode0 + PGait v4b（现在）** | **.035** | **2.09** | **.011** | 21.0 |
| GT | .030 | 2.19 | — | — |

建议的对外说法见 `NTU2PForecast/docs/ai/context/20261114-203409-session-context-20261107-20261114.md` 第 7 节：mpjpe 与上一版基本持平（高 0.46%），脚步自然度大幅提升；走向和受力反应仍待改进。

## 阶段 1 内部：摆动恢复 Stage 1 / Stage 2（只有 val）

如果导师或用户说的"阶段 1、阶段 2"指 9 月的摆动恢复，这两个阶段只有 val（198 条）数字，不能与上表的 test 混用：

| 阶段 | 最佳配置 | val mpjpe | 说明 |
|---|---|---:|---|
| 对照 | control | .21527 | 摆动 gate 未过 |
| Stage 1 | 局部速度损失 `s1_6_localvel` | .2126 | 7 个单因子全部未过摆动 gate；该阶段 L2 最佳 |
| Stage 2 | 分频带幅度损失 `s2_5_dct1_root01` | **.19014** | 1.2–2 Hz 能量从 GT 的 0.15% 提升到 57.8%，gate 全过 |

## 汇报时要注意的几点

1. **"最原始的 ReGenNet"是本仓库对 ReGenNet 的 NTU 双人移植版**，不是原作者模型在同协议下的复现；只训练了 5000 step、单 seed，其结论不能外推到"ReGenNet 本身"。
2. 阶段 1 到现在的提升不能归因于"交互建模"：没有做同架构屏蔽对方观测的消融，`inter_loss_weight=0.01` 在尺度归一化下实际等于没有 inter 项。
3. 当前主线的主要残余误差是骨盆轨迹：骨盆轨迹全对时 mpjpe .1791 → .1290（−28%）；11-13 的骨盆意图模块 v1 是负结果，待用户选择下一步。
4. 所有数字是确定性点预测口径，不代表对多模态未来的概率质量。

## 来源

- 起点、copy-last：`20260822-102600-ntu2p-diffusion-ldm-inter-full-training-result.md`
- 独立单人 baseline：`20260822-102337-ntu2p-independent-single-person-baseline-result.md`；重训 .23974 与残差精修 .22683：`20261012-104848-ntu2p-weekly-report-vs-baseline-result.md`
- 摆动恢复 .19670：`20260924-130524-ntu2p-final-10k-3seed-test-result.md`
- EMA .18816：`20260924-190149-ntu2p-lr-cosine-ema-result.md`
- root DCT 头 .18455：`20260925-181731-ntu2p-root-dct-head-inter-weight-result.md`
- v2 .18098：`20260926-005610-ntu2p-v2-final-result.md`；FD2 .18052：`20260926-165932-ntu2p-finger-dedup-result.md`
- BR:ode0 与落脚规划层：`/home/rpartx3080/CodeSpace/NTU2PForecast/docs/ai/context/20261104-182659-ntu2p-mainline-test-result.md`
- PGait v4b：`/home/rpartx3080/CodeSpace/NTU2PForecast/docs/ai/context/20261107-112431-ntu2p-phase-gait-v4b-original-protocol-test-result.md`
- 摆动恢复 Stage 1 / 2：`20260902-173335-ntu2p-articulation-recovery-stage1-result.md`、`20260902-192249-ntu2p-articulation-recovery-stage2-result.md`
- 主线说明页（9 代 test mpjpe 条形图）：https://claude.ai/artifact/PWVE9K98X6AJeXiNWa7J3A ；记录 `/home/rpartx3080/CodeSpace/NTU2PForecast/docs/ai/context/20261113-104008-ntu2p-mainline-architecture-and-scores-page-result.md`
