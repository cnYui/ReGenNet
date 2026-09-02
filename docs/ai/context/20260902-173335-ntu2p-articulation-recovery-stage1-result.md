# NTU2P 摆动恢复 Stage 1（单因子判别）结果

对应计划：`docs/ai/context/20260902-145142-ntu2p-articulation-recovery-training-plan.md` Stage 1。
驱动：`scripts/run_ntu2p_articulation_stage1.py`；汇总：`results/forecasting/ntu120_label/ntu2p_articulation_stage1/summary.md`（含全部中间 checkpoint）。
15:06–17:28 在 cuda:0 串行完成 7 个 run。

## 最终 checkpoint 结果（val 198 条）

| run | 因子 | 假设 | mpjpe | 相对 base | 帧差能量比值 | frozen | 判定 |
|---|---|---|---:|---:|---:|---:|---|
| control_inter001 | — | — | 0.2153 | −5.0% | 0.0055 | 0.358 | 对照 |
| s1_1_noreg | `delta_reg_weight=0` | H2 | 0.2158 | −4.7% | 0.0055 | 0.378 | 无效 |
| s1_2_long20k @20000 | 20000 step | H1 | 0.2151 | −5.0% | 0.0054 | 0.370 | 无效 |
| s1_3_ramp_sat5 | 饱和 ramp | H3a | 0.2190 | −3.3% | **0.0078** | **0.327** | 弱有效（+43%，未达 2 倍门槛） |
| s1_4_sinpos | 正弦位置编码 | H3b | **0.2148** | −5.2% | 0.0056 | 0.371 | 摆动无效；L2 略优 |
| s1_5_scalenorm | loss 尺度归一化 | H4 | 0.2211 | −2.4% | 0.0051 | 0.352 | 无效；L2 略差 |
| s1_6_localvel | 归一化 + 局部姿态速度损失 | H4 | **0.2126** | **−6.1%** | 0.0051 | 0.364 | 摆动无效；**L2 历史最佳** |
| s1_7_energy | 归一化 + 帧差能量匹配损失 | H5 | 0.2159 | −4.7% | 0.0051 | 0.357 | 无效（见诊断） |

全部 run 三项 L2 均低于 copy-last；全部未通过摆动 gate。
s1_2 的 20 个 checkpoint（1000–20000）帧差能量比值始终在 0.0050–0.0058，val mpjpe 在 0.2148–0.2196 之间无趋势，训练更久对摆动与 L2 都没有影响。

## 假设判定

- **H1 训练不足：否定。** 4 倍步数无任何趋势。
- **H2 残差正则压小：否定。** 去正则后能量比值与 alpha（0.79 vs 0.81）均无变化。
- **H3b 零初始化位置编码：否定**（对摆动）。正弦编码使 L2 略好。
- **H3a 线性 ramp 抑制：弱成立。** 唯一让摆动稳定上升的因子（5 个 checkpoint 均在 0.0071–0.0081），但绝对量小，且 L2 付出 +1.7% mpjpe。
- **H4 / H5：Stage 1 的实验未能有效检验**，原因见下面两个诊断。

## 诊断一：能量匹配损失在静止解处梯度消失

脚本：scratchpad `diag_grad_norms.py`（对照 checkpoint、训练集 4 个 batch、按 s1_7 的归一化常量，逐项 backward 取可训练参数梯度范数，以 mse 项为 1）：

| 项 | 归一化 loss 值 | 梯度范数 / mse |
|---|---:|---:|
| mse | 0.80 | 1.00 |
| root | 0.85 | **9.83** |
| local | 0.75 | 5.71 |
| inter | 0.48 | 2.36 |
| local_velocity | 0.78 | 2.88 |
| velocity / acceleration | 0.36 / 0.31 | 0.19 / 0.25 |
| **articulation_energy** | 0.28 | **0.34** |
| 候选：sqrt(能量) 幅度匹配 | 0.76 | **8.80** |
| 候选：局部姿态时间 std 匹配 | 0.63 | **8.81** |

- 能量 `E = mean_t ||Δ||²` 是 Δ 的二次型，`∂E/∂Δ ∝ Δ`；模型输出 Δ≈0 时梯度≈0。归一化让 loss **值**到 O(1)，但推不动参数。这是 s1_7 无效的直接原因，**不是 H5 的证据**。
- 开根号的幅度型损失（RMS 或 std）梯度不消失，梯度范数与 root 项同量级。
- **root 项梯度是 mse 的 9.8 倍**（root 只有 1 个关节，每元素梯度是全身项的 55 倍；按 loss 值归一化不能消除这一点）。这量化了 H4 的机制，也解释了 s1_5 为何几乎无效：归一化平衡了 loss 值，没有平衡梯度。

## 诊断二：GT 帧差能量的约 2/3 是拟合抖动

脚本：scratchpad `diag_spectrum.py`（val 198 条局部姿态）：

| 平滑窗口 | GT 帧差能量 | model / GT | base / GT |
|---|---:|---:|---:|
| 1（原始） | 9.47e-4 | 0.0055 | 0.0050 |
| 3 | 3.11e-4 | 0.0168 | 0.0154 |
| 5 | 2.01e-4 | 0.0260 | 0.0238 |
| 7 | 1.51e-4 | 0.0346 | 0.0317 |

GT `|acc|/|vel|` = 1.34（原始）→ 0.69（w3）→ 0.52（w5），原始序列的主频约 4 Hz，远高于人体肢体运动的 1–2 Hz。

位置域 DCT（T=50 @20 FPS，系数 k ↔ k×0.2 Hz）非 DC 能量分布与模型占比：

| 频带 | GT 占非 DC 能量 | model / GT | base / GT |
|---|---:|---:|---:|
| k1–5（≤1 Hz） | 88.7% | **0.174** | 0.160 |
| k6–10（1.2–2 Hz） | 5.8% | **0.0015** | 0.0013 |
| k11–20（2.2–4 Hz） | 2.6% | 0.0008 | 0.0007 |
| k21–49（>4 Hz） | 3.0% | 0.00006 | 0.00006 |

结论：
- 帧差能量按 ω² 加权，被抖动主导；"0.55%"夸大了差距。位置域分频带才是公平口径。
- 校正后的图景：模型复现了 GT **17% 的 ≤1 Hz 慢速形变**（即观察到的"缓慢挪向均值姿态"），在 **1.2–2 Hz 步态/手势频段只有 0.15%**，更高频段为 0。定性结论不变：没有任何摆动。
- s1_7 的目标是原始帧差能量，等于要求模型复现噪声；即便有效也会产生抖动。

## 对设计的三点修正（进入 Stage 2）

1. **指标**：`compute_ntu_articulation_metrics` 新增 `dct_low/mid/high_energy` 及 GT 对照与比值；gate 改为 `dct_low_ratio ≥ 0.40` 且 `dct_mid_ratio ≥ 0.10`（当前 0.174 / 0.0015）；帧差能量与 frozen 仅报告不进 gate。
2. **损失**：能量型改为幅度型——`--temporal_std_loss_weight`（位置域时间 std 匹配，天然抗抖动）与 `--dct_low/mid_amplitude_loss_weight`（分频带 RMS 幅度匹配，相位无关、频段可控，不匹配 high 频带以免奖励抖动）。
3. **梯度平衡**：root 项梯度约为 mse 的 10 倍，Stage 2 增加 `root_loss_weight=0.1` 的对照。

## 附带的正面结果

`s1_6_localvel`（归一化 + 局部姿态速度损失 w=1）给出本项目 NTU2P 主线迄今最佳 L2：`xyz_mse 0.03664 / xyz_mae 0.10062 / mpjpe 0.21260`，相对 base −6.1%、相对 copy-last −23.5%，五个 checkpoint 中后四个均优于对照。作为 Stage 2 公共底座保留。

## 文件

- 各 run：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s1_*_s0_*/`（含 `args.json`、`train_log.jsonl`、`eval_val_*.json`）
- 汇总：`results/forecasting/ntu120_label/ntu2p_articulation_stage1/{summary.md,summary.json,driver.log}`
- Stage 2 设计与计划：`docs/ai/context/20260902-173335-ntu2p-articulation-recovery-stage2-design-and-plan.md`
