# NTU2P 摆动恢复 Stage 2：幅度型损失设计与训练计划

前置：`docs/ai/context/20260902-173335-ntu2p-articulation-recovery-stage1-result.md`。
原计划（`20260902-145142-...-training-plan.md` Stage 2）因 Stage 1 的两个诊断做了实质修改，以本文件为准。

## 目标（按修正后的指标）

- `dct_mid_energy_ratio`（1.2–2 Hz 步态/手势频段，位置域）从 **0.0015** 提升到 **≥ 0.10**；
- `dct_low_energy_ratio`（≤1 Hz）从 **0.174** 提升到 **≥ 0.40**；
- `dct_high_energy_ratio` 不应显著上升（> 0.2 视为在产生抖动）；
- 三项 L2 仍低于 copy-last；mpjpe 相对 base 回退 ≤ 5%（用户已确认）。

## 设计

### 指标与 gate

`utils/ntu_smplx_2p_xyz.py` 新增 `dct_matrix / local_pose_dct / dct_band_energies`，频带边界 `NTU_DCT_BAND_EDGES=(1, 6, 11)`：low=k1–5、mid=k6–10、high=k≥11。`compute_ntu_articulation_metrics` 增加六个键，`articulation_ratios` 输出三个频带比值。`eval_ntu2p_residual_refiner_xyz.py` 的 `passes_articulation_gate` 改为 `dct_low ≥ 0.40 且 dct_mid ≥ 0.10`（`--articulation_gate_low_ratio/--articulation_gate_mid_ratio`）。帧差能量与 frozen 保留为报告项。

### 损失

三个新项，均为"幅度"而非"能量"，梯度在 Δ≈0 处不消失（Stage 1 诊断一）：

| 项 | 定义 | 目的 |
|---|---|---|
| `temporal_std` | `mse(std_t(local_pred), std_t(local_target))`，按 (P,J,3) | 位置域幅度，天然抗抖动；但慢速漂移也能满足，不强制振荡 |
| `dct_low_amplitude` | `mse(sqrt(E_low_pred+eps), sqrt(E_low_target+eps))` | 补足 ≤1 Hz 慢速形变幅度 |
| `dct_mid_amplitude` | 同上，k6–10 | **直接要求 1.2–2 Hz 频段有能量**，相位无关 |

不匹配 high 频带：GT 高频主要是拟合抖动，匹配它等于奖励噪声。
归一化常量沿用 copy-last 口径（copy-last 的 std 与非 DC 幅度均为 0，常量即 GT 幅度均方）。`eps=1e-6`。

### 梯度平衡

root 项梯度约为 mse 的 10 倍。增加 `root_loss_weight=0.1` 的对照 run；若有效，Stage 3 再考虑按梯度范数自动平衡。

### 公共底座

Stage 1 中唯一 L2 改进的因子（局部姿态速度损失，−6.1% vs base）与唯一弱有效的结构因子（饱和 ramp）作为所有 Stage 2 run 的底座：
`--loss_scale_normalize --local_velocity_loss_weight 1.0 --ramp_mode saturate --ramp_saturate_frames 5`。
其余条件与 Stage 1 相同（seed 0、`inter_loss_weight 0.01`、5000 step、`delta_reg_weight 0.01`、base 冻结）。

### 预期与风险

- 幅度损失迫使确定性模型对不可预测的相位"表态"，位置 L2 必然上升；权衡由 5% 容忍度界定。模型最可能学到的相位是延续观测末帧的肢体速度方向（物理上合理），并随预测时长衰减幅度。
- `temporal_std` 可能只放大慢速漂移而不产生振荡——因此 `dct_mid_amplitude` 是关键项，`s2_1/s2_2` 与 `s2_3/s2_4` 的对比可判别。
- 若 `dct_high_ratio` 明显上升，说明模型用抖动凑幅度，需加回归一化后的 acceleration 权重或缩小 mid 频带上限。
- 若所有 run 的 `dct_mid_ratio` 仍 < 0.10 且 L2 已到容忍边界，则 H5（L2 点估计对多模态未来必然坍缩）成立，转 Track B residual diffusion。

## 训练计划（7 个 run，约 1.75 h）

驱动：`scripts/run_ntu2p_articulation_stage2.py`；汇总：`results/forecasting/ntu120_label/ntu2p_articulation_stage2/summary.md`。
先用新指标重评对照组与 `s1_3 / s1_6 / s1_7` 的最终 checkpoint（输出 `eval_val_dct_*.json`，不覆写旧文件）。

| run | 在底座上新增 | 检验 |
|---|---|---|
| s2_0_combo | — | 底座本身对 DCT 指标的影响 |
| s2_1_std1 | `temporal_std 1.0` | std 幅度匹配是否只放大漂移 |
| s2_2_std3 | `temporal_std 3.0` | 权重敏感性 |
| s2_3_dct1 | `dct_low 1.0 + dct_mid 1.0` | 分频带幅度匹配能否产生 1–2 Hz 摆动 |
| s2_4_dct3 | `dct_low 3.0 + dct_mid 3.0` | 权重敏感性 / L2 代价 |
| s2_5_dct1_root01 | `dct 1.0/1.0 + root_loss_weight 0.1` | root 梯度降权是否释放 local 学习 |
| s2_6_std1_dct1 | `std 1.0 + dct 1.0/1.0` | 叠加 |

**判定规则（最终 checkpoint，同时看 5 个中间点趋势）：**
- 有效：`dct_mid_ratio ≥ 0.05`（对照的 30 倍以上）且 L2 三项低于 copy-last；
- 达标：`passes_full_gate=true`；
- 抖动告警：`dct_high_ratio > 0.2`；
- 无效：`dct_mid_ratio < 0.01`。

**Stage 3 触发：** 任一 run 达标 → 3 seed 复现 + 随机选例视频人工检查；全部无效 → Track B。
