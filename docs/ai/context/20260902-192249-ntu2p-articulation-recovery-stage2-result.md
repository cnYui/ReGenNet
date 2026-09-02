# NTU2P 摆动恢复 Stage 2（幅度型损失）结果

对应设计：`docs/ai/context/20260902-173335-ntu2p-articulation-recovery-stage2-design-and-plan.md`。
驱动：`scripts/run_ntu2p_articulation_stage2.py`；汇总：`results/forecasting/ntu120_label/ntu2p_articulation_stage2/summary.md`。
17:40–19:22 在 cuda:0 串行完成 7 个 run（每个 5000 step，seed 0）。

## 结论

**分频带幅度匹配损失（`dct_low/mid_amplitude`）解决了"整体平移、四肢不动"的问题，且三项 L2 指标同时大幅改善。** 5 个含 DCT 幅度损失的 run 全部通过新 gate；最佳 `s2_5_dct1_root01` 的 mpjpe 相对 inherited base −16.1%、相对 copy-last −31.6%，1.2–2 Hz 频段能量从 GT 的 0.15% 提升到 57.8%，frozen 比例 0.358 → 0.096（GT 0.098），高频带 < 0.01（无抖动）。

## 最终 checkpoint 结果（val 198 条）

| run | 在底座上新增 | xyz_mse | mpjpe | 相对 base | dct_low | **dct_mid** | dct_high | std 比值 | frozen | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control_inter001 | —（无底座） | 0.03875 | 0.21527 | −5.0% | 0.174 | 0.0015 | 0.0004 | 0.338 | 0.358 | ✗ |
| s2_0_combo | — | 0.03205 | 0.21160 | −6.6% | 0.184 | 0.070 | 0.0075 | 0.375 | 0.326 | ✗ |
| s2_1_std1 | std 1.0 | 0.03166 | 0.21264 | −6.1% | 0.189 | 0.097 | 0.0102 | 0.393 | — | ✗ |
| s2_2_std3 | std 3.0 | 0.02807 | 0.19669 | −13.2% | **0.622** | 0.030 | 0.0052 | 0.813 | 0.104 | ✗（mid 不足） |
| s2_3_dct1 | DCT 1.0/1.0 | 0.02577 | 0.19377 | −14.5% | 0.482 | 0.549 | 0.0082 | 0.721 | 0.096 | ✓（3000–5000） |
| s2_4_dct3 | DCT 3.0/3.0 | 0.02622 | 0.19640 | −13.3% | 0.583 | 0.564 | 0.0087 | 0.800 | — | ✓（1000–5000） |
| **s2_5_dct1_root01** | DCT 1.0/1.0 + root 0.1 | 0.02649 | **0.19014** | **−16.1%** | 0.470 | **0.578** | 0.0068 | 0.715 | 0.096 | ✓（2000–5000） |
| s2_6_std1_dct1 | std 1.0 + DCT 1.0/1.0 | **0.02550** | 0.19047 | −15.9% | 0.549 | 0.543 | 0.0069 | 0.779 | — | ✓（1000–5000） |

底座 = `--loss_scale_normalize --local_velocity_loss_weight 1.0 --ramp_mode saturate --ramp_saturate_frames 5`；其余与 Stage 1 相同。
Stage 1 最佳 L2 为 `s1_6_localvel` 的 mpjpe 0.2126；Stage 2 的 s2_5 再降 10.6%。

## 分时段与分量（s2_5 vs control）

| 指标 | copy-last | base | control | s2_5 | s2_5 vs control |
|---|---:|---:|---:|---:|---:|
| short_xyz_mse | 0.0176 | 0.0155 | 0.0151 | 0.0118 | −22% |
| mid_xyz_mse | 0.0601 | 0.0421 | 0.0399 | 0.0273 | −32% |
| long_xyz_mse | 0.1056 | 0.0648 | 0.0599 | 0.0395 | −34% |
| final_frame_error | 0.4389 | 0.3397 | 0.3177 | 0.2772 | −13% |
| root_translation_error | 0.1721 | 0.1357 | 0.1233 | 0.1165 | −6% |
| local_pose_error | 0.1591 | 0.1483 | 0.1465 | 0.1348 | −8% |
| root_mse / local_mse | — | — | 0.0210 / 0.0169 | 0.0151 / 0.0127 | −28% / −25% |
| key_joint_relation_error | 0.3273 | 0.2161 | 0.2155 | 0.1806 | −16% |

增益覆盖整个 horizon，root 与 local 同时改善，双人关节关系误差也下降。

## 视觉验证

- `results/forecasting/ntu120_label/ntu2p_articulation_stage2/joint_traj_random_cases_control_vs_s2_5.png`（脚本 `scripts/plot_ntu2p_joint_trajectories.py`）：随机 3 个案例中人物 A 的手腕/脚踝相对 root 轨迹。**control 的输出是笔直的斜线**（最后一帧的线性外推），s2_5 为与 GT 同量级、同频段的平滑摆动，前 10–25 帧相位多与 GT 一致，之后有时分叉。
- 视频：`results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s2_5_dct1_root01_s0_5000_visualization_best_improvement/tricolor_zflip/videos/`（L2 最佳 8 例）与 `..._visualization_random/tricolor_zflip/videos/`（随机 8 例，seed 0，索引 `[10, 66, 98, 103, 107, 124, 130, 194]`）；对照组同一组随机索引在 `..._inter001_s0_5000_visualization_random/tricolor_zflip/videos/`。

## 对机制的修正

Stage 1 结果文档中"L2 点估计回归到均值"的解释**不完整**。事实是：

1. control 的局部姿态残差在时间上严格线性（轨迹图），即"逐帧相同的 delta × 线性 ramp"。零初始化的 `future_pos` 使 50 个未来 query 初始完全相同，输出只能是线性漂移；振荡分量的位置误差在时间上正负抵消，对共享输出头的梯度为零——这是一个**鞍点**，不是条件均值。
2. 能量型损失在该点梯度同样为零（Stage 1 诊断一）。幅度型（开根号）损失在鞍点处梯度不消失，把模型推出鞍点后，位置 L2 的梯度重新变得有信息量，L2 随之下降——**这解释了为什么加入幅度损失反而使 L2 大幅改善**，而不是预期中的权衡。
3. `temporal_std` 只放大慢速漂移（s2_2：dct_low 0.62 但 dct_mid 0.03），与设计预判一致；`dct_mid_amplitude` 是让 1–2 Hz 摆动出现的关键项。
4. root 降权（s2_5 vs s2_3）进一步改善 mpjpe（0.1938 → 0.1901），与"root 梯度约为 mse 的 10 倍"的诊断一致。
5. 用户设定的"mpjpe 相对 base 回退 ≤ 5%"容忍度未被触发。
6. H5（多模态导致的上限）仍可能存在：dct_mid 停在 0.55–0.58、长 horizon 相位分叉；这是 Track B residual diffusion 的空间，但不再是当前阻塞项。

## 训练趋势

s2_5 五个 checkpoint 的 mpjpe：0.2158 → 0.2024 → 0.1965 → 0.1923 → 0.1901，仍在单调下降；s2_3/s2_6 同样。5000 step 大概率未收敛，Stage 3 加 20000 step 长训。

## Stage 3（已启动）

`scripts/run_ntu2p_articulation_stage3.py --config s2_5_dct1_root01 --seeds 1 2 --long_steps 20000`：seed 1、2 各 5000 step 验证可复现性，seed 0 跑 20000 step；汇总 `results/forecasting/ntu120_label/ntu2p_articulation_stage3/summary.md`。

## 文件

- 各 run：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s2_*_s0_5000/`
- 对照与 Stage 1 代表 run 的 DCT 口径重评：同目录 `eval_val_dct_000005000.json`
- 汇总：`results/forecasting/ntu120_label/ntu2p_articulation_stage2/{summary.md,summary.json,driver.log}`
