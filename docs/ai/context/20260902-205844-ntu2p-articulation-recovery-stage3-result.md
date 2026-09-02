# NTU2P 摆动恢复 Stage 3（多 seed 复现与 20000 step 长训）结果

前置：`docs/ai/context/20260902-192249-ntu2p-articulation-recovery-stage2-result.md`。
驱动：`scripts/run_ntu2p_articulation_stage3.py --config s2_5_dct1_root01 --seeds 1 2 --long_steps 20000`；汇总：`results/forecasting/ntu120_label/ntu2p_articulation_stage3/summary.md`。
19:30–20:58 在 cuda:0 串行完成（首次启动于 19:24 因 Claude Code 进程退出被中断，残缺目录改名为 `..._s1_5000_interrupted_192725` 保留，未删除）。

## 结论

1. **s2_5 配置在 3 个 seed 上稳定复现**：5000 step 的 mpjpe 为 0.1911 ± 0.0040，1.2–2 Hz 频带比值 0.59 ± 0.10，三个 seed 全部通过 `passes_full_gate`。
2. **20000 step 进一步改善 L2**：mpjpe 0.1901 → **0.1789**（相对 base −21.0%，相对 copy-last −35.6%），xyz_mse 0.0265 → 0.0234；摆动指标保持在 gate 内（low 0.595 / mid 0.513 / high 0.0098）。8000 step 后 mpjpe 在 0.178–0.185 间波动，收益主要来自 5000→8000。
3. 长训后 frozen 比例从 0.096 升到 0.123（GT 0.098），高频带从 0.0068 升到 0.0098，均仍在健康范围，但方向值得关注。

## 3 seed 复现（5000 step）

| seed | xyz_mse | mpjpe | 相对 base | dct_low | dct_mid | dct_high | frozen | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0（Stage 2） | 0.02649 | 0.19014 | −16.1% | 0.470 | 0.578 | 0.0068 | 0.096 | ✓ |
| 1 | 0.02566 | **0.18772** | −17.1% | 0.442 | 0.501 | 0.0062 | 0.086 | ✓ |
| 2 | 0.02696 | 0.19550 | −13.7% | 0.550 | **0.698** | 0.0082 | 0.085 | ✓ |
| 均值 ± 标准差 | 0.02637 ± 0.00066 | 0.1911 ± 0.0040 | — | 0.487 ± 0.056 | 0.593 ± 0.099 | — | 0.089 | 3/3 |

seed 噪声（mpjpe std 0.004）大于 Stage 2 中 s2_3 / s2_5 / s2_6 之间的差异（0.1938 / 0.1901 / 0.1905），三者应视为等价。

## 20000 step 长训（seed 0）

| step | xyz_mse | mpjpe | 相对 base | dct_low | dct_mid | dct_high | frozen |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 0.02649 | 0.19014 | −16.1% | 0.470 | 0.578 | 0.0068 | 0.096 |
| 8000 | 0.02312 | **0.17802** | −21.4% | 0.561 | 0.565 | 0.0061 | 0.123 |
| 10000 | 0.02513 | 0.18485 | −18.4% | 0.491 | 0.521 | 0.0074 | 0.121 |
| 15000 | 0.02487 | 0.18500 | −18.3% | 0.591 | 0.631 | 0.0095 | 0.104 |
| 20000 | **0.02344** | 0.17886 | −21.0% | 0.595 | 0.513 | 0.0098 | 0.123 |

全部 20 个 checkpoint 通过 gate。以 `model000020000.pt` 作为当前最终 checkpoint（与 8000 在噪声内等价，取训练终点便于复现）。

## 误差随 horizon 的分解（val 198，单位 m）

脚本：scratchpad `horizon_error.py`（结果 `results/forecasting/ntu120_label/ntu2p_articulation_stage3/error_vs_horizon_s2_5_vs_20k.{png,json}`，Stage 2 版本在 `.../ntu2p_articulation_stage2/error_vs_horizon_control_vs_s2_5.{png,json}`）。

| 第 N 帧 | 指标 | copy-last | control | s2_5 @5k | **20k** |
|---:|---|---:|---:|---:|---:|
| 10 (0.5 s) | mpjpe / root / local | 0.153 / 0.066 / 0.119 | 0.144 / 0.063 / 0.114 | 0.137 / 0.066 / 0.109 | **0.128 / 0.060 / 0.104** |
| 20 (1.0 s) | mpjpe / root / local | 0.257 / 0.134 / 0.174 | 0.213 / 0.103 / 0.160 | 0.187 / 0.103 / 0.140 | **0.178 / 0.094 / 0.138** |
| 30 (1.5 s) | mpjpe / root / local | 0.338 / 0.208 / 0.192 | 0.254 / 0.146 / 0.173 | 0.218 / 0.135 / 0.153 | **0.209 / 0.126 / 0.152** |
| 50 (2.5 s) | mpjpe / root / local | 0.439 / 0.326 / 0.194 | 0.318 / 0.222 / 0.183 | 0.277 / 0.207 / 0.184 | **0.253 / 0.179 / 0.165** |
| 50 (2.5 s) | 双人相对 root | 0.509 | 0.301 | 0.251 | **0.221** |

两条结构性结论（用户在视频最后一帧观察到的"越远偏得越多"）：

1. **root 误差随 horizon 近似线性增长、不饱和**（20k 模型约 3.5 mm/帧）。到 2.5 s 时 root 误差 0.179 是 mpjpe 0.253 的主要组成；copy-last 为 0.326，模型只追回约 45%。这是长程偏差中**可修**的部分：观测仅 10 帧（0.5 s）限制了速度估计；候选方向为 root 轨迹独立 DCT 低频头、在新损失下重测 `inter_loss_weight`（旧消融在鞍点时代完成，不再适用）。
2. **局部姿态误差在第 20–25 帧后饱和**（所有模型收敛于 0.15–0.19）。20k 模型把第 50 帧局部误差从 0.184 降到 0.165，首次明显低于 copy-last（0.194）与 control（0.183），但 1.5 s 之后的姿态本质多模态，确定性单轨迹无法覆盖。这是**不可修**的部分，应由 Track B（residual diffusion，K 样本、best-of-K/mean-of-K 协议）处理，并在评估中对 ≥1.5 s 区间改报摆动/合理性指标而非 L2。
3. 不建议用加大 `long_loss_weight` / `final_frame_loss_weight` 压长程 L2：会把长程姿态重新压回均值，即 Stage 1 之前问题在长程上的复发。

## 可视化

- 20k 最终 checkpoint 随机 8 例（seed 0，索引 `[10, 66, 98, 103, 107, 124, 130, 194]`，与 Stage 2 的 s2_5 随机组和对照随机组索引相同）：
  `results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s3_s2_5_dct1_root01_s0_20000_visualization_random/tricolor_zflip/videos/`
- 三组同索引视频可逐案例并排：对照 `..._inter001_s0_5000_visualization_random/`、s2_5 `..._s2_5_dct1_root01_s0_5000_visualization_random/`、20k 如上。

## 对 AGENTS.md 主线的影响

- 当前 NTU2P 主线最佳 checkpoint：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s3_s2_5_dct1_root01_s0_20000/model000020000.pt`。
- 评估口径：三项 L2 + `dct_low/mid/high_energy_ratio` + frozen，gate 见 Stage 2 设计；主指标报告需附 horizon 曲线。
- 下一步候选（未实现）：(a) root 轨迹 DCT 头 + `inter_loss_weight` 重测（针对可修的 root 漂移）；(b) Track B residual diffusion（针对 ≥1.5 s 的姿态多模态），以本 checkpoint 为确定性底座。

## 文件

- 各 run：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s3_s2_5_dct1_root01_s{1,2}_5000/`、`..._s0_20000/`
- 汇总：`results/forecasting/ntu120_label/ntu2p_articulation_stage3/{summary.md,summary.json,driver.log,driver_interrupted_192725.log}`
