# NTU2P residual refiner 关节摆动恢复：训练计划

对应设计：`docs/ai/context/20260902-145142-ntu2p-articulation-recovery-design.md`。

## 固定条件（全部 run 一致）

- 协议 `window_len=60, obs_len=10, pred_len=50`，manifest `results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json`，val split 198 条。
- base：`save/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_cuda_retrain_s0_5000/model000005000.pt`，冻结。
- `--device cuda:0`，`seed=0`，`batch_size=8`，`lr=3e-4`，`weight_decay=1e-4`，`clip_grad_norm=1.0`，`inter_loss_weight=0.01`。
- 除被测因子外，其余参数与 `save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000/args.json` 完全相同。
- `save_interval=1000`，每个 checkpoint 均跑标准评估（含新增摆动指标）。
- 耗时基准：5000 step ≈ 14 min，20000 step ≈ 56 min（RTX 3080）。

## 对照组

`ntu2p_residual_refiner_xyz_inter001_s0_5000/model000005000.pt`，无需重跑。当前值：
`xyz_mse 0.03875 / xyz_mae 0.10162 / mpjpe 0.21527`；`articulation_energy_ratio 0.0055`；`temporal_std_ratio 0.338`；`frozen_ratio 0.376`。

## Stage 0：工程准备（无训练，先做）

1. `train/train_ntu2p_residual_refiner_xyz.py` 新增参数，默认值均保持旧行为：
   `--delta_reg_weight`（已有）、`--loss_scale_normalize`（默认 off）、`--local_velocity_loss_weight 0.0`、`--articulation_energy_loss_weight 0.0`、`--ramp_mode linear|saturate`（默认 linear）、`--ramp_saturate_frames 5`、`--future_pos_mode learned_zero|sinusoidal`（默认 learned_zero）。
2. `model/forecasting_ntu2p_residual_xyz.py`：`ramp_mode`、`ramp_saturate_frames`、`future_pos_mode` 进入 `config()`，checkpoint 加载时读取；缺省字段按旧行为解释，保证旧 checkpoint 可加载。
3. 尺度归一化常量：在训练开始时对训练集 GT 一次性统计位置/速度/加速度/局部姿态速度的均方，写入 `args.json`。
4. 评估：`eval/eval_ntu2p_residual_refiner_xyz.py` 输出并入摆动指标与 root/local MSE 分解（复用 `eval/analyze_ntu2p_residual_refiner_articulation.py` 的计算）；新增 gate 字段 `passes_articulation_gate`。
5. 可视化：`sample/export_ntu2p_residual_refiner_xyz_visualization.py` 新增 `--selection random` 选项（固定 seed），与现有按 L2 选例并列输出。
6. 等价性测试（必须通过后才进入 Stage 1）：
   - 默认参数下加载 `inter001` checkpoint，前向输出与改动前逐元素一致（`atol=1e-6`）；
   - 新 loss 权重为 0、`loss_scale_normalize=off` 时，单步训练 loss 与改动前一致；
   - `alpha=0` 严格等价 base 输出的既有测试继续通过。

预计半天工程量。

## Stage 1：单因子判别（7 个 run，约 2.3 h GPU）

| run | 被测因子 | 假设 | 参数 | step |
|---|---|---|---|---:|
| S1-1 | 去正则 | H2 | `delta_reg_weight=0` | 5000 |
| S1-2 | 延长训练 | H1 | 与对照相同 | 20000 |
| S1-3 | 饱和 ramp | H3a | `ramp_mode=saturate, ramp_saturate_frames=5` | 5000 |
| S1-4 | 正弦位置编码 | H3b | `future_pos_mode=sinusoidal` | 5000 |
| S1-5 | 尺度归一化 | H4 | `loss_scale_normalize=on`，其余权重不变 | 5000 |
| S1-6 | 局部姿态速度损失 | H4 | `loss_scale_normalize=on, local_velocity_loss_weight=1.0` | 5000 |
| S1-7 | 摆动能量匹配损失 | H5 | `loss_scale_normalize=on, articulation_energy_loss_weight=1.0` | 5000 |

保存目录命名：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s1_<n>_<factor>_s0_<steps>`。

**判定规则（每个 run 取最终 checkpoint，同时检查 5 个中间 checkpoint 的趋势）：**
- 有效：`articulation_energy_ratio` ≥ 对照的 2 倍（≥ 0.011），且三项 L2 仍低于 `copy-last`。
- 强有效：`articulation_energy_ratio` ≥ 0.05，且 mpjpe 相对 base 回退 ≤ 5%。
- 无效/有害：摆动无提升，或任一 L2 指标高于 `copy-last`。
- S1-2 额外看：20000 step 内 val L2 是否出现回升（过拟合），以及摆动比值随 step 的斜率。
- S1-7 额外看：`acceleration` 归一化后的数值是否显著高于 GT（抖动凑能量的信号）。

## Stage 2：组合与权重扫描（3–5 个 run，约 2 h GPU）

1. S2-1：所有"有效"因子叠加，5000 step。
2. S2-2：同 S2-1，20000 step（若 S1-2 显示延长训练不过拟合）。
3. 若 `L_energy` 为关键因子：`articulation_energy_loss_weight ∈ {0.5, 2.0}` 各一个 run（1.0 已在 S2-1）。
4. 若 S2-1 出现抖动（acceleration 归一化值 > GT 的 1.5 倍）：把 `L_energy` 改为 DCT 前 16 系数幅度匹配，重跑一次。
5. 可选附加因子（仅在 1–4 后仍有余量时）：`decoder_layers=4`；root/local 双头输出。

**Stage 2 目标**：`articulation_energy_ratio ≥ 0.30`，`frozen_ratio ≤ 0.20`，三项 L2 低于 `copy-last`，mpjpe 相对 base 回退 ≤ 5%（回退阈值待用户确认，见设计 A.4）。

## Stage 3：选优、多种子与可视化（约 1.5 h）

1. 用新 gate（L2 硬约束 + 摆动硬约束）在 Stage 2 全部 checkpoint 中选优，而不是只按 L2。
2. 胜出配置补跑 seed 1、2（各 5000 或 20000 step，按 Stage 2 结论），报告 3 seed 均值与标准差。
3. 导出视频：L2 最佳 8 例 + 随机 8 例，`tricolor_zflip` 口径，与 `20260823-084520` 的 8 例并排人工检查摆动。
4. 结果写入新的时间戳 result 文档；更新 AGENTS.md / CLAUDE.md 当前研究入口。

## Stage 4：Track B residual diffusion（另开 design）

触发条件：Stage 2 后确定性路线的 `articulation_energy_ratio` 仍 < 0.30，或视频仍无可辨认的肢体摆动。
评估协议已在设计 E 节固定：mean-of-K（K=10）算 L2，单样本算摆动，同时报告 best-of-K。

## 里程碑与停止条件

- Stage 0 等价性测试不通过 → 不进入训练，先修。
- Stage 1 七个因子全部"无效" → 说明 H1–H5 之外另有原因，回到诊断，重新审视数据侧（如 GT 抖动、canonical 化）。
- Stage 2 达标 → 走 Stage 3，确定性路线作为可展示结果；Track B 转为增强项。
- Stage 2 不达标 → Stage 3 只做记录，直接启动 Stage 4。

## 需用户确认的决策

1. 相对 inherited base 允许的 mpjpe 回退上限（建议 5%）。
2. Stage 1 是否允许并行两卡/两进程（当前只有 cuda:0，默认串行，总计约 2.3 h）。
3. 是否同意本轮 `inter_loss_weight` 固定为 0.01（与最佳 checkpoint 一致），而非 AGENTS.md 所述的"下一阶段默认 0"。
