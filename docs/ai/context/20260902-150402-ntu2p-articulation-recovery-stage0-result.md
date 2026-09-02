# NTU2P 摆动恢复 Stage 0（工程准备）结果

对应计划：`docs/ai/context/20260902-145142-ntu2p-articulation-recovery-training-plan.md` Stage 0。
用户已确认三项决策：mpjpe 相对 base 回退上限 5%；本轮 `inter_loss_weight=0.01`；直接进入实施。

## 代码改动（默认值均保持历史行为）

| 文件 | 改动 |
|---|---|
| `utils/ntu_smplx_2p_xyz.py` | 新增 `compute_ntu_articulation_metrics`（帧差摆动能量、root 能量、局部姿态时间 std、frozen 比例、root/local MSE 分解，均含 GT 对照值）、`articulation_ratios`、常量 `NTU_ARTICULATION_FROZEN_EPSILON=0.001`。不改动 `compute_ntu_xyz_metrics` 及其稳定 key 集。 |
| `model/forecasting_ntu2p_residual_xyz.py` | 新增 `ramp_mode ∈ {linear, saturate}`、`ramp_saturate_frames`、`future_pos_mode ∈ {learned_zero, sinusoidal}`；ramp 与正弦位置编码以**非持久 buffer** 实现，不进入 state_dict，旧 checkpoint 原样可加载；三个字段写入 `config()`，loader 按缺省值兼容旧 checkpoint。 |
| `train/train_ntu2p_residual_refiner_xyz.py` | loss 重构为 `_raw_terms` + 固定求和顺序 `LOSS_TERM_KEYS`；新增 `--loss_scale_normalize`（以 copy-last 在训练集上的同名误差为常量，用独立数据集实例估计，不消耗训练随机流，常量写入 `args.json`）、`--local_velocity_loss_weight`、`--articulation_energy_loss_weight`、`--ramp_mode/--ramp_saturate_frames/--future_pos_mode`、`--scale_estimate_batches`。 |
| `eval/eval_ntu2p_residual_refiner_xyz.py` | 输出 `articulation_metrics`（model/base/copy_last）与 `articulation_gate`（L2 gate、摆动 gate、base 回退容忍度、`passes_full_gate`）；阈值参数默认 `0.10 / 0.20 / 0.05`。 |
| `sample/export_ntu2p_residual_refiner_xyz_visualization.py` | 新增 `--selection random`（固定 seed 均匀抽样），与原按 L2 改进选例并列。 |
| `scripts/run_ntu2p_articulation_stage1.py` | Stage 1 串行驱动：7 个 run + 对照组全部 checkpoint 评估 + 增量汇总表；可断点续跑。 |

## 等价性测试

基准：改动前用 inter001 checkpoint 在 val 前 3 个 batch（48 条）固化前向输出与 `_loss_terms` 值。改动后同一路径比对：

```text
pred   max_abs_diff = 1.192e-07
base   max_abs_diff = 0
delta  max_abs_diff = 2.980e-08
losses max_abs_diff = 9.313e-10
alpha  0.812927782535553 == 0.812927782535553
```

差异来源为 `linspace` 由运行时 CUDA 计算改为 CPU buffer 后 `.to(device)` 的 ULP 级舍入，远低于 `1e-6` 容差，判定等价。

新模式下 `alpha=0` 等价性：`ramp_mode=saturate, future_pos_mode=sinusoidal` 且 `delta_proj` 随机非零时，`max|pred - base| = 0`。

## 端到端验证

1. 新评估脚本在 inter001 上：`xyz_mse 0.03875 / xyz_mae 0.10162 / mpjpe 0.21527`（与历史一致）；`energy_ratio 0.0055`、`root_energy_ratio 0.0580`、`std_ratio 0.3381`、`frozen_ratio 0.3582`（固定 epsilon=0.001，此前分析用数据推导的 0.00107 得 0.376）；gate 判定 `passes_l2_gate=true, passes_articulation_gate=false, within_base_tolerance=true（−4.97%）, passes_full_gate=false`，符合预期。
   输出：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000/eval_val_with_articulation.json`。
2. 全部新选项打开的 2 step smoke：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_stage0_smoke_s0_2`。归一化常量（前 5 个 batch 估计）：
   `mse 6.8e-2, mae 1.2e-1, root 4.6e-2, local 2.1e-2, velocity 8.8e-4, acceleration 1.6e-3, long 1.3e-1, final 1.7e-1, inter 5.8e-1, local_velocity 9.9e-4, articulation_energy 1.6e-4`。
   其中 velocity/acceleration/local_velocity/energy 的常量比位置项小 2–3 个数量级，量化了"这些项在原权重下几乎不起作用"的判断；归一化后总 loss 从 ~0.26 升到 ~4–6，`clip_grad_norm=1.0` 使有效步长受限，是 S1-5 需要注意的副作用。
   smoke checkpoint 可被评估脚本正常加载，`model_config` 含三个新字段。
3. 随机选例导出：`results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000_visualization_random/source`，indices `[10, 66, 98, 103, 107, 124, 130, 194]`。

## 关于 ramp 的量化

`build_residual_ramp(50, "saturate", 5)` 前 8 帧 `[0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]`，平均放行 0.94；线性 ramp 平均放行 0.50。

## 下一步

启动 `scripts/run_ntu2p_articulation_stage1.py`（串行，预计约 2.3 h），汇总至 `results/forecasting/ntu120_label/ntu2p_articulation_stage1/summary.md`，完成后写 Stage 1 结果文档。
