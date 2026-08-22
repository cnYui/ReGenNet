# NTU 独立单人 baseline CUDA 训练与评测结果

## 训练配置

- 命令入口：`python -m train.train_ntu2p_independent_single_person`
- 设备：`cuda:0`，NVIDIA GeForce RTX 3080，PyTorch 1.7.1
- manifest：`results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json`
- 协议：`window_len=60, obs_len=10, pred_len=50`
- 参数共享单人模型：`num_persons=1`，5,637,285 个可训练参数
- 训练：batch 8、AdamW、学习率 `3e-4`、seed 0、5000 step
- 训练目录：`save/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_cuda_full_s0_5000`

checkpoint metadata 明确记录：

```text
representation=independent_single_person_xyz
num_persons=1
person_shared_parameters=true
protocol=ntu120_2p_o10_p50
```

## 工程验证

- CUDA 两步 smoke 通过，loss、checkpoint、val paired metrics 均 finite。
- 已加载 checkpoint 的输入隔离检查通过：只扰动 Person B 的 obs，Person A 预测逐元素不变，Person B 预测改变。
- 预测输出 shape 为 `[B,50,2,55,3]`，不是将最后一帧复制 50 次；时序差分非零。
- `py_compile` 和 `git diff --check` 通过。
- 训练入口强制要求 `--device cuda:0`，CUDA 不可用或传入 `cpu` 会立即失败；独立元数据 smoke 已确认 args/checkpoint 记录 RTX 3080 与 PyTorch 版本。

## Val checkpoint 选择

198 条 val 样本，所有 checkpoint 使用相同 copy-last 对照：

| step | xyz_mse | xyz_mae | MPJPE |
|---:|---:|---:|---:|
| 1000 | 0.0417206082 | 0.1069529379 | 0.2278022673 |
| 2000 | 0.0457454106 | 0.1146336435 | 0.2464844816 |
| 3000 | **0.0384092819** | **0.1045806971** | **0.2209684412** |
| 4000 | 0.0416264635 | 0.1078048370 | 0.2284199179 |
| 5000 | 0.0393212277 | 0.1056095821 | 0.2239838011 |
| copy-last | 0.0619918184 | 0.1267309117 | 0.2778617330 |

按预注册规则冻结 step 3000。

## Test 结果

1253 条 xsub.test 样本，使用冻结的 `model000003000.pt`：

| 方法 | xyz_mse | xyz_mae | MPJPE |
|---|---:|---:|---:|
| independent single person | **0.0490373378** | **0.1152569476** | **0.2441592532** |
| copy-last | 0.0690160168 | 0.1371341398 | 0.2960368795 |

结果文件：

```text
results/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_cuda_full_s0_5000_test_step3000/metrics_test.json
```

三项主指标均优于 copy-last，因此该基线不是 persistence 复制，而是有效的单人 future50 学习预测器。它仍不使用另一人的观测或交互损失，不能据此声称显式交互建模。
