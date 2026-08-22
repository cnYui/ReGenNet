# NTU 双人 residual refinement 关系损失消融结果

## 实验范围

- 固定模型、baseline checkpoint、manifest、seed、batch size、设备和 5000 step 协议。
- 只改变 `inter_loss_weight`：`0`、`0.01`、`0.05`、`0.1`。
- 每组均保存并评估完整 val 的 1000/2000/3000/4000/5000 checkpoint。
- 所有 checkpoint 的 `first_step_error=0`，训练过程 loss 均 finite。

## 最终 checkpoint 对比

| `inter_loss_weight` | xyz_mse | xyz_mae | mpjpe | inter consistency | 相对 lambda=0 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.038903786 | 0.101782139 | 0.215586480 | 0.012768574 | 参考 |
| 0.01 | **0.038748237** | **0.101623558** | **0.215266959** | 0.012718683 | 三项改善 |
| 0.05 | 0.038936704 | 0.101995355 | 0.215946784 | 0.012731961 | 三项退化 |
| 0.1 | 0.039243901 | 0.102351837 | 0.216827154 | **0.012688722** | 三项退化 |

所有四组最终点均超过 inherited baseline（`xyz_mse=0.041300786`、`xyz_mae=0.106072664`、`mpjpe=0.226522913`）和 `copy-last`（`0.061991818/0.126730912/0.277861735`）。

## 稳定性判断

- `lambda=0.01` 的最终 5000 step 是当前最好点，但在 2000/3000/4000 step 并未同时超过 `lambda=0` 同步 checkpoint，因此不能称为稳定收益。
- `lambda=0.05` 和 `lambda=0.1` 在中后期主指标持续落后 `lambda=0`；关系一致性改善没有转化为绝对 xyz 收益。
- 下一阶段默认仍使用 `inter_loss_weight=0`，先把确定性的 cross-person residual refiner 迁移到 residual diffusion。`lambda=0.01` 作为后续固定最终 checkpoint 的候选消融保留，不作为默认训练配置。

## 结果文件

- `results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_retrain_s0_5000/metrics_val_*.json`
- `results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter001_s0_5000/metrics_val_*.json`
- `results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter005_s0_5000/metrics_val_*.json`
- `results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_inter01_s0_5000/metrics_val_*.json`
