# NTU 双人 diffusion `L_dm + L_inter` 完整训练结果

## 对旧协议的修正

旧协议把 `L_dm-only` 同时超过 `copy-last` 的三项主指标作为训练 `L_dm + L_inter` 的前置硬门槛。这个门槛只能作为算力受限时的工程止损规则，不能作为科学上的必要条件：两个损失目标不同，`L_dm-only` 失败不蕴含加入 `L_inter` 后也失败。

因此本轮按修订协议完整训练 `L_dm + L_inter`，再与同口径 baseline 比较。

## 固定协议

| 项目 | 配置 |
|---|---|
| 数据 | NTU120 xsub 双人 SMPL-X conditioned 数据 |
| manifest | `results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_phase0_gates/manifest_seed0.json` |
| 窗口 | `window_len=60, obs_len=10, pred_len=50` |
| 条件 | 观测 10 帧 + 26 类动作标签 |
| 表示 | 双人 canonical rot6d `[56,12,T]` |
| diffusion | 1000 steps、cosine schedule、`START_X`、`FIXED_SMALL`、MSE、uniform timestep |
| 模型 | latent 256、decoder 4 层、obs encoder 2 层、4 heads、FFN 1024 |
| 优化 | batch size 8、学习率 `1e-4`、5000 steps、seed 0 |
| 设备 | `cuda:0`，NVIDIA GeForce RTX 3080 |
| 损失 | `L_dm + 1.0 * L_inter` |
| 采样 | val、`DDIM50`、`sample_seed=0` |

训练目录：

```text
save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_cuda_full_s0_5000
```

训练从随机初始化开始，5000 step 完成。最终日志 loss finite，日志和 checkpoint 均记录 `device=cuda:0`。

## Baseline 定义

主 baseline 是 `copy-last`，不是另一个训练模型：

```text
取每个样本 obs10 的最后一帧 rot6d
沿时间维复制 50 次作为 future50
使用与模型完全相同的双人 SMPL-X FK 转成 xyz
与真实 future50 计算 xyz_mse、xyz_mae、MPJPE
```

它表示“未来 50 帧保持观测结束时姿态不变”的 persistence 预测，不使用训练、动作标签或随机采样。验证集共 198 条样本，baseline 为：

```text
xyz_mse = 0.0619918184
xyz_mae = 0.1267309117
mpjpe   = 0.2778617330
```

`direct xyz Transformer` 是单独的学习型参考模型，不是主 baseline，也不替代 `copy-last`。

## 验证集结果

固定 `DDIM50` 扫描全部保存 checkpoint：

| checkpoint | xyz_mse | xyz_mae | MPJPE | 三项均超过 `copy-last` |
|---:|---:|---:|---:|---|
| 1000 | 0.1595140513 | 0.2958831408 | 0.6120102053 | 否 |
| 2000 | 0.1384407368 | 0.2723675290 | 0.5654409990 | 否 |
| 3000 | 0.1158248000 | 0.2455418319 | 0.5112328430 | 否 |
| 4000 | 0.1066152040 | 0.2387036420 | 0.4937235920 | 否 |
| 5000 | 0.1102021093 | 0.2405110948 | 0.4963321123 | 否 |

最佳 `xyz_mse`、`xyz_mae` 和 MPJPE 都是 step 4000，但三项仍分别高于 `copy-last`。因此：

```text
L_dm + L_inter 没有超过 copy-last。
```

这次结果只说明当前固定架构、损失权重 `1.0`、5000 step 和 DDIM50 口径下未过三项主 gate；不能推出所有 `L_dm + L_inter` 设计都不可能超过 baseline。

## 结论与后续边界

1. 旧协议的“`L_dm-only` 不过则不训练 `L_dm + L_inter`”作为科学判断是不充分的，已修正为必须实际训练和比较。
2. 修订后的完整训练已执行，`L_dm + L_inter` 在 val 三项主指标均未超过 `copy-last`。
3. 当前不能声称 `L_inter` 带来主指标收益；本轮相对于 `L_dm-only` 的公平比较结果是 `L_dm + L_inter` 仍未通过主 gate。
4. `copy-last` 是主 persistence baseline；direct xyz Transformer 只能作为学习型参考。
5. 未用 test 选择 checkpoint。后续若进行 test，应先在 val 上预先固定 checkpoint、采样步数和 seed，再一次性评估。

## 冻结配置后的 test 结果

根据 val 选择并冻结 `model000004000.pt`、`DDIM50`、`sample_seed=0` 后，在 1253 条 test 样本上评估：

| 方法 | xyz_mse | xyz_mae | MPJPE |
|---|---:|---:|---:|
| `copy-last` | 0.0690160168 | 0.1371341398 | 0.2960368795 |
| `L_dm + L_inter` step 4000 | 0.1257036737 | 0.2569119474 | 0.5306556957 |

test 三项同样全部未超过 `copy-last`。结果文件为：

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_cuda_full_s0_5000_test_step4000_ddim50/metrics_test.json
```

## 结果文件

```text
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_cuda_full_s0_5000_val_step1000_ddim50/metrics_val.json
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_cuda_full_s0_5000_val_step2000_ddim50/metrics_val.json
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_cuda_full_s0_5000_val_step3000_ddim50/metrics_val.json
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_cuda_full_s0_5000_val_step4000_ddim50/metrics_val.json
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_ldm_inter_cuda_full_s0_5000_val_step5000_ddim50/metrics_val.json
```
