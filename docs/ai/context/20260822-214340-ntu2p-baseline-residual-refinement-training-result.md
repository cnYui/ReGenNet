# NTU 双人 baseline residual refinement 训练结果

## 状态

- 第一阶段 xyz residual-refinement 已完成 CUDA `5000 step` 训练。
- 训练分支：`feature/ntu2p-baseline-residual-refinement`。
- baseline checkpoint：`save/forecasting/ntu120_label/ntu2p_independent_single_person_o10_p50_cuda_retrain_s0_5000/model000005000.pt`。
- 训练设备：`cuda:0`；协议固定为 `window_len=60, obs_len=10, pred_len=50`。
- baseline 冻结；`inter_loss_weight=0`；A/B 使用同一份单人模型参数。

## 实现验证

- 1-step CUDA smoke 已通过，checkpoint 保存、加载、评估链路正常。
- residual head 零初始化，首帧 delta 固定为 0；smoke 和完整评估的 `first_step_error=0`。
- 训练全程 loss、delta 正则和 alpha 均为有限值。

## Val 主指标

评估集为 manifest 中完整 `198` 条 val 样本。inherited baseline 和 `copy-last` 在所有 checkpoint 上固定，分别为：

| 模型 | xyz_mse | xyz_mae | mpjpe | alpha | 三项均超过 baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| inherited baseline | 0.041300786 | 0.106072664 | 0.226522913 | - | - |
| copy-last | 0.061991818 | 0.126730912 | 0.277861735 | - | - |
| residual step 1000 | 0.039458301 | 0.102567782 | 0.217578165 | 0.935630 | 是 |
| residual step 2000 | 0.039395435 | 0.102611308 | 0.217419528 | 0.904357 | 是 |
| residual step 3000 | 0.039429495 | 0.102370183 | 0.216900385 | 0.874586 | 是 |
| residual step 4000 | 0.038986248 | 0.101759780 | 0.215624076 | 0.847233 | 是 |
| residual step 5000 | 0.038903786 | 0.101782139 | 0.215586480 | 0.823436 | 是 |

所有 checkpoint 的 `first_step_error=0`，因此该收益不是以破坏观测窗口连续性换来的。最终 checkpoint 的完整指标见：

`results/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_retrain_s0_5000/metrics_val_005000.json`

## 结论与边界

第一阶段证明：在继承单人预测能力的前提下，显式双流 temporal self-attention 与双向 cross-person attention 可以学习出净收益。当前结论只覆盖 deterministic xyz residual refiner，不能直接外推到 diffusion 或声称关系损失已经有效。

下一步仅增加关系一致性损失，固定模型、数据、seed 和训练步数，扫描 `inter_loss_weight=0.01/0.05/0.1`，与本结果分开报告。
