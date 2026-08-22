# NTU 双人 residual refinement 关系损失消融计划

## 目标

第一阶段 `inter_loss_weight=0` 已在所有保存 checkpoint 上超过 inherited single-person baseline。下一阶段验证关系距离监督是否还能带来额外收益，同时避免把关系损失退化误判为 cross-person attention 无效。

## 固定项

- 模型：`NTU2PResidualRefinerXYZ`，继承同一份冻结 baseline checkpoint。
- 数据：`manifest_seed0.json`，`obs10 -> future50`，完整 198 条 val。
- 设备：`cuda:0`。
- seed：`0`；batch size：`8`；训练步数：`5000`；保存间隔：`1000`。
- 绝对 xyz、速度、加速度、root、local pose、长时域、末帧和 delta 正则权重保持默认值。
- 只改变 `inter_loss_weight`：`0.01`、`0.05`、`0.1`。

## 评估 gate

每个权重的 `1000/2000/3000/4000/5000` checkpoint 都评估完整 val，并记录：

- paired `xyz_mse`、`xyz_mae`、`mpjpe`；
- `first_step_error`、`alpha`、关系距离一致性；
- 相对 inherited baseline、`copy-last` 的三项主指标比较。

只有在不破坏首帧连续性且三项主指标稳定优于 `inter_loss_weight=0` refiner 时，才将关系损失带入后续 diffusion residual 设计；否则保留 `lambda=0`，将关系信息交给 cross-person attention 学习。
