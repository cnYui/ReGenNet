# NTU 双人 diffusion 未超过新 baseline 的深度排查

## 结论

新的主要 baseline 是参数共享的独立单人 xyz 预测器：Person A、B 各自只读取本人 obs10 与动作标签，分别预测 future50，再拼回 paired xyz。其冻结 test 指标为：

```text
xyz_mse = 0.0490373378
xyz_mae = 0.1152569476
mpjpe   = 0.2441592532
```

当前双人 diffusion 未超过它，不是因为双人关系信息没有价值，而是因为 diffusion 的训练目标、生成表示和最终自由采样链路与 paired xyz 评估不匹配。证据是同口径的双人 direct xyz Transformer 已达到：

```text
xyz_mse = 0.0456478547
xyz_mae = 0.1138147044
mpjpe   = 0.2387418663
```

它已经略优于 independent single-person baseline，说明“联合读取两个人的观测”本身可以带来收益。

## 固定比较口径

所有结果使用相同 manifest、`window_len=60`、`obs_len=10`、`pred_len=50`、NTU120 xsub split、双人 SMPL-X FK 和 paired `xyz_mse/xyz_mae/mpjpe`。

| 方法 | val/test | xyz_mse | xyz_mae | MPJPE |
|---|---|---:|---:|---:|
| independent single-person xyz | test | 0.0490373378 | 0.1152569476 | 0.2441592532 |
| direct joint xyz | test | 0.0456478547 | 0.1138147044 | 0.2387418663 |
| copy-last | val | 0.0619918184 | 0.1267309117 | 0.2778617330 |
| `L_dm-only` diffusion, step 4000 DDIM50 | val | 0.0557103231 | 0.1647533695 | 0.3395019423 |
| `L_dm + L_inter`, step 4000 DDIM50 | val | 0.1066152037 | 0.2387036424 | 0.4937235920 |

`L_dm-only` 只在 MSE 上优于 copy-last；MAE、MPJPE 未通过。`L_dm + L_inter` 三项均未通过 copy-last，更不可能超过新的 independent baseline。

## 证据一：输入与评估口径

独立单人入口 `train/train_ntu2p_independent_single_person.py` 使用同一 manifest，并将 paired xyz 拆成两个 `[B, T, 1, 55, 3]` 输入；Person A 的输出不依赖 Person B 的观测，A/B 共享模型参数。评估入口 `eval/eval_ntu_2p_forecasting_diffusion.py` 对 independent、direct、diffusion 使用同一 paired xyz 指标和同一 copy-last 对照。已通过输入隔离检查，未发现 A/B 拼接或指标口径错误。

## 证据二：确定性 xyz 与 rot6d diffusion 的根本差异

独立 baseline 和 direct xyz 都在最终评估空间 xyz 中直接预测，且模型输出为相对 obs 最后一帧的位移；初始输出严格等价于 copy-last，第一预测帧误差为 0。

当前 diffusion 在 canonical 双人 rot6d `[56,12,T]` 上训练 `START_X` MSE：

```text
训练输入：q_sample(real_future, t)
测试输入：纯高斯噪声 -> DDIM50 多步反向采样
```

它没有把 obs 最后一帧作为硬约束或 inpainting 条件，也没有 first-frame continuity loss。因此，即使 rot6d 去噪损失下降，最终 xyz 样本也可以从第一帧开始跳离观测结束状态。

验证集 step4000 的 first-frame 检查：

| 方法 | first_step_error | 首帧 xyz MSE（相对 obs 最后一帧） |
|---|---:|---:|
| copy-last | 约 0 | 约 0 |
| independent single-person | 0 | 0 |
| `L_dm-only` DDIM50 | 0.274053 | 0.034374 |
| `L_dm + L_inter` DDIM50 | 0.451695 | 0.088819 |

这是当前主指标失败的最直接原因：50 帧误差从第 1 帧就包含一个 deterministic baseline 没有的跳变。

## 证据三：训练目标与自由采样不完全对齐

对 step4000 checkpoint 做 teacher-forced `t=0` 检查，即把真实 future 加极小噪声后送入模型：

| 方法 | teacher-forced xyz_mse | 自由 DDIM50 xyz_mse |
|---|---:|---:|
| `L_dm-only` | 0.044903 | 0.055710 |
| `L_dm + L_inter` | 0.097134 | 0.106615 |

模型训练看到的是带真实 future 信息的 `q_sample(real_future,t)`；测试却从纯噪声开始，模型误差在多步链路中累积。训练 loss finite 或 rot MSE 下降不能推出 free sampling 的 xyz/MPJPE 通过。

## 证据四：当前 `L_inter` 权重没有按尺度校准

训练实现为：

```text
L_all = L_dm + 1.0 * (joint_mse + orient_mse + trans_mse)
```

`L_inter` 三项直接相加，量纲和归一化方式与 rot6d `L_dm` 不同。训练日志中 `L_dm` 通常约 `0.015~0.03`，而 `L_inter` 通常约 `0.08~0.25`，因此总梯度长期由交互损失主导。

这不是“关系损失无效”的证据，而是当前权重/尺度会牺牲个体 xyz 拟合。step4000 的 teacher-forced rot MSE：

```text
L_dm-only       0.008669
L_dm + L_inter  0.016175
```

自由 DDIM50 也同步恶化：`xyz_mse 0.055710 -> 0.106615`。关系辅助项虽改善了部分相对关系指标，但不足以抵消逐关节 xyz、MAE、MPJPE 的损失。

## 证据五：rot6d 输出的流形问题不是主因，但仍是风险

当前采样结果经过 `rotation_6d_to_matrix` 的 Gram-Schmidt 转换后，矩阵正交误差约 `7e-8`，所以转换函数本身不会产生非法旋转。可是原始预测的 6D 两列不再严格单位正交：

```text
L_dm-only  列范数/内积偏差均值约 0.018~0.021
L_dm+L_inter 偏差均值约 0.042~0.044
真实 target 约 1e-8
```

因此 rot6d 预测仍存在表示回归误差，尤其 `L_inter` 会进一步放大；但在当前证据中，首帧不连续和 free-sampling mismatch 是更直接的主因。

## 为什么“有动作关系”不等于当前模型必然优于 baseline

双人关系只表示联合输入包含额外可利用信息。要把额外信息转化为 paired xyz 三项收益，还需要：

1. 表示和评估空间一致，或有稳定的可逆几何映射；
2. 训练输入分布和最终采样输入分布一致；
3. 首帧连续性不被随机采样破坏；
4. 关系损失与个体运动损失处于可比较尺度；
5. 评估规则明确是单样本、样本均值还是 best-of-N。

当前 diffusion 同时违反前四项中的前三项，并且 `L_inter` 权重未校准，所以不能从“理论上有关系”推出当前实现一定超过 independent baseline。

## 后续修复优先级

当前不应继续把旧 `L_dm + L_inter` 结果作为主模型。最小可验证修复顺序是：

1. 改为 residual-to-last-frame 参数化，或在采样中硬约束第一预测帧等于 obs 最后一帧；
2. 在 rot6d 输出端显式正交化，或直接转到 xyz 空间训练；
3. 将 `L_inter` 拆项归一化，并先扫描 `inter_loss_weight=0.01/0.05/0.1`；
4. 在 val 固定随机噪声，统一比较 teacher-forced、one-step、DDIM5/10/50；
5. 重新训练后再以 independent single-person 为主 gate，同时保留 copy-last 和 direct joint xyz reference。

本轮只完成排查和证据记录，没有把未经验证的修复混入正式训练代码。
