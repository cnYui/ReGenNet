# NTU 双人 diffusion 协议修订与 baseline 定义

## 对旧协议的判断

旧协议把：

```text
L_dm-only 必须先同时超过 copy-last 的 xyz_mse、xyz_mae、mpjpe
```

设为进入 `L_dm + L_inter` 的硬门槛。

这个规则可以作为算力受限时的工程止损策略，但不能作为科学结论。`L_dm-only` 和 `L_dm + L_inter` 是两个不同模型/损失假设；前者没有超过 baseline，不蕴含后者也不可能超过 baseline。旧规则因此过早阻断了对 `L_inter` 增益的直接检验。

## 修订后的实验问题

固定相同的数据、表示、模型容量、训练预算、随机种子和采样口径，分别训练并比较：

```text
copy-last
direct xyz Transformer
L_dm-only diffusion
L_dm + 1.0 * L_inter diffusion
```

`L_dm-only` 只作为 diffusion ablation，不再作为 `L_dm + L_inter` 的前置通过条件。

## Baseline 定义

### 主 baseline：copy-last

对每个验证样本：

```text
obs_rot6d[..., -1:] -> 沿时间维复制 50 次 -> future50_rot6d
```

然后使用和模型完全相同的 `Rotation2xyz_x(..., num_person=2)` 转为双人 xyz，与真实 future50 计算：

```text
xyz_mse
xyz_mae
mpjpe
```

它表示“未来 50 帧保持观测结束时姿态不变”的 persistence baseline，不使用训练、动作标签或随机采样。

当前 val baseline（198 条样本）为：

```text
xyz_mse = 0.0619918184
xyz_mae = 0.1267309117
mpjpe   = 0.2778617330
```

### 学习型参考：direct xyz Transformer

这是独立的确定性 xyz 预测模型，不等同于 copy-last，也不替代 copy-last。它用于回答“当前数据/指标下，学习型模型是否能超过 persistence baseline”。

## 公平训练比较

`L_dm-only` 与 `L_dm + L_inter` 必须：

- 使用相同 `window_len=60, obs_len=10, pred_len=50`、相同 manifest 和 split。
- 使用相同模型宽度、batch size、学习率、训练步数、seed 和 CUDA 设备。
- 使用相同 `START_X/cosine/1000-step/uniform timestep` diffusion 配置。
- 先分别在 val 选择 checkpoint；禁止用 test 选 checkpoint 或 DDIM 步数。
- 在冻结 checkpoint 与采样配置后，再进行一次 test 评估。

为避免旧实验的恢复状态混淆，本轮 `L_dm + L_inter` 使用独立目录从随机初始化完整训练 5000 step。旧 CPU 中断目录只作为历史诊断证据，不作为本轮正式结果。

## 判定方式

三项指标都报告，不把单一 MSE 改善写成整体超过 baseline：

```text
beats_copy_last.xyz_mse
beats_copy_last.xyz_mae
beats_copy_last.mpjpe
```

只有三项主指标都不差于 copy-last，才可称为“完整主 gate 通过”；即使未通过，也必须报告每项指标和 trade-off，不能据此否定 `L_inter` 可能改善其中部分指标或交互指标。

## 执行目标

本轮完成：

```text
L_dm + L_inter，随机初始化，CUDA，5000 step，完整训练
```

训练完成后先运行固定 val 评估，再决定是否进行 test 和后续语义/视频分析。
