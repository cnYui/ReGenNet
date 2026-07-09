# NTU xyz 最优训练 validation split 理由

## 为什么必须划分 validation

当前目标是搜索六角度 loss 参数并声称“当前实验预算下最优”。

如果继续用 test set 做参数选择，会产生 test leakage：

```text
1. 多次看 test 后再选参数，test 已经变成调参集。
2. 最终 test 指标会高估泛化能力。
3. 后续论文或汇报不能严谨写成最终测试结果。
```

因此需要把原 train cache 再拆成：

```text
train_opt: 只用于训练候选模型
val_opt:  用于调参、early stopping、checkpoint selection、Pareto/front score
test:     最终模型锁定后只评估一次
```

## 推荐比例

推荐默认：

```text
val_ratio = 0.15
train_opt = 约 85%
val_opt = 约 15%
```

当前 train cache：

```text
train_full = 1956 samples
```

按 15% 估算：

```text
train_opt ≈ 1662
val_opt ≈ 294
```

## 为什么不是 20%

20% 的验证集约 391 条，验证更稳一点，但训练集降到约 1565 条。

当前 NTU120 2P cache 的类别很不均衡，历史记录里最少类只有 2 条。训练集本来就小，过大 validation 会进一步削弱少样本类别训练。

因此 20% 不作为默认，只作为敏感性检查：

```text
如果 15% validation 指标波动很大，再试 20% split 复核。
```

## 为什么不是 10%

10% 验证集约 196 条，训练集保留更多，但 validation 太小。

当前要比较：

```text
xyz_mse
mpjpe
long/final
relative/contact
FID/action accuracy
class-wise diagnostics
```

196 条对 overall metrics 勉强可用，但对 contact/action/class-wise 更不稳定。

因此 10% 只适合快速 smoke，不适合作为最优参数选择依据。

## 分层方式

必须按 action label 分层：

```text
每个 label 尽量按 85/15 拆分。
极少样本类别优先保留至少 1 条在 train_opt。
val_opt 缺失的小样本类别必须在 split_summary.json 标注。
```

原因：

```text
1. 不能为了 val 覆盖牺牲少样本类训练。
2. class-wise 指标对少样本类只作为诊断，不驱动全局参数选择。
3. overall metrics 和 copy-last gate 才是 validation 主选择依据。
```

## 最终执行建议

默认执行：

```text
val_ratio = 0.15
split_seed = 0
stratified_by = action label
```

输出：

```text
train_opt_xyz.pt
val_opt_xyz.pt
split_summary.json
```

如果后续发现 validation score 排名不稳定：

```text
1. 增加 seed=1/2 的 formal training。
2. 对 val set 做 bootstrap confidence interval。
3. 必要时补一个 val_ratio=0.20 的敏感性 split，只用于确认结论稳健性。
```
