# NTU xyz formal training 与 B-lite 调参结果

## Formal Training 目标

在 `train_opt / val_opt` 固定划分下，对 Stage A best 与 baseline 做更长预算 formal training。

固定协议：

```text
train_opt = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/train_opt_xyz.pt
val_opt = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/val_opt_xyz.pt
seed = 0,1,2
num_steps = 3000
eval_interval = 100
save_interval = 100
selection = validation score
test = 未运行
```

原因：

```text
test 只能在最终配置稳定选定后运行一次。
本轮 formal 没有产生稳定、全面优于 baseline 的配置，因此不进入最终 test。
```

## Formal Baseline

配置：

```text
velocity_loss_weight = 0.2
其它新增 loss = 0.0
```

结果：

```text
selection_score_mean = 1.001717443
selection_score_std = 0.005445368
xyz_mse = 0.021571779
xyz_mae = 0.082740905
mpjpe = 0.172388257
final_frame_error = 0.256986865
long_xyz_mse = 0.033021783
relative_root_distance_error = 0.108617873
contact_error = 0.222427123
best_steps = [1200, 1000, 1500]
```

checkpoints：

```text
seed0 = save/forecasting/ntu120_label/xyz_loss_optimal_formal/baseline_v0_s0/model000001200.pt
seed1 = save/forecasting/ntu120_label/xyz_loss_optimal_formal/baseline_v0_s1/model000001000.pt
seed2 = save/forecasting/ntu120_label/xyz_loss_optimal_formal/baseline_v0_s2/model000001500.pt
```

观察：

```text
best step 全部早于 3000，说明继续训练到 3000 会出现 validation 回退。
```

## Formal Stage A Best

配置：

```text
velocity_loss_weight = 0.2
long_loss_weight = 0.05
final_frame_loss_weight = 0.075
其它新增 loss = 0.0
```

结果：

```text
selection_score_mean = 1.001207465
selection_score_std = 0.012837559
xyz_mse = 0.021498282
xyz_mae = 0.083347645
mpjpe = 0.173642244
final_frame_error = 0.256922971
long_xyz_mse = 0.032756051
relative_root_distance_error = 0.107690318
contact_error = 0.231516586
best_steps = [1400, 2400, 800]
```

checkpoints：

```text
seed0 = save/forecasting/ntu120_label/xyz_loss_optimal_formal/stageA_lf_l050_f075_s0/model000001400.pt
seed1 = save/forecasting/ntu120_label/xyz_loss_optimal_formal/stageA_lf_l050_f075_s1/model000002400.pt
seed2 = save/forecasting/ntu120_label/xyz_loss_optimal_formal/stageA_lf_l050_f075_s2/model000000800.pt
```

对 baseline：

```text
selection_score: 略好，差距约 0.05%
xyz_mse: 略好
final_frame_error: 基本持平略好
long_xyz_mse: 略好
relative_root_distance_error: 略好
xyz_mae: 变差
mpjpe: 变差
contact_error: 明显变差
```

判断：

```text
Stage A best 不是稳定全面优于 baseline。
如果只按当前 weighted score，它略优；如果按 Pareto 和论文风险，它不能作为最终最优。
```

## B-lite 搜索

由于 Stage A best 的 contact_error 在 formal 中变差，尝试更小的 key/contact 权重。

配置基底：

```text
velocity_loss_weight = 0.2
long_loss_weight = 0.05
final_frame_loss_weight = 0.075
```

候选：

```text
key/contact = 0.0/0.01
key/contact = 0.01/0.0
key/contact = 0.01/0.01
key/contact = 0.025/0.01
key/contact = 0.01/0.025
```

seed0 初筛：

```text
key=0.01, contact=0.0:
  score = 0.977059
  xyz_mse = 0.021478802
  mpjpe = 0.170655126
  final_frame_error = 0.252493292
  contact_error = 0.221581846

key=0.025, contact=0.01:
  score = 0.981285
  xyz_mse = 0.021204962
  mpjpe = 0.173911445
  final_frame_error = 0.256818252
  contact_error = 0.218956885
```

对这两个候选补 seed1/seed2 后：

```text
key=0.01, contact=0.0:
  selection_score_mean = 0.997770270
  selection_score_std = 0.018376647
  xyz_mse = 0.021951851
  xyz_mae = 0.083480921
  mpjpe = 0.174132722
  final_frame_error = 0.260554107
  long_xyz_mse = 0.032950597
  relative_root_distance_error = 0.105995949
  contact_error = 0.234477090

key=0.025, contact=0.01:
  selection_score_mean = 0.997936134
  selection_score_std = 0.018650704
  xyz_mse = 0.021835899
  xyz_mae = 0.084164602
  mpjpe = 0.175155938
  final_frame_error = 0.260719518
  long_xyz_mse = 0.032070331
  relative_root_distance_error = 0.109123879
  contact_error = 0.234807046
```

判断：

```text
B-lite 在 seed0 有诱人结果，但多 seed 后不稳定。
它没有超过 Stage A best，也没有解决 contact_error 的 formal 风险。
不进入 formal training。
```

## 当前最严谨结论

不能说 6 个 Loss 参数已经达到最优。

当前 validation search 下：

```text
Stage A best by weighted score:
  velocity_loss_weight = 0.2
  long_loss_weight = 0.05
  final_frame_loss_weight = 0.075
  其它新增 loss = 0.0

Formal Pareto-safe baseline:
  velocity_loss_weight = 0.2
  其它新增 loss = 0.0
```

严谨表述：

```text
在当前 validation split、当前模型结构和当前搜索预算下，
long/final loss 可以带来非常小的 weighted score 改善，
但该改善未稳定覆盖 MPJPE/contact 的退化。
因此当前不应声明“六角度 loss 参数训练到最优”，也不应运行最终 test 来包装结论。
```

## 下一步建议

如果继续追求“可声明最优”，优先级如下：

```text
1. 改 score/gate：把 MPJPE 和 contact degradation gate 写死，而不是只靠 weighted score。
2. 重新做 Stage A/B 搜索，目标是 Pareto-dominant，而不是只降低加权分。
3. 或承认当前最稳配置是 baseline，六角度附加 loss 作为 ablation negative/neutral result。
4. 只有出现多 seed 稳定 Pareto 改善后，再运行最终 test。
```

