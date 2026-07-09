# NTU xyz Stage A long/final 局部网格结果

## 目的

在 validation split 上检查第一批候选 `long_loss_weight=0.05, final_frame_loss_weight=0.05` 是否已经处于局部最优附近。

数据和协议不变：

```text
train_opt = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/train_opt_xyz.pt
val_opt = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/val_opt_xyz.pt
val_ratio = 0.15
split_seed = 0
```

## 搜索范围

固定：

```text
velocity_loss_weight = 0.2
其它新增 loss = 0.0
num_steps = 1000
```

局部网格：

```text
long_loss_weight in [0.025, 0.05, 0.075, 0.1]
final_frame_loss_weight in [0.025, 0.05, 0.075, 0.1]
```

所有 16 个候选均完成 seed0；对 seed0 上有潜力的候选补 seed1/seed2。

## 多 seed 汇总

按实际 loss 权重归并，同一权重同一 seed 只保留更低 selection_score 的记录。

三 seed 已覆盖的候选：

```text
long=0.05, final=0.075:
  score_mean = 0.989713
  score_std = 0.026921
  xyz_mse = 0.021651146
  xyz_mae = 0.083504718
  mpjpe = 0.173787103
  final_frame_error = 0.258562376
  contact_error = 0.234758124

long=0.1, final=0.025:
  score_mean = 0.993433
  score_std = 0.037080
  xyz_mse = 0.021565015
  xyz_mae = 0.084020132
  mpjpe = 0.174494149
  final_frame_error = 0.263261026
  contact_error = 0.236941627

long=0.05, final=0.05:
  score_mean = 0.993639
  score_std = 0.031472
  xyz_mse = 0.021819159
  xyz_mae = 0.083863669
  mpjpe = 0.174970718
  final_frame_error = 0.261816615
  contact_error = 0.219996367

long=0.025, final=0.1:
  score_mean = 0.996723
  score_std = 0.033853
  xyz_mse = 0.021849917
  xyz_mae = 0.083456873
  mpjpe = 0.174062854
  final_frame_error = 0.262391022
  contact_error = 0.229392740

long=0.025, final=0.05:
  score_mean = 0.997038
  score_std = 0.011157
  xyz_mse = 0.021979553
  xyz_mae = 0.083292101
  mpjpe = 0.173597185
  final_frame_error = 0.260750260
  contact_error = 0.227512652

long=0.075, final=0.05:
  score_mean = 0.997244
  score_std = 0.033018
  xyz_mse = 0.021786359
  xyz_mae = 0.084010940
  mpjpe = 0.175019054
  final_frame_error = 0.265135791
  contact_error = 0.224277742

long=0.1, final=0.075:
  score_mean = 1.003293
  score_std = 0.033186
  xyz_mse = 0.021954936
  xyz_mae = 0.084571954
  mpjpe = 0.176214726
  final_frame_error = 0.264672434
  contact_error = 0.232464320

baseline:
  score_mean = 1.006076
  score_std = 0.021011
  xyz_mse = 0.021946480
  xyz_mae = 0.083573200
  mpjpe = 0.174411438
  final_frame_error = 0.264703663
  contact_error = 0.226181086
```

## 判断

当前 Stage A 多 seed 最优：

```text
velocity_loss_weight = 0.2
long_loss_weight = 0.05
final_frame_loss_weight = 0.075
其它新增 loss = 0.0
```

理由：

```text
1. selection_score_mean 最低。
2. xyz_mse / mpjpe / final_frame_error 均优于 baseline 均值。
3. contact_error 比 baseline 差，因此不能说所有维度最优。
4. 已检查周边 long/final 扰动，没有发现三 seed 更优配置。
```

边界：

```text
这仍然只是 Stage A validation-search 最优，不是最终 test 最优。
B 阶段 key/contact 和 D 阶段 action_feature 当前没有超过 Stage A 主目标。
下一步应进入 formal 训练，用更长 step 比较 baseline 与 Stage A best。
```

## 下一步

formal training：

```text
candidate = long=0.05, final=0.075
baseline = base MSE + velocity=0.2
seed = 0,1,2
num_steps = 3000
selection = validation score
test = formal 选定后只评估一次
```

