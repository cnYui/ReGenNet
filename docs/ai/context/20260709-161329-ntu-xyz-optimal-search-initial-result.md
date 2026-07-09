# NTU xyz 最优训练第一批调参结果

## Git 基线

本轮调参建立在以下提交之后：

```text
12604bd Add NTU XYZ six-angle loss optimization pipeline
678fd90 Document NTU XYZ validation split ratio
66df72b Add NTU XYZ validation search tooling
f1c61ff Add Stage B candidates from Stage A top configs
7c1bb73 Preserve NTU XYZ search leaderboard across stages
286fd84 Add Stage D candidates from Stage A top config
```

## Validation Split

固定 split：

```text
source = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/train_xyz.pt
train_opt = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/train_opt_xyz.pt
val_opt = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/val_opt_xyz.pt
split_seed = 0
val_ratio_requested = 0.15
```

实际结果：

```text
full_count = 1956
train_count = 1661
val_count = 295
val_ratio_actual = 0.1508179959
train_missing_labels = []
val_missing_labels = []
```

结论：

```text
26 个 label 在 train_opt / val_opt 中均保留。
后续调参只看 val_opt；test 不能再用于搜索。
```

## Baseline

baseline 配置：

```text
base_xyz_mse = 1.0
velocity_loss_weight = 0.2
其它新增 loss = 0.0
num_steps = 1000
seed = 0,1,2
```

三 seed 验证结果均通过 copy-last gate。

汇总：

```text
selection_score_mean = 1.006075973
selection_score_std = 0.021011126
xyz_mse_mean = 0.021946480
mpjpe_mean = 0.174411438
contact_error_mean = 0.226181086
```

## Stage A

第一批搜索：

```text
root/local: 0.1, 0.25, 0.5
long/final: 0.05 / 0.05
compact_v1: mae/root/local/long/final/acc/relative 组合
```

seed0 最优：

```text
config_id = stageA_a2_long_final_005
long_loss_weight = 0.05
final_frame_loss_weight = 0.05
best_checkpoint = save/forecasting/ntu120_label/xyz_loss_optimal/stageA_a2_long_final_005_s0/model000001000.pt
selection_score = 0.965186996
xyz_mse = 0.020764302
mpjpe = 0.172261078
contact_error = 0.200073083
```

随后补跑 seed1/seed2，三 seed 汇总：

```text
selection_score_mean = 0.993639410
selection_score_std = 0.031472086
xyz_mse_mean = 0.021819159
mpjpe_mean = 0.174970718
contact_error_mean = 0.219996367
hard_gate_pass_runs = 3 / 3
```

判断：

```text
相对 baseline，A_top 的综合 score 和 xyz_mse/contact 更好，但 mpjpe 略差。
当前只能称为第一批搜索下的验证集候选最优，不能称为最终最优。
```

## Stage B

从 Stage A top `long/final=0.05/0.05` 展开 key/contact：

```text
key/contact = 0.025/0.025, 0.05/0.05, 0.05/0.1, 0.1/0.05
seed = 0
```

最佳：

```text
config_id = stageB_long_final_005_key050_contact050
selection_score = 0.980869037
xyz_mse = 0.021145784
mpjpe = 0.173609021
contact_error = 0.213805507
```

判断：

```text
B 最佳没有超过 Stage A seed0 top。
key/contact 暂不进入正式候选，除非后续重新设计更小权重或改评分权重。
```

## Stage C

在 train_opt 训练 xyz action classifier，并用 val_opt gate：

```text
hidden_dim = 256
num_blocks = 4
num_steps = 2000
seed = 0,1,2
```

结果：

```text
seed0: top1=0.813559322, top5=0.976271186, balanced=0.552513266, handshaking=0.846153846, gate=true
seed1: top1=0.854237288, top5=0.986440678, balanced=0.690132961, handshaking=0.884615385, gate=true
seed2: top1=0.823728814, top5=0.976271186, balanced=0.573645236, handshaking=0.807692308, gate=true
```

判断：

```text
Stage C classifier gate 稳定通过。
可以进入 Stage D，但 action/FID 仍只能作为辅助诊断，不能覆盖 paired geometry 主目标。
```

## Stage D

从 Stage A top checkpoint 继续训练：

```text
resume_checkpoint = save/forecasting/ntu120_label/xyz_loss_optimal/stageA_a2_long_final_005_s0/model000001000.pt
lr = 5e-5
num_steps = 1300
action_classifier = save/forecasting/ntu120_label/xyz_loss_optimal/stageC_classifier_val_s1/classifier_model.pt
action_feature_loss_weight = 0.0, 0.005, 0.01, 0.02, 0.05
```

结果：

```text
feature=0.0:   geometry_score=0.999676415, fid_ratio=1.177723451, action_top1_gain=0.010169446
feature=0.005: geometry_score=1.008433451, fid_ratio=0.871156612, action_top1_gain=0.061016917
feature=0.01:  geometry_score=1.016845004, fid_ratio=0.824750253, action_top1_gain=0.071186423
feature=0.02:  geometry_score=1.032326590, fid_ratio=0.629022524, action_top1_gain=0.098305047
feature=0.05:  geometry_score=1.041992647, fid_ratio=0.881253213, action_top1_gain=0.098305047
```

判断：

```text
action feature loss 能改善 FID/action top1，但 geometry 明显回退。
按 paired forecasting 主目标，不接受 D 配置作为当前最优。
```

## 当前结论

当前验证集候选：

```text
primary_candidate = stageA_a2_long_final_005
loss:
  velocity_loss_weight = 0.2
  long_loss_weight = 0.05
  final_frame_loss_weight = 0.05
  其它新增 loss = 0.0
```

但它还不是最终最优，原因：

```text
1. Stage A 只做了第一批候选，没有完成 long/final 局部网格。
2. B/D 只做 seed0，且当前未超过 A_top。
3. A_top 虽然 score/xyz_mse/contact 优于 baseline，但 mpjpe 略差。
4. 尚未做 full train formal retrain 和最终一次 test。
```

## 下一步

继续搜索应优先做：

```text
1. Stage A long/final 局部网格：
   long_loss_weight in [0.025, 0.05, 0.075, 0.1]
   final_frame_loss_weight in [0.025, 0.05, 0.075, 0.1]
2. 对 top2 配置补 seed1/seed2。
3. 如果 A 局部扰动仍无法稳定改善，再进入 formal train。
4. formal train 只用 validation 选 checkpoint；最终 test 只评估一次。
```

