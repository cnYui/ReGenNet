# NTU xyz 六角度 A/B/C/D 代码实现与训练结果

## 代码实现

本轮实现了 `20260709-144942-ntu-xyz-six-angle-abcd-training-plan.md` 中的阶段 0/A/B/C/D 主链路。

修改文件：

```text
utils/ntu_smplx_2p_xyz.py
model/forecasting_ntu_xyz.py
train/train_ntu_label_xyz.py
eval/eval_ntu_label_xyz.py
```

新增文件：

```text
eval/action_xyz_classifier.py
```

新增能力：

```text
1. root/local、short/mid/long/final、acceleration、relative root、key joint/contact loss。
2. 新增完整 xyz 评估指标，并保留 copy-last 对照。
3. 新增 xyz action classifier，用于 real future gate、action accuracy、FID、diversity、class-wise FID。
4. 支持 D 阶段冻结 classifier feature loss 训练。
```

验证：

```text
compileall pass
stage0 smoke train/eval pass
xyz classifier smoke pass
```

## 阶段 C 分类器 Gate

分类器：

```text
save/forecasting/ntu120_label/xyz_action_classifier_stageC_s0/classifier_model.pt
```

真实 future40 gate：

```text
top1_acc = 0.807661612
top5_acc = 0.956105347
balanced_acc = 0.565190546
handshaking_acc = 0.970588235
classifier_gate_pass = true
```

结论：

```text
分类器可用于阶段 C/D 的辅助语义/FID 评估。
```

## Baseline

旧 baseline：

```text
save/forecasting/ntu120_label/xyz_transformer_len60_o20_p40_h256_l3_s0_1000/model000001000.pt
```

新指标口径：

```text
xyz_mse = 0.031281803
xyz_mae = 0.091381698
mpjpe = 0.189595376
long_xyz_mse = 0.034809116
final_frame_error = 0.283018141
contact_error = 0.216543612
action_top1 = 0.732641637
fid = 4.073653080
```

## A/B/C/D Tuning Chain

### 阶段 A

第一轮 A 默认权重使主指标回退，未接受。

第二轮 A_v2 仍让 `xyz_mse` 回退略超 1%，未作为最终 A。

接受的 A_best：

```text
save/forecasting/ntu120_label/xyz_loss_stageA_tune_v3_s0/model000001200.pt
```

结果：

```text
xyz_mse = 0.031521649
xyz_mae = 0.090256628
mpjpe = 0.187686249
final_frame_error = 0.273796119
contact_error = 0.209507082
fid = 3.896486579
```

判断：

```text
MSE 相对 baseline 回退低于 1%，MAE/MPJPE/final/FID 改善；A_v3 可接受。
```

### 阶段 B

接受的 B_best：

```text
save/forecasting/ntu120_label/xyz_loss_stageB_tune_s0/model000001500.pt
```

结果：

```text
xyz_mse = 0.031726739
xyz_mae = 0.090399782
mpjpe = 0.187774729
contact_error = 0.156480178
action_top1 = 0.766161203
fid = 2.751958708
```

判断：

```text
contact_error 明显改善，主几何指标相对 A_best 回退低于 1%；B 可接受。
```

### 阶段 D Tuning

接受的 D_tune：

```text
save/forecasting/ntu120_label/xyz_loss_stageD_tune_s0/model000001650.pt
```

结果：

```text
xyz_mse = 0.031026373
xyz_mae = 0.090512138
mpjpe = 0.187407460
final_frame_error = 0.275633331
contact_error = 0.151580963
action_top1 = 0.845171571
fid = 1.736539514
class_wise_fid_mean = 33.941954842
```

判断：

```text
D_tune 同时改善几何和语义/FID，可接受。
```

## Formal Chain

### A_formal

从随机初始化训练 1000 step：

```text
save/forecasting/ntu120_label/xyz_loss_stageA_formal_s0/model000001000.pt
```

结果：

```text
xyz_mse = 0.031737313
xyz_mae = 0.091127322
mpjpe = 0.189421743
long_xyz_mse = 0.034705675
final_frame_error = 0.277314855
```

判断：

```text
超过 copy-last，但 xyz_mse 相对旧 baseline 回退约 1.45%；A_formal 单独不作为最终结论。
```

### B_formal

继续 B 到 1300 step：

```text
save/forecasting/ntu120_label/xyz_loss_stageB_formal_s0/model000001300.pt
```

结果：

```text
xyz_mse = 0.031413255
xyz_mae = 0.091306860
mpjpe = 0.189305320
long_xyz_mse = 0.034743335
final_frame_error = 0.276743574
contact_error = 0.187298748
```

判断：

```text
B_formal 把 MSE 拉回到 baseline 1% 阈值内，MPJPE/long/final/contact 改善。
```

### D_formal

最终接受 checkpoint：

```text
save/forecasting/ntu120_label/xyz_loss_stageD_formal_s0/model000001500.pt
```

完整评估：

```text
results/forecasting/ntu120_label/xyz_loss_stageC_eval_s0/stageD_formal_best1500/metrics_test.json
```

几何指标：

```text
xyz_mse = 0.030939925
xyz_mae = 0.090770731
mpjpe = 0.187937536
long_xyz_mse = 0.034324502
final_frame_error = 0.273512055
root_translation_error = 0.109062294
relative_root_distance_error = 0.101139964
key_joint_relation_error = 0.160378376
contact_error = 0.164331722
beats_copy_last = true / true / true
```

语义和分布指标：

```text
action_top1 = 0.844373524
action_balanced = 0.603879667
fid = 1.823068602
class_wise_fid_mean = 33.277466152
```

对 baseline 的变化：

```text
xyz_mse: 0.031281803 -> 0.030939925，改善约 1.09%
xyz_mae: 0.091381698 -> 0.090770731，改善约 0.67%
mpjpe: 0.189595376 -> 0.187937536，改善约 0.87%
final_frame_error: 0.283018141 -> 0.273512055，改善约 3.36%
relative_root_distance_error: 0.105562203 -> 0.101139964，改善约 4.19%
fid: 4.073653080 -> 1.823068602，改善约 55.25%
action_top1: 0.732641637 -> 0.844373524，改善约 11.17 个百分点
```

结论：

```text
D_formal model000001500.pt 是当前最佳正式模型。
```

## 视频输出

数组导出：

```text
results/forecasting/ntu120_label/xyz_loss_stageD_formal_best1500_eval8
```

zflip 三色视频：

```text
results/forecasting/ntu120_label/xyz_loss_stageD_formal_best1500_tricolor_2p_videos_zflip
```

视频：

```text
case0000_A001.mp4
case0001_A002.mp4
case0002_A003.mp4
case0003_A004.mp4
case0004_A005.mp4
case0005_A006.mp4
case0006_A007.mp4
case0007_A008.mp4
```

边界：

```text
8 个视频只用于人工观感检查，不能替代 full test 指标。
当前 claim 仍应围绕 paired xyz forecasting metrics + copy-last baseline + 辅助语义/FID。
```

## 结论边界

可以写：

```text
新六角度分阶段训练在完整 test set 上相对旧 xyz baseline 小幅改善 paired geometry，并显著改善 action-classifier feature-space FID 和 action consistency。
```

不能写：

```text
FID 证明逐帧预测完全贴近真实 future。
```

也不能写：

```text
动作语义控制已经独立成功。
```

因为 action/FID 仍是辅助指标，主结论必须由 paired xyz metrics 和 copy-last gate 支撑。
