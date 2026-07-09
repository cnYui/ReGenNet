# NTU xyz 六角度参数最优训练计划

## 先回答当前参数

当前最终接受模型：

```text
save/forecasting/ntu120_label/xyz_loss_stageD_formal_s0/model000001500.pt
```

当前接受的主要训练参数是：

```text
base_xyz_mse = 1.0  # 隐式主 loss
mae_loss_weight = 0.05
root_loss_weight = 0.25
local_pose_loss_weight = 0.25
mpjpe_loss_weight = 0.0
short_loss_weight = 0.0
mid_loss_weight = 0.0
long_loss_weight = 0.05
final_frame_loss_weight = 0.05
velocity_loss_weight = 0.2
acceleration_loss_weight = 0.025
continuity_loss_weight = 0.0
first_step_loss_weight = 0.0
relative_root_loss_weight = 0.025
relative_velocity_loss_weight = 0.025
key_joint_relation_loss_weight = 0.05
contact_loss_weight = 0.05
contact_threshold = 0.15
action_feature_loss_weight = 0.02
action_logit_loss_weight = 0.0
```

当前代码默认值是：

```text
base_xyz_mse = 1.0  # 隐式主 loss
mae_loss_weight = 0.1
root_loss_weight = 1.0
local_pose_loss_weight = 1.0
mpjpe_loss_weight = 0.0
short_loss_weight = 0.0
mid_loss_weight = 0.0
long_loss_weight = 0.2
final_frame_loss_weight = 0.2
velocity_loss_weight = 0.2
acceleration_loss_weight = 0.1
continuity_loss_weight = 0.0
first_step_loss_weight = 0.0
relative_root_loss_weight = 0.2
relative_velocity_loss_weight = 0.1
key_joint_relation_loss_weight = 0.0
contact_loss_weight = 0.0
contact_threshold = 0.15
action_feature_loss_weight = 0.0
action_logit_loss_weight = 0.0
```

重要边界：

```text
当前接受参数不是全局最优。
它是本轮有限搜索下通过 gate 的 best accepted configuration。
```

## 为什么不能直接说已经最优

本轮训练已经使用完整 train/test cache：

```text
train = 1956 samples
test = 1253 samples
max_samples = -1
eval_max_samples = -1
```

但还不能说“最优”，原因是：

```text
1. 没有独立 validation split，之前 tuning 多次看了 test。
2. 没有系统搜索所有 loss 权重组合。
3. 没有多 seed 统计。
4. 没有固定同预算比较所有候选。
5. 没有证明继续训练或改变权重不能带来稳定提升。
```

如果目标是把参数训练到可以更严谨地称为“当前实验预算下最优”，必须重新设计优化协议。

## 最优定义

这里的“最优”不能定义成训练 loss 最小。训练 loss 只是加权目标，不等价于视觉拟合。

建议定义为：

```text
在固定数据、固定模型结构、固定训练预算、固定评估指标下，
在 validation set 上通过所有硬 gate，
并取得最好的综合验证分数；
最后只用一次 test set 报告最终结果。
```

硬 gate：

```text
1. loss finite，无 NaN/Inf。
2. beats_copy_last.xyz_mse = true。
3. beats_copy_last.xyz_mae = true。
4. beats_copy_last.mpjpe = true。
5. first_step_error = 0 或接近 0，不能跳帧。
6. xyz_mse / xyz_mae / mpjpe 不允许相对旧 baseline 明显恶化。
7. 固定视频 case 不能出现明显新失败类型。
```

综合验证分数建议用 normalized score，越低越好：

```text
score =
  0.30 * xyz_mse_ratio
+ 0.20 * mpjpe_ratio
+ 0.15 * final_frame_error_ratio
+ 0.10 * long_xyz_mse_ratio
+ 0.10 * relative_root_distance_error_ratio
+ 0.10 * contact_error_ratio
+ 0.05 * fid_ratio
- 0.05 * action_top1_gain
```

其中：

```text
metric_ratio = candidate_metric / baseline_metric
action_top1_gain = candidate_action_top1 - baseline_action_top1
```

如果不想把 action/FID 混入主目标，也可以使用双层选择：

```text
第一层：只按 xyz_mse / mpjpe / final / long / relation/contact 选 Pareto front。
第二层：在 Pareto front 里选 FID/action 更好的模型。
```

我更建议双层选择，因为当前任务主目标仍是 paired xyz forecasting。

## 第一步：建立 validation split

必须先停止用 test set 调参。

新增一个固定 stratified validation cache：

```text
train_full: 1956
建议拆分：
  train_opt: 约 80%-90%
  val_opt:  约 10%-20%
按 action label 分层，保证 26 类尽量覆盖。
test: 1253，只用于最终报告。
```

输出建议：

```text
results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/train_opt_xyz.pt
results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/val_opt_xyz.pt
```

验收：

```text
1. train/val/test shape 正确。
2. train/val label_counts 输出。
3. 少样本类别必须标注，因为当前 NTU 类别非常不均衡。
4. 固定 seed，不允许后续搜索时重新随机切分。
```

## 第二步：固定 baseline 和训练预算

在新 split 上重跑 baseline：

```text
baseline_config:
  base MSE + velocity_loss_weight=0.2
  其它新增 loss 关闭
```

至少跑：

```text
seed = 0, 1, 2
num_steps = 1000
batch_size = 64
eval_interval = 100
```

记录：

```text
best_by_val_xyz_mse
best_by_val_score
final_step_metrics
train_log
```

原因：

```text
后续所有参数搜索必须和 baseline 同预算、同 split、同 seed 口径比较。
```

## 第三步：阶段 A 参数搜索

阶段 A 只搜索不依赖外部模型、不依赖 joint index 的 loss：

```text
mae
root
local_pose
long
final_frame
acceleration
relative_root
relative_velocity
```

建议先固定：

```text
velocity_loss_weight = 0.2
short_loss_weight = 0.0
mid_loss_weight = 0.0
mpjpe_loss_weight = 0.0
continuity_loss_weight = 0.0
first_step_loss_weight = 0.0
```

第一轮粗搜索：

```text
mae_loss_weight: [0.0, 0.05, 0.1]
root_loss_weight: [0.0, 0.1, 0.25, 0.5]
local_pose_loss_weight: [0.0, 0.1, 0.25, 0.5]
long_loss_weight: [0.0, 0.025, 0.05, 0.1]
final_frame_loss_weight: [0.0, 0.025, 0.05, 0.1]
acceleration_loss_weight: [0.0, 0.0125, 0.025, 0.05]
relative_root_loss_weight: [0.0, 0.0125, 0.025, 0.05]
relative_velocity_loss_weight: [0.0, 0.0125, 0.025, 0.05]
```

不要全网格暴力展开。建议使用分组搜索：

```text
A1 root/local group
A2 long/final group
A3 acceleration group
A4 relative group
```

每组只保留 top 3 配置进入下一组。

每个候选训练：

```text
num_steps = 1000
seed = 0
early_stop_patience = 3 eval points
selection_metric = validation score
```

阶段 A 通过后，选：

```text
A_top3_by_val_score
A_top3_by_xyz_mse
```

如果两组不重合，保留并进入后续验证，不提前丢弃。

## 第四步：阶段 B 参数搜索

阶段 B 从 A_top 候选继续，搜索 key joint/contact：

```text
key_joint_relation_loss_weight: [0.0, 0.025, 0.05, 0.1]
contact_loss_weight: [0.0, 0.025, 0.05, 0.1]
contact_threshold: [0.10, 0.15, 0.20]
```

训练：

```text
从 A checkpoint 继续
追加 300-800 step
lr = 5e-5
selection_metric = validation score + contact_error gate
```

阶段 B 不能只看 contact：

```text
如果 contact_error 改善，但 xyz_mse/mpjpe/final 明显变差，不接受。
```

选出：

```text
B_top3
```

## 第五步：阶段 C 分类器和语义评估

分类器必须单独稳定。

训练 xyz action classifier：

```text
seed = 0, 1, 2
num_steps = 2000 或直到 val acc 不再提升
hidden_dim = 256
num_blocks = 4
```

classifier gate：

```text
real_future_top1 >= 0.80
real_future_top5 >= 0.90
balanced_acc >= 0.55
handshaking_acc >= 0.80
```

如果 gate 不稳定：

```text
1. 不进入阶段 D。
2. 只把 action/FID 当诊断。
3. 优先修分类器，而不是把不可靠 classifier loss 加回 forecasting。
```

阶段 C 对所有 A/B 候选跑：

```text
action_top1
balanced_acc
FID
class_wise_FID
diversity
copy-last 对照
```

## 第六步：阶段 D 参数搜索

只有阶段 C gate 通过后才做 D。

从 B_top3 继续，搜索：

```text
action_feature_loss_weight: [0.0, 0.005, 0.01, 0.02, 0.05]
action_logit_loss_weight: [0.0]
```

如果 feature loss 已稳定，且 action/FID 仍不足，再小心尝试：

```text
action_logit_loss_weight: [0.0025, 0.005]
```

但 logit CE 默认不推荐，因为容易让模型追分类器捷径。

阶段 D 接受条件：

```text
1. validation score 改善。
2. action_top1 或 FID 改善。
3. xyz_mse / mpjpe / final 不明显恶化。
4. 视频没有出现为了分类而牺牲真实轨迹拟合的问题。
```

## 第七步：多 seed formal training

把搜索得到的 top 配置正式重训。

候选：

```text
baseline best
A_best
B_best
D_best
```

每个候选跑：

```text
seed = 0, 1, 2
num_steps = 3000 或 5000
batch_size = 64
eval_interval = 100
save_interval = 500
selection = validation score
```

输出：

```text
mean/std over seeds
best checkpoint per seed
validation score curve
train loss curve
```

只有在多 seed 下仍稳定改善，才进入最终 test。

## 第八步：最终 test 只跑一次

最终 test 对象只能是：

```text
按 validation 选出的最终配置和 checkpoint。
```

最终报告：

```text
test xyz_mse / xyz_mae / mpjpe
short/mid/long/final
root/local
velocity/acceleration
relative/contact
action/FID/diversity
copy-last
旧 baseline
mean/std if multiple seeds
```

最终视频：

```text
固定 8-16 个 case
按动作类别补充 handshaking / hugging / pushing 等 interaction-heavy case
只作为定性诊断，不替代 full test 指标
```

## 如何判断已经最优

不是“再跑一次 loss 更低”就叫最优。建议同时满足以下条件：

```text
1. 在 validation 上，top 配置之间 score 差距小于 0.5%，且没有新的 Pareto-dominant 配置。
2. 对关键权重做局部扰动后，无法稳定改善 validation score。
3. 多 seed 下最终配置平均指标优于 baseline，且标准差不覆盖主要改善幅度。
4. 继续训练 20%-30% 额外 step 后 validation score 不再改善，或开始回退。
5. 最终 test 只评估一次，并且仍超过 copy-last 和旧 baseline。
```

局部扰动建议：

```text
每个已选权重乘以 [0.5, 1.0, 2.0]。
只改一个 group，不同时改所有 group。
```

如果局部扰动还能带来稳定提升：

```text
说明还没有到最优，回到对应阶段继续搜索。
```

## 推荐执行顺序

最简可执行版：

```text
1. 建 train_opt / val_opt cache。
2. 重跑 baseline seeds 0/1/2。
3. 阶段 A 分组搜索，选 A_top3。
4. 阶段 B 搜 key/contact，选 B_top3。
5. 训练 classifier seeds 0/1/2，确认 gate。
6. 阶段 C 评估 A/B 候选。
7. 阶段 D 搜 action_feature，选 D_top。
8. A/B/D top 配置 formal 多 seed 3000-5000 step。
9. 只对 validation 选出的最终模型跑 test。
10. 生成最终视频和结果文档。
```

如果时间有限：

```text
优先做 validation split + A/B/D top 配置三 seed formal。
不要继续只在 test 上调权重。
```
