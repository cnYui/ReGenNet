# NTU xyz 六角度最优训练执行设计

## 依据

本设计基于：

```text
docs/ai/context/20260709-152354-ntu-xyz-six-angle-optimal-training-plan.md
```

目标是把当前 best accepted configuration 升级为：

```text
在固定实验预算、固定验证协议、固定模型结构下，
可以称为当前项目内最优的 loss 参数和 checkpoint。
```

## 核心原则

不能继续用 test set 调参。

本轮之前已经多次看过 test，因此后续要做“最优训练”，第一步必须建立 validation split：

```text
train_opt 用于训练
val_opt 用于调参、选 checkpoint、早停和模型选择
test 只在最终模型确定后使用一次
```

如果没有 validation split，任何“最优”都只能是 test-tuned best，论文或汇报里不能严谨表述为最优。

## 当前起点

当前接受模型：

```text
save/forecasting/ntu120_label/xyz_loss_stageD_formal_s0/model000001500.pt
```

当前接受参数：

```text
mae_loss_weight = 0.05
root_loss_weight = 0.25
local_pose_loss_weight = 0.25
long_loss_weight = 0.05
final_frame_loss_weight = 0.05
velocity_loss_weight = 0.2
acceleration_loss_weight = 0.025
relative_root_loss_weight = 0.025
relative_velocity_loss_weight = 0.025
key_joint_relation_loss_weight = 0.05
contact_loss_weight = 0.05
contact_threshold = 0.15
action_feature_loss_weight = 0.02
action_logit_loss_weight = 0.0
```

当前代码默认值偏激进，不等同于最终接受参数。后续搜索应显式传参，不依赖默认值。

## 需要新增的工程入口

### 1. validation split 构建脚本

建议新增：

```text
scripts/split_ntu_xyz_cache_for_optimization.py
```

输入：

```text
results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/train_xyz.pt
```

输出：

```text
results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/train_opt_xyz.pt
results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/val_opt_xyz.pt
results/forecasting/ntu120_label/xyz_cache_len60_o20_p40_opt_split/split_summary.json
```

设计：

```text
1. 以 action label 分层。
2. 固定 seed。
3. 默认 val_ratio = 0.15 或 0.20。
4. 少样本类别至少保留 1 条到 train；如果 val 中缺少某类，summary 标注。
5. 不改原始 full train/test cache。
```

验收：

```text
obs_xyz / target_xyz / actions / meta 数量一致。
train_opt + val_opt = 原 train_full。
label_counts 输出到 split_summary.json。
```

### 2. 验证分数计算脚本

建议新增：

```text
scripts/score_ntu_xyz_candidate.py
```

输入：

```text
metrics_val.json
baseline_metrics_val.json
可选 semantic_metrics_val.json
```

输出：

```text
candidate_score.json
```

评分分两层：

第一层 paired geometry score：

```text
geometry_score =
  0.35 * xyz_mse_ratio
+ 0.25 * mpjpe_ratio
+ 0.15 * final_frame_error_ratio
+ 0.10 * long_xyz_mse_ratio
+ 0.10 * relative_root_distance_error_ratio
+ 0.05 * contact_error_ratio
```

第二层 semantic diagnostic：

```text
semantic_score =
  fid_ratio - action_top1_gain
```

最终选择逻辑：

```text
1. 先用 hard gate 过滤。
2. 再用 geometry_score 选 Pareto front。
3. 在 Pareto front 里优先选 FID/action 更好的配置。
```

原因：

```text
任务主目标是 paired forecasting；action/FID 不能覆盖几何退化。
```

### 3. 搜索编排脚本

建议新增：

```text
scripts/run_ntu_xyz_loss_search.py
```

职责：

```text
1. 生成候选参数配置 jsonl。
2. 调用 train.train_ntu_label_xyz。
3. 调用 eval.eval_ntu_label_xyz 在 val_opt 上评估。
4. 调用 score_ntu_xyz_candidate.py 打分。
5. 维护 leaderboard.json。
6. 自动选择每阶段 top-k 进入下一阶段。
```

不建议手工逐条跑命令，因为阶段搜索很容易漏记参数或混淆 checkpoint。

### 4. 最终结果汇总脚本

建议新增：

```text
scripts/summarize_ntu_xyz_optimal_search.py
```

输出：

```text
results/forecasting/ntu120_label/xyz_loss_optimal_search/leaderboard.md
results/forecasting/ntu120_label/xyz_loss_optimal_search/leaderboard.json
```

内容：

```text
baseline / A / B / C / D 各阶段 top 配置
每个 seed 的 best checkpoint
mean/std
是否通过 gate
最终 test 结果
视频路径
```

## 目录设计

统一根目录：

```text
save/forecasting/ntu120_label/xyz_loss_optimal/
results/forecasting/ntu120_label/xyz_loss_optimal/
```

建议结构：

```text
results/forecasting/ntu120_label/xyz_loss_optimal/
  split/
  baseline/
  stageA_search/
  stageB_search/
  stageC_classifier/
  stageC_eval/
  stageD_search/
  formal/
  final_test/
  videos/
  leaderboard.json
  leaderboard.md
```

checkpoint 目录：

```text
save/forecasting/ntu120_label/xyz_loss_optimal/
  baseline_s0/
  baseline_s1/
  baseline_s2/
  stageA_candidate_xxx_s0/
  stageB_candidate_xxx_s0/
  stageD_candidate_xxx_s0/
  formal_{config_id}_s0/
  formal_{config_id}_s1/
  formal_{config_id}_s2/
```

每个目录必须包含：

```text
args.json
train_log.jsonl
metrics_val.json
candidate_score.json
best_checkpoint.txt
```

## 执行步骤

### 第一步：构建 validation split

命令目标：

```text
train_full -> train_opt + val_opt
```

必须完成：

```text
1. split_summary.json
2. train_opt_xyz.pt
3. val_opt_xyz.pt
```

通过后才允许进入训练搜索。

### 第二步：在新 split 上重跑 baseline

baseline 不复用旧 test-tuned checkpoint。

配置：

```text
base_xyz_mse = 1.0
velocity_loss_weight = 0.2
其它新增 loss = 0.0
seed = 0,1,2
num_steps = 1000
```

输出：

```text
baseline_val_metrics_seed0/1/2
baseline_mean_std
baseline_score_ref
```

用途：

```text
后续所有 ratio 和 gate 以该 baseline 的 val 指标为基准。
```

### 第三步：阶段 A 分组搜索

搜索顺序：

```text
A1 root/local
A2 long/final
A3 acceleration
A4 relative_root/relative_velocity
```

每组策略：

```text
1. 从上一组 top3 配置展开。
2. 每个候选 seed0 跑 1000 step。
3. 每 100 step 评估 val。
4. 用 validation score 选 top3。
```

第一轮不做多 seed，目的是节省计算预算。

阶段 A 输出：

```text
A_top3_by_geometry_score
A_top3_by_xyz_mse
A_pareto_front
```

### 第四步：阶段 B key/contact 搜索

输入：

```text
A_top candidates
```

搜索：

```text
key_joint_relation_loss_weight = 0.0 / 0.025 / 0.05 / 0.1
contact_loss_weight = 0.0 / 0.025 / 0.05 / 0.1
contact_threshold = 0.10 / 0.15 / 0.20
```

训练：

```text
从 A checkpoint 继续
lr = 5e-5
追加 300-800 step
```

接受：

```text
contact_error 改善
geometry_score 不明显变差
视频无错误接触趋势
```

阶段 B 输出：

```text
B_top3
B_pareto_front
```

### 第五步：阶段 C 分类器训练和候选评估

训练 classifier：

```text
seed = 0,1,2
num_steps = 2000
hidden_dim = 256
num_blocks = 4
```

gate：

```text
top1 >= 0.80
top5 >= 0.90
balanced >= 0.55
handshaking >= 0.80
```

如果三 seed 中只有一个通过：

```text
分类器不稳定，不进入 D；先优化分类器。
```

如果至少两个 seed 通过：

```text
使用 val gate 最稳的 classifier 做阶段 C/D。
```

阶段 C 评估对象：

```text
baseline
A_top
B_top
copy-last
```

输出：

```text
action_top1
balanced_acc
FID
class_wise_FID
diversity
```

### 第六步：阶段 D action feature 搜索

输入：

```text
B_top3
通过 gate 的 classifier
```

搜索：

```text
action_feature_loss_weight = 0.0 / 0.005 / 0.01 / 0.02 / 0.05
action_logit_loss_weight = 0.0
```

只有 feature loss 稳定但语义仍不足时，才试：

```text
action_logit_loss_weight = 0.0025 / 0.005
```

接受：

```text
validation geometry_score 不退
FID/action 改善
paired xyz 主指标不被语义 loss 破坏
```

阶段 D 输出：

```text
D_top3
D_pareto_front
```

### 第七步：formal 多 seed 训练

候选：

```text
baseline_best
A_best
B_best
D_best
```

训练：

```text
seed = 0,1,2
num_steps = 3000 或 5000
batch_size = 64
eval_interval = 100
selection = val score
```

输出：

```text
每个 seed 的 best checkpoint
mean/std over seeds
learning curve
validation score curve
```

如果 D_best 平均更好但方差很大：

```text
不能直接称为最优；需要继续验证或保留 B_best 作为更稳模型。
```

### 第八步：最终 test

只允许对 validation 选出的最终配置跑 test。

最终 test 输出：

```text
metrics_test.json
semantic_metrics_test.json
final_summary.md
固定 case 视频
interaction-heavy case 视频
```

最终模型命名：

```text
save/forecasting/ntu120_label/xyz_loss_optimal/final_{config_id}_s{seed}/model_best.pt
```

## 最优判定标准

一个配置要被称为当前最优，必须满足：

```text
1. validation hard gate 全部通过。
2. validation geometry_score 最优，或处于 Pareto front 且语义/FID 显著更好。
3. 关键权重局部扰动 [0.5x, 2x] 无稳定收益。
4. 多 seed mean 优于 baseline，且 std 不覆盖主要收益。
5. 延长训练 20%-30% step 后 val score 没有继续稳定改善。
6. 最终 test 只跑一次，仍超过 copy-last 和旧 baseline。
```

如果第 6 条失败：

```text
不能称为最终最优；
需要回到 validation 协议检查是否过拟合 val。
```

## 最小可行执行版本

如果计算预算有限，先做：

```text
1. 建 train_opt/val_opt。
2. baseline seed0/1/2。
3. 用当前接受参数作为 D_current，在 train_opt 上重跑 seed0/1/2。
4. 对 D_current 做局部扰动：
   root/local: 0.125 / 0.25 / 0.5
   long/final: 0.025 / 0.05 / 0.1
   contact/key: 0.025 / 0.05 / 0.1
   action_feature: 0.01 / 0.02 / 0.05
5. 选 validation best。
6. final test 一次。
```

这不是完整搜索，但比继续用 test 调参严格得多。

## 风险和处理

### 类别不均衡

风险：

```text
某些动作类样本极少，val 指标波动大。
```

处理：

```text
输出 class-wise count。
小样本类只做诊断，不单独驱动全局参数选择。
```

### classifier shortcut

风险：

```text
action loss 提高分类器 accuracy，但破坏 paired trajectory。
```

处理：

```text
action loss 只能在 geometry gate 通过后使用。
默认不启用 action_logit_loss。
```

### 计算预算

风险：

```text
完整搜索候选过多。
```

处理：

```text
分组搜索 + top3 继承。
先 seed0 粗筛，再 seed0/1/2 formal。
```

### test leakage

风险：

```text
反复看 test 造成指标高估。
```

处理：

```text
搜索阶段禁止使用 test。
test 只在最终配置锁定后运行一次。
```

## 下一步实施清单

按顺序执行：

```text
1. 新增 scripts/split_ntu_xyz_cache_for_optimization.py。
2. 新增 scripts/score_ntu_xyz_candidate.py。
3. 新增 scripts/run_ntu_xyz_loss_search.py。
4. 新增 scripts/summarize_ntu_xyz_optimal_search.py。
5. 构建 train_opt/val_opt cache。
6. 跑 baseline seeds。
7. 跑 A/B/C/D 搜索。
8. 跑 formal 多 seed。
9. 最终 test + 视频。
10. 写最终结果文档并更新 AGENTS.md。
```
