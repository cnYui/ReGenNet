# NTU 双人 xyz 六角度 A/B/C/D 分阶段训练计划

## 依据

本计划基于：

```text
docs/ai/context/20260709-143858-ntu-xyz-six-angle-training-loss-parameter-design.md
```

目标不是单纯降低一个指标，而是让：

```text
obs20 双人 xyz + action label -> future40 双人 xyz
```

在完整 test set 上更贴近真实 future，并且视频观感不比 copy-last 差。

当前代码边界：

```text
model/forecasting_ntu_xyz.py 目前只支持 mae / velocity / continuity / first_step loss。
eval/eval_ntu_label_xyz.py 目前只支持基础 xyz 指标和 copy-last 对照。
因此正式训练 A/B/C/D 前，必须先完成新参数和新指标接口。
```

## 固定协议

数据协议固定为：

```text
window_len = 60
obs_len = 20
pred_len = 40
train cache = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/train_xyz.pt
test cache  = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/test_xyz.pt
```

硬 baseline 固定为 copy-last：

```text
copy_last_xyz = obs_xyz[:, -1] 重复 40 帧
```

当前已知 seed0 1000-step xyz 模型结果：

```text
model xyz_mse = 0.031281803, copy = 0.057248837
model xyz_mae = 0.091381698, copy = 0.120007492
model mpjpe   = 0.189595376, copy = 0.260575039
```

所有阶段都必须同时输出：

```text
model_metrics
copy_last_metrics
beats_copy_last
metrics_test.json
train_log.jsonl
args.json
```

## 总体 Gate

任何阶段的结果要被接受，必须先满足硬 gate：

```text
1. train loss finite，无 NaN/Inf。
2. compileall / smoke train / eval schema 通过。
3. 完整 test set 上 beats_copy_last.xyz_mse = true。
4. 完整 test set 上 beats_copy_last.xyz_mae = true。
5. 完整 test set 上 beats_copy_last.mpjpe = true。
6. first_step_error 不能引入跳帧；当前结构目标仍应接近 0。
7. 固定 case 视频不能出现明显新失败类型：漂移、跳帧、双人错位、接触反向、过度静止。
```

阶段间接受标准：

```text
1. 新阶段必须在自己负责的指标上优于上一接受 checkpoint。
2. xyz_mse / xyz_mae / mpjpe / long_xyz_mse / final_frame_error 不允许明显回退。
3. 如果语义或关系指标变好，但 paired geometry 明显变坏，不能接受。
4. 如果只在 8 个可视化样本上变好，full test 没变好，不能接受为正式结果。
```

建议把“明显回退”先定义为：

```text
主指标相对上一接受 checkpoint 恶化超过 1%。
阶段专属指标改善小于 2% 时，不视为稳定收益。
```

如果结果接近阈值，补充：

```text
seed = 1 / 2 重跑，或对 test set bootstrap 置信区间。
```

## 阶段 0：接口和基线准备

阶段 0 不做正式结论，只保证后续实验可控。

### 必须实现

训练参数：

```text
--root_loss_weight
--local_pose_loss_weight
--mpjpe_loss_weight
--short_loss_weight
--mid_loss_weight
--long_loss_weight
--final_frame_loss_weight
--acceleration_loss_weight
--relative_root_loss_weight
--relative_velocity_loss_weight
--key_joint_relation_loss_weight
--contact_loss_weight
--contact_threshold
--action_feature_loss_weight
--action_logit_loss_weight
--action_classifier_path
```

评估指标：

```text
short_xyz_mse
mid_xyz_mse
long_xyz_mse
final_frame_error
acceleration_error
relative_root_velocity_error
key_joint_relation_error
contact_error
action_accuracy
fid
diversity
class_wise_metrics
```

### 验收

```text
python -m compileall model train eval utils scripts
小样本 2 step train pass
小样本 eval pass
metrics key 顺序稳定
所有新增参数默认关闭时，旧 checkpoint 可正常加载和评估
```

### 如果失败

```text
1. schema 不稳定：先固定 NTU_XYZ_METRIC_KEYS，不进入训练。
2. 旧 checkpoint 不能加载：config 读取必须对新增字段使用默认值。
3. 小样本 loss 非有限：先只保留 base MSE，逐项打开新 loss 定位。
```

## 阶段 A：纯 xyz 几何、长短期和时间动态

阶段 A 只使用 pred_xyz / target_xyz / obs_xyz 可直接计算的 loss，不依赖关节语义确认和外部分类器。

### 训练目标

回答：

```text
单条预测 future 是否更贴近对应真实 future。
短期、中期、长期、最后一帧是否更稳。
速度和加速度是否更接近真实动作，而不是只学静止。
两个人 root 级相对位置和相对速度是否更合理。
```

### 默认训练参数

建议第一轮：

```text
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
action_feature_loss_weight = 0.0
action_logit_loss_weight = 0.0
```

continuity / first_step 第一轮关闭的原因：

```text
当前模型用 ramp 让 pred 第一帧结构性接近 obs 最后一帧，这两项梯度价值有限。
```

### 必评指标

主指标：

```text
xyz_mse
xyz_mae
mpjpe
beats_copy_last
```

阶段 A 指标：

```text
root_translation_error
local_pose_error
short_xyz_mse
mid_xyz_mse
long_xyz_mse
final_frame_error
velocity_error
acceleration_error
relative_root_distance_error
relative_root_velocity_error
inter_person_distance_consistency
first_step_error
```

### 通过标准

```text
1. xyz_mse / xyz_mae / mpjpe 均超过 copy-last。
2. long_xyz_mse 或 final_frame_error 至少一个相对上一 checkpoint 改善 2% 以上。
3. root_translation_error / relative_root_distance_error 不明显恶化。
4. acceleration_error 下降时，velocity_error 不能明显恶化。
5. 视频不能变成过度静止。
```

### 如果分数不达标

优化顺序：

```text
1. 先确认 base MSE + 当前旧 loss 的复现结果，排除实现错误。
2. 只打开 root/local，关闭 long/final/acceleration/relative，判断几何分解是否有效。
3. 再打开 long/final，权重从 0.1 -> 0.2 -> 0.4 做小范围搜索。
4. 再打开 acceleration，权重从 0.05 -> 0.1 -> 0.2 搜索。
5. 最后打开 relative_root / relative_velocity，权重从 0.05 -> 0.1 -> 0.2 搜索。
```

失败类型和处理：

```text
NaN/Inf：
  lr 从 3e-4 降到 1e-4，clip_grad_norm 保持 1.0，所有新增权重减半。

long/final 好了但 xyz_mae/mpjpe 变差：
  long_loss_weight 和 final_frame_loss_weight 各减半。

acceleration_error 好了但视频过平滑：
  acceleration_loss_weight 减半，检查 velocity_error 是否同步恶化。

relative 指标好了但两人整体漂移：
  relative_root_loss_weight 减半，root_loss_weight 保持或加到 1.5。

所有指标都不优于上一 checkpoint：
  回退到上一接受 checkpoint，只保留 root/local，重新训练短程 ablation。
```

## 阶段 B：关键关节关系和接触

阶段 B 只在 SMPL-X 55 joint index 确认后启用。未确认前只能实现接口并默认关闭。

### 训练目标

回答：

```text
handshaking / high-five / hugging 等互动类中，两个人的手、腕、身体关键距离是否更像真实 future。
```

### 前置条件

```text
1. 明确 SMPL-X 55 joint index 来源。
2. 把 key joint pair 全部集中到常量里。
3. 至少人工检查 3 个动作类别的视频，确认选中的手/腕/躯干关节符合预期。
```

### 默认训练参数

阶段 B 从阶段 A 最佳 checkpoint 继续训练，保留阶段 A 已接受参数，并新增：

```text
key_joint_relation_loss_weight = 0.05
contact_loss_weight = 0.05
contact_threshold = 0.15
```

如果小范围验证稳定，再尝试：

```text
key_joint_relation_loss_weight = 0.1 或 0.2
contact_loss_weight = 0.1
```

不建议一开始直接重权重，因为错关节或接触 mask 噪声会直接破坏轨迹拟合。

### 必评指标

继承阶段 A 全部指标，并新增：

```text
key_joint_relation_error
contact_error
class_wise_key_joint_relation_error
class_wise_contact_error
contact_mask_ratio
```

重点动作类别：

```text
handshaking
hugging
pushing
pat on back
point finger
touch head
```

具体类别以 NTU120 2P 当前 label 映射为准。

### 通过标准

```text
1. key_joint_relation_error 或 contact_error 相对阶段 A 最佳 checkpoint 改善 2% 以上。
2. interaction-heavy 类别的 class-wise relation error 有可解释改善。
3. xyz_mse / xyz_mae / mpjpe 不恶化超过 1%。
4. long_xyz_mse / final_frame_error 不明显恶化。
5. 视频中接触方向不能明显反向或拉扯到错误关节。
```

### 如果分数不达标

失败类型和处理：

```text
key/contact 指标异常变好，但视频明显错位：
  先怀疑 joint index 或 pair 定义错误，停止训练，人工检查关节可视化。

contact_mask_ratio 太低：
  contact_threshold 从 0.15 提到 0.20，但必须检查是否把非接触也纳入。

contact_mask_ratio 太高：
  contact_threshold 从 0.15 降到 0.10，避免把普通接近误当接触。

主指标恶化超过 1%：
  key_joint_relation_loss_weight 和 contact_loss_weight 各减半。

只有个别类别变好，整体变差：
  暂不接受阶段 B，改成只作为 class-wise diagnostic，不进入最终训练权重。
```

重新训练规则：

```text
1. 如果是 joint index 问题，修正索引后从阶段 A 最佳 checkpoint 重新跑 B。
2. 如果只是权重过大，从阶段 A 最佳 checkpoint 重新跑 B，不从失败 checkpoint 继续。
3. 如果连续两轮 B 都不能通过，最终主模型保留阶段 A，不强行加入 B。
```

## 阶段 C：动作语义和 FID 评估

阶段 C 原则上不训练 forecasting 模型，只评估阶段 A/B checkpoint 的语义和分布质量。

### 目标

回答：

```text
真实 future 是否能被动作分类器可靠识别。
预测 future 在动作识别特征空间里是否接近真实分布。
生成结果是否只是几何接近，但动作语义不对。
```

### 必备组件

```text
NTU two-person action classifier
feature extractor
FID / Diversity / class-wise metric evaluator
```

分类器必须先在真实 future 上过 gate，否则不能用 generated accuracy 或 FID 写结论。

### 评估参数

```text
action_classifier_path
fid_feature_layer
num_eval_samples = full test set
class_wise_eval = true
diversity_num_pairs = 固定随机种子抽样
seed = 0
```

### 必评指标

分类器 gate：

```text
real_future_top1_acc
real_future_top5_acc
real_future_balanced_acc
real_future_class_wise_acc
handshaking_acc
```

预测语义：

```text
pred_top1_acc
pred_top5_acc
pred_balanced_acc
pred_class_wise_acc
```

分布质量：

```text
fid
class_wise_fid
diversity
feature_mean_distance
feature_covariance_distance
```

对照必须同时输出：

```text
copy_last_action_accuracy
copy_last_fid
copy_last_diversity
```

### 通过标准

分类器 gate 建议：

```text
real_future_top1_acc >= 0.80
real_future_top5_acc >= 0.90
real_future_balanced_acc >= 0.55
handshaking_acc >= 0.80
```

预测评估接受标准：

```text
1. pred FID 优于 copy-last FID。
2. pred action accuracy 不低于 copy-last。
3. class-wise FID 不能只靠大类改善，小样本类必须单独标注不稳定。
4. FID/action accuracy 只能作为辅助，不覆盖阶段 A/B 的 paired geometry 结论。
```

### 如果分数不达标

分类器 gate 不过：

```text
1. 不进入阶段 D。
2. 检查分类器输入是否与 forecasting xyz shape、归一化、坐标方向一致。
3. 重训分类器，优先提高 balanced_acc 和 handshaking_acc。
4. 真实 future gate 通过前，不报告 generated semantic success。
```

FID 不好但几何指标好：

```text
1. 保留 A/B 模型作为 forecasting 主结果。
2. 标注为语义/分布辅助指标不足，不反向证明视觉拟合失败。
3. 检查 feature extractor 是否对两人 xyz 输入合理。
```

action accuracy 好但 FID 差：

```text
1. 说明预测像动作类别，但整体分布仍偏。
2. 不进入 D 的 logit CE 优先路线，先尝试 feature loss。
```

copy-last FID 反而更好：

```text
1. 检查分类器特征是否偏好静止或平滑。
2. 用固定视频 case 对比，确认 FID 是否与视觉观感一致。
3. 若不一致，FID 只做报告，不作为调参目标。
```

## 阶段 D：动作语义训练 loss

阶段 D 只有在阶段 C gate 通过后才允许启动。它是语义增强，不是替代几何预测。

### 训练目标

回答：

```text
在不牺牲 paired geometry 的前提下，是否能让预测 future 更符合输入 action label 的语义。
```

### 默认训练参数

从阶段 B 最佳 checkpoint 继续训练。如果阶段 B 未通过，则从阶段 A 最佳 checkpoint 继续。

第一轮只启用 feature loss：

```text
action_feature_loss_weight = 0.02
action_logit_loss_weight = 0.0
action_classifier_path = 阶段 C 通过的冻结分类器
```

如果 feature loss 稳定但语义无改善，再试：

```text
action_feature_loss_weight = 0.05
action_logit_loss_weight = 0.0
```

只有当 feature loss 不够且分类器非常稳定时，才尝试：

```text
action_feature_loss_weight = 0.02
action_logit_loss_weight = 0.01
```

分类器必须冻结：

```text
classifier.eval()
requires_grad_(False)
```

### 必评指标

继承阶段 A/B/C 全部指标，重点看：

```text
xyz_mse
xyz_mae
mpjpe
long_xyz_mse
final_frame_error
velocity_error
acceleration_error
key_joint_relation_error
contact_error
pred_top1_acc
pred_balanced_acc
fid
class_wise_fid
```

### 通过标准

```text
1. pred_top1_acc / pred_balanced_acc 或 FID 相对阶段 B 最佳 checkpoint 有改善。
2. xyz_mse / xyz_mae / mpjpe 不恶化超过 1%。
3. long_xyz_mse / final_frame_error 不恶化超过 1%。
4. 视频中动作语义更像标签，但不能牺牲对应 ground truth 拟合。
5. 如果 action accuracy 变好但 paired geometry 变差，阶段 D 不接受。
```

### 如果分数不达标

失败类型和处理：

```text
action accuracy 变好，geometry 变差：
  action_feature_loss_weight 减半；如果仍失败，放弃 D，保留 B/A。

FID 变好，视频变差：
  不接受 D，说明 feature metric 与视觉目标不一致。

loss 不稳定：
  lr 降到 1e-4 或 5e-5，action loss 权重减半。

只改善大类，小样本类恶化：
  输出 class-wise 风险，不把 D 作为统一最终模型；考虑 per-class 分析而非继续调权重。

logit CE 导致模型追求分类器捷径：
  关闭 action_logit_loss_weight，只保留 feature loss 或完全关闭语义训练。
```

重新训练规则：

```text
1. D 的失败 checkpoint 不继续使用。
2. 每次从 B 最佳或 A 最佳 checkpoint 重新开始。
3. 最多做 feature weight 0.02 / 0.05 / 0.1 三档；三档都失败则停止 D。
```

## 四阶段如何串起来训练

推荐采用“两层训练”：

```text
第一层：短程 tuning chain，用于确定参数是否值得保留。
第二层：正式 formal chain，用固定参数重跑并产出论文/汇报结果。
```

### 短程 tuning chain

顺序：

```text
P0 baseline/evaluator
  -> A_tune
  -> B_tune
  -> C_eval
  -> D_tune
```

执行规则：

```text
1. A_tune 可从当前 seed0 1000-step xyz checkpoint 继续训练 500-1000 step，用于快速判断新 loss 方向。
2. A_tune 通过后，把 A_best 作为 B_tune 起点。
3. B_tune 通过后，把 B_best 作为 C_eval 候选；B 不通过则 C_eval 使用 A_best。
4. C_eval 不训练 forecasting 模型，只决定是否允许 D。
5. D_tune 只从 B_best 或 A_best 开始，不从 C 产生新 checkpoint。
```

建议目录：

```text
save/forecasting/ntu120_label/xyz_loss_stageA_tune_s0/
save/forecasting/ntu120_label/xyz_loss_stageB_tune_s0/
results/forecasting/ntu120_label/xyz_loss_stageC_eval_s0/
save/forecasting/ntu120_label/xyz_loss_stageD_tune_s0/
```

### 正式 formal chain

当 tuning chain 确定参数后，正式链只使用已接受参数，不再临时改权重：

```text
A_formal:
  从随机初始化或项目认可的统一初始化训练，使用阶段 A 最终权重。

B_formal:
  从 A_formal best checkpoint 继续，加入阶段 B 最终权重。

C_formal_eval:
  同时评估 A_formal best 和 B_formal best，输出语义/FID/分布报告。

D_formal:
  只有 C_formal_eval 通过，才从 B_formal best 继续加入语义 loss。
```

最终模型选择规则：

```text
1. 如果 D 通过，最终模型 = D_formal best。
2. 如果 D 不通过但 B 通过，最终模型 = B_formal best。
3. 如果 B 不通过但 A 通过，最终模型 = A_formal best。
4. 如果 A 不通过，最终模型仍是现有 xyz baseline，不把新 loss 写成有效改进。
```

### 训练时长建议

tuning：

```text
A_tune: 500-1000 step
B_tune: 300-800 step
D_tune: 300-800 step
eval_interval: 100 或 250
```

formal：

```text
A_formal: 至少对齐当前 baseline 的 1000 step；如果 loss 仍下降，可扩到 3000/5000 step。
B_formal: 从 A_formal best 继续 500-1500 step。
D_formal: 从 B_formal best 继续 500-1500 step。
```

正式比较必须同预算：

```text
如果拿 A/B/D 和旧 1000-step baseline 比，必须说明训练 step 是否一致。
如果训练 step 更长，必须补充同 step 或学习曲线对比，避免把训练预算当成 loss 收益。
```

## 统一调参原则

每次失败后只改一个因素：

```text
loss 权重
lr
训练步数
起始 checkpoint
关节 pair / contact threshold
分类器版本
```

不要同时改多个因素，否则无法归因。

权重搜索优先级：

```text
1. root/local
2. long/final
3. acceleration
4. relative_root/relative_velocity
5. key_joint/contact
6. action_feature/action_logit
```

回退原则：

```text
1. 新阶段不通过就回退到上一接受 checkpoint。
2. 不从失败 checkpoint 继续堆新 loss。
3. 连续两轮阶段失败，先把该阶段降级为评估诊断，不继续硬训。
```

## 计划中的最低可交付结果

最低可交付不是 A/B/C/D 全部成功，而是：

```text
1. 阶段 A 至少给出完整的几何、长短期、动态、关系 root 级评估。
2. 所有结果保留 copy-last 对照。
3. 如果 B/C/D 未通过，要明确写成未通过原因，而不是继续调到不可解释。
4. 最终视频只用于定性验证，不替代 full test 指标。
```

## 论文或汇报边界

可以写：

```text
We progressively add paired geometric, temporal, and interaction-relation objectives, and evaluate recognition-feature metrics as complementary diagnostics.
```

不能写：

```text
FID 证明单条预测已经贴近对应真实 future。
```

也不能写：

```text
动作分类器 accuracy 提升就说明视觉预测更准确。
```

最终主结论仍必须落在：

```text
paired xyz forecasting metrics + copy-last baseline + 固定 case 视频诊断
```
