# NTU 双人 xyz 六类训练 Loss 参数设计

## 目标

当前 NTU 双人预测主线是：

```text
obs20 双人 xyz + action label -> future40 双人 xyz
shape: [B,20,2,55,3] -> [B,40,2,55,3]
```

用户目标不是只让数字变小，而是让预测视频在视觉上更贴近真实 future。因此新增训练规则必须围绕六个角度组织：

```text
1. 单样本拟合
2. 长短期误差
3. 时间动态
4. 双人关系
5. 动作语义
6. 分布质量
```

## 当前已有 Loss 参数

代码位置：

```text
model/forecasting_ntu_xyz.py::NTULabelXYZTransformer.training_loss
train/train_ntu_label_xyz.py::build_arg_parser
```

当前已有参数：

```text
mae_loss_weight
velocity_loss_weight
continuity_loss_weight
first_step_loss_weight
```

当前问题：

```text
1. 缺少 root/local 分解，无法区分整个人漂移和局部姿态错误。
2. 缺少 short/mid/long/final horizon 约束，无法针对长期发散加权。
3. 只有 velocity，没有 acceleration，容易漏掉抖动或过平滑。
4. 双人关系只在评估里有 root distance，训练时没有显式约束。
5. 动作语义和 FID 当前不参与训练/评估闭环。
6. continuity/first_step 受 ramp 结构影响，训练梯度价值有限。
```

## 总体 Loss 结构

建议保留主 MSE，同时新增分层 loss：

```text
total_loss =
  base_xyz_mse
+ mae_loss_weight * xyz_mae
+ root_loss_weight * root_position_mse
+ local_pose_loss_weight * local_pose_mse
+ short_loss_weight * short_horizon_mse
+ mid_loss_weight * mid_horizon_mse
+ long_loss_weight * long_horizon_mse
+ final_frame_loss_weight * final_frame_mse
+ velocity_loss_weight * velocity_mse
+ acceleration_loss_weight * acceleration_mse
+ continuity_loss_weight * continuity_mse
+ first_step_loss_weight * first_step_mse
+ relative_root_loss_weight * relative_root_distance_mse
+ relative_velocity_loss_weight * relative_root_velocity_mse
+ key_joint_relation_loss_weight * key_joint_pair_distance_mse
+ contact_loss_weight * contact_distance_mse
+ action_feature_loss_weight * action_classifier_feature_loss
```

其中 FID / Diversity 第一阶段只进入评估，不直接反传。

## 为什么分阶段添加

阶段化不是因为代码上不能一次实现，而是为了控制变量。

如果一次打开所有 loss 和指标，结果变好或变差时很难判断原因：

```text
root/local loss 可能改善整体漂移，但也可能压制动作幅度。
long/final loss 可能改善长期帧，但也可能牺牲短期自然性。
acceleration loss 可能减少抖动，但权重过大时会让动作过平滑。
key joint/contact loss 如果 joint index 错，会直接把模型推向错误目标。
action classifier loss 如果分类器不可靠，会牺牲逐帧拟合换取分类命中。
FID 变好不代表这一条预测贴近这一条真实 future。
```

因此阶段化的核心原则是：

```text
先加不依赖外部模型、不依赖未确认关节语义、能直接从 pred/target/obs 计算的 paired loss；
再加需要额外语义确认的交互细节；
再把 FID / action accuracy 作为评估接入；
最后才考虑把分类器特征反过来作为训练监督。
```

工程上可以一次把所有命令行参数和代码接口加好，但默认只打开阶段 A。这样后续只改权重就能逐步打开 B/C/D，不需要反复改代码。

## 1. 单样本拟合

### 目标

回答：

```text
预测的这一条 future 是否贴近对应真实 future。
```

### 参数

```text
mae_loss_weight
root_loss_weight
local_pose_loss_weight
mpjpe_loss_weight
```

### 计算方式

`mae_loss_weight`：

```text
mean(abs(pred_xyz - target_xyz))
```

`root_loss_weight`：

```text
root joint 使用 joint index 0
mean((pred_xyz[:, :, :, 0] - target_xyz[:, :, :, 0]) ** 2)
```

`local_pose_loss_weight`：

```text
以每个人 root 为原点，去掉整体平移后比较局部姿态
pred_local = pred_xyz - pred_root
target_local = target_xyz - target_root
mean((pred_local - target_local) ** 2)
```

`mpjpe_loss_weight`：

```text
mean(norm(pred_xyz - target_xyz, dim=-1))
```

### 默认建议

```text
mae_loss_weight = 0.1
root_loss_weight = 1.0
local_pose_loss_weight = 1.0
mpjpe_loss_weight = 0.0
```

说明：

```text
MPJPE 和 MSE/MAE 信息重叠，第一版先作为评估主指标，不默认进入训练。
```

## 2. 长短期误差

### 目标

回答：

```text
模型是短期好、长期崩，还是全程都差。
```

### 参数

```text
short_loss_weight
mid_loss_weight
long_loss_weight
final_frame_loss_weight
```

### 计算方式

当前 `pred_len=40`，建议按比例切分：

```text
short: frames 0:13
mid:   frames 13:26
long:  frames 26:40
```

`short/mid/long_horizon_mse`：

```text
对应时间段内 mean((pred_xyz - target_xyz) ** 2)
```

`final_frame_loss_weight`：

```text
mean((pred_xyz[:, -1] - target_xyz[:, -1]) ** 2)
```

### 默认建议

```text
short_loss_weight = 0.0
mid_loss_weight = 0.0
long_loss_weight = 0.2
final_frame_loss_weight = 0.2
```

说明：

```text
主 MSE 已覆盖全时间段，第一版只轻量加强 long/final，避免重复加权过重。
```

## 3. 时间动态

### 目标

回答：

```text
动作是否连续，速度和加速度是否接近真实 future。
```

### 参数

```text
velocity_loss_weight
acceleration_loss_weight
continuity_loss_weight
first_step_loss_weight
```

### 计算方式

`velocity_loss_weight` 当前已有：

```text
pred_full = concat(obs_last, pred_xyz)
target_full = concat(obs_last, target_xyz)
pred_vel = pred_full[:, 1:] - pred_full[:, :-1]
target_vel = target_full[:, 1:] - target_full[:, :-1]
mean((pred_vel - target_vel) ** 2)
```

`acceleration_loss_weight`：

```text
pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
target_acc = target_vel[:, 1:] - target_vel[:, :-1]
mean((pred_acc - target_acc) ** 2)
```

`continuity_loss_weight` 当前已有：

```text
mean((pred_xyz[:, 0] - obs_xyz[:, -1]) ** 2)
```

`first_step_loss_weight` 当前已有：

```text
mean((pred_xyz[:, 0] - target_xyz[:, 0]) ** 2)
```

### 默认建议

```text
velocity_loss_weight = 0.2
acceleration_loss_weight = 0.1
continuity_loss_weight = 0.0
first_step_loss_weight = 0.0
```

说明：

```text
当前模型用 ramp 强制 pred 第一帧等于 obs 最后一帧，因此 continuity/first_step 的梯度价值有限。保留参数，但默认关闭或降权。
```

## 4. 双人关系

### 目标

回答：

```text
两个人之间的相对位置、相对速度和接触关系是否像真实互动。
```

### 参数

```text
relative_root_loss_weight
relative_velocity_loss_weight
key_joint_relation_loss_weight
contact_loss_weight
```

### 计算方式

`relative_root_loss_weight`：

```text
pred_dist = norm(pred_person1_root - pred_person2_root)
target_dist = norm(target_person1_root - target_person2_root)
mean((pred_dist - target_dist) ** 2)
```

`relative_velocity_loss_weight`：

```text
pred_rel = pred_person1_root - pred_person2_root
target_rel = target_person1_root - target_person2_root
pred_rel_vel = pred_rel[:, 1:] - pred_rel[:, :-1]
target_rel_vel = target_rel[:, 1:] - target_rel[:, :-1]
mean((pred_rel_vel - target_rel_vel) ** 2)
```

`key_joint_relation_loss_weight`：

```text
选定关键关节对，比较 pairwise distance 时间序列
例如 hand-hand、wrist-wrist、hand-torso
mean((pred_pair_distance - target_pair_distance) ** 2)
```

`contact_loss_weight`：

```text
只在真实距离小于 contact_threshold 的关节对上生效
mask = target_pair_distance < contact_threshold
mean(mask * (pred_pair_distance - target_pair_distance) ** 2)
```

### 默认建议

```text
relative_root_loss_weight = 0.2
relative_velocity_loss_weight = 0.1
key_joint_relation_loss_weight = 0.2
contact_loss_weight = 0.1
contact_threshold = 0.15
```

### 关键关节对

第一版需要先确认 SMPL-X 55 joint index。未确认前只应把索引集中放在常量里，并在文档注明来源。

建议接口：

```text
NTU_INTERACTION_JOINT_PAIRS = (
  ("left_hand", "right_hand"),
  ("right_hand", "left_hand"),
  ("left_wrist", "right_wrist"),
  ("right_wrist", "left_wrist"),
)
```

如果 joint index 不可靠，第一版先实现接口和默认关闭：

```text
key_joint_relation_loss_weight = 0.0
contact_loss_weight = 0.0
```

## 5. 动作语义

### 目标

回答：

```text
预测 future 是否像输入 action label 对应的动作类别。
```

### 参数

```text
action_feature_loss_weight
action_logit_loss_weight
action_classifier_path
```

### 计算方式

第一阶段不建议直接训练时启用，只作为评估接入：

```text
pred_future -> action classifier -> accuracy / feature
target_future -> action classifier -> feature
```

训练阶段如果启用，可选两种：

```text
1. feature loss:
   mean((classifier_feature(pred) - classifier_feature(target)) ** 2)

2. logit CE loss:
   cross_entropy(classifier_logits(pred), action_label)
```

### 默认建议

```text
action_feature_loss_weight = 0.0
action_logit_loss_weight = 0.0
```

说明：

```text
动作分类器会引入额外模型依赖，第一版先做评估指标；训练时启用必须冻结分类器，并单独记录 classifier gate。
```

## 6. 分布质量

### 目标

回答：

```text
预测动作整体分布是否像真实动作，是否模式坍缩。
```

### 指标

```text
fid_eval
diversity_eval
class_wise_fid_eval
```

### 计算方式

FID：

```text
真实 future 和预测 future 分别送入动作识别模型抽 feature
计算两组 feature 的均值和协方差
FID = ||mu_real - mu_pred||^2 + Tr(sigma_real + sigma_pred - 2 * sqrt(sigma_real * sigma_pred))
```

Diversity：

```text
随机抽两条预测 feature，计算平均 L2 距离
```

class-wise FID：

```text
按 action label 分组后分别计算 FID
```

### 训练参数建议

第一版不加入训练 loss 参数：

```text
fid_loss_weight = 不实现
diversity_loss_weight = 不实现
```

原因：

```text
FID 是分布统计，不是逐样本可稳定反传的 paired forecasting loss。
当前模型是 deterministic predictor，Diversity 主要来自不同输入样本，不适合作为直接训练目标。
```

## 需要新增的训练命令参数

建议在 `train/train_ntu_label_xyz.py::build_arg_parser` 新增：

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

其中第一版建议默认真正启用：

```text
root_loss_weight = 1.0
local_pose_loss_weight = 1.0
long_loss_weight = 0.2
final_frame_loss_weight = 0.2
acceleration_loss_weight = 0.1
relative_root_loss_weight = 0.2
relative_velocity_loss_weight = 0.1
```

第一版默认关闭：

```text
mpjpe_loss_weight = 0.0
short_loss_weight = 0.0
mid_loss_weight = 0.0
key_joint_relation_loss_weight = 0.0
contact_loss_weight = 0.0
action_feature_loss_weight = 0.0
action_logit_loss_weight = 0.0
```

## 需要新增的评估指标

建议扩展 `utils/ntu_smplx_2p_xyz.py::NTU_XYZ_METRIC_KEYS`：

```text
short_xyz_mse
mid_xyz_mse
long_xyz_mse
final_frame_error
acceleration_error
relative_root_velocity_error
key_joint_relation_error
contact_error
```

建议新增独立评估入口或扩展 `eval/eval_ntu_label_xyz.py`：

```text
action_accuracy
fid
diversity
class_wise_fid
class_wise_accuracy
```

## 实现顺序

### 阶段 A：纯 xyz loss 和指标

添加原因：

```text
阶段 A 直接针对当前最主要的问题：预测 future 在几何位置、局部姿态、长期帧和时间动态上不够贴近真实 future。
这些 loss 只依赖 pred_xyz / target_xyz / obs_xyz，不需要外部分类器，也不需要确认手腕/手掌等关节语义。
因此它是最低风险、最容易归因的一组改动。
```

改动：

```text
model/forecasting_ntu_xyz.py
utils/ntu_smplx_2p_xyz.py
train/train_ntu_label_xyz.py
eval/eval_ntu_label_xyz.py
```

内容：

```text
root/local
short/mid/long/final
acceleration
relative root / relative velocity
```

验收：

```text
compileall pass
小样本 train 2 step pass
eval metrics json schema 稳定
copy-last 仍同步输出
```

### 阶段 B：关键关节关系

添加原因：

```text
阶段 A 的 relative root 只能约束两个人整体距离，不能保证 handshaking / high-five / hugging 等动作的手部接触和局部互动正确。
阶段 B 用 key joint pair 和 contact loss 补足这一点。
但它依赖 SMPL-X 55 joint index；如果索引不可靠，约束错关节会比不加更糟，所以必须独立成阶段。
```

前置：

```text
确认 SMPL-X 55 joint index。
```

内容：

```text
key_joint_relation_loss
contact_loss
key_joint_relation_error
contact_error
```

验收：

```text
handshaking / high-five / hugging 等类别的 class-wise relation error 有可解释输出。
```

### 阶段 C：动作语义和 FID 评估

添加原因：

```text
阶段 C 回答原 ReGenNet 风格的问题：预测动作像不像真实动作分布，是否符合动作类别。
FID / action accuracy 很重要，但它们是分布和语义指标，不是 paired future 逐帧拟合指标。
先作为评估接入，可以判断它们是否和视频观感一致，同时不影响训练主目标。
```

内容：

```text
加载/训练 NTU two-person action classifier
action accuracy
FID
diversity
class-wise metrics
```

验收：

```text
真实 future 的 classifier accuracy 必须先通过 gate，否则不能用 generated accuracy / FID 写结论。
```

### 阶段 D：动作语义训练 loss

添加原因：

```text
只有当阶段 C 证明分类器本身可靠，并且 FID/action accuracy 与视觉质量有解释关系时，才值得把 action feature/logit loss 加入训练。
否则分类器 loss 可能让模型更像某个动作类别，但更不贴近对应 ground truth。
阶段 D 的目标是语义增强，不是替代 paired geometric loss。
```

只有阶段 C 通过后再做：

```text
action_feature_loss
action_logit_loss
```

验收：

```text
不能只看 action accuracy 变好；必须同时检查 xyz_mse/mpjpe/final_frame/error/copy-last 是否恶化。
```

## 不建议第一版做的事

```text
1. 不把 FID 直接写成训练 loss。
2. 不默认启用 action classifier loss。
3. 不在 joint index 未确认前默认启用 key joint/contact loss。
4. 不用 Diversity/Multimodality 作为 deterministic predictor 的主训练目标。
5. 不只报告新增指标而不保留 copy-last 对照。
```

## 论文表述边界

可以写：

```text
We optimize paired geometric, temporal, and interaction-aware losses, and report recognition-feature-based metrics such as FID as complementary generation-quality diagnostics.
```

不能写：

```text
FID 证明预测逐帧贴近真实 future。
```

因为 FID 是分布指标，不是 paired forecasting accuracy 指标。
