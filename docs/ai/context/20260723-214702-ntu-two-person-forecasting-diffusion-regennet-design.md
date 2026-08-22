# NTU 双人 10->50 ReGenNet 风格扩散预测设计

## 状态

已确认设计，尚未开始实现或训练。

## 目标

在 NTU120 双人数据上新建一条预测主线：

```text
双人前 10 帧 + 动作标签 -> 双人后 50 帧
```

该主线使用条件扩散和 ReGenNet 风格的 Transformer Decoder。扩散只作用于后 50 帧；前 10 帧和动作标签始终作为条件。训练从第一个正式 step 起优化论文形式的总损失：

```text
L_all = L_dm + lambda_inter * L_inter
lambda_inter = 1
```

当前确定性 NTU xyz Transformer、其 checkpoint 和评估结果保持不变，作为公平 baseline，不被替换或覆盖。

## 已确认边界

### 协议

```text
window_len = 60
obs_len = 10
pred_len = 50
数据集 = NTU120 xsub train/test
条件 = 双人 obs10 + 26 类动作标签
目标 = 双人 future50
```

`T >= 60` 的现有数据 gate 保持有效。现有 `obs20/pred40` xyz cache、checkpoint、分类器和生成结果不能复用到本协议。

### 双人顺序

当前 H5 的既有约定暂定为：

```text
raw motion[..., 0:3] = Person A
raw motion[..., 3:6] = Person B
```

历史 `preprocess/actor_reactor.py` 的设计会把 actor 放在前、reactor 放在后；但现存 `xsub.train.h5` 和 `xsub.test.h5` 未保存逐样本角色标签、属性或生成清单。因此本主线使用 Person A / Person B 的中性命名，不把当前文件独立证明为 actor/reactor 监督。

这不阻碍预测训练，前提是同一序列中 Person A / B 在 obs10 与 future50 之间身份连续。论文中 actor/reactor 的方向性解释属于待审计假设，不作为当前结果的强声明。

### 表示

原始 H5 是：

```text
[T, 56, 6]
前 55 个 slot: 两人的 axis-angle，各 3 维
第 56 个 slot: 两人的 root translation，各 3 维
```

新数据适配器在读取后将每人的 pose 从 axis-angle 转为真正的 rotation-6D，生成模型使用：

```text
[B, 56, 12, T]
slot 0:55, channel 0:6  = Person A rot6d
slot 0:55, channel 6:12 = Person B rot6d
slot 55, channel 0:3    = Person A translation
slot 55, channel 6:9    = Person B translation
其他 translation padding channel 为 0
```

这与 `Rotation2xyz_x(..., pose_rep="rot6d", num_person=2)` 的双人拆分约定一致，也与论文采用 6D rotation representation 的口径一致。不得沿用旧 ForecastingCMDM 把 raw `[56, 6]` 标为单人 `rot6d` 的做法。

## 模型

### 条件扩散定义

令：

```text
C = {双人 obs10, action label}
Y_0 = 双人真实 future50 的 rot6d + translation
```

训练随机采样 `t` 和高斯噪声 `epsilon`：

```text
Y_t = sqrt(alpha_bar_t) * Y_0 + sqrt(1 - alpha_bar_t) * epsilon
Y0_hat = F(Y_t, t, C)
```

使用 1000 step cosine beta schedule。正式论文式训练不使用 `one_step_noise_prob` 的纯高斯近似分支；该分支只能作为后续采样对齐实验，不能混入主结果。

### Decoder 架构

采用 ReGenNet 风格而非逐字复刻的 Decoder：

```text
双人 obs10 -- 输入投影 + obs Encoder -------------------+
动作标签 -- action embedding ----------------------------+--> memory
timestep -- timestep embedding --------------------------+

双人 noisy future50 -- 输入投影 + 位置编码 --> Decoder target
                                                   |
                                                   v
                                      Transformer Decoder cross-attention
                                                   |
                                                   v
                                           双人 clean future50, Y0_hat
```

保留的 ReGenNet 核心：

```text
1. noisy target motion 的条件扩散去噪。
2. timestep embedding。
3. action embedding 与 classifier-free condition dropout。
4. Transformer Decoder 去噪骨干。
5. 网络直接预测 clean x0。
```

不把论文的 directional causal mask 设为主模型硬约束。论文使用该 mask 是为了在线 reactor 生成时不能看到未来 actor motion；本任务条件中只有 obs10，不包含未来任一人的动作，不存在该泄漏路径。future50 的 noisy target 可以使用非因果 self-attention 来维持整段动作一致性。

causal mask 保留为后续结构消融项，不影响主线结论。

## 损失

### 扩散损失

```text
L_dm = mean((Y0_hat - Y_0)^2)
```

该项在真正的双人 rot6d + translation tensor 上计算，等价于论文直接预测 clean x0 的 MSE 目标。

### 显式双人交互损失

由 `Y_0` 与 `Y0_hat` 分别通过可微 SMPL-X FK 得到：

```text
J_A, J_B: [B, 50, 55, 3] 的全局 joint xyz
R_A, R_B: [B, 50, 3, 3] 的 root rotation matrix
g_A, g_B: [B, 50, 3] 的 root translation
```

使用完整相对量，不能预先取长度：

```text
r_joint = J_A - J_B
R_rel = transpose(R_A) @ R_B
r_trans = g_A - g_B

L_joint  = mean((r_joint_hat - r_joint)^2)
L_orient = mean((R_rel_hat - R_rel)^2)
L_trans  = mean((r_trans_hat - r_trans)^2)
L_inter  = L_joint + L_orient + L_trans
L_all    = L_dm + 1.0 * L_inter
```

这对应论文中相对 body pose、relative orientation、relative translation 的三部分。每项按 batch、时间和自身坐标维度求平均后再相加，避免旧 `masked_l2` 对三维输入额外除以序列长度的问题。

现有确定性 xyz 的速度、接触、动作 classifier feature 等附加 loss 不进入论文式正式扩散模型。它们只能作为后续扩展或消融；加入后不得再称为严格 `L_dm + L_inter` 主模型。

## 三阶段训练和评估

### 阶段 0：数据与公式 gate

不训练正式模型。必须通过：

```text
1. 10/50 dataset 与双人 rot6d adapter shape 正确。
2. Person A / B 在 obs 与 future 间不交换。
3. raw rotvec -> rot6d -> SMPL-X FK 可得到 [B, T, 2, 55, 3]。
4. q_sample 在 t=0、高噪 t 和完整 shape 下有限。
5. Y0_hat == Y_0 时，L_dm、L_joint、L_orient、L_trans 都为 0。
6. 对 Person A/B 或 root translation 的受控扰动只改变相应 L_inter 子项。
7. 反向传播后模型参数与 FK 前的预测 tensor 均有有限梯度。
```

### 阶段 1：公平 baseline

建立三类基线，均使用 `obs10 -> future50`：

```text
A. copy-last：复制第 10 帧 50 次，不训练。
B. direct xyz Transformer：重训确定性双人预测器。
C. diffusion L_dm-only：新扩散架构，但 L_inter=0。
```

阶段 1 回答：扩散网络本身能否从噪声恢复双人 future50，且是否超过不预测的 copy-last。现有 `obs20 -> future40` 模型不能作为数值 baseline。

### 阶段 2：正式论文式扩散模型

从随机初始化开始，以阶段 1 的 diffusion L_dm-only 相同网络宽度、训练步数、seed 集合、优化器、batch protocol 和采样设定训练：

```text
L_all = L_dm + 1.0 * L_inter
```

不得以“先训 L_dm、后加入 L_inter”的 warm start 作为正式论文式结果。可以做短 smoke 验证数值，但正式模型从 step 1 起优化总损失。

硬门槛：

```text
1. full xsub test 的 MSE、MAE、MPJPE 必须超过 copy-last。
2. relative joint、relative root、relative orientation、contact 等 held-out 互动指标必须优于 L_dm-only diffusion。
3. 主预测指标不能因 L_inter 出现实质性退化。
4. 只有同时超过同协议 direct xyz Transformer，才可声明扩散主线优于确定性预测 baseline。
```

### 阶段 3：论文式生成评估与模型选择

阶段 3 默认不是给扩散网络添加新 loss，而是检查阶段 2 模型是否具有论文要求的生成质量：

```text
1. 使用只在真实 future50 训练的新 action classifier，并先通过 real future50 分类 gate。
2. 多随机种子采样后计算 FID、action accuracy、Diversity、Multimodality。
3. 在 xsub test 的未见主体上计算所有指标。
4. 比较 DDIM 步数、生成质量和推理延迟。
5. 生成双人三色视频：蓝色 obs10、橙色 generated future50、绿色 real future50。
6. 报告 w.o. L_inter 消融，即阶段 1C 与阶段 2 的对比。
```

FID、动作识别和多样性是论文使用的评估指标，不是主训练损失。若几何预测已通过但这些辅助指标不足，可新开“语义扩展”分支，例如冻结 classifier 的 feature loss；该分支必须与论文式主模型分开报告。

## 指标优先级

主结论必须按以下顺序建立：

```text
1. paired future50 的 MSE / MAE / MPJPE 优于 copy-last。
2. 速度、加速度、首帧连续性、末帧误差没有明显退化。
3. 双人完整相对关节、相对朝向、相对根平移和接触关系改善。
4. FID、动作识别、Diversity、Multimodality 作为生成质量辅助证据。
5. 双人视频作为定性审计，不替代全 test 定量指标。
```

论文以 FID、action accuracy、Diversity、Multimodality 和 train/test-conditioned 泛化为主，因为 actor -> reactor 任务存在多个合理反应，不适合逐帧 paired MSE 作为唯一结论。当前 forecasting 任务有同一条样本的真实 future50，必须把 paired 指标和 copy-last 放在主位置。

## 新增实现边界

计划新增而非修改旧主线：

```text
data_loaders/forecasting/ntu_2p_diffusion.py
utils/ntu_2p_rot6d.py
model/forecasting_ntu_2p_diffusion.py
train/train_ntu_2p_forecasting_diffusion.py
eval/eval_ntu_2p_forecasting_diffusion.py
sample/sample_ntu_2p_forecasting_diffusion.py
scripts/check_ntu_2p_diffusion_gates.py
```

具体文件名可在实现计划中根据现有模块复用情况微调，但不得覆盖：

```text
model/forecasting_ntu_xyz.py
train/train_ntu_label_xyz.py
train/train_label_forecasting_diffusion.py
save/forecasting/ntu120_label/xyz_loss_stageD_formal_s0/
```

## 风险与边界

```text
1. 当前 Person A/B 的 actor/reactor 语义未能从 H5 独立审计，结果中不能超出该证据范围。
2. rotation-6D 扩散、SMPL-X FK 和 L_inter 会显著提高显存与训练耗时；必须先通过小 batch gradient gate。
3. 扩散的 paired MSE 可能弱于直接回归，但在分布、多样性或互动关系上更强；报告必须同时给出两类证据。
4. 不能只因 FID 或动作识别改善就宣布预测成功；copy-last 对照和 paired future50 指标是硬边界。
5. label swap 只能说明条件敏感性，不能作为 counterfactual 预测正确性的主证据。
```

## 实现前的下一步

基于本设计再写逐文件实现计划，包含每个模块的输入输出 shape、单元测试、smoke command、正式训练命令和结果目录命名。通过计划审阅后才修改代码。
