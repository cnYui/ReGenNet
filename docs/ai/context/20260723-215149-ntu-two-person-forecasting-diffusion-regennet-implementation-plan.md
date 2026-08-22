# NTU 双人 10->50 ReGenNet 风格扩散预测完整实施计划

## 状态与依据

本计划落实 `20260723-214702-ntu-two-person-forecasting-diffusion-regennet-design.md`。尚未修改实现代码，尚未开始训练。

三个正式阶段为：

1. 阶段 1：建立公平 baseline。
2. 阶段 2：从随机初始化正式训练 `L_dm + L_inter`。
3. 阶段 3：论文式生成质量评估与模型选择。

实施前增加阶段 0 工程 gate。阶段 0 不产生任何可报告模型结论。

## 固定协议

### 任务与数据

```text
数据集：NTU120 双人 xsub
window_len：60
obs_len：10
pred_len：50
条件：双人 obs10 + 26 类动作标签
目标：双人 future50
```

当前 H5 中暂定：

```text
raw motion[..., 0:3] = Person A axis-angle / translation
raw motion[..., 3:6] = Person B axis-angle / translation
```

Person A/B 只表示 H5 当前通道顺序。预处理历史上意图将 actor 放在前、reactor 放在后，但现有 H5 未保存逐样本角色元数据或生成清单。因此训练可以使用 A/B 的稳定顺序，结果不能单独宣称已验证 actor/reactor 监督。

### 数据拆分纪律

```text
xsub.train：训练来源。
xsub.train 内部：按 sample_id 的动作类别分层切分 train/val。
xsub.test：只在模型、配置、DDIM 步数冻结后执行正式评估。
```

同一个 sample_id 的不同随机窗口不得跨 train/val。切分 manifest 必须保存 sample_id、action、seed、样本数和路径。test 不得用于调 loss、选择 checkpoint、选择 DDIM 步数或挑视频样例。

### 输出路径

所有新实验使用独立目录：

```text
save/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_<experiment>/
results/forecasting/ntu120_label/ntu2p_diffusion_o10_p50_<experiment>/
```

不得覆盖现有 `obs20/pred40` checkpoint、cache、视频或 Stage D 结果。

## 数学与表示

### Canonical 双人 rot6d

原 H5 的 `[B,56,6,T]` 是两人 axis-angle，不是单人 rot6d。新数据适配器把每人的前三维 axis-angle 转为 rotation-6D，模型输入和目标统一为 `[B,56,12,T]`：

```text
slot 0:55, channel 0:6  = Person A rot6d
slot 0:55, channel 6:12 = Person B rot6d
slot 55, channel 0:3    = Person A translation
slot 55, channel 6:9    = Person B translation
未使用的 translation padding channel = 0
```

以 `pose_rep=rot6d,num_person=2` 调用 `Rotation2xyz_x`。这与 ReGenNet 采用 6D rotation representation 的口径对齐。

### 条件扩散

```text
C = {双人 obs10, action label}
Y_0 = 双人真实 future50 的 canonical rot6d tensor
t ~ Uniform({0,...,999})
epsilon ~ N(0,I)
Y_t = sqrt(alpha_bar_t) * Y_0 + sqrt(1-alpha_bar_t) * epsilon
Y0_hat = F(Y_t,t,C)
```

正式主模型固定使用：

```text
diffusion_steps = 1000
noise_schedule = cosine
mean_type = START_X
variance = FIXED_SMALL
loss_type = MSE
timestep_sampling = uniform
one_step_noise_prob = 0
```

高噪重采样或直接纯高斯 `t=999` 训练不进入正式论文式结果；若需要，只能单列后续扩展实验。

### 总损失

```text
L_dm = mean((Y0_hat - Y_0)^2)
L_all = L_dm + 1.0 * L_inter
```

对 target 与 prediction 分别进行可微 SMPL-X FK，得到：

```text
J_A, J_B: [B,50,55,3] 全局 joint xyz
R_A, R_B: [B,50,3,3] root rotation matrix
g_A, g_B: [B,50,3] root translation
```

交互量与损失为：

```text
r_joint = J_A - J_B
R_rel = transpose(R_A) @ R_B
r_trans = g_A - g_B

L_joint  = mean((r_joint_hat-r_joint)^2)
L_orient = mean((R_rel_hat-R_rel)^2)
L_trans  = mean((r_trans_hat-r_trans)^2)
L_inter  = L_joint + L_orient + L_trans
```

不得在比较前对向量或相对旋转取长度。旧 `GaussianDiffusion.masked_l2()` 的维度假定也不能用于本交互 loss。

## 新增模块

### `utils/ntu_2p_rot6d.py`

职责：唯一管理双人 raw rotvec、canonical rot6d、FK 与交互 loss。

计划接口：

```text
check_raw_ntu_2p_motion(value, seq_len)
raw_ntu_2p_to_rot6d(value)
check_ntu_2p_rot6d(value, seq_len)
split_ntu_2p_rot6d(value)
join_ntu_2p_rot6d(person_a, person_b)
ntu_2p_rot6d_to_xyz(value, converter)
root_rotation_matrices(value)
interaction_targets(value, converter)
interaction_loss(pred, target, converter)
```

所有函数检查 shape、dtype、device 和 finite。translation 只能读取 slot 55；输出 xyz 固定为 `[B,T,2,55,3]`。

### `data_loaders/forecasting/ntu_2p_diffusion.py`

职责：读取现有 H5，截取连续 60 帧，转为 canonical 双人 rot6d，并输出固定预测窗口。

单样本输出：

```text
obs_motion: [56,12,10]
future: [56,12,50]
action: [1]
mask: [1,1,50]
meta: sample_id,start,length,action,action_code,split
```

规则：train 可随机选连续窗口；val/test 使用确定性窗口；所有 split 依赖 sample_id manifest；不得写入或伪造 actor/reactor 标签。

### `model/forecasting_ntu_2p_diffusion.py`

新增 `NTU2PForecastingDiffusionDecoder`。

接口：

```text
forward(noisy_future,timesteps,y) -> pred_xstart
noisy_future: [B,56,12,50]
y['obs_motion']: [B,56,12,10]
y['action']: [B,1]
pred_xstart: [B,56,12,50]
```

结构：

```text
obs10 -> InputProcess -> position encoding -> TransformerEncoder
timestep/action/obs summary/obs tokens -> memory
noisy future50 -> InputProcess -> future position encoding -> Decoder target
TransformerDecoder(tgt=future_tokens,memory=memory,tgt_mask=None)
OutputProcess -> clean future50
```

主模型不使用 causal future mask。论文使用 mask 是为了阻止在线生成看到未来 actor 条件；本模型条件只含 obs10，不存在该泄漏。`causal_future_mask` 仅作为后续独立消融开关，默认关闭。

模型提供 `config()`、shape/finite 检查和 action classifier-free condition dropout。SMPL-X FK 不放入模型，避免训练、采样、评估产生重复实现。

### `train/train_ntu_2p_forecasting_diffusion.py`

单步训练固定为：

```text
1. 取 future=Y_0、obs10、action、mask。
2. 从 uniform sampler 采样 t 和 importance weight。
3. noise=randn_like(Y_0)，Y_t=q_sample(Y_0,t,noise)。
4. Y0_hat=model(Y_t,t,{obs10,action,mask})。
5. 计算 L_dm。
6. inter_loss_weight=0 时只用 L_dm；为 1 时计算完整 L_inter。
7. 按 diffusion weight 求 batch mean，finite 检查，backward，clip，step。
```

日志字段：

```text
train_loss
rot_mse
joint_mse
orient_mse
trans_mse
inter_loss
t_mean
lr
effective_batch_size
```

checkpoint 除 state_dict 外记录：model/diffusion/loss config、representation=`two_person_rot6d`、person order=`person_a_then_person_b_assumed`、manifest path/hash、step 和 seed。

### `eval/eval_ntu_2p_forecasting_diffusion.py`

支持 copy-last、direct checkpoint 与 diffusion checkpoint。所有模型必须在相同 val/test metadata 上比较。

paired 主指标：

```text
xyz_mse, xyz_mae, mpjpe
first_step_error, velocity_error, acceleration_error, final_frame_error
relative_joint_vector_error
relative_root_translation_error
relative_orientation_error
contact_error
```

paired 指标使用预先固定 seed，并额外报告多个 seed 的 mean/std。禁止 best-of-K，避免给扩散模型不公平的 oracle 优势。

### `sample/sample_ntu_2p_forecasting_diffusion.py`

从标准高斯噪声开始，以 DDIM 或 DDPM 反向采样。保存 obs10、real future50、generated future50 的 rot6d、xyz、Person A/B metadata 和 sampling config。

### `scripts/check_ntu_2p_diffusion_gates.py`

阶段 0 自动 gate，任意失败返回非零退出码：

```text
raw/rot6d/xyz shape
Person A/B split/join round-trip
rotvec -> rot6d -> FK finite
q_sample low/high timestep finite
pred==target 时所有 loss 为 0
受控 A/B、translation、root orientation 扰动的 loss 响应
L_all.backward() 后梯度 finite 且非空
train/val/test manifest 无 sample_id 泄漏
```

## 阶段 0：工程与数学 gate

阶段 0 不训练正式模型，顺序如下：

1. 扫描 `T>=60` 样本数、26 类覆盖和长度分布。
2. 建立 sample_id 级别 train/val manifest，检查三 split 不重叠。
3. 抽取固定样本，验证 obs10 与 future50 重新拼接为同一连续 60 帧。
4. 验证 canonical rot6d 和 FK 输出 shape。
5. 验证 q_sample、L_dm、L_inter 的零值、受控扰动和梯度。
6. 执行单 batch forward、两 optimizer step、checkpoint reload。
7. 执行 DDIM2 和 DDPM2，输出必须 finite、非全零、非逐元素复制 obs 最后一帧。
8. 小规模 val eval，生成 copy-last 与模型指标文件。

阶段 0 任一失败，停止后续阶段并新建诊断文档。

## 阶段 1：公平 baseline

阶段 1 的目标是证明基础预测与扩散训练成立，不评价 `L_inter`。

### 1A：copy-last

```text
future50 每帧 = obs 的第 10 帧
```

无需训练。必须保存 full val/test 全部 paired 指标。这是所有学习模型的最低硬门槛。

### 1B：direct xyz Transformer

重训现有 `NTULabelXYZTransformer`，但使用 `obs_len=10,pred_len=50`。历史 `obs20,pred40` Stage D checkpoint 不能做数值比较。

direct 模型使用当前几何预测配方；依赖未验证 future50 classifier 的 action feature loss 在本阶段关闭。模型选择在 val 上完成，冻结后才运行 test。

### 1C：diffusion L_dm-only

使用新 diffusion Decoder，全部模型、优化器、步数、batch、noise schedule 与阶段 2 一致，仅设置：

```text
inter_loss_weight=0
loss=L_dm
```

它是阶段 2 的 `w.o. L_inter` 严格消融，不能使用阶段 2 checkpoint 反向初始化。

### 阶段 1 顺序与 gate

```text
copy-last -> direct smoke -> direct formal -> diffusion L_dm smoke -> diffusion L_dm formal
```

先用 seed 0 完成闭环。通过 val gate 后，再以相同 seed 集合复跑 direct 与两种 diffusion，最终报告 mean/std。

进入阶段 2 的条件：

```text
所有 loss/指标/output finite。
diffusion L_dm-only 能从噪声生成双人 future50。
至少一个学习模型在 val 的 xyz_mse、xyz_mae、mpjpe 上超过 copy-last。
若 L_dm-only diffusion 未超过 copy-last，先诊断表示、采样或训练对齐，不进入阶段 2。
```

## 阶段 2：正式论文式 L_dm + L_inter

### 训练规则

从随机初始化开始。模型宽度、训练步数、seed、optimizer、batch、noise schedule、采样配置均与阶段 1C 相同，唯一正式变量为：

```text
inter_loss_weight=1
loss=L_dm+L_inter
```

正式模型从第一个 optimizer step 起计算总损失。短 smoke 只能检查数值；不得以 L_dm-only warm start 后的模型作为论文式正式结果。

### 日志、checkpoint 和选择

每个训练 checkpoint 在内部 val 上记录 paired 和 interaction 指标。checkpoint 选择规则预注册为：

```text
先通过 xyz_mse、xyz_mae、mpjpe 的 copy-last gate。
通过者中优先 interaction 指标改善且 paired 指标最优者。
禁止根据 test、FID 或挑选视频确定 checkpoint。
```

### 阶段 2 硬门槛

full xsub.test 上，对比 copy-last、direct、diffusion L_dm-only、diffusion L_dm+L_inter：

```text
1. xyz_mse < copy-last xyz_mse。
2. xyz_mae <= copy-last xyz_mae。
3. mpjpe < copy-last mpjpe。
4. relative joint、root translation、orientation、contact 指标整体优于 L_dm-only。
5. L_inter 不得造成主 paired 指标实质性退化。
```

结论层级严格区分：

```text
超过 copy-last：具备基本预测价值。
优于 L_dm-only：显式交互损失有增益。
优于 direct xyz：扩散主线优于确定性 baseline。
```

必需消融只有 `L_dm-only`。causal future mask 等结构消融必须在阶段 2 通过后独立进行，避免同时改变 loss 与结构而不能归因。

## 阶段 3：论文式生成评估与模型选择

阶段 3 默认不向扩散网络增加损失。它检查通过阶段 2 的模型是否也有论文要求的生成质量。

### 3A：真实 future50 分类器 gate

当前 future40 classifier 不可复用。训练仅使用真实双人 future50 xyz 的新 classifier：

```text
训练：内部 train
选择：内部 val
最终 gate：xsub.test 真实 future50
```

记录 top1、top5、balanced accuracy、每类样本数、每类 accuracy、majority baseline 和 confusion matrix。只有真实 future50 gate 显著优于 chance/majority 且输出有限时，才允许引用 generated action accuracy 和 FID。该 classifier 只评估，不反向训练阶段 2 的扩散模型。

### 3B：多样本生成质量

```text
paired 主指标：固定 seed，并报告多个 seed mean/std，禁止 best-of-K。
FID/action/Diversity/Multimodality：对预注册条件和 seed 多次采样。
```

正式论文式评估目标：每次 1000 个条件样本、20 次独立重复、报告均值和 95% 置信区间。工程 smoke 可减少样本和重复，但不能替代正式表格。

### 3C：DDIM、DDPM 和延迟

对冻结 checkpoint 比较：

```text
DDIM 5 / 10 / 50 step
必要时 DDPM 1000 step 作为慢速参考
```

每种配置记录 paired、interaction、FID、action、Diversity、Multimodality 和单样本平均延迟。推理配置在 val 或预注册规则上选择，test 只报告。

### 3D：双人视频

输出真双人 SMPL-X skeleton：

```text
蓝色：双人 obs10
橙色：双人 generated future50
绿色：双人 real future50
```

case 选择在生成前固定：按动作类别分层，并覆盖低、中、高 paired error 分位数。视频仅做 Person A/B、连续性、接触和异常姿势审计，不能替代全 test 定量结论。

### 阶段 3 输出

统一比较表至少含：

```text
MSE, MAE, MPJPE
first, velocity, acceleration, final error
relative joint, root, orientation, contact error
FID, action accuracy, Diversity, Multimodality
sampling latency
```

若几何预测通过但 FID/动作指标不足，应如实报告。不得直接加 classifier loss 追逐这些指标；任何语义 feature loss 必须新开扩展分支，并声明不再是严格 `L_dm+L_inter` 主模型。

## 验证与回归保护

单元级：raw 通道拆分、rotvec 到 rot6d、split/join、FK、零值 loss、三类受控扰动、梯度。

集成级：dataset/collate、forward、train step、checkpoint reload、DDIM/DDPM、评估样本数和 metadata 一致。

回归保护：不改 `model/forecasting_ntu_xyz.py`、`train/train_ntu_label_xyz.py`、`train/train_label_forecasting_diffusion.py`；任何新 checkpoint metadata 必须标记 `num_person=2`、`representation=two_person_rot6d`、`obs_len=10`、`pred_len=50`。

## 实施顺序

1. 实现 `utils/ntu_2p_rot6d.py` 与单元 gate。
2. 实现 dataset、manifest、collate。
3. 实现扩散 Decoder。
4. 实现训练入口、日志和 checkpoint schema。
5. 实现阶段 0 自动检查，全部通过后继续。
6. 实现统一 eval/sample，先输出 copy-last。
7. 执行阶段 1 smoke 和 formal baseline。
8. 执行阶段 2 smoke 和正式总损失训练。
9. 阶段 2 通过后训练 future50 classifier。
10. 执行阶段 3 多样本、FID、DDIM、视频和消融评估。
11. 每个阶段结束新建时间戳结果文档，不覆盖本计划或设计文档。

## 停止条件

以下任一情况不进入下一阶段：

```text
阶段 0 出现 shape、FK、零值 loss 或梯度失败。
阶段 1 的 L_dm-only diffusion 不能超过 copy-last。
阶段 2 的 L_inter 未改善 held-out interaction，或实质性损害主 paired 指标。
阶段 3 的真实 future50 classifier gate 失败，却试图报告其 FID/action 结论。
```

停止时必须新建诊断文档，记录证据、未通过 gate、受影响结论与下一步候选，不得将未通过 gate 的模型写为正式论文结果。
