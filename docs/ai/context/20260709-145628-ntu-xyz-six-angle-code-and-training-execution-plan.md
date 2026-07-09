# NTU xyz 六角度 loss 代码修改与训练执行计划

## 目标

根据 `20260709-144942-ntu-xyz-six-angle-abcd-training-plan.md` 开始落地代码，并跑完整 A/B/C/D 链路。

## 代码改动范围

计划修改：

```text
utils/ntu_smplx_2p_xyz.py
model/forecasting_ntu_xyz.py
train/train_ntu_label_xyz.py
eval/eval_ntu_label_xyz.py
```

计划新增：

```text
eval/action_xyz_classifier.py
```

原因：

```text
旧 action_consistency_classifier.py 面向 rotvec future [B,56,6,T]。
当前最终路线是 xyz skeleton [B,T,2,55,3]，不能混用旧分类器输入口径。
```

## 实现边界

阶段 A 默认启用：

```text
mae/root/local/long/final/velocity/acceleration/relative_root/relative_velocity
```

阶段 B 默认从阶段 A checkpoint 继续：

```text
key_joint_relation/contact
```

关键关节第一版只使用 SMPL/SMPL-X body 前 22 个关节中较稳定的 wrist 对：

```text
left_wrist = 20
right_wrist = 21
```

原因：

```text
完整手指 joint index 未在本地上下文里确认，先不用手指索引做训练约束。
```

阶段 C 新增 xyz action classifier：

```text
real future classifier gate
pred/copy action accuracy
pred/copy FID
pred/copy diversity
class-wise FID
```

阶段 D 只在阶段 C 分类器 gate 通过后启用：

```text
action_feature_loss
action_logit_loss 可选，默认关闭
```

## 训练执行目录

使用固定 Python：

```text
/home/rpartx3080/.local/micromamba/envs/regennet/bin/python
```

数据：

```text
train_xyz_cache = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/train_xyz.pt
eval_xyz_cache  = results/forecasting/ntu120_label/xyz_cache_len60_o20_p40/test_xyz.pt
```

输出：

```text
save/forecasting/ntu120_label/xyz_loss_stage0_smoke/
save/forecasting/ntu120_label/xyz_loss_stageA_tune_s0/
save/forecasting/ntu120_label/xyz_loss_stageB_tune_s0/
save/forecasting/ntu120_label/xyz_action_classifier_stageC_s0/
results/forecasting/ntu120_label/xyz_loss_stageC_eval_s0/
save/forecasting/ntu120_label/xyz_loss_stageD_tune_s0/
```

## 验收

```text
1. compileall 通过。
2. smoke train/eval 通过。
3. A/B/C/D 每阶段产出 metrics_test.json 或对应语义评估 json。
4. 每阶段都保留 copy-last 对照。
5. 最终新增结果文档记录通过/失败和回退判断。
```
