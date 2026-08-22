# NTU 双人 10->50 独立单人预测 baseline 设计

## 目标

新增一个比 `copy-last` 更有信息量、但不建模两人关系的学习型 baseline：

```text
Person A：只用 A 的 obs10 + 同一动作标签，预测 A 的 future50
Person B：只用 B 的 obs10 + 同一动作标签，预测 B 的 future50
双人结果：将 A/B 预测按原顺序拼回，统一计算 paired 双人指标
```

它替代“同一帧复制 50 次”作为最基础的学习型参考；`copy-last` 仍保留为无训练 persistence baseline，不能删除或改名。

## 固定定义

- 数据、manifest、窗口和拆分沿用 NTU 双人 diffusion 主线：`window_len=60, obs_len=10, pred_len=50`。
- 模型结构沿用现有 `NTULabelXYZTransformer` 的 Transformer encoder/decoder、位置编码、动作条件和几何 loss 配方。
- 单人模型设置 `num_persons=1`，输入输出为 `[B,T,1,55,3]`；不把另一人的观测拼入输入。
- A/B 使用同一套参数（参数共享），训练时把 `[B, T, 2, 55, 3]` 沿 person 维展开成 `[2B, T, 1, 55, 3]`，因此不会因训练两套网络增加容量。
- action 标签可作为动作条件；由于两人来自同一个样本，A/B 使用同一标签，但预测分支之间没有 feature、hidden state 或 target 交互。
- 单人训练关闭所有需要两人的交互项：`relative_root_loss_weight=0`、`relative_velocity_loss_weight=0`、`key_joint_relation_loss_weight=0`、`contact_loss_weight=0`，以及 action feature/logit loss 保持 0。
- 单人预测结果仅在拼回双人后计算 `xyz_mse`、`xyz_mae`、`mpjpe`、速度/加速度、末帧、相对关系和接触等 paired 指标；不把单人 loss 数值直接与双人 loss 比较。

## 公平性边界

该 baseline 与双人 direct Transformer 的差别必须明确记录：

```text
独立单人 baseline：num_persons=1，不能看另一人的 obs，不计算交互训练项
双人 direct reference：num_persons=2，可联合编码两人的 obs，保留既有双人几何配置
```

两者不是同容量消融。独立单人 baseline 用来回答“仅靠每个人自己的历史，能达到什么水平”；双人 direct 用来回答“联合确定性预测器能达到什么水平”。扩散模型仍按 `copy-last -> independent-single-person -> direct-two-person -> L_dm-only -> L_dm+L_inter` 报告，避免把两个问题混成一个结论。

## 实施方案

1. 在 `model/forecasting_ntu_xyz.py` 增加单人模型工厂或明确的 `num_persons=1` 配置校验，不复制模型架构。
2. 在训练脚本中增加 `--independent_person_baseline`（或等价独立入口）：把双人 xyz batch 展开为单人 batch，训练并保存 `representation=independent_single_person_xyz`、`num_persons=1`、`person_shared_parameters=true`。
3. 在双人 diffusion evaluator 增加 `--mode independent_single_person`，加载单人 checkpoint，分别预测 A/B 后拼回 `[B,T,2,55,3]`；主结果保存 paired metrics，数组保留 A/B 预测顺序，便于后续按人诊断。
4. 训练/选择只看固定 val；冻结 checkpoint 后再跑 test。所有结果写入新的 `ntu2p_independent_single_person_o10_p50_*` 目录，不覆盖历史目录。
5. 增加 shape、person split/join、单人 forward、拼回顺序和 paired metric 的 smoke gate。

## 预注册判定

- 该 baseline 必须能生成随时间变化的 future50；逐帧等于 obs 最后一帧只能作为初始化或训练失败诊断，不能冒充学习结果。
- 主表同时列出 `copy-last` 与 independent-single-person；三项主指标分别报告，不以单一 MSE 改善宣称整体优于 baseline。
- 不依据该 baseline 的 paired 交互指标宣称“建模交互”；它恰好用于量化没有显式关系建模时的下限。
