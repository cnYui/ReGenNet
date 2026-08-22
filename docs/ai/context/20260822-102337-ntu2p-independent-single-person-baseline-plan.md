# NTU 独立单人 baseline 实施计划

## 当前状态

设计、实现、CUDA 训练和冻结 checkpoint 的 test 评估均已完成。结果见同时间戳的 `*-result.md`。

## 步骤

1. [完成] 复用 `NTULabelXYZTransformer`，允许 `num_persons=1` 的配置安全保存/加载。
2. [完成] 新增独立单人训练入口；双人 batch 沿 person 维展开，动作标签复制到两个分支。
3. [完成] evaluator 增加 `independent_single_person` 模式，A/B 分别 forward 后拼接为双人预测。
4. [完成] CUDA smoke、5000 step CUDA 正式训练、checkpoint reload 和输入隔离检查。
5. [完成] 冻结 val 最优 step 3000，在 1253 条 test 样本上完成一次评估。

## 取舍

- 参数共享而非 A/B 两套模型：控制容量，避免 baseline 因参数翻倍而失去可比性。
- 保留动作标签：与 diffusion 主线条件一致；禁止另一人的动作或姿态作为输入。
- 只扩展现有脚本的必要分支：减少新 checkpoint schema 和重复评测代码。
