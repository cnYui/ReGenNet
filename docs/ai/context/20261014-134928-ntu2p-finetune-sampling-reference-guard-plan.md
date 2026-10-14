# 端到端候选的随机评价边界与冒烟记录

## 已完成验证

F3 CUDA200-step 冒烟完成，训练39.00秒，梯度有限，完整H-val终点评价完成。记录位于 `results/forecasting/ntu120_label/ntu2p_graphdiff/subjval/finetune_smoke200/`。200步只验证流水线，不作采纳证据。F1/F2/F3 的5000-step正式筛选已启动，网络、损失、学习率及采纳线在运行期间保持固定。

## 静态审查发现与设计

原随机采样评价脚本针对冻结 C4 登记。它把 `prepared['draft']` 作为 FD2 参照，并在该轨迹上叠加原 FD2 的 OOF 残差。F 模型的 draft 已微调，原 FD2 则位于独立 reference；直接复用该脚本会把微调底座误报为原参照，并改变 bootstrap 中心。即使标为范围外诊断，这个语义仍有问题。

在进入采样前拒绝 `fine_tune_prior=True` 的 checkpoint，明确 F 随机评价需要单独设计原 FD2 reference 与对应残差参照。此改动只限制尚未登记的评价入口，不修改模型、训练或已有 C4 产物。F3 点预测仍由正式统一评价脚本使用冻结 reference 正常比较。若本轮 F3 值得继续研究随机模式，先新建评价设计与计划，再实现支持。

验证仅需确认冻结 C4 可通过该模式检查、F3 被拒绝，并运行语法/ruff；无需重复耗时采样。
