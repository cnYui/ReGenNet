# 端到端 F1/F2/F3 实现与 CPU 结构验证

依据 `20261014-120112-ntu2p-end-to-end-graphdiff-followup-design-and-plan.md`，已实现端到端候选；本文件只报告结构与训练流程验证，不报告模型收益。

## 实现

- `model/forecasting_ntu2p_graphdiff.py` 新增 `fine_tune_prior=False`，旧 C 默认模式保留。F 的 `prior` 可训练，`reference` 为原 FD2 的独立完整副本，永久冻结、始终 eval，全部权重保存在 checkpoint；评估 `details['prior']` 始终来自 reference。
- F 的点预测、局部速度与接触图几何允许梯度返回 InterMixer；构造残差目标仍在 `torch.no_grad()` 内。reference 的复制不消耗随机数。
- `train/train_ntu2p_graphdiff.py` 新增 F1（无图/腿流/扩散）、F2（图+腿流）、F3（F2+扩散）。三者可训练参数均为 4,146,400，固定 batch8，纠正器 lr=3e-4，InterMixer lr=3e-5。F 模式和两组学习率写入 args；尺度估计显式 eval，避免底座 dropout 改变常量或训练随机流。
- `scripts/run_ntu2p_graphdiff.py` 支持 F 候选，独立候选 run 名并校验微调/双学习率记录；F 请求记录额外保存固定训练配置。C 默认候选、既有历史产物、评估 helper、损失公式与采纳阈值未修改。

## 验证证据

新增 `scripts/check_ntu2p_graphdiff_finetune.py`，从真实训练入口分别执行 F1/F2/F3 的 CPU 两步训练。结果 **66/66 通过**：

1. 修改前保存 C1–C4 的实际 seed0×5000 EMA checkpoint 输出。修改后 eval/train 的全部 details tensor 逐位相同，最大差异 0；train 随机流也逐位相同，C 不含新增 reference 权重。
2. F1/F2/F3 的参数初始化相同；复制 reference 不消耗额外随机数。真实训练开始前，参数、sigma、主/辅助损失尺度和全局随机流一致。
3. 每个候选两步都有有限非零底座梯度；新输出头有有限非零梯度，F2/F3 的图与腿专用流也有。点轨迹/局部特征/场景几何保留梯度，残差目标停止梯度。
4. 训练会改变底座；reference 无梯度、参数不变，eval 的 `prior` 与原 FD2 输出逐位一致。AdamW 两组覆盖所有可训练参数、无重复、排除 reference。
5. EMA 对底座按预期平均、对 reference 直接复制。raw/EMA checkpoint 重载后的全部输出逐位相同，reference 完整保留。

结果 JSON：`results/forecasting/ntu120_label/ntu2p_graphdiff/finetune_structural_checks_20261014-134355.json`。

旧模式兼容快照：`results/forecasting/ntu120_label/ntu2p_graphdiff/finetune_legacy_snapshot_20261014-133324.pt`，含修改前模型源码 SHA256、实际 checkpoint 路径与 eval/train 输出。

四个相关 Python 文件的 `py_compile` 通过。验证只使用 CPU，没有启动 CUDA 正式训练。

## 后续执行

F3 CUDA 200-step 冒烟可由主代理立即运行；驱动显式使用 cuda:0，并在终点附同口径 val 评估：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /home/rpartx3080/.local/micromamba/envs/regennet/bin/python scripts/run_ntu2p_graphdiff.py --protocol subjval --candidates F3 --steps 200 --workers 1 --tag finetune_smoke200 --seeds 0
```

通过后独立标签完成 5000-step 筛选：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /home/rpartx3080/.local/micromamba/envs/regennet/bin/python scripts/run_ntu2p_graphdiff.py --protocol subjval --candidates F1 F2 F3 --steps 5000 --workers 2 --tag finetune_screen5000 --seeds 0
```

F 阶段仍须遵循原采纳线和固定案例审查。接触图没有接触前禁止后退的硬保证，腿分支不是相位状态机；F3 点指标变好也不能自动证明随机扩散质量改善。
