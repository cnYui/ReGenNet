# GRPO（JiT-RL）梯度平衡修订

关联设计：`20261018-201532-ntu2p-jit-flow-rl-design-and-plan.md` 3.5 节。修订发生在任何 RL 结果评价之前。

## 发现

按设计实现的第一版（对数似然与 KL 按元素取均值，与 Flow-GRPO 实现一致；另加权重 1 的端点 FD2 配方损失）在 J4 上跑到约 430 次迭代时，RL-A（只奖励精度）与 RL-F（完整奖励）的 train 奖励分项逐项几乎相同（acc 0.149→0.159、rel 0.216→0.224、ant 0.016→0.013，两者到小数点后 3 位一致）。两者共享数据与噪声种子，唯一共同的驱动是端点损失，说明奖励几乎不影响更新。

`scripts/diag_ntu2p_jitflow_grpo_grad_balance.py`（J4 EMA，train batch 4 条件 × 8 样本）实测梯度范数：逐元素均值形式的策略梯度 0.082，端点配方损失 3.88（约 1/47）；按元素求和（精确高斯对数似然）为 281（约 72 倍，反过来淹没端点项）。Adam 跟随总梯度方向，均值形式下 RL 实际不起作用。

## 修订

`train/train_ntu2p_jitflow_grpo.py` 新增 `--pg_scale`（默认 50）：PG+βKL 项整体乘 50，使其初始梯度与端点项同量级；比值与裁剪仍用均值形式（比值保持在 1 附近，数值稳定）。倍数只由上述 train 梯度诊断确定。其余设置不变（G=16、batch 8、10 步、ρ=0.7、β=0.04、clip 0.2、2 次内更新、lr 2e-5、端点损失权重 1、1500 次迭代、EMA 0.999）。

第一版两次 run 保存在 `save/forecasting/ntu120_label/ntu2p_jitflow_subjval_J4-RL{A,F}-tmix_s0_5000_grpo1500_invalid_unscaled_pg/`（日志 `..._invalid_unscaled_pg.log`），作废，不评价。修订版 RL-F、RL-A 依次重跑（单卡同时还在跑 3 seed 复现，两个 GRPO 并发会显存不足）。
