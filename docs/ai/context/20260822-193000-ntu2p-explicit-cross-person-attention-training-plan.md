# NTU 双人显式跨人 attention 训练计划

## 背景

本轮依据 `20260822-191956-ntu2p-explicit-cross-person-attention-architecture-spec.md`，将现有双人 diffusion 的混合双人 token 改为保留人物来源的 A/B token，并开始 CUDA 训练。工作区已有用户改动，均不回退。

## 设计决策

1. 新增 `model/two_person_transformer.py`，只负责 A/B token 的人物内 temporal self-attention、双向共享 cross-attention、future-to-memory attention 和 FFN。
2. `model/forecasting_ntu_2p_diffusion.py` 使用 `split_ntu_2p_rot6d` 拆分人物；A/B 共享输入投影和输出投影，最后通过 `join_ntu_2p_rot6d` 恢复原始 `[B,56,12,50]` 接口。
3. memory 固定为 timestep、action、A/B summary、A/B contextual obs 共 24 个 token；future decoder 保持非因果联合去噪，除非命令显式启用 causal mask。
4. 用新的 `model_type=ntu2p_forecasting_diffusion_cross_person` 标识 checkpoint，避免与旧混合 token 权重误加载；checkpoint config 记录显式 attention 开关和 token 数。
5. 训练仍使用既有 `L_dm + inter_loss_weight * L_inter`、manifest、seed 和 CUDA 强制策略。首轮先用现有正式协议的 5000 step 配置；若显存不足，记录实际限制后再调整 batch 或 gradient accumulation，不静默切换 CPU。

## 验证门槛

- `split -> shared projection -> output projection -> join` 形状、有限值和梯度 smoke 通过。
- 修改 Person B 的 obs 时，cross attention 开启后 Person A 输出发生变化；模型正常训练模式下 dropout 关闭以便测试确定性。
- `cuda:0` 完成小 batch forward/backward，记录参数量和显存。
- 训练日志与 checkpoint 必须包含新架构配置、manifest hash、CUDA 元数据和 finite loss。
- 训练完成后按既有 val/test 评估入口与 `copy-last`、independent single-person baseline 对照；本轮不把训练启动写成性能提升结论。

## 执行顺序

1. 实现 token attention 模块和 diffusion 模型入口。
2. 运行静态检查、CPU shape smoke 和 CUDA forward/backward smoke。
3. 以 `cuda:0` 启动 5000 step 训练，保存 checkpoint 和日志。
4. 训练完成后记录结果文档；若训练仍在运行，先记录启动状态和可复现命令。
