# ReGenNet AI 入口

## 稳定约束

- 默认使用中文；文档、计划、总结和代码注释均使用中文，除非用户明确要求英文。
- 实现前必须先完成 design / plan；把上下文、决策、取舍和结果写入 `docs/ai/context/`。
- 新增上下文只创建 `YYYYMMDD-HHMMSS-文件名.md`，不覆写、重命名或删除历史文件。
- 代码注释写原因，不写过程；优先函数式、复用现有模块，保持 KISS/DRY。

## 当前研究入口

- 已完成主线：InterHuman 双人动作预测，固定 `150/30/120` 协议；详细结果见 `docs/ai/context/20260614-185753-regennet-key-context-extracted-from-agents.md` 及其引用文档。
- 当前 NTU 双人 xyz 主线：必须使用真实双人 skeleton/xyz 口径，并将模型与 `copy-last` 同时评估；主指标为 paired `xyz_mse/xyz_mae/mpjpe` 等，动作分类、FID 和视频仅作辅助证据。
- 新增独立单人学习型 baseline：复用 `NTULabelXYZTransformer`，`num_persons=1` 且 A/B 参数共享；每个人只看自己的 obs10 与动作标签，分别预测 future50 后拼回双人。它与 `copy-last`、联合双人 direct reference 分开报告，不能声称显式交互建模。
- 当前推进中的 NTU 双人 ReGenNet 风格 diffusion：`window_len=60, obs_len=10, pred_len=50`，双人 raw axis-angle 转 canonical rot6d，条件为 obs10 + 26 类动作标签，目标为 future50，训练目标为 `L_dm + L_inter`。协议和实现见 `docs/ai/context/20260723-214702-ntu-two-person-forecasting-diffusion-regennet-design.md` 与 `docs/ai/context/20260723-215149-ntu-two-person-forecasting-diffusion-regennet-implementation-plan.md`。
- `L_dm-only` 不再是进入 `L_dm + L_inter` 的科学前置 gate；两者必须实际训练后比较。修订协议及 `L_dm + 1.0 * L_inter` CUDA 5000-step 完整训练结果见 `docs/ai/context/20260822-100014-ntu2p-diffusion-protocol-revision-plan.md` 与 `docs/ai/context/20260822-102600-ntu2p-diffusion-ldm-inter-full-training-result.md`。本轮 val 的五个 checkpoint 在 `xyz_mse/xyz_mae/mpjpe` 均未超过 `copy-last`。
- NTU 双人 diffusion 训练必须显式使用 `--device cuda:0`；禁止 CUDA 不可用时静默回退 CPU。设备策略见 `docs/ai/context/20260821-221534-ntu2p-diffusion-cuda-training-enforcement-result.md`。
- 独立单人 baseline 已用 `cuda:0` 完成 5000 step；设计、训练和 test 结果见 `docs/ai/context/20260822-102337-ntu2p-independent-single-person-baseline-result.md`。
- 当前新的主要比较 gate 是 `L_dm + L_inter` 双人联合模型同时超过 independent single-person xyz baseline 的 `xyz_mse/xyz_mae/mpjpe`；`copy-last` 继续作为非学习型 persistence 参考，direct joint xyz 作为联合输入学习型 reference。旧 diffusion 未过 gate 的深度排查见 `docs/ai/context/20260822-115553-ntu2p-diffusion-vs-independent-baseline-deep-audit.md`：问题集中在纯噪声自由采样与训练目标不对齐、首帧连续性未约束、`L_inter` 尺度过强及 rot6d 回归误差，不能归因于双人关系信息无效。
- NTU 双人显式跨人 attention diffusion 已实现并完成 CUDA `5000 step` 训练：共享单人投影、人物内 temporal self-attention、双向 cross-attention、24-token memory 和分人物 future decoder。实现、训练计划与结果见 `docs/ai/context/20260822-193000-ntu2p-explicit-cross-person-attention-training-plan.md`、`docs/ai/context/20260822-193000-ntu2p-explicit-cross-person-attention-training-result.md`；当前五个 val checkpoint 的 `DDIM50` paired xyz 主指标仍未超过 `copy-last`，尚不能声称 attention 带来性能提升。
- NTU 双人 baseline residual-refinement 已完成 CUDA `5000 step`：冻结 independent single-person xyz baseline，新增双流 temporal self-attention、双向 cross-person attention 和零初始化 residual head。1000/2000/3000/4000/5000 五个 val checkpoint 均同时超过 inherited baseline 与 `copy-last` 的 paired `xyz_mse/xyz_mae/mpjpe`，首帧误差保持 0。结果见 `docs/ai/context/20260822-214340-ntu2p-baseline-residual-refinement-training-result.md`；下一步固定其他条件扫描 `inter_loss_weight=0.01/0.05/0.1`，计划见 `docs/ai/context/20260822-214340-ntu2p-residual-inter-loss-ablation-plan.md`。
- NTU 双人 residual refinement 的 `inter_loss_weight=0.01/0.05/0.1` 消融已完成。`0.01` 的最终 5000 step 最好，但没有在所有中间 checkpoint 稳定超过 `lambda=0`；`0.05/0.1` 中后期主指标退化。下一阶段默认保留 `inter_loss_weight=0`，先迁移到 residual diffusion；完整结果见 `docs/ai/context/20260822-222701-ntu2p-residual-inter-loss-ablation-result.md`。
- 当前最佳 residual refiner 的架构图、可复用 xyz 导出脚本和 8 个最佳案例视频已生成；架构与指标记录见 `docs/ai/context/20260823-084520-ntu2p-residual-refiner-architecture-visualization-result.md`。

## 解释边界

- InterHuman 本地 actor/reactor 顺序标签不是动作语义类别，不能当作“握手”等动作监督。
- 不声称 multi-person forecasting、interaction-aware 或 explicit relation 首创；relation-aware 只按已有同口径实验的相对收益表述。
- 历史 CMDM、旧 NTU120 长窗口和单 skeleton 视频记录已归档，不作为当前协议或结论；压缩前入口全文见 `docs/ai/context/20260822-100457-agents-content-archive.md`。
- 20260822 已按用户授权清理 `save/**/*.pt` 权重与 checkpoint，保留配置、指标、日志和诊断文件；执行记录见 `docs/ai/context/20260822-205033-save-weight-cleanup-plan.md`。

## 文档入口

- 长期项目记忆：`docs/ai/context/20260614-185753-regennet-key-context-extracted-from-agents.md`
- 阶段性设计、实验和结果：`docs/ai/context/` 下按时间戳文件
- 本次压缩计划与结果：`docs/ai/context/20260822-100457-agents-slimming-plan.md`、`docs/ai/context/20260822-100457-agents-slimming-result.md`
