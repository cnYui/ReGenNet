# ReGenNet AI 入口

## 稳定约束

- 默认使用中文；文档、计划、总结和代码注释均使用中文，除非用户明确要求英文。
- 实现前必须先完成 design / plan；把上下文、决策、取舍和结果写入 `docs/ai/context/`。
- 新增上下文只创建 `YYYYMMDD-HHMMSS-文件名.md`，不覆写、重命名或删除历史文件。
- 代码注释写原因，不写过程；优先函数式、复用现有模块，保持 KISS/DRY。

## 当前研究入口

- 已完成主线：InterHuman 双人动作预测，固定 `150/30/120` 协议；详细结果见 `docs/ai/context/20260614-185753-regennet-key-context-extracted-from-agents.md` 及其引用文档。
- 当前 NTU 双人 xyz 主线：必须使用真实双人 skeleton/xyz 口径，并将模型与 `copy-last` 同时评估；主指标为 paired `xyz_mse/xyz_mae/mpjpe` 等，动作分类、FID 和视频仅作辅助证据。
- 当前推进中的 NTU 双人 ReGenNet 风格 diffusion：`window_len=60, obs_len=10, pred_len=50`，双人 raw axis-angle 转 canonical rot6d，条件为 obs10 + 26 类动作标签，目标为 future50，训练目标为 `L_dm + L_inter`。协议和实现见 `docs/ai/context/20260723-214702-ntu-two-person-forecasting-diffusion-regennet-design.md` 与 `docs/ai/context/20260723-215149-ntu-two-person-forecasting-diffusion-regennet-implementation-plan.md`。
- NTU 双人 diffusion 训练必须显式使用 `--device cuda:0`；禁止 CUDA 不可用时静默回退 CPU。设备策略见 `docs/ai/context/20260821-221534-ntu2p-diffusion-cuda-training-enforcement-result.md`。

## 解释边界

- InterHuman 本地 actor/reactor 顺序标签不是动作语义类别，不能当作“握手”等动作监督。
- 不声称 multi-person forecasting、interaction-aware 或 explicit relation 首创；relation-aware 只按已有同口径实验的相对收益表述。
- 历史 CMDM、旧 NTU120 长窗口和单 skeleton 视频记录已归档，不作为当前协议或结论；压缩前入口全文见 `docs/ai/context/20260822-100457-agents-content-archive.md`。

## 文档入口

- 长期项目记忆：`docs/ai/context/20260614-185753-regennet-key-context-extracted-from-agents.md`
- 阶段性设计、实验和结果：`docs/ai/context/` 下按时间戳文件
- 本次压缩计划与结果：`docs/ai/context/20260822-100457-agents-slimming-plan.md`、`docs/ai/context/20260822-100457-agents-slimming-result.md`
