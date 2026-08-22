# AGENTS.md 压缩与归档计划

## 目标

- 压缩根目录 `AGENTS.md`，使其只承担会话入口和稳定约束，不再承载逐阶段实验日志。
- 将压缩前完整内容归档到本目录，保证历史信息可追溯。
- 对当前主线只保留必要的协议、设备和论文结论边界；具体实现与结果继续引用已有时间戳文档。

## 设计

- `AGENTS.md` 保留中文偏好、design/plan 前置、上下文文件命名与不可覆盖规则。
- `AGENTS.md` 保留当前有效研究入口：InterHuman 150/30/120、NTU 双人 xyz，以及正在推进的 NTU 双人 10/50 ReGenNet 风格 diffusion。
- `AGENTS.md` 不再保留旧 ForecastingCMDM 阶段的逐步指标、checkpoint、视频目录和已废弃协议；这些内容归档并由原有 context 文档承载。
- 新增归档文档保存压缩前全文；新增结果文档记录压缩后的内容和验证。

## 取舍

- 不删除、重命名或覆写历史 `docs/ai/context/` 文档。
- 不修改当前工作区中与本任务无关的训练、评估、采样和数据文件。
- 不把过期实验结论继续写成当前主线；当前 NTU diffusion 强制使用 CUDA，协议为 `window_len=60, obs_len=10, pred_len=50`，formal 与 exploratory 结果分开。
