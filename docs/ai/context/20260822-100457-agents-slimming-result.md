# AGENTS.md 压缩与归档结果

## 已完成

- 新建 `20260822-100457-agents-content-archive.md`，完整保存本次修改前 `AGENTS.md` 内容。
- 压缩根目录 `AGENTS.md`，保留稳定协作规则、当前研究主线、评估与论文边界及权威文档索引。
- 将旧阶段细节、过期协议和历史 checkpoint 交由归档文档及既有 `docs/ai/context/` 文档承载。

## 当前入口保留内容

- 默认中文、实现前先做 design/plan、所有上下文使用时间戳新文件保存。
- InterHuman 150/30/120 是已完成的主要 forecasting 研究线；结构化 baseline 以已有 context 结果为准。
- NTU 双人 xyz 的主指标必须和 copy-last 比较，必须使用真实双人 xyz/skeleton 口径；action/FID/视频只能作为辅助证据。
- 当前推进中的 NTU 双人 ReGenNet 风格 diffusion 使用 `window_len=60, obs_len=10, pred_len=50`、双人 rot6d、`L_dm + L_inter`，训练必须显式运行在 `cuda:0`。
- 不得把 InterHuman actor/reactor 顺序标签当动作语义类别，也不得声称 multi-person forecasting、interaction-aware 或 explicit relation 首创。

## 校验

- 已确认压缩前 `AGENTS.md` 为 29 行、约 12.8 KB；归档代码块与压缩前版本逐字一致。
- 已确认入口引用的长期记忆、当前 diffusion 设计、实现计划和 CUDA 设备结果文档均存在。
- 已确认新入口与归档文档存在，且 `git diff --check` 通过。
