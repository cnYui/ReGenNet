<!-- prune:keep -->

# 提交 JiT-RL 工作；NTU2PForecast 已发布到 GitHub

用户 2026-11-05："你把代码提pr往github提交一下"。确认时选择：ReGenNet 分支 `feat/ntu2p-jit-flow-rl` 上的工作"一起提交并自动合并"（按 `scripts/ship_pr.py`）；NTU2PForecast "新建公开仓库"。

## 本仓库本次提交的内容

- 2026-10-19 JiT-RL 轮（JiT 流 + 双人接触图 + GRPO，按预登记不采纳）的代码与文档：`model/forecasting_ntu2p_jitflow.py`、`train/train_ntu2p_jitflow{,_grpo}.py`、`eval/eval_ntu2p_jitflow_samples.py`、`utils/ntu2p_interaction_timing.py`、相关 `scripts/`，以及 `eval/eval_ntu2p_v2.py`、`model/forecasting_ntu2p_v2.py` 的读出分派改动；结果入口已在 `AGENTS.md`，详见 `20261019-034335-ntu2p-jitflow-rl-final-result.md`。
- 同期未提交的文档：2026-10-12 周报对比、2026-10-18 FD2 架构图。
- 迁移指引 `20261104-173814-ntu2p-moved-to-ntu2pforecast-pointer.md` 与本文件。

这些文件在 2026-11-03 已按字节原样提取到 NTU2PForecast，两边内容一致；之后的改动只在 NTU2PForecast。

## NTU2PForecast 的后续状态（补充指引文档中"截至 2026-11-04"之后）

- GitHub 公开仓库 https://github.com/cnYui/NTU2PForecast ，首个 PR https://github.com/cnYui/NTU2PForecast/pull/1 （由用户审阅合并，无自动合并 CI）。新增 `LICENSE`（上游 MIT），因为提取的文件含上游代码。
- 原协议 test 已完成：主线（扩散 BR:ode0 + 落脚规划层）mpjpe .17971 ± .00057，FD2 .18052（−0.45%）；xyz_mse +4.8%；滑行帧比 .203 → .030。
- 关节级交互图 G1/G2 不采纳；G1 的单 seed 精度副发现补 seed1/2 后未复现。
