# CLAUDE.md 入口建立、摆动恢复工作 PR 合并与本地同步记录

本文记录 20260902 会话中研究工作之外的工程与仓库操作，研究内容本身见文末索引。

## 一、Claude Code 入口文件

- 事实：Claude Code 只自动读取 `CLAUDE.md`（与 `CLAUDE.local.md`），不读取 `AGENTS.md`。
- 决策（用户指定）：新建 `CLAUDE.md` 并**复制** `AGENTS.md` 全文，而非 `@AGENTS.md` 引用或软链接。
- 由此产生的约定：**两份文件内容必须保持一致，修改其一必须同步另一个**。本会话内每次更新 `AGENTS.md` 的"当前研究入口"时均同步写入 `CLAUDE.md`。已将该约定写入两份文件的"稳定约束"。

## 二、会话中断与续跑

- 19:24 启动的 Stage 3 驱动随 Claude Code 进程退出被终止（seed 1 跑到 1392 step）。残缺目录改名为 `save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s3_s2_5_dct1_root01_s1_5000_interrupted_192725`，旧日志改名为 `driver_interrupted_192725.log`，均未删除；19:29 用同一命令重启，驱动可续跑，Stage 3 于 20:58 完成。
- 教训：训练子进程 stdout 为块缓冲，驱动日志看不到实时 step，进度应读各 run 的 `train_log.jsonl`。

## 三、代码提交、PR 与合并

远端布局：`origin`=liangxuy/ReGenNet（原作者，不提交）、`fork`=cnYui/ReGenNet（用户仓库，gh 以 cnYui 登录）、`human`、`backup`。

| 步骤 | 内容 |
|---|---|
| 分支 | `feature/ntu2p-articulation-recovery`，自本地 `main`（`b987371`）切出 |
| 提交 | `fa7e9d6` "Recover limb articulation in NTU2P residual refiner via DCT amplitude losses"：22 个文件，+1775 / −45 |
| 提交内容 | 代码：`utils/ntu_smplx_2p_xyz.py`、`model/forecasting_ntu2p_residual_xyz.py`、`train/train_ntu2p_residual_refiner_xyz.py`、`eval/eval_ntu2p_residual_refiner_xyz.py`、`sample/export_ntu2p_residual_refiner_xyz_visualization.py`、`eval/analyze_ntu2p_residual_refiner_articulation.py`、`scripts/run_ntu2p_articulation_stage{1,2,3}.py`、`scripts/plot_ntu2p_joint_trajectories.py`；文档：本日 10 份 `docs/ai/context/20260902-*` 研究记录、`AGENTS.md`、`CLAUDE.md` |
| 刻意未提交 | `docs/ai/presentation/`（含 `node_modules`）、`docs/ai/context/20260827-184636-two-person-motion-prediction-presentation-plan.md`、`20260902-142122/142149-claude-code-cli-install-*.md`（非本会话产物，由用户决定） |
| 提交前检查 | `py_compile` 全部通过；`git diff --cached --check` 通过；暂存内容密钥扫描无命中（"token" 命中均为 `24-token memory`、`token_type` 等代码词） |
| 推送 | `fork/feature/ntu2p-articulation-recovery` |
| PR | https://github.com/cnYui/ReGenNet/pull/1 ，base `main`，共 4 个提交：本次 `fa7e9d6` + 此前只推过 feature 分支、未进入 `fork/main` 的 3 个 residual refiner 提交（`1464ecc`…`b987371`）。未直接推送 `main`，全部经 PR 合并 |
| 合并 | GitHub merge commit `c77d6d5`，2026-09-02T13:04:47Z |
| 本地同步 | `git pull --ff-only fork main`，本地 `main` = `fork/main` = `c77d6d5`，跟踪文件工作区干净 |
| 分支保留 | 本地与远端 feature 分支均未删除 |

## 四、上游跟踪修正

- 发现：本地 `main` 原跟踪 `human/main`（cnYui/human_forecasting），与实际同步目标 `fork` 不一致，`git pull` 会拉错仓库。
- 操作（用户确认）：`git branch -u fork/main main`；现状态 `main...fork/main`，无领先/落后。`human` 远端保留，仅不再作为 `main` 上游。

## 五、本日研究记录索引（按时间）

1. `20260902-143842-ntu2p-residual-refiner-regression-to-mean-diagnosis.md` —— 代码走查诊断与验证计划
2. `20260902-144108-ntu2p-residual-refiner-regression-to-mean-quantitative-result.md` —— 帧差能量/root 能量/frozen 定量确认
3. `20260902-145142-ntu2p-articulation-recovery-design.md` —— 调整设计（评估 gate、损失、结构、step、Track B 协议）
4. `20260902-145142-ntu2p-articulation-recovery-training-plan.md` —— 分阶段训练计划
5. `20260902-150402-ntu2p-articulation-recovery-stage0-result.md` —— 工程准备与等价性测试
6. `20260902-173335-ntu2p-articulation-recovery-stage1-result.md` —— 7 个单因子全部无效；梯度消失与 GT 抖动两个诊断
7. `20260902-173335-ntu2p-articulation-recovery-stage2-design-and-plan.md` —— 指标改 DCT 分频带、损失改幅度型
8. `20260902-192249-ntu2p-articulation-recovery-stage2-result.md` —— 问题解决；鞍点机制修正
9. `20260902-205844-ntu2p-articulation-recovery-stage3-result.md` —— 3 seed 复现、20000 step、horizon 分解
10. `20260902-212338-ntu2p-training-step-budget-design.md` —— 步数预算：筛选 5000 / 最终 10000 × 3 seed
11. 本文 —— 入口文件、中断续跑、PR/合并/同步、跟踪修正

## 六、当前状态与待办

- 当前主线最佳 checkpoint：`save/forecasting/ntu120_label/ntu2p_residual_refiner_xyz_artic_s3_s2_5_dct1_root01_s0_20000/model000020000.pt`（权重在忽略目录，未入库）。
- 本文及配套的 `AGENTS.md` / `CLAUDE.md` 约定更新尚未提交，待用户决定直接提交 `main` 或再走 PR。
- 已确认的下一次训练：s2_5 配置 3 seed × 10000 step；presentation 前补 test split 单次评估与独立验证。
