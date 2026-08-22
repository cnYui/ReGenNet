# 同步到 cnYui GitHub 仓库计划

## 目标

将当前工作区已有的 NTU 双人 diffusion 实现、对应实验文档、`AGENTS.md` 压缩归档以及可视化适配修改，完整提交并推送到 `cnYui/ReGenNet`。

## 范围

- 当前分支：`feature/ntu2p-diffusion-o10-p50`
- 目标远程：`fork`（`https://github.com/cnYui/ReGenNet.git`）
- 目标远程分支：同名 `feature/ntu2p-diffusion-o10-p50`
- 纳入当前工作区所有已跟踪修改和新增源代码/文档文件。
- 不修改远程 `main`，不强制推送，不删除远程分支。

## 验证

- 提交前执行所有变更 Python 文件的 `py_compile`。
- 执行 `git diff --check`。
- 推送后读取远程分支 commit，确认与本地提交一致。

## 取舍

- 当前工作区的代码与文档属于同一 NTU 双人 diffusion 实验链路，不能只推送 `AGENTS.md`，否则远程分支不可复现。
- 保留现有 `origin`（`liangxuy/ReGenNet`）不变，只使用用户指定的 `cnYui` 账号远程 `fork`。
