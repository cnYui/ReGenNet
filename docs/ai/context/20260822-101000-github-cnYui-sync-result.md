# 同步到 cnYui GitHub 仓库结果

## 已完成

- 本地分支 `feature/ntu2p-diffusion-o10-p50` 已提交 NTU 双人 diffusion 实现、实验记录、可视化适配和 `AGENTS.md` 压缩归档。
- 已推送到 `fork`：`https://github.com/cnYui/ReGenNet.git`。
- 远程新建并跟踪分支：`feature/ntu2p-diffusion-o10-p50`。

## 提交

```text
7aaa88f Add NTU two-person diffusion forecasting pipeline
```

## 校验

- 使用项目 `regennet` 环境对新增和修改 Python 文件执行 `py_compile`，通过。
- `git diff --check` 通过。
- 推送返回成功，并提供该分支的 GitHub Pull Request 创建链接。

## 边界

- 本次只新建 feature 分支，未修改、合并或强制推送远程 `main`。
