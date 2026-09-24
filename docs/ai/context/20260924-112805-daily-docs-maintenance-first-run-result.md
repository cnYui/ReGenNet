# 每日文档维护：首次正式运行结果（手动触发）

用户要求手动正式跑一次。触发方式与 crontab 完全相同：`env -i`，只给 HOME/LANG/PATH，用 `/bin/sh` 执行 crontab 那一行（`flock` + 按月日志 `~/.local/state/regennet-docs-maintenance/202609.log`）。设计与实现见 `20260924-110849-daily-docs-maintenance-cron-design-and-plan.md`、`20260924-111530-daily-docs-maintenance-cron-result.md`、`20260924-111820-daily-docs-maintenance-review-followup.md`。

## 结果

| 项 | 结果 |
|---|---|
| 起止 | 2026-09-24 11:25:57 开始，约 5 分钟结束，rc=0 |
| 清理 | 截止 20260909，删除 111 个、保留 124 个，与预演一致 |
| 压缩 | 精简 1 条（residual-refinement 条目中已完成的 `inter_loss_weight` 扫描"下一步"），50→50 行、14462→14418 字节，全部 context 路径保留；归档 `20260924-112557-agents-md-compression.md` |
| 本机审查 | 通过，1 个 minor：保留文档 `20260902-221812-...-pr-sync-result.md` 用通配写法 `20260902-142122/142149-claude-code-cli-install-*.md` 提到两份本次被删的文档。这不是完整文件名，按规则不构成引用，两份文档只能从 git 历史找回。接受，不改规则 |
| PR | https://github.com/cnYui/ReGenNet/pull/7 ，CI 约定检查（含保留期与悬空引用）通过后自动合并 |
| 同步 | 主工作区 main 快进到 `8797ed0`（跟踪文件无改动）；临时 worktree、`chore/docs-maintenance-20260924` 本地分支与远端分支均已删除 |
| context 目录 | 235 → 125 个文件（−111 + 1 份归档） |

## 发现并修正的问题

日志顺序错乱：cron 把输出重定向到文件时，Python 的 stdout 是块缓冲。脚本自己打印的"压缩结果""清理：/压缩："等行一直留在缓冲区，直到进程退出才写入日志，排到了 ship_pr.py 子进程输出之后。内容没有丢，只是顺序不对。修正：`daily_docs_maintenance.py` 的 `main()` 开头调用 `sys.stdout.reconfigure(line_buffering=True)`，已用父子进程交错输出到文件的用例验证顺序正确。
