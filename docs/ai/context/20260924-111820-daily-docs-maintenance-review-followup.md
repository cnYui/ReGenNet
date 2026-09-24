# 每日文档维护：PR #5 本机审查意见的跟进

PR #5（`20260924-110849-daily-docs-maintenance-cron-design-and-plan.md`、`20260924-111530-daily-docs-maintenance-cron-result.md`）已合并。其本机 Claude Code 审查结论为"通过"，另有 1 个 major、2 个 minor，经核实全部成立，本次一并处理。

| 级别 | 问题 | 处理 |
|---|---|---|
| major | CI 允许任意 PR 删除 context 文档，只要不留悬空引用，不校验保留期；AGENTS.md 的"删除只由每日任务按规则执行"只能靠自觉 | `check_repo_conventions.py`：被删文件必须有日期前缀且早于 `今天 − 14 天`，否则拦截。比清理脚本的 15 天放宽一天，因为 CI 按 UTC 取日期，本机（JST）03:30 运行时 UTC 仍是前一天。边界已测：JST 09-25 运行，清理截止 20260910；CI（UTC 09-24）截止同为 20260910，删除 20260909 放行、20260910 拦截。改名 = 删除 + 新增，所以保留期内的文档也不能改名 |
| minor | 清理脚本跳过 context 子目录文件（既不清理也不算引用来源），CI 悬空引用检查却扫全树，两者口径不一致 | `prune_ai_context.py`：context 子目录文件与其它仓库文件一样算作引用来源（仍不参与清理）。子目录没有日期前缀，按新规则 CI 也不允许删除它们，两边一致 |
| minor | `--dry-run` 的说明没有写明清理会在临时 worktree 中真实执行 | 更新 docstring 与 `--help`：真实执行，但不提交、不推送，worktree 结束即丢弃 |

同时让 `check_repo_conventions.py` 从 `prune_ai_context.py` 导入 `CONTEXT_DIR`、`DATE_PREFIX`、`DEFAULT_DAYS`，避免两处各写一份。

验证：约定检查临时仓库用例 15/15（原 10 条 + 删除保留期内文档、删除无日期文件、改名保留期内文档、截止前一天、截止当天）；清理单元用例、子目录引用保护用例、`sync` 4 个场景、ruff（py37）均通过；真实仓库预演仍为删除 111 个（保留数变为 123，因为新增了 2 份当天文档）。

PR #5 合并后，`ship_pr.py` 在本 worktree 中运行 sync，把主工作区的 main 快进到 `5565036`，保留了当前分支，没有尝试 `switch`。这是 worktree 兼容 sync 在真实仓库上的第一次运行，结果正确。
