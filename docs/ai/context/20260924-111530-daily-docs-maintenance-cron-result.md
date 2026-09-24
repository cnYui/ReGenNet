# 每日 docs/ai/context 清理 + AGENTS.md 压缩定时任务：实现与验证结果

设计与取舍见 `20260924-110849-daily-docs-maintenance-cron-design-and-plan.md`。

## 一、为什么不用 desktop routine

从本 SSH 远程会话用 scheduled-tasks 建了只读探测 routine（`hostname; pwd; …`）并"立即运行"，3 分钟内没有启动任何会话，`list_task_runs` 为 0；用户已有的全部 routine 都跑在 Windows 路径下。结论：desktop routine 不能在这台远程机上执行。探测任务已删除（Windows 侧保留其 SKILL.md，无害）。改用本机 crontab + 本机已登录的 `claude -p`。

## 二、改动

| 文件 | 内容 |
|---|---|
| `scripts/prune_ai_context.py`（新） | 清理：文件名日期超过 15 天且无引用、无 `<!-- prune:keep -->` 的 context 文件删除；引用 = context 外任一 git 跟踪文件或已保留 context 文档中出现该文件名（闭包传递，按子串匹配）；`--dry-run`、`--today` |
| `scripts/daily_docs_maintenance.py`（新） | 编排：前置检查 → 临时 worktree（基于 `fork/main`）→ 清理 → `claude -p` 压缩（`dontAsk` + 白名单）→ 校验改动只在 AGENTS.md 与 context（context 只删、归档只新建一份）→ 提交 → `ship_pr.py` → `finally` 清理 worktree 与临时分支；`--dry-run` |
| `scripts/check_repo_conventions.py` | context 历史文件仍不得修改；允许删除，但 head 树中不得再有文件引用被删文件名（`git grep -F`）；context 部分按 `--no-renames` 比较，改名 = 删除 + 新增 |
| `scripts/ship_pr.py` | 抽出 `ask_claude()` 供审查与压缩共用；`sync` 支持 worktree：main 在别的工作树检出时到那里 `merge --ff-only`（该工作树有跟踪文件修改则跳过），不再 `fetch main:main` / `switch main`；跳过删除别的工作树已检出的分支 |
| `AGENTS.md` | 稳定约束改为"不修改、不重命名；删除只由每日维护任务执行"；文档入口加本组记录 |

## 三、验证

| 验证 | 结果 |
|---|---|
| ruff 0.16.8（py37，E9/F/B）检查 4 个脚本 | 通过 |
| 清理规则纯函数用例（保留期边界、标记、外部引用、两级闭包、附件、无日期、二进制） | 通过 |
| 改动范围校验 `unexpected_changes`（CLAUDE.md、修改旧 context、多余新文件、有改无归档、有归档无改动） | 通过 |
| 约定检查 10 个临时仓库用例 | 10/10：删未引用文档通过；删 2 个 + 新增归档通过；删仍被 AGENTS.md / 其它 context 文档引用的拦截；修改拦截；命名不合规拦截；相似内容删除 + 新增不误判为改名；改名仍被引用的文档拦截；CLAUDE.md 软链接被替换拦截；只改 AGENTS.md 通过 |
| `sync` 4 个场景（本地裸仓库模拟 fork + CI merge commit） | 全部符合预期：worktree 中 sync 快进主工作区 main、保留当前分支与别的工作树的分支；主工作区脏时不快进；移除 worktree 后在主工作区 sync 删除已合并分支；main 未被任何工作树检出时仍走 `fetch main:main` + `switch` |
| 旧版 `sync` 在 worktree 中复现（git 2.34.1） | `fetch fork main:main` **未被拒绝**，直接移动了主工作区检出的 main，使主工作区凭空出现反向改动；随后 `git switch main` 报 already checked out 退出 1。新版不再触碰别处检出的分支 |
| 真实仓库清理预演（`--dry-run --today 20260924`） | 截止 20260909，删除 111、保留 121（保留期内 3、外部引用 29、闭包间接引用 89）；只看直接引用会删 197 个 |
| cron 同等精简环境（`env -i`，只给 HOME/LANG/PATH）完整 `--dry-run` | exit 0：`gh`/`git ls-remote`/`claude -p`（claude.ai 登录态）均可用；清理删 111；压缩精简 2 条（residual-refinement 条目删去已完成的 `inter_loss_weight` 扫描"下一步"、Stage 2 条目删去已完成的 Stage 3 计划并把驱动/汇总路径并入 Stage 3 条目），49→49 行、13549→13464 字节，全部 context 路径保留；归档逐条照抄原文、给出精简后写法、并入去向和 9 类刻意保留内容；临时 worktree 与分支已清理 |

## 四、调度

本机 crontab（系统时区 Asia/Tokyo）：

```
# ReGenNet 每日 docs/ai/context 清理 + AGENTS.md 压缩（scripts/daily_docs_maintenance.py）
PATH=/home/rpartx3080/.local/bin:/home/rpartx3080/.nvm/versions/node/v22.12.0/bin:/usr/local/bin:/usr/bin:/bin
LANG=C.UTF-8
30 3 * * * mkdir -p $HOME/.local/state/regennet-docs-maintenance && cd /home/rpartx3080/CodeSpace/ReGenNet && flock -n /tmp/regennet-docs-maintenance.lock python3 scripts/daily_docs_maintenance.py >> $HOME/.local/state/regennet-docs-maintenance/$(date +\%Y\%m).log 2>&1
```

- 每天 03:30，避开用户 Windows 侧 routine（00:00、01:00、02:30、06:00、06:25）；`flock -n` 防止上一轮未结束时重入。
- cron 默认 PATH 找不到 nvm 下的 `claude` 与 `~/.local/bin/gh`，因此显式设置；升级 node 版本后需同步改这一行。
- 运行记录只在日志（按月追加）和 GitHub PR 上，不出现在 desktop 的 Routines 面板。
- 手动试跑：`cd /home/rpartx3080/CodeSpace/ReGenNet && python3 scripts/daily_docs_maintenance.py --dry-run`；停用：`crontab -e` 删除该行。

## 五、已知限制

- 首次正式运行会删除约 111 个 2026-09-09 之前的 context 文档（均可从 git 历史找回）；此后每天通常只删少量随归档过期而失去引用的文档。
- 压缩判断依赖模型；护栏是：只允许改 AGENTS.md + 新建一份归档（脚本事后校验）、ship_pr.py 的本机审查、CI 悬空引用检查。
- 依赖本机 claude.ai 登录态与 gh token；任一过期时当天任务在前置检查处失败并记入日志，不会产生半成品 PR。
