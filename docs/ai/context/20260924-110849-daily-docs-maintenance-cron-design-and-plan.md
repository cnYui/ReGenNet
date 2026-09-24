# 每日 docs/ai/context 清理 + AGENTS.md 压缩定时任务：设计与计划

## 一、背景与目标

用户在 Windows 本机为其它仓库（github-pr-automation、ai-builder-lab-miniprogram、sub2api）配置了 Claude desktop routine：每天在独立 worktree 里先清理 `docs/ai/context/` 超期文档（保留 15 天、带引用/标记护栏），再压缩 AGENTS.md 并完整归档移除内容，合成一个 PR 合并进 main，最后快进同步本地 main。本次要求给 ReGenNet（远程 Linux 机 `/home/rpartx3080/CodeSpace/ReGenNet`）也配一个同样的定时任务。

## 二、关键约束与发现

1. **desktop routine 不能在这台 SSH 远程机上运行**：从本会话（SSH 远程会话）用 scheduled-tasks 建了一个只读探测 routine 并"立即运行"，3 分钟内没有启动任何会话；已有的所有 routine 也都在 Windows 路径下运行。探测任务已删除。
   → 改用本机 **crontab**，调用本机已登录的 `claude -p`（claude.ai 登录态，与 `scripts/ship_pr.py` 的审查一致，不用 API key）。Windows 关机也不影响。
2. **现行规则禁止删除 context 历史文件**：`AGENTS.md` 稳定约束写"不覆写、重命名或删除历史文件"；`scripts/check_repo_conventions.py` 对 `docs/ai/context/` 下任何非新增改动报错；`ship_pr.py` 的审查提示把"违反稳定约束"定为 blocking。三处都需要调整，否则清理 PR 无法合并。
3. **ReGenNet 的入口是"引用链"式的**：AGENTS.md 常写"详见 X 及其引用文档"。只看直接引用（本机 routine 的做法）在今天会删 197/229 个文档，并切断这些引用链；按"引用闭包"保护只删 112 个（2026-09-24 预演，截止 20260909）。
4. **`ship_pr.py sync` 在 worktree 中会失败**：main 已在主工作区检出时，git 拒绝 `fetch fork main:main` 覆盖它，也拒绝 `switch main` 和删除别的工作树已检出的分支。定时任务在独立 worktree 里调用 `ship_pr.py`，合并后必然在 sync 报错。

## 三、设计

### 1. `scripts/prune_ai_context.py`（清理，确定性）

- 只处理 git 跟踪的、直接位于 `docs/ai/context/` 的文件；日期取文件名 `YYYYMMDD` 前缀（checkout 会重置 mtime，不能用 mtime）。
- 保留（任一命中）：
  1. 日期 ≥ 今天 − 15 天，或文件名没有日期前缀；
  2. 正文含 `<!-- prune:keep -->`；
  3. 文件名出现在 context 目录之外的任一 git 跟踪文本文件中（AGENTS.md、README、`docs/ai/presentation/`、脚本等）；
  4. 文件名出现在已保留的 context 文档中（闭包传递）。
- 引用判定用"文件名子串"匹配，与 CI 的 `git grep -F` 口径一致；附件（png/svg/mmd）同样适用。
- 其余删除；`--dry-run` 只打印。输出截止日期、删除清单、保留原因统计。
- 被删文件都在 git 历史中，可用 `git log --diff-filter=D --name-only -- docs/ai/context/` 找回。

### 2. `scripts/check_repo_conventions.py`（CI 约定放宽但加护栏）

- context 历史文件仍然**不得修改**；**允许删除**，但删除后 head 树里不得再有任何文件引用被删文件名（悬空引用即报错）。
- context 部分改用 `--no-renames` 比较：改名视为"删除 + 新增"，分别受上述两条规则约束，避免清理与新归档被相似度误判为改名。
- 这样清理脚本删的一定能通过（闭包保证无悬空引用），手工误删仍在用的文档会被拦住。

### 3. `scripts/ship_pr.py sync`（worktree 兼容）

- 用 `git worktree list --porcelain` 找出各分支检出位置。
- main 在别的工作树检出时：那个工作树跟踪文件干净就在那里 `merge --ff-only`，否则提示跳过；不再 `fetch main:main`、不 `switch`。
- 删除上游已消失的分支时，跳过在别的工作树检出的分支。

### 4. `scripts/daily_docs_maintenance.py`（编排，确定性 + 一次 Claude 判断）

与本机 routine 相同的流程，但把确定性步骤写成脚本，只把"压缩 AGENTS.md"交给 Claude：

1. 前置检查：`gh auth status`、`git ls-remote --exit-code <main 上游 remote> main`、`claude` 可用；任一失败直接退出。
2. `fetch` 后基于 `<remote>/main` 建临时 worktree 与分支 `chore/docs-maintenance-YYYYMMDD`（重名加 `-2`…）。
3. 先清理：worktree 内运行 `scripts/prune_ai_context.py`（必须在压缩前，才能按 AGENTS.md 完整引用集保护）。
4. 再压缩：`claude -p` 在 worktree 内执行压缩提示，`--permission-mode dontAsk` + 白名单（Read/Edit/Write/Glob/Grep + 只读 git 与 wc/date），`--json-schema` 返回 `changed/removed_count/archive/summary`。
5. 校验改动只在 `AGENTS.md` 与 `docs/ai/context/`，且 CLAUDE.md 软链接未被动；否则中止不提交。无改动就结束。
6. 提交后在 worktree 内调用 `python3 scripts/ship_pr.py --title … --body …`：本机审查 → 推送 → 建 PR → CI 约定检查 → 自动合并 → sync 主工作区 main。
7. `finally` 中移除 worktree、`worktree prune`、删除本地临时分支；打印最终报告。`--dry-run` 只做 1–5 并打印 diff 统计，不提交不推送。

压缩提示的判据（ReGenNet 定制，宁可少删）：

- 只能改 `## 当前研究入口` 与 `## 文档入口`；`# 标题`、引用块、`## 稳定约束`、`## 解释边界` 原样保留。
- 可移除/精简：已被后续条目明确取代的中间状态与"下一步"计划；重复表述；引用文档中已完整保存的过程细节（数值表、逐步经过）——精简时保留结论与文档路径。
- 必须保留：现行协议、指标口径、gate 定义、当前最佳 checkpoint 及关键数值、训练预算等仍生效的决策；负结果与坑；未完成事项；用户确认过的参数；每条保留条目至少一个 context 文档引用（它决定了该文档受清理保护）。
- 移除内容逐条完整照抄进新的 `docs/ai/context/<YYYYMMDD-HHMMSS>-agents-md-compression.md`（含前后行数/字节、命中判据、并入去向、刻意保留的内容）。无明确可移除内容就不改。

### 5. 调度

- crontab：每天 03:30（JST），`flock -n` 防重入，日志按月追加到 `~/.local/state/regennet-docs-maintenance/YYYYMM.log`。
- crontab 里显式设置 PATH（`~/.local/bin`、nvm node bin），cron 默认 PATH 找不到 `claude`/`gh`。
- 避开用户其它 routine（00:00、01:00、02:30、06:00、06:25）。

### 6. AGENTS.md

- 稳定约束：context 历史文件"不修改、不重命名"；删除只由每日清理任务按保留期与引用闭包执行，要长期保留的文档需被入口引用或加 `<!-- prune:keep -->`。
- 文档入口：加本组设计/结果文档。

## 四、取舍

- **cron 而非 desktop routine**：desktop routine 在 SSH 远程机上跑不起来；cron 不依赖 Windows 开机，但运行记录不在 Routines 面板里，只在日志文件和 GitHub PR 上。
- **脚本编排而非整段提示**：本机 routine 把全流程写成提示、以 bypass 权限运行。这里把确定性步骤写成脚本（保证 worktree 必清理、只提交两个区域），Claude 只在 `dontAsk` 白名单内做压缩判断，权限面更小。
- **引用闭包而非直接引用**：删得少（112 vs 197），但不会切断"及其引用文档"式的链；随着压缩把旧条目移出入口，对应文档会在归档过期（15 天）后自然进入清理。
- **不固定模型**：与 `ship_pr.py` 一致，使用本机 Claude Code 默认模型设置。

## 五、验证计划

1. `prune_ai_context.py`：真实仓库 `--dry-run` 与预演数字一致；临时仓库构造"保留期内/标记/外部引用/闭包引用/附件/无日期"用例。
2. `check_repo_conventions.py`：临时仓库用例——删除未被引用的文档通过；删除仍被引用的文档拦截；修改拦截；新增命名不合规拦截；删除 + 相似内容新增不误判。
3. `ship_pr.py sync`：临时仓库 + 裸远端模拟"main 在主工作区检出、在 worktree 中 sync"。
4. 编排脚本：在 cron 同等的精简环境（`env -i`）下 `--dry-run` 全流程，确认 claude/gh/git 凭据可用且 worktree 被清理。
5. ruff（py37 规则）+ Python 3.7 `py_compile`。
6. 本改动经 `python3 scripts/ship_pr.py` 合并后再安装 crontab。
