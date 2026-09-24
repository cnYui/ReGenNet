# CLAUDE.md 软链接与 PR 本地审查 + 自动合并流水线：实现与本地验证结果

设计与取舍见 `20260924-104641-claude-md-symlink-and-pr-auto-merge-design-and-plan.md`。本文记录实现与合并前的本地验证；端到端（真实 PR）结果另起文档记录。

## 一、改动

| 文件 | 内容 |
|---|---|
| `CLAUDE.md` | 核对与 `AGENTS.md` 逐字节一致（`cmp` 通过）后删除，改为软链接 `CLAUDE.md -> AGENTS.md`（相对路径，git mode 120000） |
| `AGENTS.md` | 顶部加"只修改 `AGENTS.md`"提示；"两份保持一致"规则改为软链接规则；提交流程改为 `python3 scripts/ship_pr.py`；文档入口加本组记录 |
| `scripts/check_repo_conventions.py` | 约定检查：CLAUDE.md 软链接、`docs/ai/context/` 只增 + `.md` 命名、改动 py 文件的 ruff 新增问题（py37，E9/F63/F7/F82，merge-base 与 head 做差） |
| `scripts/ship_pr.py` | `ship`（默认）/ `review` / `sync`；审查用本机 `claude -p`（claude.ai 登录态，无 API key） |
| `.github/workflows/pr-auto-merge.yml` | `约定检查` job → `自动合并` job（轮询 `claude-review` status，`gh pr merge --merge --delete-branch --match-head-commit`） |

## 二、本机审查的调用方式

```
claude -p <审查提示> --output-format json --json-schema <verdict/summary/findings 模式>
  --tools Read,Grep,Glob,Bash --allowedTools Read,Grep,Glob,"Bash(git diff:*)","Bash(git log:*)","Bash(git show:*)"
  --permission-mode dontAsk --strict-mcp-config --no-session-persistence
```

- `dontAsk` + 白名单：审查过程只能读代码和 git 历史，不能改工作区；不加载 MCP，减少耗时与外部依赖。
- 连通性实测：`loggedIn=true, authMethod=claude.ai`，`structured_output` 按模式返回。
- 放行规则 `decide()`：`verdict=block` 或任一 finding 为 `blocking` 即阻断，两者不一致时从严。
- 原始结论存 `.git/claude-review/<head>-<merge-base 前 12 位>.json`（不入库），head 与 merge-base 都不变时重跑直接复用，不重复审查；同时以 PR 评论形式留痕。

## 三、本地验证

| 验证 | 结果 |
|---|---|
| actionlint 1.7.12 检查 workflow | 通过 |
| ruff 0.16.8（py37，E9/F/B）检查两个新脚本 | 通过（修正 2 处 B904 异常链） |
| 项目环境 Python 3.7.13 `py_compile` 两个脚本 | 通过；`--help` 可运行 |
| 约定检查 12 个临时仓库用例 | 12/12 符合预期：正常改动通过（含遗留文件行号下移、新增 png 附件）；软链接被替换/改目标/删除、context 修改/删除/重命名、context 新文档命名不合规、新增 walrus、遗留文件新增 F821 均拦截；遗留文件改名、删除 py 文件不误报；`--skip-python` 与找不到 ruff 的退出码正确 |
| `sync` 3 个场景（本地裸仓库模拟 fork + 模拟 CI merge commit 合并并删远端分支） | 全部符合预期：脏工作区时只快进 `main` 引用、不切分支、不删分支；干净时切回 `main`、删除已合并分支、保留远端已删但未合并的分支；在 `main` 上时 `--ff-only` 快进 |
| 纯函数断言 | 远端 URL 解析（https/ssh/无 .git）、`decide` 从严、`parse_gone_branches`、评论渲染（`|` 转义）均通过 |
| 真实仓库 `sync` 影响预演 | 现有 5 个本地分支没有上游消失的，`sync` 不会删除任何已有分支 |

## 四、本机审查首次实测与修正

对提交 `abd58e7` 运行 `python3 scripts/ship_pr.py review`：结论"通过"，给出 1 个 major、2 个 minor，经核实全部成立并处理：

| 级别 | 问题 | 处理 |
|---|---|---|
| major | `wait_for_merge` 在"全部检查 pass/skipping"时判定未合并；但合并 job 依赖约定检查，其 check 要稍后才出现，间隙内会误报失败并跳过 sync | 改为以合并 job（`自动合并`，常量 `MERGE_JOB_NAME`）自身结束为准 |
| minor | 审查缓存只以 head SHA 为 key，`fork/main` 前进后审查范围已变却会复用旧结论 | key 改为 head + merge-base |
| minor | 设计文档第 7 条写 `git branch -d`，实现为先 `merge-base --is-ancestor` 确认已并入 main 再 `git branch -D`（避免 `-d` 按当前 HEAD 判断而误拒） | 以实现为准，在此注明；设计文档作为历史记录不改 |

## 五、基线与已知限制

- 全仓 ruff 基线 13 处 F821，均在上游遗留文件（`actor-x/src/evaluate/tables/easy_table_A2M.py`、`actor-x/src/models/modeltype/kgan.py`、`data_loaders/humanml/motion_loaders/model_motion_loaders.py`、`model/transformer_utils.py`）；按设计只拦截新增问题，未修改这些文件。
- 本地未安装 ruff，`ship_pr.py` 在本地跳过 Python 检查，由 CI 兜底。
- 未经 `ship_pr.py` 的推送不会自动合并（CI 等 5 分钟后以明确信息失败），这是有意的门槛。
- 推送 `.github/workflows/*` 需要 gh token 带 `workflow` scope：`gh auth refresh -h github.com -s workflow --insecure-storage`（保持明文存储，兼容 git 使用的旧版 gh 2.4.0 凭据助手）。
