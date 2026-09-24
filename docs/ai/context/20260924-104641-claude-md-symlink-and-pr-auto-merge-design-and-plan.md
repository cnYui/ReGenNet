# CLAUDE.md 软链接与 PR 本地审查 + 自动合并流水线：设计与计划

## 一、背景与目标

用户要求（20260924）：

1. 清空 `CLAUDE.md`，改为指向 `AGENTS.md` 的软链接，只维护 `AGENTS.md`；并在入口中写明"之后只改 `AGENTS.md`"。
2. 写 GitHub CI：提交 PR 后自动审查代码 → 合并到 `main` → 删除已合并的旧分支 → 自动同步本地 `main`。
3. 审查**不用 API key**，直接用本机已登录的 Claude Code（与 desktop 同一账号、同一订阅额度）。
4. 本地同步方式：用户选"提交脚本等待并同步"；完成后按流程提交，并用这次 PR 做端到端验证。

## 二、现状（已核实）

| 项 | 状态 |
|---|---|
| `CLAUDE.md` / `AGENTS.md` | 两份普通文件，工作区逐字节一致；已提交版本仅"握手"引号 1 处不同 |
| 远端 | `fork`=cnYui/ReGenNet（公开，fork 自 liangxuy/ReGenNet），本地 `main` 上游 `fork/main` |
| GitHub 设置 | Actions 已启用；默认 workflow 权限 `read`；`main` 无分支保护；允许 merge commit；`delete_branch_on_merge=false`；无任何 secret |
| gh | 以 cnYui 登录，token scopes 为 `gist, read:org, repo`，**缺 `workflow`**（推送 `.github/workflows/*` 会被拒） |
| git 凭据 | 全局 `credential.https://github.com.helper` 指向仓库内旧版 `gh-local/usr/bin/gh`（2.4.0，只能读明文 `hosts.yml`） |
| Claude Code CLI | `claude` 2.1.280，`loggedIn=true, authMethod=claude.ai`；`claude -p --output-format json --json-schema` 已实测返回 `structured_output` |
| Python | 项目环境 `~/.local/micromamba/envs/regennet` 为 3.7.13；系统 `python3` 3.10.12（无 pip） |
| 静态检查基线 | ruff 0.16.8（`--target-version py37 --select E9,F63,F7,F82`）全仓 265 个 py 文件共 13 处 F821，集中在 4 个上游遗留文件 |

## 三、决策与取舍

1. **软链接而非 `@AGENTS.md` 导入**：用户明确指定。官方文档（code.claude.com/docs/en/memory）确认 Claude Code 通过软链接读取 `CLAUDE.md`；Edit/Write 工具拒绝穿过软链接写入并提示改目标文件，天然契合"只改 `AGENTS.md`"。唯一限制是 Windows 克隆会变成一行文本，本项目只在 Linux 使用，可接受。由于软链接没有独立内容，"在 CLAUDE.md 中加入提示"落实为写在 `AGENTS.md` 顶部，打开任一文件都先看到。
2. **审查放在本机而非 GitHub 云端**：云端 runner 无法使用 desktop 的登录态，除非配置 `CLAUDE_CODE_OAUTH_TOKEN`/`ANTHROPIC_API_KEY` secret，用户不希望如此。因此本地脚本调用 `claude -p` 做审查，把结论写回 GitHub：
   - commit status `claude-review=success`（打在被审查的 head SHA 上）作为 CI 合并的门槛；
   - PR 评论贴出审查摘要与问题列表，留痕。
   审查不通过（存在 blocking 问题）时脚本不推送、不建 PR。
3. **CI 只做确定性检查 + 合并**：免费、可复现；语义审查由第 2 条负责。CI 检查内容：
   - `CLAUDE.md` 在 head 提交中必须是指向 `AGENTS.md` 的软链接（防止被工具替换成普通文件后两份内容再次分叉）；
   - `docs/ai/context/` 只允许新增，不允许修改、删除、重命名；新增 `.md` 必须符合 `YYYYMMDD-HHMMSS-名称.md`（落实 AGENTS.md 稳定约束；非 md 附件如 png/svg 只要求只增）；
   - 改动的 py 文件做 ruff 检查（目标 Python 3.7：语法错误、3.8+ 语法、未定义名），**只拦截本 PR 新引入的问题**：同时检查 merge-base 与 head 两个版本并做差，避免因 4 个上游遗留文件的历史问题阻塞无关 PR。
4. **合并方式**：`gh pr merge --merge --delete-branch --match-head-commit <sha>`。merge commit 与已有历史一致，且保证本地 `git branch -d` 能判定已合并；`--match-head-commit` 防止检查后又被推入新提交；带 `--repo` 时 `--delete-branch` 只删远端分支。
5. **安全边界**：仅对"同仓库分支 + 作者为仓库所有者 + 非草稿"的 PR 自动合并；外部 fork PR 只跑检查（其 token 只读，也拿不到合并权限）。不用 `pull_request_target`。草稿 PR 作为"暂不合并"的逃生口。
6. **状态竞态**：新建 PR 时脚本先打 status 再建 PR，无竞态；向已有 PR 推送新提交时 push 先触发 CI，status 随后才打上，因此 CI 合并 job 轮询 status 最多 5 分钟。未经脚本审查的推送会在超时后以明确信息失败；之后再跑脚本会补打 status 并重跑失败的 run。
7. **本地同步**：GitHub 云端无法修改本机仓库；自托管 runner 在公开仓库上有安全风险，已否决。脚本等待 PR 合并后：`fetch --prune` → 本地 `main` 快进到 `fork/main` → 删除上游已消失且已并入 `main` 的本地分支（`git branch -d`，未合并的拒删）。工作区有未提交的跟踪文件修改时不切分支，只快进 `main` 引用。
8. **实现语言**：两个脚本均用 Python 标准库（与 `scripts/` 现有风格一致，JSON 处理与纯函数便于测试），语法兼容 3.7，系统 `python3` 与项目环境都能运行。

## 四、设计

### 4.1 文件

| 文件 | 作用 |
|---|---|
| `CLAUDE.md` | 软链接 → `AGENTS.md` |
| `AGENTS.md` | 顶部加提示；替换"两份保持一致"规则为软链接规则；提交流程改为脚本流程；文档入口加本记录 |
| `scripts/check_repo_conventions.py` | 约定检查（软链接 / context 只增 / ruff 新增问题），CI 与本地共用 |
| `scripts/ship_pr.py` | `ship`（默认）/ `review` / `sync` 三个子命令 |
| `.github/workflows/pr-auto-merge.yml` | `checks` → `merge` 两个 job |

### 4.2 `scripts/ship_pr.py ship` 流程

1. 当前分支不能是 `main`；跟踪文件无未提交修改（保证审查内容 = 推送内容）。
2. 从 `main` 的上游解析远端名（`fork`）与仓库 slug（`cnYui/ReGenNet`）。
3. 本地先跑约定检查（软链接、context 只增），不过则退出。
4. `claude -p` 审查 `git diff <remote>/main...HEAD`：只读工具（Read/Grep/Glob、`git diff/log/show`），`--json-schema` 约束输出 `{verdict, summary, findings[]}`，`verdict=block` 当且仅当存在 blocking 问题。结果存 `.git/claude-review/<sha>.json`。
5. 通过后 `git push -u`，给 head SHA 打 `claude-review` status，建 PR（已有则复用；可透传 `gh pr create` 参数，默认 `--fill`），发审查评论。
6. 若是已有 PR 且该 SHA 的 CI run 已失败（例如先前手动推送过），重跑该 run。
7. 轮询 PR 状态与 checks：MERGED → 进入 sync；CLOSED、检查失败、或全部结束仍未合并 → 报错退出。
8. `sync`。

### 4.3 workflow

- 触发：`pull_request` 到 `main`，`opened/synchronize/reopened/ready_for_review`；同一 PR 并发取消旧 run。
- `checks`（`contents: read`）：checkout（`fetch-depth: 0`）→ pipx 安装固定版本 ruff → 运行 `check_repo_conventions.py --base <base.sha> --head <head.sha>`。
- `merge`（`contents: write, pull-requests: write, statuses: read`，带安全条件）：轮询 `claude-review` status → `gh pr merge`。

## 五、实施与验证计划

1. 核对两文件一致后建立软链接，更新 `AGENTS.md`。
2. 编写两个脚本与 workflow。
3. 本地验证：
   - actionlint 检查 workflow；ruff（py37）检查新脚本；
   - 在临时 git 仓库构造用例测试约定检查：软链接被替换、context 历史文件被修改/删除/改名、新文档命名不合规、新增 walrus 语法、遗留问题不误报；
   - 在临时仓库 + 本地裸仓库模拟远端，测试 `sync`：快进、删除已合并分支、保留未合并分支、脏工作区不切分支。
4. 一次性手动步骤（用户）：`gh auth refresh -h github.com -s workflow --insecure-storage`。加 `--insecure-storage` 是为了让 token 继续留在明文 `hosts.yml`：git 的凭据助手是旧版 gh 2.4.0，读不到系统密钥环。
5. 端到端：新建 feature 分支提交 → `python3 scripts/ship_pr.py` → 观察本地审查、status、CI 检查、自动合并、远端分支删除、本地 main 同步与本地分支删除。
6. 结果写入 `docs/ai/context/<时间戳>-claude-md-symlink-and-pr-auto-merge-result.md`，并更新 `AGENTS.md` 入口。

## 六、风险

- 本地审查消耗订阅额度，大 PR 耗时较长；脚本设总超时。
- AI 审查可能误报 blocking：此时脚本停止，由用户判断后修改，或手动在 GitHub 合并（CI 不强制，只是不自动合并）。
- `ubuntu-latest`、`actions/checkout@v7` 等浮动版本未来可能变化；ruff 固定版本以保证可复现。
