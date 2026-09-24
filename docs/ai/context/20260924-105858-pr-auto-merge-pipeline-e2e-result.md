# PR 本地审查 + 自动合并流水线：首次端到端验证结果

设计见 `20260924-104641-claude-md-symlink-and-pr-auto-merge-design-and-plan.md`，实现与本地验证见 `20260924-105223-claude-md-symlink-and-pr-auto-merge-result.md`。

## 一、一次性准备

- 用户执行 `gh auth refresh -h github.com -s workflow --insecure-storage`：token scopes 变为 `gist, read:org, repo, workflow`，仍存于 `~/.config/gh/hosts.yml`，git 使用的旧版 gh 2.4.0 凭据助手可正常推送 `.github/workflows/*`。

## 二、端到端过程（PR https://github.com/cnYui/ReGenNet/pull/2 ）

命令：在 `feature/claude-md-symlink-pr-auto-merge` 上运行 `python3 scripts/ship_pr.py --title ... --body ...`。

| 环节 | 结果 |
|---|---|
| 本机审查 | 复用 `.git/claude-review/a062f12…-c77d6d58271c.json`（此前对同一 head + merge-base 已审查，结论通过） |
| 推送 | 成功推送新分支（`workflow` scope 生效） |
| status | head `a062f12` 上 `claude-review=success` |
| 建 PR | #2，2026-09-24T01:56:12Z |
| PR 评论 | cnYui 发布审查摘要与问题表 |
| CI `约定检查` | pass，01:56:20 → 01:56:26；日志确认 pipx 安装 ruff 0.16.8 并执行（未跳过 Python 检查），输出"✓ 仓库约定检查通过" |
| CI `自动合并` | pass，01:56:29 → 01:56:36；首轮即读到 `claude-review: success` |
| 合并 | `mergedBy=app/github-actions`，01:56:33，merge commit `69b3355`，从建 PR 到合并 21 秒 |
| 远端分支 | 已删除（API 返回 Branch not found） |
| 本地同步 | `fetch --prune` 删除 `fork/feature/...` 跟踪引用 → `main` 快进 `c77d6d5..69b3355` → 切回 `main` → 删除本地 feature 分支；`main...fork/main` 一致 |
| CLAUDE.md | `main` 上为 `120000` 软链接 `CLAUDE.md -> AGENTS.md` |

本记录本身通过同一流程提交（第二次使用，起点为合并后的新 `main`）。

## 三、本机审查三轮的产出

本机审查在合并前共跑三轮，前两轮各发现 1 个经核实成立的 major（`wait_for_merge` 误判竞态、重跑后立即轮询误判），均已修复，详见实现结果文档第四节。第三轮（`a062f12`）无 blocking/major，两个 minor 经核实**暂不修改**：

| minor | 核实结论 |
|---|---|
| ruff JSON 的 `code` 可能为 null，`sorted` 时与 str 比较抛 TypeError | 当前固定 ruff 0.16.8，实测语法错误输出 `code="invalid-syntax"`（字符串），不会触发；升级 ruff 版本时需复查 |
| 以临时目录为参数调用 ruff，`build/`、`dist/`、`venv/`、`node_modules/` 等默认排除目录下的 py 改动会被静默跳过 | 本仓库代码不在这些目录；如将来有需要，改为逐个传文件并加 `--force-exclude` 相关设置 |

处理原则：blocking 必须修复后才推送；major/minor 逐条核实，成立的 major 当轮修复，minor 记录后择机处理，避免"改一次、审一次"无限循环。

## 四、日常用法

```bash
# 在 feature 分支提交后
python3 scripts/ship_pr.py            # 默认 --fill；可透传 gh pr create 参数，如 --title/--body/--draft
python3 scripts/ship_pr.py review     # 只审查
python3 scripts/ship_pr.py sync       # 只同步本地 main 并清理已合并分支
```

- 草稿 PR 只跑检查不合并，转为 ready 后触发合并。
- 不经脚本推送的提交不会自动合并（CI 等待 `claude-review` 5 分钟后失败）；之后在该分支运行脚本会补审、补打 status 并重跑失败的 run。
