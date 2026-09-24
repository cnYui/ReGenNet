#!/usr/bin/env python3
"""每日文档维护：先清理 docs/ai/context，再用本机 Claude Code 压缩 AGENTS.md，合成一个 PR 自动合并。

用法：python3 scripts/daily_docs_maintenance.py [--dry-run]
  --dry-run  只做清理与压缩并打印改动，不提交、不推送

流程：前置检查 → 基于 <remote>/main 建临时 worktree → prune_ai_context.py 清理 → claude -p 压缩 AGENTS.md
→ 校验改动只在 AGENTS.md 与 docs/ai/context/ → 提交 → ship_pr.py（本机审查、推送、CI 自动合并、同步本地 main）
→ 无论成败都移除 worktree 与临时分支。

必须先清理再压缩：清理要按压缩前 AGENTS.md 的完整引用集保护文档。
只在临时 worktree 中改动，主工作区可能有并行会话的未提交工作，只在合并后由 ship_pr.py sync 快进。
由 crontab 每天调用，设计见 docs/ai/context/20260924-110849-daily-docs-maintenance-cron-design-and-plan.md。
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from prune_ai_context import CONTEXT_DIR, DEFAULT_DAYS, prune
from ship_pr import MAIN, ShipError, ask_claude, main_remote, parse_github_slug, run, run_visible

REPO = Path(__file__).resolve().parent.parent
BRANCH_PREFIX = "chore/docs-maintenance-"
SLIM_TIMEOUT_S = 30 * 60
# 压缩只需读仓库、改 AGENTS.md、写一份归档；改动范围由 unexpected_changes 事后兜底
SLIM_TOOLS = ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
SLIM_ALLOWED = ["Read", "Edit", "Write", "Glob", "Grep", "Bash(git diff:*)", "Bash(git log:*)",
                "Bash(git show:*)", "Bash(wc:*)", "Bash(date:*)"]
SLIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["changed", "removed", "archive", "summary"],
    "properties": {
        "changed": {"type": "boolean"},
        "removed": {"type": "integer"},
        "archive": {"type": "string"},
        "summary": {"type": "string"},
    },
}
SLIM_PROMPT = """你在 ReGenNet 仓库的临时 worktree 中做每日 AGENTS.md 压缩（瘦身）。这是例行维护：不要另写 design/plan/结果文档，\
docs/ai/context 的超期文档已由脚本清理完毕，不要删除或修改任何已有文件。

## 只允许的改动
1. 编辑 `AGENTS.md`（`CLAUDE.md` 是指向它的软链接，不要碰）；
2. 仅当 AGENTS.md 确有改动时，新建且只新建一个归档文件 `{archive}`。
压缩前 AGENTS.md 为 {lines} 行、{size} 字节。

## 范围
`# ReGenNet AI 入口` 标题与引用块、`## 稳定约束`、`## 解释边界` 原样保留；只压缩 `## 当前研究入口` 与 `## 文档入口`。
不新增事实，不改写保留内容的含义，保持中文与既有写法。

## 判据（标准：下一个会话读不到这条，会不会做错事或重复踩坑？）
可以移除或精简（转入归档）：
- 已被后续条目明确取代的中间状态与"下一步"计划（后续条目已记录其完成或放弃）；
- 与其它条目重复的表述；
- 引用文档中已完整保存的过程细节（逐步经过、成串数值）——精简时保留结论、关键数值与该条原有的全部 `docs/ai/context/` 路径。
必须保留：
- 现行协议、指标口径、gate 定义、当前最佳 checkpoint 路径及其关键数值、训练步数预算等仍生效的决策；
- 负结果、坑与反直觉事实（读起来像"已解决的历史"的最容易被误删）；
- 未完成事项；用户确认过的参数与授权记录。
入口里的文档路径决定该文档能否免于每日清理：整条移除后，它引用的文档只受归档保护，归档 15 天后过期即被清理（git 历史可找回）。
有疑问就保留，宁可少删一条，不可多删一条。没有明确符合判据的内容时不改 AGENTS.md、不建归档，changed 返回 false。

## 归档 `{archive}`（仅在 AGENTS.md 有改动时）
① 标题、时间（+09:00）、压缩前后行数与字节数（`wc -lc AGENTS.md`）；② 每条被移除或精简的原文完整照抄（不摘要）及命中的判据，\
精简的条目附精简后的写法；③ 若有事实并入其它条目，写明并入了什么、并到哪里；④「本次刻意保留的内容」：看似可删但保留的条目及原因。

## 输出
changed：AGENTS.md 是否有改动；removed：移除或精简的条目数；archive：归档相对路径（无改动为空字符串）；summary：中文 2–4 句说明做了什么、保留了什么。
"""
PR_FOOTER = "🤖 Generated with [Claude Code](https://claude.com/claude-code)"


def git(*args: str) -> str:
    return run("git", "-C", str(REPO), *args)


def pick_branch(remote: str, date: str) -> str:
    """同一天重跑时本地或远端可能留有同名分支，加序号避开。"""
    for n in range(1, 100):
        name = BRANCH_PREFIX + date + ("" if n == 1 else f"-{n}")
        local = run("git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False)
        if not local and not git("ls-remote", "--heads", remote, name):
            return name
    raise ShipError(f"{BRANCH_PREFIX}{date} 的序号已用尽")


def file_stats(path: Path) -> Tuple[int, int]:
    data = path.read_bytes()
    return data.count(b"\n"), len(data)


def worktree_changes(tree: Path) -> List[Tuple[str, str]]:
    out = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "-z", "--untracked-files=all"],
                         check=True, capture_output=True).stdout.decode("utf-8")
    return [(entry[:2], entry[3:]) for entry in out.split("\0") if entry]


def unexpected_changes(changes: List[Tuple[str, str]], archive: str) -> List[str]:
    """只接受：AGENTS.md 修改、context 文件删除、新建且仅新建归档；AGENTS.md 改了就必须有归档。"""
    problems = [f"{code} {path}" for code, path in changes
                if not ((path == "AGENTS.md" and code == " M")
                        or (path.startswith(CONTEXT_DIR) and code == " D")
                        or (path == archive and code == "??"))]
    touched = {path for _, path in changes}
    if "AGENTS.md" in touched and archive not in touched:
        problems.append(f"AGENTS.md 已改动但没有归档 {archive}")
    if archive in touched and "AGENTS.md" not in touched:
        problems.append(f"新建了归档 {archive} 但 AGENTS.md 没有改动")
    return problems


def slim(tree: Path, archive: str) -> Dict[str, object]:
    lines, size = file_stats(tree / "AGENTS.md")
    print(f"\n== 压缩 AGENTS.md（{lines} 行，{size} 字节）==", flush=True)
    result = ask_claude(SLIM_PROMPT.format(archive=archive, lines=lines, size=size), SLIM_SCHEMA,
                        SLIM_TOOLS, SLIM_ALLOWED, cwd=str(tree), timeout_s=SLIM_TIMEOUT_S, what="压缩")
    print(f"压缩结果：{result}")
    return result


def describe(cutoff: str, deleted: int, before: Tuple[int, int], after: Tuple[int, int],
             result: Dict[str, object], archive: str) -> Tuple[str, str]:
    """返回 (清理说明, 压缩说明)，提交信息与 PR 正文共用。"""
    pruned = f"清理：保留 {DEFAULT_DAYS} 天、截止 {cutoff}、删除 {deleted} 个文件" if deleted else "清理：本段无变化"
    slimmed = (f"压缩：AGENTS.md {before[0]}→{after[0]} 行（{before[1]}→{after[1]} 字节）、"
               f"移除或精简 {result['removed']} 条、归档到 {archive}" if before != after else "压缩：本段无变化")
    return pruned, slimmed


def ship(tree: Path, branch: str, today: str, pruned: str, slimmed: str, summary: str) -> None:
    title = f"chore: docs/ai/context 清理 + AGENTS.md 压缩 ({today})"
    run("git", "-C", str(tree), "add", "-A", "--", "AGENTS.md", CONTEXT_DIR)
    run("git", "-C", str(tree), "commit", "-q", "-m",
        f"{title}\n\n{pruned}\n{slimmed}\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    body = "\n".join([
        "每日文档维护（`scripts/daily_docs_maintenance.py`，crontab 触发）。", "",
        f"- {pruned}（被删文件可从 git 历史找回）", f"- {slimmed}", "",
        summary, "", PR_FOOTER])
    # ship_pr.py 以当前目录为仓库，必须在 worktree 内以子进程运行
    code = subprocess.run([sys.executable, "scripts/ship_pr.py", "--title", title, "--body", body],
                          cwd=str(tree)).returncode
    if code != 0:
        raise ShipError(f"ship_pr.py 失败（exit={code}），PR 状态见上方输出；分支 {branch}")


def cleanup(tmp: Path, tree: Path, branch: str) -> None:
    run("git", "-C", str(REPO), "worktree", "remove", "--force", str(tree), check=False)
    run("git", "-C", str(REPO), "worktree", "prune", check=False)
    run("git", "-C", str(REPO), "branch", "-D", branch, check=False)
    shutil.rmtree(str(tmp), ignore_errors=True)


def maintain(dry_run: bool) -> None:
    now = datetime.datetime.now()
    today = now.strftime("%Y%m%d")
    print(f"== 每日文档维护 {now:%Y-%m-%d %H:%M:%S}{'（dry-run）' if dry_run else ''} ==", flush=True)

    os.chdir(str(REPO))  # ship_pr 的辅助函数以当前目录为仓库；cron 的起始目录不可依赖
    run("gh", "auth", "status")
    remote = main_remote()
    git("ls-remote", "--exit-code", remote, MAIN)
    if shutil.which("claude") is None:
        raise ShipError("找不到 claude 命令")
    run_visible("git", "-C", str(REPO), "fetch", "-q", remote, MAIN)

    branch = pick_branch(remote, today)
    tmp = Path(tempfile.mkdtemp(prefix="regennet-docs-maintenance-"))
    tree = tmp / "repo"
    archive = f"{CONTEXT_DIR}{now:%Y%m%d-%H%M%S}-agents-md-compression.md"
    try:
        run_visible("git", "-C", str(REPO), "worktree", "add", "-q", "-b", branch, str(tree), f"{remote}/{MAIN}")
        print("\n== 清理 docs/ai/context ==", flush=True)
        cutoff, deleted = prune(tree, DEFAULT_DAYS, now.date(), dry_run=False)
        before = file_stats(tree / "AGENTS.md")
        result = slim(tree, archive)
        after = file_stats(tree / "AGENTS.md")

        changes = worktree_changes(tree)
        problems = unexpected_changes(changes, archive)
        if problems:
            raise ShipError("改动超出允许范围，未提交：\n" + "\n".join("  ✗ " + p for p in problems))
        pruned, slimmed = describe(cutoff, len(deleted), before, after, result, archive)
        print(f"\n{pruned}\n{slimmed}")
        if not changes:
            print("无过期文档且 AGENTS.md 无需压缩，不建 PR")
            return
        if dry_run:
            print(run("git", "-C", str(tree), "diff", "--stat", "--", "AGENTS.md"))
            print(run("git", "-C", str(tree), "diff", "--", "AGENTS.md"))
            if (tree / archive).exists():
                print(f"\n---- {archive} ----\n" + (tree / archive).read_text(encoding="utf-8"))
            print("dry-run：不提交、不推送")
            return
        ship(tree, branch, today, pruned, slimmed, str(result.get("summary", "")))
        slug = parse_github_slug(git("remote", "get-url", remote))
        pr = run("gh", "pr", "list", "--repo", slug, "--head", branch, "--state", "all",
                 "--json", "url,state", "--jq", '.[0] | "\\(.url) \\(.state)"')
        print(f"\nPR：{pr}")
    finally:
        cleanup(tmp, tree, branch)
        print(f"已清理临时 worktree；本地 {MAIN} = {git('rev-parse', '--short', MAIN)}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="每日 docs/ai/context 清理 + AGENTS.md 压缩")
    parser.add_argument("--dry-run", action="store_true", help="只做清理与压缩并打印改动，不提交、不推送")
    args = parser.parse_args(argv)
    try:
        maintain(args.dry_run)
    except ShipError as error:
        print(f"✗ {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
