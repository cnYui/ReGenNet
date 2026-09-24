#!/usr/bin/env python3
"""用本机 Claude Code 审查当前分支，通过后推送并建 PR，等 CI 自动合并后同步本地 main。

用法（在已提交的 feature 分支上）：
  python3 scripts/ship_pr.py [gh pr create 参数...]  审查 → 推送 → 建 PR → 等待合并 → 同步（默认 --fill）
  python3 scripts/ship_pr.py review                 只做本机审查，不推送
  python3 scripts/ship_pr.py sync                   只同步：本地 main 快进到远端，删除已合并且远端已删的分支

审查走本机 Claude Code 登录态（与 desktop 同一账号），不需要 API key；结论以 commit status
`claude-review` 写回 GitHub，.github/workflows/pr-auto-merge.yml 据此决定是否合并。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from check_repo_conventions import run_checks

MAIN = "main"
# 与 .github/workflows/pr-auto-merge.yml 中的 REVIEW_CONTEXT 保持一致
STATUS_CONTEXT = "claude-review"
WORKFLOW_FILE = "pr-auto-merge.yml"
MERGE_JOB_NAME = "自动合并"  # 与 workflow 中合并 job 的 name 保持一致
REVIEW_TIMEOUT_S = 30 * 60
MERGE_TIMEOUT_S = 30 * 60
POLL_S = 15
# 审查只需读代码和 git 历史；dontAsk 下未列出的工具一律拒绝，避免审查过程改动工作区
REVIEW_TOOLS = ["Read", "Grep", "Glob", "Bash"]
REVIEW_ALLOWED = ["Read", "Grep", "Glob", "Bash(git diff:*)", "Bash(git log:*)", "Bash(git show:*)"]
REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "summary", "findings"],
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "block"]},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "file", "description"],
                "properties": {
                    "severity": {"type": "string", "enum": ["blocking", "major", "minor"]},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "description": {"type": "string"},
                },
            },
        },
    },
}
REVIEW_PROMPT = """你在审查 ReGenNet 仓库一个即将自动合并进 main 的改动，改动范围是 `git diff {base}...HEAD`。

步骤：先用 `git diff --stat {base}...HEAD` 和 `git log {base}..HEAD` 了解范围，再逐文件阅读 diff，必要时用 Read/Grep 查看上下文。
结果数据、日志、图片等生成物只确认没有误提交大文件或密钥即可，重点审查代码、脚本、配置和文档。

只报告你能从代码中确认的问题，不报告风格偏好。severity 定义：
- blocking：会导致崩溃或错误结果、破坏训练/评估协议或指标口径、违反 CLAUDE.md「稳定约束」或「解释边界」、提交了密钥或凭据；
- major：很可能是缺陷，但影响有限或需要作者确认；
- minor：可读性、注释、命名等小问题。
verdict 为 block 当且仅当存在 blocking 问题。summary 与 description 用中文，summary 用 2–4 句概括改动与结论。不要修改任何文件。
"""

class ShipError(Exception):
    pass


def run(*cmd: str, check: bool = True) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise ShipError(f"命令失败：{' '.join(cmd)}\n{proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


def run_visible(*cmd: str) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        raise ShipError(f"命令失败：{' '.join(cmd)}")


def parse_github_slug(url: str) -> str:
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url)
    if not match:
        raise ShipError(f"无法从远端地址解析 GitHub 仓库：{url}")
    return match.group(1)


def parse_gone_branches(for_each_ref: str) -> List[str]:
    """输入 `%(refname:short) %(upstream:track)` 格式，返回上游已被删除的本地分支。"""
    return [line.split(" ", 1)[0] for line in for_each_ref.splitlines() if line.endswith(" [gone]")]


def decide(review: Dict[str, object]) -> str:
    """模型给出的 verdict 与 findings 任一指向阻断即阻断，防止两者不一致时误放行。"""
    findings = review.get("findings") or []
    blocked = review.get("verdict") == "block" or any(f.get("severity") == "blocking" for f in findings)
    return "block" if blocked else "approve"


def render_review(review: Dict[str, object], sha: str) -> str:
    label = {"approve": "✅ 通过", "block": "⛔ 阻断"}[decide(review)]
    lines = [f"### 本机 Claude Code 审查：{label}", "", f"审查提交：`{sha}`", "", str(review.get("summary", ""))]
    findings = review.get("findings") or []
    if findings:
        lines += ["", "| 级别 | 位置 | 问题 |", "|---|---|---|"]
        for f in findings:
            where = f"{f.get('file', '')}:{f['line']}" if f.get("line") else str(f.get("file", ""))
            text = str(f.get("description", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {f.get('severity')} | `{where}` | {text} |")
    return "\n".join(lines)


def current_branch() -> str:
    return run("git", "branch", "--show-current")


def tracked_dirty() -> bool:
    return bool(run("git", "status", "--porcelain", "--untracked-files=no"))


def main_remote() -> str:
    upstream = run("git", "rev-parse", "--abbrev-ref", f"{MAIN}@{{upstream}}")
    return upstream.split("/", 1)[0]


def ask_claude(prompt: str, schema: Dict[str, object], tools: List[str], allowed: List[str],
               cwd: str, timeout_s: int, what: str) -> Dict[str, object]:
    """用本机 claude -p 执行一次结构化任务；dontAsk 下白名单外的工具一律拒绝，不会停下来等人确认。"""
    if shutil.which("claude") is None:
        raise ShipError("找不到 claude 命令，请先安装并登录 Claude Code CLI")
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(schema),
             "--tools", ",".join(tools), "--allowedTools", ",".join(allowed),
             "--permission-mode", "dontAsk", "--strict-mcp-config", "--no-session-persistence"],
            capture_output=True, text=True, timeout=timeout_s, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise ShipError(f"{what}超过 {timeout_s // 60} 分钟未完成") from None
    try:
        result = json.loads(proc.stdout)
    except ValueError:
        raise ShipError(f"{what}输出无法解析（exit={proc.returncode}）：{(proc.stderr or proc.stdout)[:500]}") from None
    output = result.get("structured_output")
    if result.get("is_error") or not isinstance(output, dict):
        raise ShipError(f"{what}失败：{str(result.get('result'))[:500]}")
    return output


def review_branch(remote: str) -> Tuple[str, Dict[str, object]]:
    run("git", "fetch", remote, MAIN)
    base = f"{remote}/{MAIN}"
    head = run("git", "rev-parse", "HEAD")
    if not run("git", "diff", "--name-only", f"{base}...HEAD"):
        raise ShipError(f"当前分支相对 {base} 没有改动")
    errors = run_checks(base, "HEAD", shutil.which("ruff"))
    if errors:
        raise ShipError("仓库约定检查未通过：\n" + "\n".join("  ✗ " + e for e in errors))
    # 审查范围由 head 与 merge-base 共同决定，两者都不变时结论可复用；推送或建 PR 失败后重跑不必再等一次审查
    merge_base = run("git", "merge-base", base, "HEAD")
    cache = Path(run("git", "rev-parse", "--git-dir")) / "claude-review" / f"{head}-{merge_base[:12]}.json"
    if cache.exists():
        review = json.loads(cache.read_text(encoding="utf-8"))
        print(f"复用该提交已有的审查结论：{cache}")
        print(render_review(review, head))
        return head, review
    print(f"本机 Claude Code 审查中（{base}...{head[:8]}），通常需要几分钟 …", flush=True)
    review = ask_claude(REVIEW_PROMPT.format(base=base), REVIEW_SCHEMA, REVIEW_TOOLS, REVIEW_ALLOWED,
                        cwd=run("git", "rev-parse", "--show-toplevel"), timeout_s=REVIEW_TIMEOUT_S, what="审查")

    # 留存原始结论，便于复用和事后对照 PR 评论
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_review(review, head))
    return head, review


def find_open_pr(slug: str, branch: str) -> Optional[str]:
    number = run("gh", "pr", "list", "--repo", slug, "--head", branch, "--state", "open",
                 "--json", "number", "--jq", ".[0].number // empty")
    return number or None


def rerun_failed_run(slug: str, sha: str) -> None:
    """该提交此前若未经本脚本审查就已推送，CI 会因缺少 status 失败；补打 status 后重跑才能合并。"""
    runs = json.loads(run("gh", "run", "list", "--repo", slug, "--workflow", WORKFLOW_FILE, "--commit", sha,
                          "--json", "databaseId,status,conclusion", "--limit", "1") or "[]")
    if runs and runs[0]["status"] == "completed" and runs[0]["conclusion"] != "success":
        run_id = str(runs[0]["databaseId"])
        run_visible("gh", "run", "rerun", run_id, "--repo", slug)
        # 重跑生效前 gh pr checks 仍返回上次的失败结果，等 run 重新排队后再开始轮询，避免误判
        for _ in range(12):
            if run("gh", "run", "view", run_id, "--repo", slug, "--json", "status", "--jq", ".status") != "completed":
                return
            time.sleep(5)
        raise ShipError(f"已请求重跑 run {run_id}，但 60 秒内未开始；请在 GitHub 上确认")


def check_buckets(slug: str, pr: str) -> Dict[str, str]:
    # 有检查失败或未完成时 gh 的退出码非 0，但 JSON 照常输出，因此不按退出码判断
    proc = subprocess.run(["gh", "pr", "checks", pr, "--repo", slug, "--json", "name,bucket"],
                          capture_output=True, text=True)
    try:
        return {c["name"]: c["bucket"] for c in json.loads(proc.stdout)}
    except ValueError:
        return {}  # 刚建 PR 时检查尚未注册


def pr_state(slug: str, pr: str) -> str:
    return run("gh", "pr", "view", pr, "--repo", slug, "--json", "state", "--jq", ".state")


def wait_for_merge(slug: str, pr: str) -> None:
    print(f"等待 CI 检查与自动合并：https://github.com/{slug}/pull/{pr}", flush=True)
    deadline = time.time() + MERGE_TIMEOUT_S
    while True:
        state = pr_state(slug, pr)
        if state == "MERGED":
            print("✓ PR 已合并，远端分支已由 CI 删除")
            return
        if state == "CLOSED":
            raise ShipError("PR 已被关闭且未合并")
        buckets = check_buckets(slug, pr)
        failed = sorted(name for name, bucket in buckets.items() if bucket in ("fail", "cancel"))
        if failed:
            raise ShipError(f"CI 未通过：{', '.join(failed)}；详情：gh pr checks {pr} --repo {slug}")
        # 合并 job 要等约定检查结束才出现，只能以它自身结束为准；它结束仍未合并说明被跳过（草稿 PR 或非仓库所有者）
        if buckets.get(MERGE_JOB_NAME) in ("pass", "skipping") and pr_state(slug, pr) != "MERGED":
            raise ShipError("CI 已结束但没有自动合并（草稿 PR 或作者不是仓库所有者？）")
        if time.time() > deadline:
            raise ShipError(f"等待超过 {MERGE_TIMEOUT_S // 60} 分钟仍未合并")
        time.sleep(POLL_S)


def parse_worktree_branches(porcelain: str) -> Dict[str, str]:
    """解析 `git worktree list --porcelain`，返回 {分支名: 检出它的工作树路径}。"""
    branches = {}
    path = ""
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree "):]
        elif line.startswith("branch refs/heads/"):
            branches[line[len("branch refs/heads/"):]] = path
    return branches


def is_merged_into_main(branch: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", branch, MAIN]).returncode == 0


def sync(remote: str) -> None:
    run_visible("git", "fetch", remote, "--prune")
    branch = current_branch()
    dirty = tracked_dirty()
    gone = parse_gone_branches(run("git", "for-each-ref", "--format=%(refname:short) %(upstream:track)",
                                   "refs/heads"))
    here = run("git", "rev-parse", "--show-toplevel")
    # git 拒绝在别的工作树里覆盖、切换到或删除已检出的分支，这些分支只能在所在工作树里处理
    elsewhere = {name: path for name, path in
                 parse_worktree_branches(run("git", "worktree", "list", "--porcelain")).items()
                 if path != here}
    if branch == MAIN:
        if dirty:
            print("! 工作区有未提交的跟踪文件修改，main 未快进；提交或 stash 后重跑 sync")
            return
        run_visible("git", "merge", "--ff-only", f"{remote}/{MAIN}")
    elif MAIN in elsewhere:
        main_tree = elsewhere[MAIN]
        if run("git", "-C", main_tree, "status", "--porcelain", "--untracked-files=no"):
            print(f"! {main_tree} 有未提交的跟踪文件修改，其中的 {MAIN} 未快进；在那里提交或 stash 后重跑 sync")
        else:
            run_visible("git", "-C", main_tree, "merge", "--ff-only", f"{remote}/{MAIN}")
    else:
        # 不切分支也能快进 main 引用；非快进时 git 会拒绝，不会丢提交
        run_visible("git", "fetch", remote, f"{MAIN}:{MAIN}")
        if branch in gone and is_merged_into_main(branch) and not dirty:
            run_visible("git", "switch", MAIN)
    for name in gone:
        if name == current_branch():
            reason = (f"{MAIN} 已在工作树 {elsewhere[MAIN]} 检出，无法在此切回" if MAIN in elsewhere
                      else f"工作区有未提交修改或仍有提交未进入 {MAIN}")
            print(f"! 保留当前分支 {name}（远端分支已删除）：{reason}")
        elif name in elsewhere:
            print(f"! 保留 {name}：已在工作树 {elsewhere[name]} 检出")
        elif is_merged_into_main(name):
            # 已确认全部提交都在 main 中，用 -D 避免 -d 按当前 HEAD 判断合并状态而误拒
            run_visible("git", "branch", "-D", name)
        else:
            print(f"! 保留 {name}：远端分支已删除，但仍有提交未进入 {MAIN}")
    print(f"✓ 本地 {MAIN} = {run('git', 'rev-parse', '--short', MAIN)}，当前分支 {current_branch()}")


def ship(gh_args: List[str]) -> None:
    branch = current_branch()
    if not branch or branch == MAIN:
        raise ShipError(f"请在 feature 分支上运行，不能直接从 {MAIN} 提交")
    if tracked_dirty():
        raise ShipError("有未提交的跟踪文件修改；先提交，保证审查的内容就是推送的内容")
    remote = main_remote()
    slug = parse_github_slug(run("git", "remote", "get-url", remote))

    head, review = review_branch(remote)
    if decide(review) == "block":
        raise ShipError("审查发现阻断级问题，未推送；修复后重新运行")

    run_visible("git", "push", "-u", remote, branch)
    # 先打 status 再建 PR：CI 在 PR 打开时就能看到审查结论
    run("gh", "api", "-X", "POST", f"repos/{slug}/statuses/{head}", "-f", "state=success",
        "-f", f"context={STATUS_CONTEXT}", "-f", "description=本机 Claude Code 审查通过")
    pr = find_open_pr(slug, branch)
    existed = pr is not None
    if pr is None:
        url = run("gh", "pr", "create", "--repo", slug, "--base", MAIN, "--head", branch, *(gh_args or ["--fill"]))
        pr = url.rstrip("/").rsplit("/", 1)[-1]
        print(f"已创建 PR：{url}")
    run("gh", "pr", "comment", pr, "--repo", slug, "--body", render_review(review, head))
    if existed:
        rerun_failed_run(slug, head)
    wait_for_merge(slug, pr)
    sync(remote)


def main(argv: List[str]) -> int:
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    try:
        if argv[:1] == ["sync"]:
            sync(main_remote())
        elif argv[:1] == ["review"]:
            _, review = review_branch(main_remote())
            return 0 if decide(review) == "approve" else 1
        else:
            ship(argv)
    except ShipError as error:
        print(f"✗ {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:  # 约定检查内部的 git 调用
        print(f"✗ 命令失败：{' '.join(error.cmd)}\n{(error.stderr or '').strip()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
