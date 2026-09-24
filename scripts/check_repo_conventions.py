#!/usr/bin/env python3
"""仓库约定检查：PR 合并前的 CI 与本地 scripts/ship_pr.py 共用同一套规则。

检查 base...head（以 merge-base 为起点）之间的改动：
1. CLAUDE.md 必须是指向 AGENTS.md 的软链接；
2. docs/ai/context/ 历史文件不得修改；只能删除文件名日期已超过保留期的文件（每日清理由
   scripts/prune_ai_context.py 执行），且删除后不得留下悬空引用；新增 .md 必须命名为 YYYYMMDD-HHMMSS-名称.md；
3. 改动的 .py 文件不得新增 ruff 问题（目标 Python 3.7）。

用法：python3 scripts/check_repo_conventions.py --base fork/main [--head HEAD] [--ruff ruff | --skip-python]
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Counter, Dict, List, Optional, Tuple

from prune_ai_context import CONTEXT_DIR, DATE_PREFIX, DEFAULT_DAYS

CONTEXT_MD_NAME = re.compile(r"^\d{8}-\d{6}-[^/]+\.md$")
SYMLINK_MODE = "120000"
# 项目运行环境是 Python 3.7.13，按 py37 检查才能拦住 3.8+ 语法；只选会导致运行失败的规则，避免风格噪音
RUFF_ARGS = ["check", "--isolated", "--no-cache", "--target-version", "py37",
             "--select", "E9,F63,F7,F82", "--output-format", "json"]

Change = Tuple[str, str, str]  # (状态字母, 旧路径, 新路径)；非改名时新旧路径相同
FindingKey = Tuple[str, str, str]  # (路径, 规则, 信息)；不含行号，改动引起的行号漂移不算新问题


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def parse_name_status(raw: str) -> List[Change]:
    """解析 `git diff --name-status -M -z` 输出；改名/复制条目带两个路径。"""
    tokens = raw.split("\0")
    changes = []
    i = 0
    while i < len(tokens) and tokens[i]:
        status = tokens[i][0]
        if status in "RC":
            changes.append((status, tokens[i + 1], tokens[i + 2]))
            i += 3
        else:
            changes.append((status, tokens[i + 1], tokens[i + 1]))
            i += 2
    return changes


def check_claude_symlink(head: str) -> List[str]:
    entry = git("ls-tree", head, "--", "CLAUDE.md").split()
    if not entry:
        return ["CLAUDE.md 不存在：它应是指向 AGENTS.md 的软链接（ln -s AGENTS.md CLAUDE.md）"]
    mode, _, blob = entry[:3]
    if mode == SYMLINK_MODE and git("cat-file", "blob", blob) == "AGENTS.md":
        return []
    return [f"CLAUDE.md 必须是指向 AGENTS.md 的软链接（当前 git mode={mode}）；"
            "请把改动合并进 AGENTS.md，再执行 ln -sfn AGENTS.md CLAUDE.md"]


def dangling_references(head: str, names: List[str]) -> Dict[str, List[str]]:
    """返回 head 树中仍引用这些文件名的文件；与 prune_ai_context.py 一样按文件名子串匹配。"""
    if not names:
        return {}
    patterns = [arg for name in names for arg in ("-e", name)]
    proc = subprocess.run(["git", "-c", "core.quotePath=false", "grep", "-I", "-o", "-F", *patterns, head, "--"],
                          capture_output=True, text=True)
    if proc.returncode not in (0, 1):  # 1 表示没有匹配
        raise RuntimeError(f"git grep 失败：{proc.stderr.strip()}")
    found: Dict[str, List[str]] = {}
    for line in proc.stdout.splitlines():
        path, name = line[len(head) + 1:].rsplit(":", 1)  # 输出格式为 <head>:<路径>:<匹配>
        if path not in found.setdefault(name, []):
            found[name].append(path)
    return found


def deletion_cutoff(today: datetime.date) -> str:
    """文件名日期早于该值才允许删除。CI 按 UTC 取日期，比本机（JST）最多早一天，
    因此比清理脚本放宽一天，保证每日任务按本机日期删除的文件都能通过。"""
    return (today - datetime.timedelta(days=DEFAULT_DAYS - 1)).strftime("%Y%m%d")


def check_context_docs(changes: List[Change], head: str, today: Optional[datetime.date] = None) -> List[str]:
    """changes 须按 --no-renames 取得：改名即"删除 + 新增"，两半分别受约束。"""
    errors = []
    deleted = []
    cutoff = deletion_cutoff(today or datetime.date.today())
    for status, path, _ in changes:
        if not path.startswith(CONTEXT_DIR):
            continue
        if status == "D":
            name = path[len(CONTEXT_DIR):]
            match = DATE_PREFIX.match(name)
            if not match or match.group(1) >= cutoff:
                errors.append(f"{path} 的文件名日期未早于 {cutoff}（保留期 {DEFAULT_DAYS} 天）或没有日期前缀，不得删除")
            deleted.append(name)
        elif status != "A":
            action = {"M": "修改", "T": "改变类型"}.get(status, status)
            errors.append(f"{CONTEXT_DIR} 历史文件不得修改，本 PR {action}了 {path}")
        elif path.endswith(".md") and not CONTEXT_MD_NAME.match(path[len(CONTEXT_DIR):]):
            errors.append(f"新增 context 文档必须命名为 YYYYMMDD-HHMMSS-名称.md：{path}")
    for name, paths in sorted(dangling_references(head, deleted).items()):
        errors.append(f"删除了 {CONTEXT_DIR}{name}，但它仍被引用：{', '.join(paths)}")
    return errors


def ruff_findings(ruff: str, rev: str, files: List[Tuple[str, str]]) -> Counter[FindingKey]:
    """把 rev 中的文件物化到临时目录再检查，base 与 head 走同一路径，结果才能直接做差。

    files 为 (rev 中的路径, 结果中使用的路径)，改名文件据此对齐到新路径。
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        for rev_path, out_path in files:
            dst = root / out_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(subprocess.run(["git", "show", f"{rev}:{rev_path}"],
                                           check=True, capture_output=True).stdout)
        proc = subprocess.run([ruff, *RUFF_ARGS, str(root)], capture_output=True, text=True)
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"ruff 运行失败：{proc.stderr.strip()}")
        return collections.Counter(
            (Path(os.path.relpath(item["filename"], root)).as_posix(), item["code"], item["message"])
            for item in json.loads(proc.stdout or "[]"))


def new_findings(head_found: Counter[FindingKey], base_found: Counter[FindingKey]) -> List[FindingKey]:
    return sorted((head_found - base_found).elements())


def check_python(ruff: str, merge_base: str, head: str, changes: List[Change]) -> List[str]:
    py = [(status, old, new) for status, old, new in changes if status != "D" and new.endswith(".py")]
    if not py:
        return []
    head_found = ruff_findings(ruff, head, [(new, new) for _, _, new in py])
    base_found = ruff_findings(ruff, merge_base, [(old, new) for status, old, new in py if status != "A"])
    return [f"{path}: {code} {message}（本 PR 新引入，目标 Python 3.7）"
            for path, code, message in new_findings(head_found, base_found)]


def run_checks(base: str, head: str = "HEAD", ruff: Optional[str] = None) -> List[str]:
    """返回全部违规信息；ruff 为 None 时跳过 Python 检查（本地未安装 ruff 时由 CI 兜底）。"""
    merge_base = git("merge-base", base, head).strip()
    changes = parse_name_status(git("diff", "--name-status", "-M", "-z", merge_base, head))
    # 清理删除的旧文档与新增归档可能被相似度判成改名，context 部分按不识别改名比较
    context_changes = parse_name_status(git("diff", "--name-status", "--no-renames", "-z",
                                            merge_base, head, "--", CONTEXT_DIR))
    errors = check_claude_symlink(head) + check_context_docs(context_changes, head)
    if ruff is not None:
        errors += check_python(ruff, merge_base, head, changes)
    return errors


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ReGenNet 仓库约定检查")
    parser.add_argument("--base", required=True, help="PR 目标分支的提交，如 fork/main 或 PR base SHA")
    parser.add_argument("--head", default="HEAD", help="待合并的提交，默认 HEAD")
    parser.add_argument("--ruff", default="ruff", help="ruff 可执行文件")
    parser.add_argument("--skip-python", action="store_true", help="跳过 ruff 检查")
    args = parser.parse_args(argv)

    ruff = None
    if not args.skip_python:
        ruff = shutil.which(args.ruff)
        if ruff is None:
            print(f"✗ 找不到 ruff（{args.ruff}）；本地可加 --skip-python，CI 中由 workflow 安装", file=sys.stderr)
            return 2
    errors = run_checks(args.base, args.head, ruff)
    # 在 GitHub Actions 中输出 ::error::，违规会直接显示在 PR 的检查摘要里
    prefix = "::error::" if os.environ.get("GITHUB_ACTIONS") == "true" else "✗ "
    for error in errors:
        print(prefix + error)
    if errors:
        return 1
    print("✓ 仓库约定检查通过" + ("（已跳过 Python 检查）" if ruff is None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
