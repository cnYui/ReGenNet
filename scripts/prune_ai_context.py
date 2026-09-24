#!/usr/bin/env python3
"""清理 docs/ai/context/ 中超过保留期、且不再被任何保留内容引用的文档。

保留规则（任一命中即保留）：
1. 文件名日期（YYYYMMDD 前缀）在保留期内，或文件名没有日期前缀；
2. 正文含 `<!-- prune:keep -->` 标记；
3. 文件名出现在 context 目录之外的任一 git 跟踪文件中（AGENTS.md、README、脚本等）；
4. 文件名出现在已保留的 context 文档中（按闭包传递）——入口常写"详见 X 及其引用文档"，只看直接引用会切断这条链。

引用按文件名子串匹配，与 check_repo_conventions.py 的悬空引用检查（git grep -F）口径一致。
只处理 git 跟踪的文件；删除的文件留在 git 历史中：git log --diff-filter=D --name-only -- docs/ai/context/

用法：python3 scripts/prune_ai_context.py [--repo-root .] [--days 15] [--today YYYYMMDD] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import datetime
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

CONTEXT_DIR = "docs/ai/context/"
KEEP_MARKER = "<!-- prune:keep -->"
DATE_PREFIX = re.compile(r"^(\d{8})")
DEFAULT_DAYS = 15


def tracked_files(root: Path) -> List[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def read_text(path: Path) -> Optional[str]:
    """二进制或非 UTF-8 文件不可能按文件名引用文档，返回 None 跳过。"""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def mentioned(names: Iterable[str], text: str) -> Set[str]:
    return {name for name in names if name in text}


def plan_prune(docs: Dict[str, Optional[str]], outside: Dict[str, str],
               cutoff: str) -> Tuple[List[str], Dict[str, str]]:
    """纯函数：docs 为 {文件名: 正文或 None}，outside 为 {context 外路径: 正文}。

    返回 (待删除文件名, {保留文件名: 原因})。
    """
    reasons: Dict[str, str] = {}
    for name, text in docs.items():
        match = DATE_PREFIX.match(name)
        if not match:
            reasons[name] = "无日期前缀"
        elif match.group(1) >= cutoff:
            reasons[name] = "保留期内"
        elif text is not None and KEEP_MARKER in text:
            reasons[name] = "prune:keep 标记"
    for path, text in sorted(outside.items()):
        for name in sorted(mentioned(docs, text)):
            reasons.setdefault(name, f"被 {path} 引用")
    queue = sorted(reasons)
    while queue:
        referrer = queue.pop()
        text = docs[referrer]
        if text is None:
            continue
        for name in sorted(mentioned(docs, text) - set(reasons)):
            reasons[name] = f"经 {referrer} 间接引用"
            queue.append(name)
    return sorted(set(docs) - set(reasons)), reasons


def collect(root: Path) -> Tuple[Dict[str, Optional[str]], Dict[str, str]]:
    docs: Dict[str, Optional[str]] = {}
    outside: Dict[str, str] = {}
    for rel in tracked_files(root):
        path = root / rel
        if not path.is_file():  # 已在工作区删除但未提交
            continue
        if rel.startswith(CONTEXT_DIR):
            if "/" not in rel[len(CONTEXT_DIR):]:
                docs[path.name] = read_text(path)
        else:
            text = read_text(path)
            if text is not None:
                outside[rel] = text
    return docs, outside


def prune(root: Path, days: int, today: datetime.date, dry_run: bool) -> Tuple[str, List[str]]:
    """执行清理并打印报告，返回 (截止日期 YYYYMMDD, 删除的文件名)。"""
    cutoff = (today - datetime.timedelta(days=days)).strftime("%Y%m%d")
    docs, outside = collect(root)
    doomed, reasons = plan_prune(docs, outside, cutoff)
    if not dry_run:
        for name in doomed:
            (root / CONTEXT_DIR / name).unlink()

    print(f"保留 {days} 天，截止 {cutoff}（文件名日期早于此且无引用的删除）")
    print(f"{'将删除' if dry_run else '已删除'} {len(doomed)} 个文件，保留 {len(reasons)} 个")
    for name in doomed:
        print(f"  - {name}")
    counts = collections.Counter(
        "外部引用" if r.startswith("被 ") else "间接引用" if r.startswith("经 ") else r for r in reasons.values())
    print("保留原因：" + "，".join(f"{kind} {n}" for kind, n in sorted(counts.items())))
    for name in sorted(reasons):
        print(f"  = {name}（{reasons[name]}）")
    return cutoff, doomed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="清理 docs/ai/context/ 超期且无引用的文档")
    parser.add_argument("--repo-root", default=".", help="仓库根目录，默认当前目录")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"保留天数，默认 {DEFAULT_DAYS}")
    parser.add_argument("--today", help="以 YYYYMMDD 作为今天（测试用），默认本地日期")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不删除")
    args = parser.parse_args(argv)

    root = Path(subprocess.run(["git", "-C", args.repo_root, "rev-parse", "--show-toplevel"],
                               check=True, capture_output=True, text=True).stdout.strip())
    today = (datetime.datetime.strptime(args.today, "%Y%m%d").date() if args.today
             else datetime.date.today())
    prune(root, args.days, today, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
