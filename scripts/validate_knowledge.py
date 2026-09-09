#!/usr/bin/env python3
"""Validate structured Markdown records before vector ingestion."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REQUIRED_FIELDS = ("城市", "来源", "核验日期", "地址", "评分", "类型")


def validate_document(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    errors: list[str] = []
    if not re.search(r"^#\s+\S+", raw, flags=re.MULTILINE):
        errors.append("缺少一级标题")
    for field in REQUIRED_FIELDS:
        if not re.search(rf"^\s*[-*]\s*{field}[:：]\s*\S+", raw, flags=re.MULTILINE):
            errors.append(f"缺少字段：{field}")
    rating = re.search(r"^\s*[-*]\s*评分[:：]\s*([0-9.]+)", raw, flags=re.MULTILINE)
    if rating and not 0 <= float(rating.group(1)) <= 5:
        errors.append("评分必须在 0 到 5 之间")
    source = re.search(r"^\s*[-*]\s*来源[:：]\s*(\S+)", raw, flags=re.MULTILINE)
    if source and not source.group(1).startswith(("https://", "http://")):
        errors.append("来源必须是 HTTP(S) URL")
    verified = re.search(r"^\s*[-*]\s*核验日期[:：]\s*(\S+)", raw, flags=re.MULTILINE)
    if verified and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", verified.group(1)):
        errors.append("核验日期必须是 YYYY-MM-DD")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="resource/knowledge/generated", help="Markdown root")
    args = parser.parse_args()
    root = Path(args.dir)
    paths = sorted(root.rglob("*.md"))
    failures = 0
    for path in paths:
        errors = validate_document(path)
        if errors:
            failures += 1
            print(f"FAIL {path}: {'; '.join(errors)}")
        else:
            print(f"OK   {path}")
    print(f"Validated {len(paths)} documents; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
