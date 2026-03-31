#!/usr/bin/env python3
"""Normalize the dedicated PyPI README for publishing."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "README.pypi.md"
BASE = "https://github.com/l-forster/open-jet/blob/main/"
LINK_RE = re.compile(r"(?P<prefix>!?\[[^\]]*\]\()(?P<target>[^)]+)(?P<suffix>\))")


def rewrite_target(target: str) -> str:
    if target.startswith(("http://", "https://", "#", "mailto:")):
        return target
    if target.startswith("./"):
        target = target[2:]
    return BASE + target


def rewrite_links(text: str) -> str:
    return LINK_RE.sub(
        lambda match: (
            match.group("prefix")
            + rewrite_target(match.group("target"))
            + match.group("suffix")
        ),
        text,
    )


def main() -> None:
    body = TARGET.read_text(encoding="utf-8").rstrip() + "\n"
    TARGET.write_text(rewrite_links(body), encoding="utf-8")


if __name__ == "__main__":
    main()
