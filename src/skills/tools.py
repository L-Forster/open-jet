from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import SkillRegistry


def skills_list(*, root: Path | None = None) -> dict[str, Any]:
    registry = SkillRegistry(root or Path.cwd())
    result = registry.discover()
    return {
        "skills": [skill.compact_dict() for skill in result.skills],
        "diagnostics": [
            diagnostic.to_dict()
            for diagnostic in result.diagnostics
            if diagnostic.code in {"duplicate_shadowed", "unsupported_platform", "malformed_yaml", "missing_description", "invalid_name"}
        ],
    }


def skill_view(name: str, *, file_path: str | None = None, root: Path | None = None) -> dict[str, Any]:
    registry = SkillRegistry(root or Path.cwd())
    skill = registry.get(name)
    if skill is None:
        raise ValueError(f"unknown skill: {name}")
    if not file_path:
        return {
            "name": skill.name,
            "path": str(skill.path),
            "format": skill.format,
            "content": skill.raw_content,
        }
    if skill.format != "standard":
        raise ValueError("file_path is only supported for standard skill directories")
    target = _safe_skill_file(skill.root, file_path)
    return {
        "name": skill.name,
        "path": str(target),
        "format": skill.format,
        "content": target.read_text(encoding="utf-8", errors="replace"),
    }


def _safe_skill_file(skill_root: Path, requested: str) -> Path:
    text = str(requested).strip()
    if not text:
        raise ValueError("file_path cannot be empty")
    requested_path = Path(text)
    if requested_path.is_absolute():
        raise ValueError("file_path must be relative to the skill root")
    if any(part in {"", ".", ".."} for part in requested_path.parts):
        raise ValueError("file_path cannot contain empty, current, or parent directory segments")
    root = skill_root.resolve(strict=True)
    target = (root / requested_path).resolve(strict=True)
    if not _is_relative_to(target, root):
        raise ValueError("file_path escapes the skill root")
    if not target.is_file():
        raise ValueError("file_path must point to a file")
    return target


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
