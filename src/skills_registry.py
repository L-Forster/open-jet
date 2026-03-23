from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SkillSummary:
    name: str
    path: Path
    use: str
    tags: tuple[str, ...] = ()


def openjet_home() -> Path:
    override = os.environ.get("OPENJET_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".openjet"


def local_openjet_home(root: Path) -> Path:
    return root / ".openjet"


def active_skills_home(root: Path) -> Path:
    local_home = local_openjet_home(root)
    if (local_home / "skills").exists():
        return local_home
    return openjet_home()


def skills_dir(root: Path) -> Path:
    return active_skills_home(root) / "skills"


def skills_manifest_path(root: Path) -> Path:
    return active_skills_home(root) / "skills.md"


def available_skill_names(root: Path) -> list[str]:
    return [summary.name for summary in skill_summaries(root)]


def resolve_skill_path(root: Path, name: str) -> Path | None:
    normalized = Path(str(name).strip()).stem
    if not normalized:
        return None
    candidate = skills_dir(root) / f"{normalized}.md"
    if candidate.exists():
        return candidate
    return None


def skill_summaries(root: Path) -> list[SkillSummary]:
    directory = skills_dir(root)
    if not directory.exists():
        return []
    summaries: list[SkillSummary] = []
    for path in sorted(directory.glob("*.md")):
        body = _load_doc(path)
        if not body:
            continue
        metadata, content = _split_frontmatter(body)
        tags = metadata.get("tags")
        normalized_tags = tuple(str(tag).strip() for tag in tags if str(tag).strip()) if isinstance(tags, list) else ()
        summaries.append(
            SkillSummary(
                name=path.stem,
                path=path,
                use=_skill_use_summary(metadata, content),
                tags=normalized_tags,
            )
        )
    return summaries


def render_skills_manifest(root: Path) -> str:
    directory = skills_dir(root)
    manifest = skills_manifest_path(root)
    summaries = skill_summaries(root)

    lines = [
        "# Skills",
        "",
        "This file is an index only.",
        "If a skill looks relevant to the current task, load the matching skill `.md` file before following it.",
        "Do not assume the short summary here contains the full instructions.",
        "",
        f"skills_dir: {directory}",
        f"skills_manifest: {manifest}",
        "",
    ]
    if not summaries:
        lines.append("No skills are currently available.")
        return "\n".join(lines).strip()

    lines.append("Available skills:")
    for summary in summaries:
        lines.append(f"- name: {summary.name}")
        lines.append(f"  file: {summary.path}")
        if summary.tags:
            lines.append(f"  tags: {', '.join(summary.tags)}")
        lines.append(f"  use: {summary.use}")
    return "\n".join(lines).strip()


def sync_skills_manifest(root: Path) -> Path:
    manifest = skills_manifest_path(root)
    try:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(render_skills_manifest(root) + "\n", encoding="utf-8")
    except Exception:
        return manifest
    return manifest


def _load_doc(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return ""


def _split_frontmatter(body: str) -> tuple[dict[str, Any], str]:
    if not body.startswith("---\n"):
        return {}, body
    try:
        _, rest = body.split("---\n", 1)
        frontmatter, content = rest.split("\n---\n", 1)
        loaded = yaml.safe_load(frontmatter) or {}
        if isinstance(loaded, dict):
            return loaded, content
    except Exception:
        return {}, body
    return {}, body


def _skill_use_summary(metadata: dict[str, Any], content: str) -> str:
    explicit = str(metadata.get("use") or "").strip()
    if explicit:
        return explicit
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "No summary provided."
