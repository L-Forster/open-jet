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
    source: str
    dir_label: str
    tags: tuple[str, ...] = ()


def openjet_home() -> Path:
    override = os.environ.get("OPENJET_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".openjet"


def local_openjet_home(root: Path) -> Path:
    return root / ".openjet"


def active_skills_home(root: Path) -> Path:
    del root
    return openjet_home()


def skills_dir(root: Path) -> Path:
    del root
    return openjet_home() / "skills"


def project_skills_dir(root: Path) -> Path:
    return local_openjet_home(root) / "skills"


def skills_manifest_path(root: Path) -> Path:
    del root
    return openjet_home() / "skills.md"


def available_skill_names(root: Path) -> list[str]:
    return [summary.name for summary in skill_summaries(root)]


def resolve_skill_path(root: Path, name: str) -> Path | None:
    normalized = Path(str(name).strip()).stem
    if not normalized:
        return None
    for _, directory, _ in reversed(_skill_source_dirs(root)):
        candidate = directory / f"{normalized}.md"
        if candidate.exists():
            return candidate
    return None


def skill_summaries(root: Path) -> list[SkillSummary]:
    summaries_by_name: dict[str, SkillSummary] = {}
    for source, directory, dir_label in _skill_source_dirs(root):
        for path in sorted(directory.glob("*.md")):
            body = _load_doc(path)
            if not body:
                continue
            metadata, content = _split_frontmatter(body)
            tags = metadata.get("tags")
            normalized_tags = tuple(str(tag).strip() for tag in tags if str(tag).strip()) if isinstance(tags, list) else ()
            summaries_by_name[path.stem] = SkillSummary(
                name=path.stem,
                path=path,
                use=_skill_use_summary(metadata, content),
                source=source,
                dir_label=dir_label,
                tags=normalized_tags,
            )
    return [summaries_by_name[name] for name in sorted(summaries_by_name)]


def render_skills_manifest(root: Path) -> str:
    summaries = skill_summaries(root)
    project_dir = project_skills_dir(root)

    lines = [
        "# Skills",
        "",
        "This file is an index only.",
        "If a skill looks relevant to the current task, load it by skill name before following it.",
        "Do not assume the short summary here contains the full instructions.",
        "Do not rely on absolute filesystem paths in this index.",
        "",
        "global_skills_dir: ~/.openjet/skills",
        f"project_skills_dir: .openjet/skills ({'present' if project_dir.exists() else 'absent'})",
        "merge_policy: project skills overlay global skills with the same name.",
        "",
    ]
    if not summaries:
        lines.append("No skills are currently available.")
        return "\n".join(lines).strip()

    lines.append("Available skills:")
    for summary in summaries:
        lines.append(f"- name: {summary.name}")
        lines.append(f"  dir: {summary.dir_label}")
        lines.append(f"  source: {summary.source}")
        lines.append(f"  load_name: {summary.name}")
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


def _skill_source_dirs(root: Path) -> list[tuple[str, Path, str]]:
    sources: list[tuple[str, Path, str]] = []
    global_dir = skills_dir(root)
    if global_dir.exists():
        sources.append(("global", global_dir, "~/.openjet/skills"))
    local_dir = project_skills_dir(root)
    if local_dir.exists():
        sources.append(("project", local_dir, ".openjet/skills"))
    return sources


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
