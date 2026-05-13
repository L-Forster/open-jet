from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .app_paths import openjet_install_root
from .skills.discovery import SourceLocation, default_skill_sources, discover_skills
from .skills.model import Skill


@dataclass(frozen=True)
class SkillSummary:
    name: str
    path: Path
    use: str
    source: str
    dir_label: str
    tags: tuple[str, ...] = ()
    format: str = "legacy"
    root: Path | None = None
    description: str = ""
    allowed_tools: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()


def openjet_home() -> Path:
    return openjet_install_root()


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
    normalized = Path(str(name).strip()).stem.lower()
    if not normalized:
        return None
    for skill in _discover(root).skills:
        if skill.name == normalized:
            return skill.path
    return None


def skill_summaries(root: Path) -> list[SkillSummary]:
    return [_summary_from_skill(skill) for skill in _discover(root).skills]


def render_skills_manifest(root: Path) -> str:
    summaries = skill_summaries(root)
    project_dir = project_skills_dir(root)

    lines = [
        "# Skills",
        "",
        "This file is an index only.",
        "If a skill looks relevant to the current task, call skill_view with its name before following it.",
        "Do not assume the short summary here contains the full instructions.",
        "Do not rely on absolute filesystem paths in this index.",
        "",
        "global_skills_dir: <install>/skills",
        f"project_skills_dir: .openjet/skills ({'present' if project_dir.exists() else 'absent'})",
        "project_agent_skills_dir: .agents/skills",
        "user_skills_dirs: ~/.openjet/skills, ~/.agents/skills",
        "merge_policy: project skills overlay global skills with the same name.",
        "skill_loading: progressive_disclosure via skill_view(name).",
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
        lines.append(f"  format: {summary.format}")
        lines.append(f"  load_name: {summary.name}")
        if summary.tags:
            lines.append(f"  tags: {', '.join(summary.tags)}")
        if summary.allowed_tools:
            lines.append(f"  allowed_tools: {', '.join(summary.allowed_tools)}")
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


def _discover(root: Path):
    return discover_skills(root, sources=tuple(_skill_source_locations(root)))


def _skill_source_locations(root: Path) -> list[SourceLocation]:
    sources = list(default_skill_sources(root, bundled_dir=skills_dir(root)))
    # Preserve tests and installed-package behavior that patch this module's
    # openjet_install_root() by making the bundled source explicit here.
    if not _project_is_under_home(root):
        sources = [source for source in sources if source.kind != "user"]
    return sources


def _skill_source_dirs(root: Path) -> list[tuple[str, Path, str]]:
    dirs: list[tuple[str, Path, str]] = []
    for source in _skill_source_locations(root):
        if not source.path.exists():
            continue
        dirs.append((_legacy_source_label(source.kind), source.path, source.label))
    return dirs


def _summary_from_skill(skill: Skill) -> SkillSummary:
    source = _legacy_source_label(skill.source_kind)
    return SkillSummary(
        name=skill.name,
        path=skill.path,
        use=skill.description if skill.format == "legacy" else _compact_summary(skill.description),
        source=source,
        dir_label=skill.source_label,
        tags=skill.metadata.tags,
        format=skill.format,
        root=skill.root,
        description=skill.description,
        allowed_tools=skill.metadata.allowed_tools,
        platforms=skill.metadata.platforms,
    )


def _project_is_under_home(root: Path) -> bool:
    try:
        Path(root).expanduser().resolve().relative_to(Path.home().resolve())
        return True
    except Exception:
        return False


def _compact_summary(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _legacy_source_label(source_kind: str) -> str:
    if source_kind == "project":
        return "project"
    if source_kind == "user":
        return "user"
    return "global"


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
    explicit = str(metadata.get("use") or metadata.get("description") or "").strip()
    if explicit:
        return explicit
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "No summary provided."
