from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..app_paths import openjet_install_root
from .model import Skill, SkillDiagnostic
from .parser import parse_skill_file


@dataclass(frozen=True)
class SourceLocation:
    name: str
    kind: str
    path: Path
    label: str
    precedence: int


@dataclass(frozen=True)
class DiscoveryResult:
    skills: tuple[Skill, ...]
    diagnostics: tuple[SkillDiagnostic, ...]
    all_skills: tuple[Skill, ...]
    sources: tuple[SourceLocation, ...]

    def by_name(self) -> dict[str, Skill]:
        return {skill.name: skill for skill in self.skills}


def default_skill_sources(
    root: Path,
    *,
    home: Path | None = None,
    bundled_dir: Path | None = None,
) -> tuple[SourceLocation, ...]:
    project = Path(root).expanduser()
    user_home = Path(home).expanduser() if home is not None else Path.home()
    bundled = Path(bundled_dir).expanduser() if bundled_dir is not None else openjet_install_root() / "skills"
    return (
        SourceLocation("project_openjet", "project", project / ".openjet" / "skills", ".openjet/skills", 0),
        SourceLocation("project_agents", "project", project / ".agents" / "skills", ".agents/skills", 1),
        SourceLocation("user_openjet", "user", user_home / ".openjet" / "skills", "~/.openjet/skills", 2),
        SourceLocation("user_agents", "user", user_home / ".agents" / "skills", "~/.agents/skills", 3),
        SourceLocation("bundled", "bundled", bundled, "<install>/skills", 4),
    )


def discover_skills(
    root: Path,
    *,
    sources: Iterable[SourceLocation] | None = None,
    home: Path | None = None,
    bundled_dir: Path | None = None,
    include_invalid: bool = False,
    include_unsupported: bool = False,
) -> DiscoveryResult:
    source_locations = tuple(sources or default_skill_sources(root, home=home, bundled_dir=bundled_dir))
    active_by_name: dict[str, Skill] = {}
    diagnostics: list[SkillDiagnostic] = []
    all_skills: list[Skill] = []

    for source in sorted(source_locations, key=lambda item: item.precedence):
        for candidate_path, skill_format in _iter_skill_candidates(source.path):
            skill = parse_skill_file(
                candidate_path,
                source=source.name,
                source_kind=source.kind,
                source_label=source.label,
                format=skill_format,
            )
            all_skills.append(skill)
            diagnostics.extend(skill.diagnostics)

            if skill.name in active_by_name:
                previous = active_by_name[skill.name]
                diagnostics.append(
                    SkillDiagnostic(
                        code="duplicate_shadowed",
                        severity="warning",
                        message=(
                            f"Skill {skill.name!r} from {source.label} is shadowed by "
                            f"{previous.source_label}."
                        ),
                        path=skill.path,
                        skill_name=skill.name,
                    )
                )
                continue
            if skill.has_errors and not include_invalid:
                continue
            if not skill.supported_on_current_platform and not include_unsupported:
                continue
            active_by_name[skill.name] = skill

    skills = tuple(active_by_name[name] for name in sorted(active_by_name))
    return DiscoveryResult(
        skills=skills,
        diagnostics=tuple(diagnostics),
        all_skills=tuple(all_skills),
        sources=source_locations,
    )


def _iter_skill_candidates(directory: Path) -> list[tuple[Path, str]]:
    if not directory.exists() or not directory.is_dir():
        return []
    standard: list[Path] = []
    legacy: list[Path] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        if child.is_dir():
            skill_md = child / "SKILL.md"
            if skill_md.is_file():
                standard.append(skill_md)
        elif child.is_file() and child.suffix.lower() == ".md":
            legacy.append(child)
    return [(path, "standard") for path in standard] + [(path, "legacy") for path in legacy]
