from __future__ import annotations

from pathlib import Path

from .discovery import DiscoveryResult, SourceLocation, discover_skills
from .model import Skill, SkillDiagnostic, normalize_skill_name


class SkillRegistry:
    def __init__(
        self,
        root: Path | None = None,
        *,
        sources: tuple[SourceLocation, ...] | None = None,
        home: Path | None = None,
        bundled_dir: Path | None = None,
    ) -> None:
        self.root = Path(root or Path.cwd())
        self.sources = sources
        self.home = home
        self.bundled_dir = bundled_dir

    def discover(self, *, include_invalid: bool = False, include_unsupported: bool = False) -> DiscoveryResult:
        return discover_skills(
            self.root,
            sources=self.sources,
            home=self.home,
            bundled_dir=self.bundled_dir,
            include_invalid=include_invalid,
            include_unsupported=include_unsupported,
        )

    def list(self) -> tuple[Skill, ...]:
        return self.discover().skills

    def diagnostics(self) -> tuple[SkillDiagnostic, ...]:
        return self.discover(include_invalid=True, include_unsupported=True).diagnostics

    def get(self, name: str) -> Skill | None:
        needle = normalize_skill_name(name)
        if not needle:
            return None
        return self.discover().by_name().get(needle)

    def get_any(self, name: str) -> Skill | None:
        needle = normalize_skill_name(name)
        if not needle:
            return None
        for skill in self.discover(include_invalid=True, include_unsupported=True).all_skills:
            if normalize_skill_name(skill.name) == needle:
                return skill
        return None

    def render_catalog(self) -> str:
        result = self.discover()
        lines = [
            "# Skills",
            "",
            "This file is an index only.",
            "If a skill looks relevant to the current task, call skill_view with its name before following it.",
            "Do not assume the short summary here contains the full instructions.",
            "Do not rely on absolute filesystem paths in this index.",
            "",
            "skill_loading: progressive_disclosure",
            "merge_policy: project skills overlay user and bundled skills with the same name.",
            "",
        ]
        if not result.skills:
            lines.append("No skills are currently available.")
            return "\n".join(lines).strip()
        lines.append("Available skills:")
        for skill in result.skills:
            lines.append(f"- name: {skill.name}")
            lines.append(f"  source: {skill.source_kind}")
            lines.append(f"  dir: {skill.source_label}")
            lines.append(f"  format: {skill.format}")
            lines.append(f"  load_name: {skill.name}")
            lines.append(f"  use: {skill.description}")
        return "\n".join(lines).strip()
