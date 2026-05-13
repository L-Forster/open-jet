from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model import SkillDiagnostic, validate_skill_name
from .registry import SkillRegistry


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    diagnostics: tuple[SkillDiagnostic, ...]
    path: Path | None = None


def validate_skill(root: Path, name: str) -> ValidationResult:
    registry = SkillRegistry(root)
    skill = registry.get_any(name)
    if skill is None:
        return ValidationResult(
            ok=False,
            diagnostics=(
                SkillDiagnostic(
                    code="not_found",
                    severity="error",
                    message=f"Skill not found: {name}",
                    skill_name=name,
                ),
            ),
        )
    diagnostics = tuple(
        diagnostic
        for diagnostic in registry.diagnostics()
        if diagnostic.skill_name == skill.name or diagnostic.path == skill.path
    )
    return ValidationResult(
        ok=not any(diagnostic.severity == "error" for diagnostic in diagnostics),
        diagnostics=diagnostics,
        path=skill.path,
    )


def validate_all(root: Path) -> ValidationResult:
    registry = SkillRegistry(root)
    diagnostics = registry.diagnostics()
    return ValidationResult(
        ok=not any(diagnostic.severity == "error" for diagnostic in diagnostics),
        diagnostics=diagnostics,
    )


def create_skill_scaffold(root: Path, name: str) -> Path:
    skill_name = str(name).strip().lower()
    if not validate_skill_name(skill_name):
        raise ValueError(
            "Skill names must be lowercase letters, numbers, underscores, or hyphens; "
            "they must start and end with a letter or number."
        )
    skill_root = Path(root) / ".openjet" / "skills" / skill_name
    if skill_root.exists():
        raise ValueError(f"Skill already exists: {skill_name}")
    skill_root.mkdir(parents=True)
    for directory in ("references", "scripts", "assets"):
        (skill_root / directory).mkdir()
    (skill_root / "SKILL.md").write_text(_skill_template(skill_name), encoding="utf-8")
    return skill_root


def format_diagnostics(diagnostics: tuple[SkillDiagnostic, ...]) -> str:
    if not diagnostics:
        return "No skill diagnostics."
    lines: list[str] = []
    for diagnostic in diagnostics:
        path = f" | path={diagnostic.path}" if diagnostic.path else ""
        skill = f" | skill={diagnostic.skill_name}" if diagnostic.skill_name else ""
        lines.append(f"- {diagnostic.severity}: {diagnostic.code}{skill}{path} | {diagnostic.message}")
    return "\n".join(lines)


def _skill_template(name: str) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: Use this skill when the task matches the workflow described here.\n"
        "version: 0.1.0\n"
        "platforms:\n"
        "  - macos\n"
        "  - linux\n"
        "  - windows\n"
        "metadata:\n"
        "  openjet:\n"
        "    skill_format: agent-skills\n"
        "---\n"
        f"# {name}\n"
        "\n"
        "Use this skill when the task needs this workflow.\n"
        "\n"
        "## Workflow\n"
        "\n"
        "1. Inspect the relevant project files.\n"
        "2. Apply the smallest safe change.\n"
        "3. Run focused verification.\n"
    )
