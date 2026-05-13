from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .model import Skill, SkillDiagnostic, SkillFormat, SkillMetadata, current_platform, validate_skill_name


_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z", re.DOTALL)


def parse_skill_file(
    path: Path,
    *,
    source: str,
    source_kind: str,
    source_label: str,
    format: SkillFormat,
) -> Skill:
    raw_content = _read_text(path)
    frontmatter, content, yaml_error = split_frontmatter(raw_content)
    diagnostics: list[SkillDiagnostic] = []
    candidate_name = path.parent.name if format == "standard" else path.stem
    metadata_name = _string(frontmatter.get("name")) or candidate_name
    name = metadata_name.strip()

    if yaml_error:
        diagnostics.append(
            SkillDiagnostic(
                code="malformed_yaml",
                severity="error",
                message=f"Malformed YAML frontmatter: {yaml_error}",
                path=path,
                skill_name=candidate_name,
            )
        )
    if not validate_skill_name(name):
        diagnostics.append(
            SkillDiagnostic(
                code="invalid_name",
                severity="error",
                message=(
                    "Skill names must be lowercase letters, numbers, underscores, or hyphens; "
                    "they must start and end with a letter or number."
                ),
                path=path,
                skill_name=name or candidate_name,
            )
        )
    if format == "standard" and name != candidate_name and validate_skill_name(name) and validate_skill_name(candidate_name):
        diagnostics.append(
            SkillDiagnostic(
                code="name_mismatch",
                severity="warning",
                message=f"Frontmatter name {name!r} does not match directory name {candidate_name!r}.",
                path=path,
                skill_name=name,
            )
        )

    explicit_description = _string(frontmatter.get("description"))
    legacy_use = _string(frontmatter.get("use"))
    description = explicit_description or legacy_use or _first_content_line(content) or "No description provided."
    if format == "standard" and not explicit_description:
        diagnostics.append(
            SkillDiagnostic(
                code="missing_description",
                severity="warning",
                message="Standard skills should include a non-empty description in YAML frontmatter.",
                path=path,
                skill_name=name or candidate_name,
            )
        )

    platforms = _string_tuple(frontmatter.get("platforms"), lower=True)
    unsupported_platforms = tuple(platform for platform in platforms if platform not in {"macos", "linux", "windows"})
    if unsupported_platforms:
        diagnostics.append(
            SkillDiagnostic(
                code="unsupported_platform",
                severity="warning",
                message=f"Unknown platform value(s): {', '.join(unsupported_platforms)}.",
                path=path,
                skill_name=name or candidate_name,
            )
        )
    if platforms and current_platform() not in platforms:
        diagnostics.append(
            SkillDiagnostic(
                code="unsupported_platform",
                severity="warning",
                message=f"Skill is not supported on current platform {current_platform()}.",
                path=path,
                skill_name=name or candidate_name,
            )
        )

    metadata = frontmatter.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    root = path.parent if format == "standard" else path.parent
    skill_metadata = SkillMetadata(
        name=name,
        description=description,
        version=_string(frontmatter.get("version")),
        author=_string(frontmatter.get("author")),
        license=_string(frontmatter.get("license")),
        platforms=platforms,
        metadata_openjet=_nested_dict(metadata_dict, "openjet"),
        metadata_hermes=_nested_dict(metadata_dict, "hermes"),
        required_environment_variables=_string_tuple(frontmatter.get("required_environment_variables")),
        allowed_tools=_string_tuple(frontmatter.get("allowed-tools", frontmatter.get("allowed_tools"))),
        tags=_string_tuple(frontmatter.get("tags"), lower=True),
        mode=_string(frontmatter.get("mode")).lower(),
        raw_frontmatter=frontmatter,
    )
    return Skill(
        name=name or candidate_name,
        root=root,
        path=path,
        source=source,
        source_kind=source_kind,
        source_label=source_label,
        format=format,
        metadata=skill_metadata,
        content=content.strip(),
        raw_content=raw_content.strip(),
        diagnostics=tuple(diagnostics),
    )


def split_frontmatter(raw_content: str) -> tuple[dict[str, Any], str, str | None]:
    if not raw_content.startswith("---"):
        return {}, raw_content, None
    match = _FRONTMATTER_RE.match(raw_content)
    if not match:
        return {}, raw_content, "frontmatter is missing a closing --- delimiter"
    frontmatter_text, content = match.groups()
    try:
        loaded = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        return {}, content, str(exc).strip()
    if not isinstance(loaded, dict):
        return {}, content, "frontmatter must be a YAML mapping"
    return loaded, content, None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_tuple(value: Any, *, lower: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        return ()
    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        text = str(raw).strip()
        if lower:
            text = text.lower()
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return tuple(items)


def _nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _first_content_line(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""
