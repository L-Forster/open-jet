from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


SUPPORTED_PLATFORMS: tuple[str, ...] = ("macos", "linux", "windows")
SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,78}[a-z0-9])?$")

SkillFormat = Literal["legacy", "standard"]
DiagnosticSeverity = Literal["info", "warning", "error"]


def current_platform() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def normalize_skill_name(name: str) -> str:
    return Path(str(name).strip()).stem.lower()


def validate_skill_name(name: str) -> bool:
    text = str(name).strip()
    return bool(SKILL_NAME_RE.fullmatch(text))


@dataclass(frozen=True)
class SkillDiagnostic:
    code: str
    message: str
    severity: DiagnosticSeverity = "warning"
    path: Path | None = None
    skill_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.skill_name:
            payload["skill_name"] = self.skill_name
        if self.path is not None:
            payload["path"] = str(self.path)
        return payload


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    version: str = ""
    author: str = ""
    license: str = ""
    platforms: tuple[str, ...] = ()
    metadata_openjet: dict[str, Any] = field(default_factory=dict)
    metadata_hermes: dict[str, Any] = field(default_factory=dict)
    required_environment_variables: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    mode: str = ""
    raw_frontmatter: dict[str, Any] = field(default_factory=dict)

    def compact_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "description": self.description,
        }
        if self.version:
            payload["version"] = self.version
        if self.author:
            payload["author"] = self.author
        if self.license:
            payload["license"] = self.license
        if self.platforms:
            payload["platforms"] = list(self.platforms)
        if self.required_environment_variables:
            payload["required_environment_variables"] = list(self.required_environment_variables)
        if self.allowed_tools:
            payload["allowed_tools"] = list(self.allowed_tools)
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.mode:
            payload["mode"] = self.mode
        return payload


@dataclass(frozen=True)
class Skill:
    name: str
    root: Path
    path: Path
    source: str
    source_kind: str
    source_label: str
    format: SkillFormat
    metadata: SkillMetadata
    content: str
    raw_content: str
    diagnostics: tuple[SkillDiagnostic, ...] = ()

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def supported_on_current_platform(self) -> bool:
        platforms = self.metadata.platforms
        return not platforms or current_platform() in platforms

    @property
    def has_errors(self) -> bool:
        return any(diagnostic.severity == "error" for diagnostic in self.diagnostics)

    def compact_dict(self) -> dict[str, object]:
        payload = self.metadata.compact_dict()
        payload.update(
            {
                "name": self.name,
                "source": self.source,
                "source_kind": self.source_kind,
                "source_label": self.source_label,
                "format": self.format,
                "path": str(self.path),
            }
        )
        return payload
