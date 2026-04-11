from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_WORKFLOW_MODES = {"chat", "code", "review", "debug"}


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    path: Path
    body: str
    mode: str = "chat"
    devices: tuple[str, ...] = ()
    interval_seconds: int | None = None
    allow_shell: bool = False
    require_plan: bool = False
    require_verification: bool | None = None
    skills: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    source: str = "repo"


@dataclass(frozen=True)
class WorkflowDiscoveryIssue:
    path: Path
    error: str


def discover_workflow_specs(root: Path) -> list[WorkflowSpec]:
    specs, _ = discover_workflow_index(root)
    return specs


def discover_workflow_issues(root: Path) -> list[WorkflowDiscoveryIssue]:
    _, issues = discover_workflow_index(root)
    return issues


def discover_workflow_index(root: Path) -> tuple[list[WorkflowSpec], list[WorkflowDiscoveryIssue]]:
    repo_dir = root / "workflows"
    local_dir = root / ".openjet" / "workflows"
    discovered: dict[str, WorkflowSpec] = {}
    issues: list[WorkflowDiscoveryIssue] = []
    for path in sorted(repo_dir.glob("*.md")):
        try:
            spec = parse_workflow_markdown(path, source="repo")
        except ValueError as exc:
            issues.append(WorkflowDiscoveryIssue(path=path.resolve(), error=str(exc)))
            continue
        discovered[spec.name.lower()] = spec
    for path in sorted(local_dir.glob("*.md")):
        try:
            spec = parse_workflow_markdown(path, source="local_override")
        except ValueError as exc:
            issues.append(WorkflowDiscoveryIssue(path=path.resolve(), error=str(exc)))
            continue
        discovered[spec.name.lower()] = spec
    return sorted(discovered.values(), key=lambda item: item.name.lower()), issues


def load_workflow_spec(root: Path, name: str) -> WorkflowSpec:
    needle = name.strip().lower()
    if not needle:
        raise ValueError("workflow name is required")
    for spec in discover_workflow_specs(root):
        if spec.name.lower() == needle:
            return spec
    raise ValueError(f"unknown workflow: {name}")


def parse_workflow_markdown(path: Path, *, source: str = "repo") -> WorkflowSpec:
    text = path.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(text, path=path)
    name = str(metadata.get("name") or path.stem).strip()
    if not name:
        raise ValueError(f"workflow name cannot be empty: {path}")
    mode = str(metadata.get("mode") or "chat").strip().lower() or "chat"
    if mode not in VALID_WORKFLOW_MODES:
        raise ValueError(f"workflow mode must be one of: {', '.join(sorted(VALID_WORKFLOW_MODES))}")
    return WorkflowSpec(
        name=name,
        path=path.resolve(),
        body=body.strip(),
        mode=mode,
        devices=_normalize_string_list(metadata.get("devices"), field_name="devices"),
        interval_seconds=_normalize_interval(metadata.get("interval_seconds")),
        allow_shell=bool(metadata.get("allow_shell", False)),
        require_plan=bool(metadata.get("require_plan", False)),
        require_verification=_normalize_optional_bool(metadata.get("require_verification"), field_name="require_verification"),
        skills=_normalize_string_list(metadata.get("skills"), field_name="skills"),
        files=_normalize_string_list(metadata.get("files"), field_name="files"),
        source=source,
    )


def _parse_frontmatter(text: str, *, path: Path) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    closing_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        raise ValueError(f"workflow frontmatter is not terminated: {path}")
    frontmatter = "\n".join(lines[1:closing_index]).strip()
    body = "\n".join(lines[closing_index + 1 :])
    if not frontmatter:
        return {}, body
    try:
        loaded = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid workflow frontmatter: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"workflow frontmatter must be a mapping: {path}")
    return loaded, body


def _normalize_string_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"workflow {field_name} must be a string or list of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return tuple(normalized)


def _normalize_interval(value: object) -> int | None:
    if value in {None, ""}:
        return None
    if not isinstance(value, int):
        raise ValueError("workflow interval_seconds must be an integer")
    if value <= 0:
        raise ValueError("workflow interval_seconds must be greater than zero")
    return value


def _normalize_optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"workflow {field_name} must be boolean")
