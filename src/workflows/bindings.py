from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..device_sources import DeviceSource, resolve_device_source
from .specs import WorkflowSpec
from .state import load_workflow_assignment


@dataclass(frozen=True)
class WorkflowBindings:
    source: str
    requested_ids: tuple[str, ...]
    sources: tuple[DeviceSource, ...]

    @property
    def primary_refs(self) -> tuple[str, ...]:
        return tuple(source.primary_ref for source in self.sources)


def resolve_workflow_bindings(
    root: Path,
    spec: WorkflowSpec,
    cfg: Mapping[str, object] | None,
    *,
    cli_device_ids: Sequence[str] | None = None,
) -> WorkflowBindings:
    requested_ids, source = _binding_candidates(root, spec, cli_device_ids=cli_device_ids)
    sources = validate_workflow_device_ids(cfg, requested_ids)
    return WorkflowBindings(source=source, requested_ids=requested_ids, sources=sources)


def validate_workflow_device_ids(
    cfg: Mapping[str, object] | None,
    device_ids: Sequence[str],
) -> tuple[DeviceSource, ...]:
    normalized = _normalize_requested_ids(device_ids)
    resolved: list[DeviceSource] = []
    seen: set[str] = set()
    for device_id in normalized:
        source = resolve_device_source(device_id, cfg)
        if source is None:
            raise ValueError(f"unknown device reference: {device_id}")
        if not source.enabled:
            raise ValueError(f"device {source.primary_ref} is disabled")
        key = source.primary_ref.lower()
        if key in seen:
            continue
        seen.add(key)
        resolved.append(source)
    return tuple(resolved)


def _binding_candidates(
    root: Path,
    spec: WorkflowSpec,
    *,
    cli_device_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], str]:
    cli_ids = _normalize_requested_ids(cli_device_ids or ())
    if cli_ids:
        return cli_ids, "cli"
    assigned_ids = load_workflow_assignment(root, spec.name)
    if assigned_ids:
        return assigned_ids, "assignment"
    return _normalize_requested_ids(spec.devices), "workflow"


def _normalize_requested_ids(device_ids: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in device_ids:
        text = str(item).strip().lstrip("@")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return tuple(normalized)
