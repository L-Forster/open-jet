from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from ..config import load_config
from ..device_sources import ensure_devices_registry
from ..sdk import OpenJetSession
from ..tool_executor import format_tool_args
from .bindings import WorkflowBindings, resolve_workflow_bindings
from .context import allowed_tools_for_workflow, build_workflow_context
from .specs import WorkflowSpec


@dataclass(frozen=True)
class WorkflowRunResult:
    name: str
    spec_path: str
    success: bool
    started_at: str
    finished_at: str
    binding_source: str
    bound_devices: tuple[str, ...] = field(default_factory=tuple)
    registry_path: str | None = None
    response_text: str = ""
    tool_calls: tuple[str, ...] = field(default_factory=tuple)
    tool_details: tuple[str, ...] = field(default_factory=tuple)
    payload_paths: tuple[str, ...] = field(default_factory=tuple)
    docs_loaded: tuple[str, ...] = field(default_factory=tuple)
    preloaded_files: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None


async def run_workflow(
    root: Path,
    spec: WorkflowSpec,
    *,
    cfg: Mapping[str, object] | None = None,
    cli_device_ids: list[str] | None = None,
) -> WorkflowRunResult:
    started_at = _utcnow()
    resolved_cfg = dict(cfg or load_config())
    session: OpenJetSession | None = None
    context = None
    bindings = WorkflowBindings(source="workflow", requested_ids=(), sources=())
    try:
        ensure_devices_registry(root, cfg=resolved_cfg)
        bindings = resolve_workflow_bindings(root, spec, resolved_cfg, cli_device_ids=cli_device_ids)
        session = await OpenJetSession.create(
            cfg=resolved_cfg,
            system_prompt="",
            root=root,
            allowed_tools=allowed_tools_for_workflow(spec),
        )
        context = await build_workflow_context(
            root,
            spec,
            bindings,
            cfg=resolved_cfg,
            current_context_tokens=session.agent.estimated_context_tokens(),
            effective_window=session.agent.context_window_tokens or 2048,
        )
        session.add_turn_context(context.messages)
        response = await session.run(_workflow_run_prompt(spec, bindings))
        registry_path = ensure_devices_registry(root, cfg=resolved_cfg)
        tool_calls = tuple(result.tool_call.name for result in response.tool_results)
        tool_details = tuple(
            f"{result.tool_call.name}: {format_tool_args(result.tool_call)}"
            for result in response.tool_results
        )
        payload_paths = tuple(
            sorted(
                {
                    str(result.meta.get("payload_ref")).strip()
                    for result in response.tool_results
                    if str(result.meta.get("payload_ref") or "").strip()
                }
            )
        )
        return WorkflowRunResult(
            name=spec.name,
            spec_path=str(spec.path),
            success=True,
            started_at=started_at,
            finished_at=_utcnow(),
            binding_source=bindings.source,
            bound_devices=bindings.primary_refs,
            registry_path=str(registry_path) if registry_path else None,
            response_text=response.text.strip(),
            tool_calls=tool_calls,
            tool_details=tool_details,
            payload_paths=payload_paths,
            docs_loaded=context.docs_loaded if context else (),
            preloaded_files=context.preloaded_files if context else (),
        )
    except Exception as exc:
        registry_path = ensure_devices_registry(root, cfg=resolved_cfg)
        return WorkflowRunResult(
            name=spec.name,
            spec_path=str(spec.path),
            success=False,
            started_at=started_at,
            finished_at=_utcnow(),
            binding_source=bindings.source,
            bound_devices=bindings.primary_refs,
            registry_path=str(registry_path) if registry_path else None,
            docs_loaded=context.docs_loaded if context else (),
            preloaded_files=context.preloaded_files if context else (),
            error=str(exc).strip() or type(exc).__name__,
        )
    finally:
        if session is not None:
            await session.close()


def _workflow_run_prompt(spec: WorkflowSpec, bindings: WorkflowBindings) -> str:
    lines = [
        f"Run workflow `{spec.name}` now.",
        "Use the workflow document and device registry already loaded in context.",
        "Read or capture device data only when relevant to the workflow instructions.",
    ]
    if bindings.primary_refs:
        lines.append(f"Bound device ids for this run: {', '.join(bindings.primary_refs)}.")
    else:
        lines.append("No device ids are explicitly bound for this run.")
    lines.append("Return a concise run summary with any useful observations or failures.")
    return "\n".join(lines)


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
