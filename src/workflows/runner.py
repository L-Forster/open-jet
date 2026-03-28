from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from ..config import load_config
from ..device_sources import ensure_devices_registry
from ..executor import load_file
from ..harness import HarnessState, build_turn_context, set_preferred_skills, update_state_for_user_message
from ..runtime_limits import MIN_TOKEN_BUDGET, derive_context_budget, read_memory_snapshot
from ..sdk import OpenJetSession
from ..tool_executor import format_tool_args
from ..tools.registry import workflow_default_tool_names, workflow_optional_tool_names
from .bindings import WorkflowBindings, resolve_workflow_bindings
from .specs import WorkflowSpec


WORKFLOW_BASE_TOOLS = set(workflow_default_tool_names())
WORKFLOW_OPTIONAL_TOOLS = set(workflow_optional_tool_names())
MAX_PRELOAD_FILE_TOKENS = 1200
PRELOAD_RESERVE_TOKENS = 384


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


@dataclass(frozen=True)
class WorkflowTurnContext:
    messages: list[dict[str, str]]
    docs_loaded: tuple[str, ...]
    preloaded_files: tuple[str, ...] = field(default_factory=tuple)


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
        context = await _build_workflow_turn_context(
            root,
            spec,
            bindings,
            cfg=resolved_cfg,
            current_context_tokens=(
                session.agent.persistent_context_tokens()
                + session.agent.runtime_overhead_tokens()
            ),
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


def allowed_tools_for_workflow(spec: WorkflowSpec) -> set[str]:
    allowed = set(WORKFLOW_BASE_TOOLS)
    if spec.allow_shell:
        allowed |= set(WORKFLOW_OPTIONAL_TOOLS)
    return allowed


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


async def _build_workflow_turn_context(
    root: Path,
    spec: WorkflowSpec,
    bindings: WorkflowBindings,
    *,
    cfg: Mapping[str, object] | None,
    current_context_tokens: int,
    effective_window: int,
):
    state = update_state_for_user_message(
        HarnessState(),
        f"Run workflow {spec.name}",
        mode=spec.mode,
        files=list(spec.files),
    )
    if spec.skills:
        state = set_preferred_skills(state, list(spec.skills))
    workflow_messages = [{"role": "system", "content": _render_workflow_doc(spec)}]
    preloaded_messages, preloaded_files = await _preload_workflow_files(
        root,
        spec,
        current_context_tokens=current_context_tokens,
        effective_window=effective_window,
    )
    context = build_turn_context(
        root=root,
        state=state,
        current_context_tokens=current_context_tokens,
        effective_window=effective_window,
        memory_snapshot=read_memory_snapshot(),
        layered_config=dict(cfg.get("layered_context") or {}) if isinstance(cfg, Mapping) else None,
        cfg=cfg,
        referenced_device_ids=bindings.primary_refs or bindings.requested_ids,
        extra_system_messages=[*workflow_messages, *preloaded_messages],
        extra_docs_loaded=preloaded_files,
    )
    return WorkflowTurnContext(
        messages=context.messages,
        docs_loaded=tuple(context.docs_loaded),
        preloaded_files=tuple(preloaded_files),
    )


async def _preload_workflow_files(
    root: Path,
    spec: WorkflowSpec,
    *,
    current_context_tokens: int,
    effective_window: int,
) -> tuple[list[dict[str, str]], list[str]]:
    if not spec.files:
        return [], []
    budget = derive_context_budget(effective_window)
    remaining_budget = max(0, budget.prompt_tokens - current_context_tokens - PRELOAD_RESERVE_TOKENS)
    if remaining_budget < MIN_TOKEN_BUDGET * len(spec.files):
        raise ValueError(
            "workflow file preload budget is too small for the configured files; "
            "reduce `files:` or increase the context window"
        )
    messages: list[dict[str, str]] = []
    preloaded: list[str] = []
    files_left = len(spec.files)
    for raw_path in spec.files:
        budget_for_file = min(
            MAX_PRELOAD_FILE_TOKENS,
            max(MIN_TOKEN_BUDGET, remaining_budget // max(1, files_left)),
        )
        path = Path(raw_path)
        resolved = path if path.is_absolute() else (root / path)
        loaded = await load_file(str(resolved), max_tokens=budget_for_file)
        if not loaded.ok:
            raise ValueError(f"workflow file preload failed for {raw_path}: {loaded.detail}")
        messages.append(
            {
                "role": "system",
                "content": (
                    "Workflow-configured file context:\n"
                    f"path: {loaded.path}\n"
                    f"tokens_estimated: {loaded.estimated_tokens}\n"
                    f"tokens_loaded: {loaded.returned_tokens}\n"
                    f"token_budget: {loaded.token_budget}\n"
                    f"truncated: {'yes' if loaded.truncated else 'no'}\n"
                    "content:\n"
                    f"{loaded.content}"
                ),
            }
        )
        preloaded.append(loaded.path)
        remaining_budget = max(0, remaining_budget - loaded.returned_tokens)
        files_left -= 1
    return messages, preloaded


def _render_workflow_doc(spec: WorkflowSpec) -> str:
    lines = [
        f"WORKFLOW DOCUMENT: {spec.name}",
        f"path: {spec.path}",
        f"mode: {spec.mode}",
        f"allow_shell: {'true' if spec.allow_shell else 'false'}",
        f"devices: {', '.join(spec.devices) if spec.devices else 'none'}",
        f"skills: {', '.join(spec.skills) if spec.skills else 'none'}",
        f"files: {', '.join(spec.files) if spec.files else 'none'}",
        "instructions:",
        spec.body.strip() or "(empty workflow body)",
    ]
    return "\n".join(lines)
