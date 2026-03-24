from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..device_sources import DEVICE_TOOL_NAMES, ensure_devices_registry, format_device_registry_prompt
from ..executor import load_file
from ..harness import HarnessState, build_turn_context, set_preferred_skills, update_state_for_user_message
from ..runtime_limits import MIN_TOKEN_BUDGET, derive_context_budget, read_memory_snapshot
from .bindings import WorkflowBindings
from .specs import WorkflowSpec

READ_ONLY_WORKFLOW_TOOLS = {
    "read_file",
    "load_file",
    "glob",
    "grep",
    "list_directory",
    "system_info",
}
MAX_PRELOAD_FILE_TOKENS = 1200
PRELOAD_RESERVE_TOKENS = 384


@dataclass(frozen=True)
class WorkflowContext:
    messages: list[dict[str, str]]
    docs_loaded: tuple[str, ...]
    preloaded_files: tuple[str, ...]
    registry_path: Path | None


def allowed_tools_for_workflow(spec: WorkflowSpec) -> set[str]:
    allowed = set(READ_ONLY_WORKFLOW_TOOLS) | set(DEVICE_TOOL_NAMES)
    if spec.allow_shell:
        allowed.add("shell")
    return allowed


async def build_workflow_context(
    root: Path,
    spec: WorkflowSpec,
    bindings: WorkflowBindings,
    *,
    cfg: Mapping[str, object] | None,
    current_context_tokens: int,
    effective_window: int,
) -> WorkflowContext:
    state = update_state_for_user_message(
        HarnessState(),
        f"Run workflow {spec.name}",
        mode=spec.mode,
        files=list(spec.files),
    )
    if spec.skills:
        state = set_preferred_skills(state, list(spec.skills))
    harness_context = build_turn_context(
        root=root,
        state=state,
        current_context_tokens=current_context_tokens,
        effective_window=effective_window,
        memory_snapshot=read_memory_snapshot(),
        layered_config=dict(cfg.get("layered_context") or {}) if isinstance(cfg, Mapping) else None,
    )
    messages = list(harness_context.messages)
    messages.append({"role": "system", "content": _render_workflow_doc(spec)})
    registry_path = ensure_devices_registry(root, cfg=cfg)
    if registry_path is not None:
        messages.append(
            {
                "role": "system",
                "content": format_device_registry_prompt(
                    registry_path,
                    referenced_ids=bindings.primary_refs or bindings.requested_ids,
                ),
            }
        )
    preload_messages, preloaded_files = await _preload_workflow_files(
        root,
        spec,
        current_context_tokens=current_context_tokens + harness_context.docs_tokens,
        effective_window=effective_window,
    )
    messages.extend(preload_messages)
    return WorkflowContext(
        messages=messages,
        docs_loaded=tuple(harness_context.docs_loaded),
        preloaded_files=tuple(preloaded_files),
        registry_path=registry_path,
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
