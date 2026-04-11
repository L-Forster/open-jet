from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import load_config, save_config
from .device_sources import (
    collect_device_observation,
    list_device_sources,
    resolve_device_source,
    set_device_enabled,
    sync_devices_registry,
)
from .executor import (
    DEFAULT_SHELL_TIMEOUT_SECONDS,
    EditFileResult,
    _normalize_tool_path,
    _validate_edited_content,
    edit_file,
    glob_files,
    grep_files,
    list_directory,
    load_file,
    read_file,
    read_system_info,
    run_shell,
    write_file,
)
from .multimodal import content_to_plain_text
from .observation import ObservationStore, observation_to_agent_content
from .peripherals import PeripheralKind, PeripheralTransport
from .peripherals.system import device_discovery_hint
from .persistent_memory import update_persistent_memory
from .runtime_protocol import ToolCall
from .tools.registry import bind_tool_executor, get_tool_spec

if TYPE_CHECKING:
    from .swap_manager import SwapManager


@dataclass(frozen=True)
class ToolExecutionResult:
    output: str
    meta: dict
    context_content: Any | None = None

    @property
    def ok(self) -> bool:
        return bool(self.meta.get("ok", False))


class ToolArgumentError(ValueError):
    pass


_swap_manager: SwapManager | None = None
_MISSING = object()
_DEVICE_TOOLS = {
    "camera_snapshot": (PeripheralKind.CAMERA, None, True, None),
    "microphone_record": (PeripheralKind.MICROPHONE, "duration_seconds", True, None),
    "sensor_read": (PeripheralKind.SENSOR, None, False, PeripheralTransport.GPIO),
    "gpio_read": (PeripheralKind.SENSOR, None, False, PeripheralTransport.GPIO),
}
_UNIFIED_DIFF_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def set_swap_manager(manager: SwapManager | None) -> None:
    global _swap_manager
    _swap_manager = manager


def get_swap_manager() -> SwapManager | None:
    return _swap_manager


async def execute_tool(tool_call: ToolCall) -> ToolExecutionResult:
    args = tool_call.arguments
    if not isinstance(args, dict):
        return _invalid(tool_call.name)
    try:
        spec = get_tool_spec(tool_call.name)
        if spec is None or spec.executor is None:
            return ToolExecutionResult(output=f"Unknown tool: {tool_call.name}", meta={"ok": False})
        result = spec.executor(args)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, ToolExecutionResult):
            raise TypeError(f"tool {tool_call.name} returned invalid result type: {type(result).__name__}")
        return result
    except ToolArgumentError as exc:
        return _invalid(tool_call.name, str(exc))
    except ValueError as exc:
        return _error(str(exc))


def format_tool_args(tool_call: ToolCall) -> str:
    if not isinstance(tool_call.arguments, dict):
        return str(tool_call.arguments)
    args = tool_call.arguments
    match tool_call.name:
        case "shell":
            command = args.get("command", str(args))
            timeout_seconds = args.get("timeout_seconds")
            mode = str(args.get("resource_mode", "normal") or "normal").strip()
            return f"$ {command} (timeout: {timeout_seconds}s, mode: {mode})" if isinstance(timeout_seconds, int) else f"$ {command} (mode: {mode})"
        case "system_info":
            return f"scope={str(args.get('scope', 'summary') or 'summary').strip()}"
        case "device_list":
            kind = str(args.get("kind", "") or "").strip()
            return f"kind={kind}" if kind else "active devices"
        case "camera_snapshot" | "microphone_record" | "sensor_read" | "gpio_read":
            source = str(args.get("source", "") or "").strip() or "<auto>"
            duration = args.get("duration_seconds")
            return f"{source} ({duration}s)" if tool_call.name == "microphone_record" and isinstance(duration, int) else source
        case "microphone_set_enabled":
            source = str(args.get("source", "") or "").strip() or "<auto>"
            return f"{source} -> {'on' if args.get('enabled') else 'off'}"
        case "memory":
            location = str(args.get("location", "project") or "project").strip()
            action = str(args.get("action", "") or "").strip()
            scope = str(args.get("scope", "") or "").strip()
            return " ".join(part for part in (action, location, scope) if part)
        case "read_file" | "write_file" | "load_file" | "edit_file":
            return str(args.get("path", str(args)))
        case "glob" | "grep":
            return str(args.get("pattern", str(args)))
        case "list_directory":
            return str(args.get("path", ".") or ".")
        case _:
            return str(args)


async def _shell_result(args: dict[str, Any]) -> ToolExecutionResult:
    command = _str_arg(args, "command", required=True, allow_empty=False)
    timeout_seconds = _int_arg(args, "timeout_seconds", allow_none=True)
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ToolArgumentError("timeout_seconds must be > 0")
    resource_mode = str(args.get("resource_mode", "normal") or "normal").strip().lower()
    if resource_mode not in {"normal", "auto", "unload_first"}:
        raise ToolArgumentError("resource_mode must be normal, auto, or unload_first")
    estimated_ram_mb = _int_arg(args, "estimated_ram_mb", allow_none=True)
    estimated_vram_mb = _int_arg(args, "estimated_vram_mb", allow_none=True)
    reload_delay_seconds = _int_arg(
        args,
        "reload_delay_seconds",
        default=0,
        minimum=0,
        minimum_message="reload_delay_seconds must be >= 0",
    )
    timeout = timeout_seconds or DEFAULT_SHELL_TIMEOUT_SECONDS
    plan = _swap_manager.plan_unload(estimated_ram_mb or 0) if _swap_manager is not None else None
    use_swap = resource_mode == "unload_first" and _swap_manager is not None
    if resource_mode == "auto" and plan is not None:
        use_swap = plan.unload_recommended

    if use_swap and _swap_manager is not None:
        swap = await _swap_manager.run_with_swap(command, timeout=timeout, reload_delay_seconds=reload_delay_seconds)
        res = swap.exec_result
        swap_meta = {
            "swap_mode": resource_mode,
            "swap_available": True,
            "swap_attempted": True,
            "swap_plan_recommended": True if plan is None else plan.unload_recommended,
            "estimated_ram_mb": estimated_ram_mb,
            "estimated_vram_mb": estimated_vram_mb,
            "reload_delay_seconds": reload_delay_seconds,
            "swapped": swap.unloaded,
            "swap_reload_ok": swap.reload_ok,
            "swap_restore_ok": swap.restore_ok,
            "swap_duration_s": swap.swap_duration_s,
        }
    else:
        res = await run_shell(command, timeout_seconds=timeout)
        swap_meta = {
            "swap_mode": resource_mode,
            "swap_available": _swap_manager is not None,
            "swap_attempted": False,
            "swap_plan_recommended": plan.unload_recommended if plan is not None else False,
            "estimated_ram_mb": estimated_ram_mb,
            "estimated_vram_mb": estimated_vram_mb,
            "reload_delay_seconds": reload_delay_seconds,
        }

    resource_hint = _resource_failure_hint(res)
    output = res.summary
    for note in (_swap_summary_line(swap_meta), resource_hint):
        if note:
            output = f"{output}\n{note}"
    return ToolExecutionResult(
        output=output,
        meta={
            "ok": res.ok,
            "exit_code": res.exit_code,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "timed_out": res.timed_out,
            "timeout_seconds": res.timeout_seconds,
            "resource_failure_detected": bool(resource_hint),
            **swap_meta,
        },
    )


def _device_tool_result(tool_name: str, args: dict[str, Any]) -> ToolExecutionResult:
    kind, duration_field, supports_output_path, required_transport = _DEVICE_TOOLS[tool_name]
    kwargs: dict[str, Any] = {"source_ref": _str_arg(args, "source"), "expected_kind": kind}
    if duration_field is not None:
        kwargs["duration_seconds"] = _int_arg(args, duration_field, default=5, minimum=1, minimum_message=f"{duration_field} must be > 0")
    if supports_output_path:
        kwargs["output_path"] = _str_arg(args, "output_path")
    if required_transport is not None:
        kwargs["required_transport"] = required_transport
    try:
        return _device_observation_result(**kwargs)
    except RuntimeError as exc:
        return _error(str(exc))


def _microphone_set_enabled_result(args: dict[str, Any]) -> ToolExecutionResult:
    enabled = args.get("enabled", _MISSING)
    if not isinstance(enabled, bool):
        raise ToolArgumentError("enabled must be boolean")
    cfg = _tool_config()
    source = _select_device_source(cfg, source_ref=_str_arg(args, "source"), expected_kind=PeripheralKind.MICROPHONE)
    updated = set_device_enabled(cfg, reference=source.primary_ref, enabled=enabled)
    save_config(cfg)
    registry_path = _sync_tool_device_registry(cfg)
    state = "enabled" if updated.enabled else "disabled"
    return ToolExecutionResult(
        output=f"Microphone @{updated.primary_ref} is now {state}.",
        meta={"ok": True, "source_ref": updated.primary_ref, "enabled": updated.enabled, "registry_path": str(registry_path) if registry_path else None},
    )


async def _memory_result(args: dict[str, Any]) -> ToolExecutionResult:
    location = args.get("location", "project")
    scope = args.get("scope", "")
    action = args.get("action", "")
    content = args.get("content", "")
    if not isinstance(scope, str) or not scope.strip() or not isinstance(action, str) or not action.strip():
        raise ToolArgumentError("required: scope, action")
    if not isinstance(location, str):
        raise ToolArgumentError("location must be string")
    if not isinstance(content, str):
        raise ToolArgumentError("content must be string")
    text = await update_persistent_memory(
        Path.cwd(),
        scope=scope,
        action=action,
        content=content,
        location=location,
    )
    ok = not text.startswith("Skipped ")
    status = "completed" if ok else "skipped"
    return ToolExecutionResult(
        output=text,
        meta={"ok": ok, "status": status, "location": location, "scope": scope, "action": action},
    )


async def _control_tool_result(args: dict[str, Any]) -> ToolExecutionResult:
    del args
    return ToolExecutionResult(
        output="Control tool should be handled by the session runtime before execution.",
        meta={"ok": False, "status": "control_tool_unhandled"},
    )


async def _load_file_result(args: dict[str, Any]) -> ToolExecutionResult:
    loaded = await load_file(_str_arg(args, "path", required=True, allow_empty=False), max_tokens=_int_arg(args, "max_tokens", allow_none=True))
    if not loaded.ok:
        return ToolExecutionResult(output=loaded.detail, meta={"ok": False})
    return ToolExecutionResult(
        output=f"{loaded.summary}\ncontent:\n{loaded.content}",
        meta={
            "ok": True,
            "truncated": loaded.truncated,
            "estimated_tokens": loaded.estimated_tokens,
            "returned_tokens": loaded.returned_tokens,
            "token_budget": loaded.token_budget,
            "mem_available_mb": loaded.mem_available_mb,
        },
    )


async def _edit_file_result(args: dict[str, Any]) -> ToolExecutionResult:
    path = _str_arg(args, "path", required=True, allow_empty=False)
    patch = args.get("patch")
    if patch is not None:
        if not isinstance(patch, str) or not patch.strip():
            raise ToolArgumentError("patch must be a non-empty string")
        if _looks_like_unified_diff_patch(patch):
            result = _apply_unified_diff_patch(path, patch)
        else:
            result = await edit_file(path, patch=patch, return_result=True)
    else:
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        if not isinstance(old_string, str) or not old_string:
            raise ToolArgumentError("required: patch or old_string")
        if not isinstance(new_string, str):
            raise ToolArgumentError("required: new_string")
        result = await edit_file(path, old_string=old_string, new_string=new_string, replace_all=bool(args.get("replace_all", False)), return_result=True)
    return ToolExecutionResult(
        output=result.output,
        meta={
            "ok": result.ok,
            "internal_retry": result.internal_retry,
            "replacements": result.replacements,
            "match_strategy": result.match_strategy,
            "validation_error": result.validation_error,
        },
    )


def _looks_like_unified_diff_patch(patch: str) -> bool:
    return any(
        line.startswith("@@ ") or line.startswith("--- ") or line.startswith("+++ ")
        for line in patch.splitlines()
    )


def _apply_unified_diff_patch(path: str, patch: str) -> EditFileResult:
    raw_path = path.strip()
    if not raw_path:
        return EditFileResult(ok=False, output="Error: path is empty.")

    p = _normalize_tool_path(raw_path)
    if not p.exists():
        return EditFileResult(ok=False, output=f"Error: file not found: {path}")
    if p.is_dir():
        return EditFileResult(ok=False, output=f"Error: path is a directory: {path}")

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        hunks = _parse_unified_diff_patch(patch)
        new_content = _apply_unified_diff_hunks(content, hunks, path=path)
    except ValueError as exc:
        return EditFileResult(ok=False, output=f"Error: {exc}")
    except Exception as exc:
        return EditFileResult(ok=False, output=f"Error reading {path}: {exc}")

    validation_error = _validate_edited_content(p, new_content)
    if validation_error:
        return EditFileResult(
            ok=False,
            output=validation_error,
            internal_retry=True,
            replacements=len(hunks),
            match_strategy="line-numbered-diff",
            validation_error=validation_error,
        )

    try:
        p.write_text(new_content, encoding="utf-8")
    except Exception as exc:
        return EditFileResult(ok=False, output=f"Error writing {path}: {exc}")

    return EditFileResult(
        ok=True,
        output=f"Edited {path}: {len(hunks)} replacement(s) made via line-numbered-diff.",
        replacements=len(hunks),
        match_strategy="line-numbered-diff",
    )


def _parse_unified_diff_patch(patch: str) -> list[tuple[int, int, str, str]]:
    lines = patch.splitlines(keepends=True)
    idx = 0
    hunks: list[tuple[int, int, str, str]] = []

    while idx < len(lines):
        line = lines[idx]
        if line.startswith(("diff ", "index ", "--- ", "+++ ")) or not line.strip():
            idx += 1
            continue
        match = _UNIFIED_DIFF_HUNK_RE.match(line.rstrip("\r\n"))
        if not match:
            raise ValueError("patch must contain unified diff hunks with @@ -old,+new @@ headers.")
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        idx += 1
        hunk_lines: list[str] = []
        while idx < len(lines):
            candidate = lines[idx]
            if candidate.startswith("@@ "):
                break
            if candidate.startswith(("diff ", "index ", "--- ", "+++ ")):
                break
            hunk_lines.append(candidate)
            idx += 1
        old_parts: list[str] = []
        new_parts: list[str] = []
        for hunk_line in hunk_lines:
            if hunk_line.startswith("\\ No newline at end of file"):
                continue
            if not hunk_line:
                raise ValueError("invalid unified diff hunk line.")
            prefix = hunk_line[0]
            body = hunk_line[1:]
            if prefix in {" ", "-"}:
                old_parts.append(body)
            if prefix in {" ", "+"}:
                new_parts.append(body)
            if prefix not in {" ", "-", "+"}:
                raise ValueError("invalid unified diff hunk line prefix.")
        hunks.append((old_start, old_count, "".join(old_parts), "".join(new_parts)))

    if not hunks:
        raise ValueError("patch did not contain any unified diff hunks.")
    return hunks


def _apply_unified_diff_hunks(
    content: str,
    hunks: list[tuple[int, int, str, str]],
    *,
    path: str,
) -> str:
    lines = content.splitlines(keepends=True)
    for old_start, old_count, old_text, new_text in reversed(hunks):
        start_idx = max(0, old_start - 1)
        end_idx = start_idx + old_count
        if end_idx > len(lines):
            raise ValueError(f"hunk range is out of bounds for {path}")
        existing = "".join(lines[start_idx:end_idx])
        if existing != old_text:
            raise ValueError(f"hunk old text did not match {path} at line {old_start}")
        replacement = new_text.splitlines(keepends=True)
        lines[start_idx:end_idx] = replacement
    return "".join(lines)


async def _text_result_async(func, *func_args, error_prefix: str, **func_kwargs) -> ToolExecutionResult:
    text = await func(*func_args, **func_kwargs)
    return ToolExecutionResult(output=text, meta={"ok": not text.startswith(error_prefix)})


def _invalid(tool_name: str, detail: str | None = None) -> ToolExecutionResult:
    return ToolExecutionResult(output=f"Error: invalid arguments for {tool_name}{f' ({detail})' if detail else ''}", meta={"ok": False})


def _error(detail: str) -> ToolExecutionResult:
    return ToolExecutionResult(output=f"Error: {detail}", meta={"ok": False})


def _str_arg(args: dict[str, Any], name: str, *, required: bool = False, allow_empty: bool = True, default: object = _MISSING) -> str | None:
    value = args.get(name, default)
    if value is _MISSING:
        if required:
            raise ToolArgumentError(f"required: {name}")
        return None
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolArgumentError(f"{name} must be string")
    if required and not allow_empty and not value.strip():
        raise ToolArgumentError(f"required: {name}")
    return value


def _int_arg(
    args: dict[str, Any],
    name: str,
    *,
    default: object = _MISSING,
    allow_none: bool = False,
    minimum: int | None = None,
    minimum_message: str | None = None,
) -> int | None:
    value = args.get(name, default)
    if value is _MISSING:
        if allow_none:
            return None
        raise ToolArgumentError(f"required: {name}")
    if value is None:
        if allow_none:
            return None
        raise ToolArgumentError(f"{name} must be int")
    if not isinstance(value, int):
        raise ToolArgumentError(f"{name} must be int")
    if minimum is not None and value < minimum:
        raise ToolArgumentError(minimum_message or f"{name} must be >= {minimum}")
    return value


def _swap_summary_line(meta: dict[str, Any]) -> str:
    mode = str(meta.get("swap_mode", "normal") or "normal")
    if meta.get("swapped"):
        delay = meta.get("reload_delay_seconds", 0)
        duration = meta.get("swap_duration_s")
        return (
            f"[swap path used: mode={mode}"
            f"{f', reload_delay={delay}s' if isinstance(delay, int) and delay > 0 else ''}"
            f"{f', total={duration}s' if isinstance(duration, (int, float)) else ''}]"
        )
    if meta.get("swap_attempted") and meta.get("swap_available"):
        return "[swap skipped: unload was requested but the swap cycle could not complete, so the command ran directly]"
    if mode == "unload_first":
        return "[swap unavailable: unload-first was requested but no swap-capable local runtime is active]"
    if mode == "auto" and meta.get("swap_plan_recommended"):
        return "[swap suggested: auto mode recommended unload, but no swap-capable local runtime is active]"
    return ""


def _resource_failure_hint(result) -> str:
    if result.ok:
        return ""
    haystack = "\n".join(part for part in (result.stdout, result.stderr) if part).lower()
    if any(
        pattern in haystack
        for pattern in (
            "out of memory",
            "cuda out of memory",
            "cuda error",
            "cublas_status_alloc_failed",
            "cannot allocate memory",
            "std::bad_alloc",
            "memoryerror",
            "killed",
        )
    ):
        return (
            "[resource failure detected: consider `system_info` and, if needed, retry the shell command "
            "with `resource_mode` set to `unload_first`]"
        )
    return ""


def _device_list_output(*, kind: str | None) -> tuple[str, dict[str, object]]:
    cfg = _tool_config()
    sources = list_device_sources(cfg)
    registry_path = _sync_tool_device_registry(cfg)
    parsed_kind = _parse_device_kind(kind)
    if parsed_kind is not None:
        sources = [source for source in sources if source.device.kind is parsed_kind]
    sources = [source for source in sources if source.enabled]
    if not sources:
        label = f"{parsed_kind.value} " if parsed_kind is not None else ""
        message = f"No active {label}device sources detected.".strip()
        hint = device_discovery_hint()
        if hint:
            message = f"{message}\n{hint}"
        return message, {"ok": True, "count": 0, "registry_path": str(registry_path) if registry_path else None}
    lines = ["Discovered active device sources:"]
    for source in sources:
        refs = ", ".join(f"@{ref}" for ref in source.refs)
        path = f" | path={source.device.path}" if source.device.path else ""
        lines.append(
            f"- @{source.primary_ref}: {source.device.label} | kind={source.device.kind.value} "
            f"| transport={source.device.transport.value} | state={'enabled' if source.enabled else 'disabled'} | refs={refs}{path}"
        )
    return "\n".join(lines), {
        "ok": True,
        "count": len(sources),
        "kind": parsed_kind.value if parsed_kind else None,
        "registry_path": str(registry_path) if registry_path else None,
    }


def _device_observation_result(
    *,
    source_ref: str | None,
    expected_kind: PeripheralKind,
    duration_seconds: int = 5,
    output_path: str | None = None,
    required_transport: PeripheralTransport | None = None,
) -> ToolExecutionResult:
    cfg = _tool_config()
    source = _select_device_source(cfg, source_ref=source_ref, expected_kind=expected_kind)
    if required_transport is not None and source.device.transport is not required_transport:
        raise ValueError(
            f"{expected_kind.value} source {source.primary_ref} uses {source.device.transport.value}; "
            f"only {required_transport.value} is supported right now"
        )
    store = ObservationStore()
    observation = collect_device_observation(source, store=store, cfg=cfg, duration_seconds=duration_seconds, output_path=output_path)
    registry_path = _sync_tool_device_registry(cfg, store=store)
    context_content = observation_to_agent_content(observation, store=store)
    return ToolExecutionResult(
        output=content_to_plain_text(context_content),
        context_content=context_content,
        meta={
            "ok": True,
            "source_ref": source.primary_ref,
            "source_id": source.device.id,
            "kind": source.device.kind.value,
            "transport": source.device.transport.value,
            "payload_ref": observation.payload_ref,
            "registry_path": str(registry_path) if registry_path else None,
        },
    )


def _tool_config() -> dict[str, object]:
    try:
        cfg = load_config()
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _select_device_source(
    cfg: dict[str, object],
    *,
    source_ref: str | None,
    expected_kind: PeripheralKind,
):
    if isinstance(source_ref, str) and source_ref.strip():
        source = resolve_device_source(source_ref, cfg)
        if source is None:
            raise ValueError(f"unknown device reference: {source_ref}")
        if source.device.kind is not expected_kind:
            raise ValueError(f"source {source.primary_ref} is not a {expected_kind.value}")
        return source
    candidates = [source for source in list_device_sources(cfg) if source.device.kind is expected_kind]
    if not candidates:
        raise ValueError(f"no {expected_kind.value} sources detected")
    if len(candidates) > 1:
        raise ValueError(f"multiple {expected_kind.value} sources detected; specify source")
    return candidates[0]


def _parse_device_kind(kind: str | None) -> PeripheralKind | None:
    text = str(kind or "").strip().lower()
    if not text:
        return None
    try:
        return PeripheralKind(text)
    except ValueError as exc:
        raise ValueError("kind must be one of: camera, microphone, speaker, sensor") from exc


def _sync_tool_device_registry(cfg: dict[str, object], *, store: ObservationStore | None = None) -> Path | None:
    try:
        return sync_devices_registry(cfg, store=store)
    except Exception:
        return None


async def _system_info_result(args: dict[str, Any]) -> ToolExecutionResult:
    return await _text_result_async(read_system_info, scope=_str_arg(args, "scope"), error_prefix="Error:")


async def _read_file_result(args: dict[str, Any]) -> ToolExecutionResult:
    return await _text_result_async(
        read_file,
        _str_arg(args, "path", required=True, allow_empty=False),
        error_prefix="Error:",
    )


async def _write_file_result(args: dict[str, Any]) -> ToolExecutionResult:
    path = args.get("path", "")
    content = args.get("content", "")
    if not isinstance(path, str) or not path.strip() or not isinstance(content, str):
        raise ToolArgumentError("required: path, content")
    return await _text_result_async(write_file, path, content, error_prefix="Error")


async def _glob_result(args: dict[str, Any]) -> ToolExecutionResult:
    return await _text_result_async(
        glob_files,
        _str_arg(args, "pattern", required=True, allow_empty=False),
        path=args.get("path"),
        error_prefix="Error",
    )


async def _grep_result(args: dict[str, Any]) -> ToolExecutionResult:
    return await _text_result_async(
        grep_files,
        _str_arg(args, "pattern", required=True, allow_empty=False),
        path=args.get("path"),
        glob_filter=args.get("glob"),
        ignore_case=bool(args.get("ignore_case", False)),
        error_prefix="Error",
    )


async def _list_directory_result(args: dict[str, Any]) -> ToolExecutionResult:
    return await _text_result_async(list_directory, path=args.get("path"), error_prefix="Error")


def _device_list_result(args: dict[str, Any]) -> ToolExecutionResult:
    text, meta = _device_list_output(kind=_str_arg(args, "kind"))
    return ToolExecutionResult(output=text, meta=meta)


def _bind_tool_executors() -> None:
    bind_tool_executor("shell", _shell_result)
    bind_tool_executor("system_info", _system_info_result)
    bind_tool_executor("device_list", _device_list_result)
    bind_tool_executor("camera_snapshot", lambda args: _device_tool_result("camera_snapshot", args))
    bind_tool_executor("microphone_record", lambda args: _device_tool_result("microphone_record", args))
    bind_tool_executor("sensor_read", lambda args: _device_tool_result("sensor_read", args))
    bind_tool_executor("gpio_read", lambda args: _device_tool_result("gpio_read", args))
    bind_tool_executor("microphone_set_enabled", _microphone_set_enabled_result)
    bind_tool_executor("read_file", _read_file_result)
    bind_tool_executor("memory", _memory_result)
    bind_tool_executor("todo_write", _control_tool_result)
    bind_tool_executor("todo_complete", _control_tool_result)
    bind_tool_executor("todo_clear", _control_tool_result)
    bind_tool_executor("exit_plan_mode", _control_tool_result)
    bind_tool_executor("verify_skip", _control_tool_result)
    bind_tool_executor("write_file", _write_file_result)
    bind_tool_executor("load_file", _load_file_result)
    bind_tool_executor("edit_file", _edit_file_result)
    bind_tool_executor("glob", _glob_result)
    bind_tool_executor("grep", _grep_result)
    bind_tool_executor("list_directory", _list_directory_result)


_bind_tool_executors()
