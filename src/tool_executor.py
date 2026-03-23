from __future__ import annotations

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
    edit_file,
    glob_files,
    grep_files,
    list_directory,
    load_file,
    read_system_info,
    read_file,
    run_shell,
    write_file,
)
from .multimodal import content_to_plain_text
from .observation import ObservationStore, observation_to_agent_content
from .peripherals import PeripheralKind, PeripheralTransport
from .peripherals.system import device_discovery_hint
from .persistent_memory import update_persistent_memory
from .runtime_protocol import ToolCall

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


# Optional SwapManager — set by app.py at startup when available.
_swap_manager: SwapManager | None = None


def set_swap_manager(manager: SwapManager | None) -> None:
    """Register a SwapManager for memory-guarded shell execution."""
    global _swap_manager
    _swap_manager = manager


async def execute_tool(tool_call: ToolCall) -> ToolExecutionResult:
    if not isinstance(tool_call.arguments, dict):
        return ToolExecutionResult(
            output=f"Error: invalid arguments for {tool_call.name}",
            meta={"ok": False},
        )

    if tool_call.name == "shell":
        command = tool_call.arguments.get("command", "")
        timeout_seconds = tool_call.arguments.get("timeout_seconds")
        resource_mode = str(tool_call.arguments.get("resource_mode", "normal") or "normal").strip().lower()
        estimated_ram_mb = tool_call.arguments.get("estimated_ram_mb")
        estimated_vram_mb = tool_call.arguments.get("estimated_vram_mb")
        reload_delay_seconds = tool_call.arguments.get("reload_delay_seconds", 0)
        if not isinstance(command, str) or not command.strip():
            return ToolExecutionResult(
                output="Error: invalid arguments for shell (required: command)",
                meta={"ok": False},
            )
        if timeout_seconds is not None:
            if not isinstance(timeout_seconds, int):
                return ToolExecutionResult(
                    output="Error: invalid arguments for shell (timeout_seconds must be int)",
                    meta={"ok": False},
                )
            if timeout_seconds <= 0:
                return ToolExecutionResult(
                    output="Error: invalid arguments for shell (timeout_seconds must be > 0)",
                    meta={"ok": False},
                )
        if resource_mode not in {"normal", "auto", "unload_first"}:
            return ToolExecutionResult(
                output="Error: invalid arguments for shell (resource_mode must be normal, auto, or unload_first)",
                meta={"ok": False},
            )
        if estimated_ram_mb is not None and not isinstance(estimated_ram_mb, int):
            return ToolExecutionResult(
                output="Error: invalid arguments for shell (estimated_ram_mb must be int)",
                meta={"ok": False},
            )
        if estimated_vram_mb is not None and not isinstance(estimated_vram_mb, int):
            return ToolExecutionResult(
                output="Error: invalid arguments for shell (estimated_vram_mb must be int)",
                meta={"ok": False},
            )
        if not isinstance(reload_delay_seconds, int):
            return ToolExecutionResult(
                output="Error: invalid arguments for shell (reload_delay_seconds must be int)",
                meta={"ok": False},
            )
        if reload_delay_seconds < 0:
            return ToolExecutionResult(
                output="Error: invalid arguments for shell (reload_delay_seconds must be >= 0)",
                meta={"ok": False},
            )
        effective_timeout = timeout_seconds or DEFAULT_SHELL_TIMEOUT_SECONDS
        swap_meta: dict = {}
        swap_plan = _swap_manager.plan_unload(estimated_ram_mb or 0) if _swap_manager is not None else None
        use_swap = False
        if resource_mode == "unload_first":
            use_swap = _swap_manager is not None
        elif resource_mode == "auto" and swap_plan is not None:
            use_swap = swap_plan.unload_recommended

        if use_swap and _swap_manager is not None:
            swap_result = await _swap_manager.run_with_swap(
                command,
                timeout=effective_timeout,
                reload_delay_seconds=reload_delay_seconds,
            )
            res = swap_result.exec_result
            swap_meta = {
                "swap_mode": resource_mode,
                "swap_available": True,
                "swap_attempted": True,
                "swap_plan_recommended": True if swap_plan is None else swap_plan.unload_recommended,
                "estimated_ram_mb": estimated_ram_mb,
                "estimated_vram_mb": estimated_vram_mb,
                "reload_delay_seconds": reload_delay_seconds,
                "swapped": swap_result.unloaded,
                "swap_reload_ok": swap_result.reload_ok,
                "swap_restore_ok": swap_result.restore_ok,
                "swap_duration_s": swap_result.swap_duration_s,
            }
        else:
            res = await run_shell(command, timeout_seconds=effective_timeout)
            swap_meta = {
                "swap_mode": resource_mode,
                "swap_available": _swap_manager is not None,
                "swap_attempted": False,
                "swap_plan_recommended": swap_plan.unload_recommended if swap_plan is not None else False,
                "estimated_ram_mb": estimated_ram_mb,
                "estimated_vram_mb": estimated_vram_mb,
                "reload_delay_seconds": reload_delay_seconds,
            }

        result_output = res.summary
        swap_note = _swap_summary_line(swap_meta)
        if swap_note:
            result_output = f"{result_output}\n{swap_note}"
        resource_hint = _resource_failure_hint(res)
        if resource_hint:
            result_output = f"{result_output}\n{resource_hint}"
        return ToolExecutionResult(
            output=result_output,
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

    if tool_call.name == "system_info":
        scope = tool_call.arguments.get("scope")
        if scope is not None and not isinstance(scope, str):
            return ToolExecutionResult(
                output="Error: invalid arguments for system_info (scope must be string)",
                meta={"ok": False},
            )
        text = await read_system_info(scope=scope)
        return ToolExecutionResult(output=text, meta={"ok": not text.startswith("Error:")})

    if tool_call.name == "device_list":
        kind = tool_call.arguments.get("kind")
        if kind is not None and not isinstance(kind, str):
            return ToolExecutionResult(
                output="Error: invalid arguments for device_list (kind must be string)",
                meta={"ok": False},
            )
        try:
            text, meta = _device_list_output(kind=kind)
        except ValueError as exc:
            return ToolExecutionResult(output=f"Error: {exc}", meta={"ok": False})
        return ToolExecutionResult(output=text, meta=meta)

    if tool_call.name == "camera_snapshot":
        source_ref = tool_call.arguments.get("source")
        output_path = tool_call.arguments.get("output_path")
        if source_ref is not None and not isinstance(source_ref, str):
            return ToolExecutionResult(
                output="Error: invalid arguments for camera_snapshot (source must be string)",
                meta={"ok": False},
            )
        if output_path is not None and not isinstance(output_path, str):
            return ToolExecutionResult(
                output="Error: invalid arguments for camera_snapshot (output_path must be string)",
                meta={"ok": False},
            )
        try:
            result = _device_observation_result(
                source_ref=source_ref,
                expected_kind=PeripheralKind.CAMERA,
                output_path=output_path,
            )
        except (ValueError, RuntimeError) as exc:
            return ToolExecutionResult(output=f"Error: {exc}", meta={"ok": False})
        return result

    if tool_call.name == "microphone_record":
        source_ref = tool_call.arguments.get("source")
        duration_seconds = tool_call.arguments.get("duration_seconds", 5)
        output_path = tool_call.arguments.get("output_path")
        if source_ref is not None and not isinstance(source_ref, str):
            return ToolExecutionResult(
                output="Error: invalid arguments for microphone_record (source must be string)",
                meta={"ok": False},
            )
        if not isinstance(duration_seconds, int):
            return ToolExecutionResult(
                output="Error: invalid arguments for microphone_record (duration_seconds must be int)",
                meta={"ok": False},
            )
        if duration_seconds <= 0:
            return ToolExecutionResult(
                output="Error: invalid arguments for microphone_record (duration_seconds must be > 0)",
                meta={"ok": False},
            )
        if output_path is not None and not isinstance(output_path, str):
            return ToolExecutionResult(
                output="Error: invalid arguments for microphone_record (output_path must be string)",
                meta={"ok": False},
            )
        try:
            result = _device_observation_result(
                source_ref=source_ref,
                expected_kind=PeripheralKind.MICROPHONE,
                duration_seconds=duration_seconds,
                output_path=output_path,
            )
        except (ValueError, RuntimeError) as exc:
            return ToolExecutionResult(output=f"Error: {exc}", meta={"ok": False})
        return result

    if tool_call.name == "microphone_set_enabled":
        source_ref = tool_call.arguments.get("source")
        enabled = tool_call.arguments.get("enabled")
        if source_ref is not None and not isinstance(source_ref, str):
            return ToolExecutionResult(
                output="Error: invalid arguments for microphone_set_enabled (source must be string)",
                meta={"ok": False},
            )
        if not isinstance(enabled, bool):
            return ToolExecutionResult(
                output="Error: invalid arguments for microphone_set_enabled (enabled must be boolean)",
                meta={"ok": False},
            )
        try:
            cfg = _tool_config()
            source = _select_device_source(cfg, source_ref=source_ref, expected_kind=PeripheralKind.MICROPHONE)
            updated = set_device_enabled(cfg, reference=source.primary_ref, enabled=enabled)
            save_config(cfg)
            registry_path = _sync_tool_device_registry(cfg)
        except ValueError as exc:
            return ToolExecutionResult(output=f"Error: {exc}", meta={"ok": False})
        state = "enabled" if updated.enabled else "disabled"
        return ToolExecutionResult(
            output=f"Microphone @{updated.primary_ref} is now {state}.",
            meta={
                "ok": True,
                "source_ref": updated.primary_ref,
                "enabled": updated.enabled,
                "registry_path": str(registry_path) if registry_path else None,
            },
        )

    if tool_call.name in {"sensor_read", "gpio_read"}:
        source_ref = tool_call.arguments.get("source")
        if source_ref is not None and not isinstance(source_ref, str):
            return ToolExecutionResult(
                output=f"Error: invalid arguments for {tool_call.name} (source must be string)",
                meta={"ok": False},
            )
        try:
            result = _device_observation_result(
                source_ref=source_ref,
                expected_kind=PeripheralKind.SENSOR,
                required_transport=PeripheralTransport.GPIO,
            )
        except (ValueError, RuntimeError) as exc:
            return ToolExecutionResult(output=f"Error: {exc}", meta={"ok": False})
        return result

    if tool_call.name == "read_file":
        path = tool_call.arguments.get("path", "")
        if not isinstance(path, str) or not path.strip():
            return ToolExecutionResult(
                output="Error: invalid arguments for read_file (required: path)",
                meta={"ok": False},
            )
        text = await read_file(path)
        return ToolExecutionResult(output=text, meta={"ok": not text.startswith("Error:")})

    if tool_call.name == "memory":
        scope = tool_call.arguments.get("scope", "")
        action = tool_call.arguments.get("action", "")
        content = tool_call.arguments.get("content", "")
        if not isinstance(scope, str) or not scope.strip() or not isinstance(action, str) or not action.strip():
            return ToolExecutionResult(
                output="Error: invalid arguments for memory (required: scope, action)",
                meta={"ok": False},
            )
        if not isinstance(content, str):
            return ToolExecutionResult(
                output="Error: invalid arguments for memory (content must be string)",
                meta={"ok": False},
            )
        try:
            text = await update_persistent_memory(
                Path.cwd(),
                scope=scope,
                action=action,
                content=content,
            )
        except ValueError as exc:
            return ToolExecutionResult(output=f"Error: {exc}", meta={"ok": False})
        return ToolExecutionResult(output=text, meta={"ok": True, "scope": scope, "action": action})

    if tool_call.name == "write_file":
        path = tool_call.arguments.get("path", "")
        content = tool_call.arguments.get("content", "")
        if not isinstance(path, str) or not path.strip() or not isinstance(content, str):
            return ToolExecutionResult(
                output="Error: invalid arguments for write_file (required: path, content)",
                meta={"ok": False},
            )
        text = await write_file(path, content)
        return ToolExecutionResult(output=text, meta={"ok": not text.startswith("Error")})

    if tool_call.name == "load_file":
        path = tool_call.arguments.get("path", "")
        max_tokens = tool_call.arguments.get("max_tokens")
        if not isinstance(path, str) or not path.strip():
            return ToolExecutionResult(
                output="Error: invalid arguments for load_file (required: path)",
                meta={"ok": False},
            )
        if max_tokens is not None and not isinstance(max_tokens, int):
            return ToolExecutionResult(
                output="Error: invalid arguments for load_file (max_tokens must be int)",
                meta={"ok": False},
            )
        loaded = await load_file(path, max_tokens=max_tokens)
        if not loaded.ok:
            return ToolExecutionResult(output=loaded.detail, meta={"ok": False})
        payload = f"{loaded.summary}\ncontent:\n{loaded.content}"
        return ToolExecutionResult(
            output=payload,
            meta={
                "ok": True,
                "truncated": loaded.truncated,
                "estimated_tokens": loaded.estimated_tokens,
                "returned_tokens": loaded.returned_tokens,
                "token_budget": loaded.token_budget,
                "mem_available_mb": loaded.mem_available_mb,
            },
        )

    if tool_call.name == "edit_file":
        path = tool_call.arguments.get("path", "")
        patch = tool_call.arguments.get("patch")
        old_string = tool_call.arguments.get("old_string", "")
        new_string = tool_call.arguments.get("new_string", "")
        replace_all_flag = tool_call.arguments.get("replace_all", False)
        if not isinstance(path, str) or not path.strip():
            return ToolExecutionResult(
                output="Error: invalid arguments for edit_file (required: path)",
                meta={"ok": False},
            )
        if patch is not None:
            if not isinstance(patch, str) or not patch.strip():
                return ToolExecutionResult(
                    output="Error: invalid arguments for edit_file (patch must be a non-empty string)",
                    meta={"ok": False},
                )
            result = await edit_file(path, patch=patch, return_result=True)
        else:
            if not isinstance(old_string, str) or not old_string:
                return ToolExecutionResult(
                    output="Error: invalid arguments for edit_file (required: patch or old_string)",
                    meta={"ok": False},
                )
            if not isinstance(new_string, str):
                return ToolExecutionResult(
                    output="Error: invalid arguments for edit_file (required: new_string)",
                    meta={"ok": False},
                )
            result = await edit_file(
                path,
                old_string=old_string,
                new_string=new_string,
                replace_all=bool(replace_all_flag),
                return_result=True,
            )
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

    if tool_call.name == "glob":
        pattern = tool_call.arguments.get("pattern", "")
        search_path = tool_call.arguments.get("path")
        if not isinstance(pattern, str) or not pattern.strip():
            return ToolExecutionResult(
                output="Error: invalid arguments for glob (required: pattern)",
                meta={"ok": False},
            )
        text = await glob_files(pattern, path=search_path)
        return ToolExecutionResult(output=text, meta={"ok": not text.startswith("Error")})

    if tool_call.name == "grep":
        pattern = tool_call.arguments.get("pattern", "")
        search_path = tool_call.arguments.get("path")
        glob_filter = tool_call.arguments.get("glob")
        ignore_case = tool_call.arguments.get("ignore_case", False)
        if not isinstance(pattern, str) or not pattern.strip():
            return ToolExecutionResult(
                output="Error: invalid arguments for grep (required: pattern)",
                meta={"ok": False},
            )
        text = await grep_files(
            pattern,
            path=search_path,
            glob_filter=glob_filter,
            ignore_case=bool(ignore_case),
        )
        return ToolExecutionResult(output=text, meta={"ok": not text.startswith("Error")})

    if tool_call.name == "list_directory":
        dir_path = tool_call.arguments.get("path")
        text = await list_directory(path=dir_path)
        return ToolExecutionResult(output=text, meta={"ok": not text.startswith("Error")})

    return ToolExecutionResult(output=f"Unknown tool: {tool_call.name}", meta={"ok": False})


def format_tool_args(tool_call: ToolCall) -> str:
    if tool_call.name == "shell":
        command = tool_call.arguments.get("command", str(tool_call.arguments))
        timeout_seconds = tool_call.arguments.get("timeout_seconds")
        resource_mode = str(tool_call.arguments.get("resource_mode", "normal") or "normal").strip()
        if isinstance(timeout_seconds, int):
            return f"$ {command} (timeout: {timeout_seconds}s, mode: {resource_mode})"
        return f"$ {command} (mode: {resource_mode})"
    if tool_call.name == "system_info":
        scope = str(tool_call.arguments.get("scope", "summary") or "summary").strip()
        return f"scope={scope}"
    if tool_call.name == "device_list":
        kind = str(tool_call.arguments.get("kind", "") or "").strip()
        return f"kind={kind}" if kind else "all devices"
    if tool_call.name in {"camera_snapshot", "microphone_record", "sensor_read", "gpio_read"}:
        source = str(tool_call.arguments.get("source", "") or "").strip()
        if tool_call.name == "microphone_record":
            duration_seconds = tool_call.arguments.get("duration_seconds")
            if isinstance(duration_seconds, int):
                return f"{source or '<auto>'} ({duration_seconds}s)"
        return source or "<auto>"
    if tool_call.name == "microphone_set_enabled":
        source = str(tool_call.arguments.get("source", "") or "").strip() or "<auto>"
        enabled = tool_call.arguments.get("enabled")
        return f"{source} -> {'on' if enabled else 'off'}"
    if tool_call.name == "memory":
        scope = str(tool_call.arguments.get("scope", ""))
        action = str(tool_call.arguments.get("action", ""))
        return f"{action} {scope}".strip()
    if tool_call.name in {"read_file", "write_file", "load_file", "edit_file"}:
        return str(tool_call.arguments.get("path", str(tool_call.arguments)))
    if tool_call.name in {"glob", "grep"}:
        return str(tool_call.arguments.get("pattern", str(tool_call.arguments)))
    if tool_call.name == "list_directory":
        return str(tool_call.arguments.get("path", ".") or ".")
    return str(tool_call.arguments)


def _swap_summary_line(meta: dict) -> str:
    if not meta:
        return ""
    mode = str(meta.get("swap_mode", "normal") or "normal")
    if meta.get("swapped"):
        delay = meta.get("reload_delay_seconds", 0)
        duration = meta.get("swap_duration_s")
        delay_part = f", reload_delay={delay}s" if isinstance(delay, int) and delay > 0 else ""
        duration_part = f", total={duration}s" if isinstance(duration, (int, float)) else ""
        return f"[swap path used: mode={mode}{delay_part}{duration_part}]"
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
    if not haystack:
        return ""
    patterns = (
        "out of memory",
        "cuda out of memory",
        "cuda error",
        "cublas_status_alloc_failed",
        "cannot allocate memory",
        "std::bad_alloc",
        "memoryerror",
        "killed",
    )
    if not any(pattern in haystack for pattern in patterns):
        return ""
    return (
        "[resource failure detected: consider `system_info` and, if needed, retry the shell command "
        "with `resource_mode` set to `unload_first`]"
    )


def _device_list_output(*, kind: str | None) -> tuple[str, dict[str, object]]:
    cfg = _tool_config()
    sources = list_device_sources(cfg)
    registry_path = _sync_tool_device_registry(cfg)
    parsed_kind = _parse_device_kind(kind)
    if parsed_kind is not None:
        sources = [source for source in sources if source.device.kind is parsed_kind]

    if not sources:
        label = f"{parsed_kind.value} " if parsed_kind is not None else ""
        message = f"No {label}device sources detected.".strip()
        hint = device_discovery_hint()
        if hint:
            message = f"{message}\n{hint}"
        return (
            message,
            {
                "ok": True,
                "count": 0,
                "registry_path": str(registry_path) if registry_path else None,
            },
        )

    lines = ["Discovered device sources:"]
    for source in sources:
        refs = ", ".join(f"@{ref}" for ref in source.refs)
        path = f" | path={source.device.path}" if source.device.path else ""
        lines.append(
            f"- @{source.primary_ref}: {source.device.label} | kind={source.device.kind.value} "
            f"| transport={source.device.transport.value} | state={'enabled' if source.enabled else 'disabled'} | refs={refs}{path}"
        )
    return (
        "\n".join(lines),
        {
            "ok": True,
            "count": len(sources),
            "kind": parsed_kind.value if parsed_kind else None,
            "registry_path": str(registry_path) if registry_path else None,
        },
    )


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
    observation = collect_device_observation(
        source,
        store=store,
        cfg=cfg,
        duration_seconds=duration_seconds,
        output_path=output_path,
    )
    registry_path = _sync_tool_device_registry(cfg, store=store)
    context_content = observation_to_agent_content(observation, store=store)
    output = content_to_plain_text(context_content)
    return ToolExecutionResult(
        output=output,
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


def _sync_tool_device_registry(
    cfg: dict[str, object],
    *,
    store: ObservationStore | None = None,
) -> Path | None:
    try:
        return sync_devices_registry(cfg, store=store)
    except Exception:
        return None
