from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
from .persistent_memory import update_persistent_memory
from .runtime_protocol import ToolCall

if TYPE_CHECKING:
    from .swap_manager import SwapManager


@dataclass(frozen=True)
class ToolExecutionResult:
    output: str
    meta: dict

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
