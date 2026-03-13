from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .executor import (
    DEFAULT_SHELL_TIMEOUT_SECONDS,
    edit_file,
    glob_files,
    grep_files,
    list_directory,
    load_file,
    read_file,
    run_shell,
    write_file,
)
from .persistent_memory import update_persistent_memory
from .runtime_protocol import ToolCall


@dataclass(frozen=True)
class ToolExecutionResult:
    output: str
    meta: dict

    @property
    def ok(self) -> bool:
        return bool(self.meta.get("ok", False))


async def execute_tool(tool_call: ToolCall) -> ToolExecutionResult:
    if not isinstance(tool_call.arguments, dict):
        return ToolExecutionResult(
            output=f"Error: invalid arguments for {tool_call.name}",
            meta={"ok": False},
        )

    if tool_call.name == "shell":
        command = tool_call.arguments.get("command", "")
        timeout_seconds = tool_call.arguments.get("timeout_seconds")
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
        res = await run_shell(command, timeout_seconds=timeout_seconds or DEFAULT_SHELL_TIMEOUT_SECONDS)
        return ToolExecutionResult(
            output=res.summary,
            meta={
                "ok": res.ok,
                "exit_code": res.exit_code,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "timed_out": res.timed_out,
                "timeout_seconds": res.timeout_seconds,
            },
        )

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
        if isinstance(timeout_seconds, int):
            return f"$ {command} (timeout: {timeout_seconds}s)"
        return f"$ {command}"
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
