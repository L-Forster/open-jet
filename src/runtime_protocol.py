from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .runtime_limits import estimate_tokens

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run a shell command. For heavy local work you may set "
                "resource_mode=unload_first to unload the local model before the command "
                "and reload it afterward."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                    },
                    "resource_mode": {
                        "type": "string",
                        "description": "Resource strategy: normal, auto, or unload_first",
                    },
                    "estimated_ram_mb": {
                        "type": "integer",
                        "description": "Optional estimated RAM needed by the command in MB",
                    },
                    "estimated_vram_mb": {
                        "type": "integer",
                        "description": "Optional estimated VRAM needed by the command in MB",
                    },
                    "reload_delay_seconds": {
                        "type": "integer",
                        "description": "Optional delay before reloading the model after an unload-run cycle",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": (
                "Inspect local system resource information such as RAM, disk, load, "
                "and GPU memory when detectable. Use this before heavy shell commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Info scope: summary, memory, gpu, disk, or all",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory",
            "description": "Read or update persistent cross-session memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Memory scope: user or agent",
                    },
                    "action": {
                        "type": "string",
                        "description": "Operation: read, append, replace, or clear",
                    },
                    "content": {
                        "type": "string",
                        "description": "Memory content for append or replace",
                    },
                },
                "required": ["scope", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_file",
            "description": "Load a text file into context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit a file using one or more strict SEARCH/REPLACE blocks. "
                "Prefer patch over full rewrites. Use this exact format: "
                "<<<<<<< SEARCH\\n...existing text...\\n=======\\n...replacement text...\\n>>>>>>> REPLACE"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path",
                    },
                    "patch": {
                        "type": "string",
                        "description": (
                            "One or more SEARCH/REPLACE blocks. "
                            "This is the preferred edit format."
                        ),
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Deprecated legacy exact text to replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Deprecated legacy replacement text",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all matches for the legacy fields",
                    },
                },
                "required": ["path", "patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by glob.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern",
                    },
                    "path": {
                        "type": "string",
                        "description": "Search root",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern",
                    },
                    "path": {
                        "type": "string",
                        "description": "Search root",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Ignore case",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path",
                    },
                },
                "required": [],
            },
        },
    },
]


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None


@dataclass
class StreamChunk:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.IGNORECASE | re.DOTALL)
_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.IGNORECASE | re.DOTALL)


def tool_schema_token_estimate() -> int:
    payload = json.dumps(TOOLS, ensure_ascii=True, separators=(",", ":"))
    return estimate_tokens(payload)


def _coerce_reasoning_parameter(raw: str) -> object:
    value = raw.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _extract_reasoning_tool_calls(reasoning_text: str) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    if not reasoning_text.strip():
        return tool_calls

    for block in _TOOL_CALL_BLOCK_RE.findall(reasoning_text):
        match = _FUNCTION_BLOCK_RE.search(block)
        if not match:
            continue
        name = match.group(1).strip()
        if not name:
            continue
        arguments: dict[str, object] = {}
        for param_name, param_value in _PARAMETER_BLOCK_RE.findall(match.group(2)):
            key = param_name.strip()
            if not key:
                continue
            arguments[key] = _coerce_reasoning_parameter(param_value)
        tool_calls.append(ToolCall(name=name, arguments=arguments))
    return tool_calls


def _extract_reasoning_display_text(reasoning_text: str) -> str:
    if not reasoning_text.strip():
        return ""
    visible = reasoning_text
    marker = re.search(r"<tool_call>\s*", visible, flags=re.IGNORECASE)
    if marker:
        visible = visible[: marker.start()]
    visible = visible.strip()
    return visible


async def stream_openai_chat(
    http: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    use_tools: bool = True,
    headers: dict[str, str] | None = None,
    extra_body: dict | None = None,
) -> AsyncIterator[StreamChunk]:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if use_tools:
        payload["tools"] = TOOLS
    if extra_body:
        payload.update(extra_body)

    tool_id_by_index: dict[int, str] = {}
    tool_name_by_index: dict[int, str] = {}
    tool_args_by_index: dict[int, str] = {}
    reasoning_buf = ""
    saw_text = False

    async with http.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
    ) as resp:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                body = await resp.aread()
                text = body.decode("utf-8", errors="replace").strip()
                if text:
                    detail = f": {text[:500]}"
            except Exception:
                detail = ""
            raise RuntimeError(f"{exc}{detail}") from exc

        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[6:]
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {}) or {}
            finish_reason = choice.get("finish_reason")

            text = delta.get("content", "") or ""
            if text:
                saw_text = True
                yield StreamChunk(text=text)

            reasoning_text = delta.get("reasoning_content", "") or ""
            if reasoning_text:
                reasoning_buf += reasoning_text

            for tc in delta.get("tool_calls", []) or []:
                idx = int(tc.get("index", 0))
                tc_id = tc.get("id", "")
                if tc_id:
                    tool_id_by_index[idx] = tc_id
                fn = tc.get("function", {}) or {}
                name_part = fn.get("name", "")
                args_part = fn.get("arguments", "")
                if name_part:
                    tool_name_by_index[idx] = name_part
                if isinstance(args_part, str) and args_part:
                    tool_args_by_index[idx] = tool_args_by_index.get(idx, "") + args_part

            if finish_reason is None:
                continue

            tool_calls: list[ToolCall] = []
            for idx in sorted(tool_name_by_index):
                name = tool_name_by_index.get(idx, "")
                if not name:
                    continue
                args_raw = tool_args_by_index.get(idx, "")
                if args_raw:
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                else:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        name=name,
                        arguments=args,
                        id=tool_id_by_index.get(idx),
                    )
                )

            if not tool_calls and reasoning_buf:
                tool_calls = _extract_reasoning_tool_calls(reasoning_buf)

            if tool_calls and not saw_text and reasoning_buf:
                display_text = _extract_reasoning_display_text(reasoning_buf)
                if display_text:
                    yield StreamChunk(text=display_text)

            yield StreamChunk(tool_calls=tool_calls, done=True)
            return
