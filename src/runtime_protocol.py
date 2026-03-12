from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command.",
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
                },
                "required": ["command"],
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
            "description": "Edit a file by exact string replacement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all matches",
                    },
                },
                "required": ["path", "old_string", "new_string"],
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


async def stream_openai_chat(
    http: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    use_tools: bool = True,
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

    async with http.stream("POST", f"{base_url}/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
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
                yield StreamChunk(text=text)

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

            yield StreamChunk(tool_calls=tool_calls, done=True)
            return
