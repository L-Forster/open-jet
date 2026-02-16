"""Thin async HTTP client for Ollama /api/chat with streaming and tool-call support."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

# -- Tool definitions exposed to the model -----------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command on the Jetson device. Always requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Always requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


# -- Data types --------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class StreamChunk:
    """One piece of a streaming response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False


# -- Client ------------------------------------------------------------------


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def chat_stream(
        self, messages: list[dict], *, use_tools: bool = True
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion, yielding chunks of text and/or tool calls."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if use_tools:
            payload["tools"] = TOOLS

        async with self._http.stream(
            "POST", "/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                chunk = _parse_chunk(data)
                yield chunk
                if chunk.done:
                    return


def _parse_chunk(data: dict) -> StreamChunk:
    """Parse a single JSON line from the Ollama streaming response."""
    done = data.get("done", False)
    msg = data.get("message", {})

    text = msg.get("content", "")

    tool_calls: list[ToolCall] = []
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        if name:
            tool_calls.append(ToolCall(name=name, arguments=args))

    return StreamChunk(text=text, tool_calls=tool_calls, done=done)
