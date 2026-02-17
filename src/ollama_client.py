"""HTTP client for llama-server's OpenAI-compatible API. Manages the server lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

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
            "name": "load_file",
            "description": "Load a text/code file into context with RAM-safe truncation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to a text/code file",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Optional max token budget for loaded content",
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


def _find_llama_server() -> str:
    path = shutil.which("llama-server")
    if path:
        return path
    from pathlib import Path
    candidate = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("llama-server not found on PATH or ~/llama.cpp/build/bin/")


class OllamaClient:
    """Starts llama-server and streams chat completions via its OpenAI API."""

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        context_window_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.context_window_tokens = max(512, int(context_window_tokens))
        self.base_url = f"http://{host}:{port}"
        self._http = httpx.AsyncClient(timeout=120.0)
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        await self._stop_server()
        binary = _find_llama_server()
        env = os.environ.copy()
        bin_dir = os.path.dirname(binary)
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")

        self._proc = await asyncio.create_subprocess_exec(
            binary, "-m", self.model,
            "--host", self.host, "--port", str(self.port),
            "-ngl", "20", "-c", str(self.context_window_tokens),
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        deadline = asyncio.get_event_loop().time() + 120.0
        while asyncio.get_event_loop().time() < deadline:
            if self._proc.returncode is not None:
                err = (await self._proc.stderr.read()).decode()
                raise RuntimeError(f"llama-server exited: {err[:500]}")
            try:
                r = await self._http.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(1.0)
        raise TimeoutError("llama-server did not become ready")

    async def close(self) -> None:
        await self._stop_server()
        await self._http.aclose()

    async def reset_kv_cache(self) -> None:
        # llama.cpp clears KV state on server restart.
        await self._stop_server()
        await self.start()

    async def _stop_server(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
        self._proc = None

    async def chat_stream(
        self, messages: list[dict], *, use_tools: bool = True
    ) -> AsyncIterator[StreamChunk]:
        payload: dict = {
            "model": "local",
            "messages": messages,
            "stream": True,
        }
        if use_tools:
            payload["tools"] = TOOLS

        # Tool arguments may arrive split across multiple stream deltas.
        # Accumulate them by tool-call index and emit only once finalized.
        tool_id_by_index: dict[int, str] = {}
        tool_name_by_index: dict[int, str] = {}
        tool_args_by_index: dict[int, str] = {}

        async with self._http.stream(
            "POST", f"{self.base_url}/v1/chat/completions", json=payload
        ) as resp:
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
