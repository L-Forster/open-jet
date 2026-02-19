"""HTTP client for llama-server's OpenAI-compatible API. Manages the server lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path
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
        device: str = "auto",
        gpu_layers: int = 20,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.context_window_tokens = max(512, int(context_window_tokens))
        self.device = device.strip().lower() if device else "auto"
        self.gpu_layers = max(0, int(gpu_layers))
        self.base_url = f"http://{host}:{port}"
        self._http = httpx.AsyncClient(timeout=120.0)
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        await self._stop_server()
        await self._cleanup_stale_inference_processes()
        binary = _find_llama_server()
        env = os.environ.copy()
        bin_dir = os.path.dirname(binary)
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")

        resolved_device = self._resolve_device()
        requested_ngl = self.gpu_layers if resolved_device == "cuda" else 0
        requested_ctx = self.context_window_tokens
        try:
            await self._start_once(binary=binary, env=env, ngl=requested_ngl, ctx=requested_ctx)
        except RuntimeError as exc:
            # Retry once after aggressive same-user cleanup to release stale CUDA/NvMap allocations.
            if not self._is_model_load_alloc_failure(str(exc)):
                raise
            await self._stop_server()
            await self._cleanup_stale_inference_processes()
            await self._start_once(binary=binary, env=env, ngl=requested_ngl, ctx=requested_ctx)
        self.gpu_layers = requested_ngl
        self.context_window_tokens = requested_ctx

    async def _start_once(
        self,
        *,
        binary: str,
        env: dict[str, str],
        ngl: int,
        ctx: int,
    ) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            binary,
            "-m",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--parallel",
            "1",
            "--no-kv-unified",
            "--no-cont-batching",
            "-b",
            "512",
            "-ub",
            "128",
            "-ngl",
            str(ngl),
            "-c",
            str(ctx),
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        deadline = asyncio.get_event_loop().time() + 120.0
        while asyncio.get_event_loop().time() < deadline:
            if self._proc.returncode is not None:
                err = (await self._proc.stderr.read()).decode(errors="replace").strip()
                if len(err) > 8000:
                    keep = 4000
                    omitted = len(err) - (keep * 2)
                    err = (
                        f"{err[:keep]}\n"
                        f"... <{omitted} chars omitted> ...\n"
                        f"{err[-keep:]}"
                    )
                detail = err or "no stderr output"
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.returncode}: {detail}"
                )
            try:
                r = await self._http.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(1.0)
        raise TimeoutError("llama-server did not become ready")

    async def _cleanup_stale_inference_processes(self) -> None:
        stale_pids = self._find_stale_inference_pids()
        if not stale_pids:
            return

        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue

        await asyncio.sleep(0.8)

        for pid in stale_pids:
            if not self._pid_exists(pid):
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue

        # Give the driver a short window to release CUDA/NvMap allocations.
        await asyncio.sleep(0.6)

    def _find_stale_inference_pids(self) -> list[int]:
        stale: list[int] = []
        uid = os.getuid()
        current_pid = os.getpid()

        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == current_pid:
                continue

            status_path = entry / "status"
            try:
                status = status_path.read_text(errors="ignore")
            except (FileNotFoundError, PermissionError, OSError):
                continue
            uid_line = next((line for line in status.splitlines() if line.startswith("Uid:")), "")
            if not uid_line:
                continue
            parts_uid = uid_line.split()
            if len(parts_uid) < 2:
                continue
            try:
                real_uid = int(parts_uid[1])
            except ValueError:
                continue
            if real_uid != uid:
                continue

            cmdline_path = entry / "cmdline"
            try:
                raw = cmdline_path.read_bytes()
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if not raw:
                continue

            parts = [p for p in raw.decode(errors="ignore").split("\x00") if p]
            if not parts:
                continue
            joined = " ".join(parts).lower()
            if "llama-server" not in joined and "ollama" not in joined:
                continue

            stale.append(pid)

        return stale

    def _pid_exists(self, pid: int) -> bool:
        return Path(f"/proc/{pid}").exists()

    def _is_model_load_alloc_failure(self, message: str) -> bool:
        lowered = message.lower()
        markers = (
            "cudamalloc failed",
            "failed to allocate cuda",
            "unable to allocate cuda",
            "nvmapmemallocinternaltagged",
            "out of memory",
            "error loading model",
            "failed to load model",
        )
        return any(marker in lowered for marker in markers)

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

    def _resolve_device(self) -> str:
        if self.device in {"cuda", "cpu"}:
            return self.device
        if os.path.exists("/usr/local/cuda") or os.path.exists("/dev/nvhost-gpu"):
            return "cuda"
        return "cpu"

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
