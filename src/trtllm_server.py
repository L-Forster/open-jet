"""HTTP client for trtllm-serve's OpenAI-compatible API. Manages server lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sysconfig
from pathlib import Path
from typing import AsyncIterator

import httpx

from .llama_server import TOOLS, StreamChunk, ToolCall


def _find_trtllm_serve() -> str:
    scripts_dir = Path(sysconfig.get_path("scripts") or "")
    candidate = scripts_dir / "trtllm-serve"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    raise FileNotFoundError(
        f"trtllm-serve not found at expected install location: {candidate}. "
        "Install tensorrt_llm in the same environment as open-jet."
    )


class TrtllmServerClient:
    """Starts trtllm-serve and streams chat completions via its OpenAI API."""

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        context_window_tokens: int = 2048,
        backend: str = "pytorch",
        config_path: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.context_window_tokens = max(512, int(context_window_tokens))
        # Keep compatibility with existing app status/logging paths.
        self.gpu_layers = 0
        self.backend = (backend or "pytorch").strip()
        self.config_path = (config_path or "").strip() or None
        self.trust_remote_code = bool(trust_remote_code)
        self.base_url = f"http://{host}:{port}"
        self._http = httpx.AsyncClient(timeout=120.0)
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        await self._stop_server()
        binary = _find_trtllm_serve()
        env = os.environ.copy()

        cmd = [
            binary,
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--backend",
            self.backend,
        ]
        if self.trust_remote_code:
            cmd.append("--trust_remote_code")
        if self.config_path:
            config = Path(self.config_path).expanduser()
            if not config.is_file():
                raise FileNotFoundError(f"TensorRT-LLM config file not found: {config}")
            cmd.extend(["--config", str(config)])

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
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
                    f"trtllm-serve exited with code {self._proc.returncode}: {detail}"
                )
            try:
                r = await self._http.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    return
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
            ):
                pass
            await asyncio.sleep(1.0)
        raise TimeoutError("trtllm-serve did not become ready")

    async def close(self) -> None:
        await self._stop_server()
        await self._http.aclose()

    async def reset_kv_cache(self) -> None:
        # Best effort reset by restarting the serving process.
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
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if use_tools:
            payload["tools"] = TOOLS

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
