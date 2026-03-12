"""HTTP client for llama-server's OpenAI-compatible API. Manages the server lifecycle."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from pathlib import Path
from typing import AsyncIterator

import httpx

from .runtime_protocol import StreamChunk, stream_openai_chat


def _find_llama_server() -> str:
    path = shutil.which("llama-server")
    if path:
        return path
    from pathlib import Path
    candidate = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("llama-server not found on PATH or ~/llama.cpp/build/bin/")


class LlamaServerClient:
    """Starts llama-server and streams chat completions via its OpenAI API."""

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        context_window_tokens: int = 2048,
        device: str = "auto",
        gpu_layers: int = 99,
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

    @staticmethod
    def _ensure_jetson_clocks_sudoers() -> None:
        """Install a passwordless sudoers rule for jetson_clocks.

        Only runs once — skipped if the rule already exists.
        Prompts the user for their password via ``sudo`` on first install.
        """
        import subprocess

        sudoers_file = Path("/etc/sudoers.d/open-jet-clocks")
        if sudoers_file.exists():
            return

        rule = "ALL ALL=(ALL) NOPASSWD: /usr/bin/jetson_clocks\n"
        try:
            subprocess.run(
                ["sudo", "tee", str(sudoers_file)],
                input=rule.encode(), capture_output=True, timeout=30,
            )
            subprocess.run(
                ["sudo", "chmod", "0440", str(sudoers_file)],
                capture_output=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def _maximize_gpu_clocks() -> None:
        """Pin GPU and EMC clocks to maximum frequency on Jetson."""
        import subprocess
        result = subprocess.run(
            ["sudo", "-n", "jetson_clocks"],
            timeout=5, capture_output=True,
        )
        if result.returncode != 0:
            # Fallback: write directly to GPU devfreq sysfs.
            gpu_devfreq = Path("/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu")
            try:
                max_freq = (gpu_devfreq / "max_freq").read_text().strip()
                (gpu_devfreq / "min_freq").write_text(max_freq)
            except (PermissionError, OSError):
                pass

    @staticmethod
    def _compact_memory() -> None:
        """Drop filesystem caches and compact free memory.

        On Jetson (unified memory) CUDA needs large contiguous allocations.
        Dropping caches frees pagecache/slab memory so the kernel can coalesce
        free pages into larger contiguous regions.
        """
        try:
            Path("/proc/sys/vm/drop_caches").write_text("3")
            Path("/proc/sys/vm/compact_memory").write_text("1")
        except PermissionError:
            # Not running as root — try via sudo non-interactively.
            import subprocess
            subprocess.run(
                ["sudo", "-n", "sh", "-c",
                 "echo 3 > /proc/sys/vm/drop_caches && echo 1 > /proc/sys/vm/compact_memory"],
                timeout=5, capture_output=True,
            )
        except OSError:
            pass

    async def start(self) -> None:
        await self._stop_server()
        await self._cleanup_stale_inference_processes()
        self._ensure_jetson_clocks_sudoers()
        self._maximize_gpu_clocks()
        self._compact_memory()
        binary = _find_llama_server()
        env = os.environ.copy()
        bin_dir = os.path.dirname(binary)
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        # Use cudaMallocManaged on Jetson unified memory so CUDA doesn't
        # need physically contiguous pages for large allocations.
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"

        resolved_device = self._resolve_device()
        requested_ngl = self.gpu_layers if resolved_device == "cuda" else 0
        requested_ctx = self.context_window_tokens
        # Keep model/offload settings unchanged; only reduce launch batch pressure if KV alloc fails.
        launch_attempts = ((512, 128), (256, 64), (128, 32))
        last_err: RuntimeError | None = None
        for batch, ubatch in launch_attempts:
            try:
                await self._start_once(
                    binary=binary,
                    env=env,
                    ngl=requested_ngl,
                    ctx=requested_ctx,
                    batch=batch,
                    ubatch=ubatch,
                )
                last_err = None
                break
            except RuntimeError as exc:
                last_err = exc
                if not self._is_model_load_alloc_failure(str(exc)):
                    raise
                await self._stop_server()
                await self._cleanup_stale_inference_processes()
                continue

        if last_err is not None:
            raise last_err
        self.gpu_layers = requested_ngl
        self.context_window_tokens = requested_ctx

    async def _start_once(
        self,
        *,
        binary: str,
        env: dict[str, str],
        ngl: int,
        ctx: int,
        batch: int,
        ubatch: int,
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
            "--no-mmap",
            "--flash-attn", "on",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "-b",
            str(batch),
            "-ub",
            str(ubatch),
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
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
            ):
                # During startup, the socket may accept then drop before readiness.
                # Treat transient transport errors the same as "not ready yet".
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
            exe_name = Path(parts[0]).name.lower()
            if exe_name not in {"llama-server", "ollama"}:
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
        async for chunk in stream_openai_chat(
            self._http,
            base_url=self.base_url,
            model="local",
            messages=messages,
            use_tools=use_tools,
        ):
            yield chunk
