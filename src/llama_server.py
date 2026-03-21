"""HTTP client for llama-server's OpenAI-compatible API. Manages the server lifecycle."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import httpx

from .airgap import apply_airgap_env, assert_endpoint_allowed
from .runtime_protocol import StreamChunk, stream_openai_chat


_FRAGMENTED_LFB_MB = 64
_JETSON_VMM_CHUNK_MB = "1"
_JETSON_VMM_RESERVE_MB = "4096"


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
        airgapped: bool = False,
        diagnostics_hook: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.context_window_tokens = max(512, int(context_window_tokens))
        self.device = device.strip().lower() if device else "auto"
        self.gpu_layers = max(0, int(gpu_layers))
        self.airgapped = bool(airgapped)
        self.base_url = f"http://{host}:{port}"
        self._http = httpx.AsyncClient(timeout=120.0)
        self._proc: asyncio.subprocess.Process | None = None
        self.reasoning_mode = "default"
        self._diagnostics_hook = diagnostics_hook

    def set_reasoning_mode(self, mode: str) -> None:
        normalized = mode.strip().lower()
        if normalized not in {"default", "on", "off"}:
            raise ValueError("reasoning mode must be one of: default, on, off")
        self.reasoning_mode = normalized

    def reasoning_status(self) -> str:
        return self.reasoning_mode

    def _emit_diagnostic(self, event_type: str, **data: Any) -> None:
        if not self._diagnostics_hook:
            return
        try:
            self._diagnostics_hook(event_type, data)
        except Exception:
            pass

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
        import subprocess

        try:
            subprocess.run(["sync"], timeout=5, capture_output=True)
        except (OSError, subprocess.TimeoutExpired):
            pass

        try:
            Path("/proc/sys/vm/drop_caches").write_text("3")
            Path("/proc/sys/vm/compact_memory").write_text("1")
        except PermissionError:
            return
        except OSError:
            return

    @staticmethod
    def _is_jetson_platform() -> bool:
        return Path("/etc/nv_tegra_release").exists() or Path("/dev/nvhost-gpu").exists()

    @staticmethod
    def _largest_free_block_mb_from_text(text: str, *, page_size_kb: int) -> float | None:
        max_order = -1
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("Node "):
                continue
            parts = line.split()
            counts = parts[4:]
            for order, count_text in enumerate(counts):
                try:
                    count = int(count_text)
                except ValueError:
                    continue
                if count > 0:
                    max_order = max(max_order, order)
        if max_order < 0:
            return None
        return (page_size_kb * (2 ** max_order)) / 1024.0

    @classmethod
    def _largest_free_block_mb(cls) -> float | None:
        try:
            buddyinfo = Path("/proc/buddyinfo").read_text(encoding="utf-8")
        except OSError:
            return None
        page_size_kb = int(os.sysconf("SC_PAGE_SIZE")) // 1024
        if page_size_kb <= 0:
            page_size_kb = 4
        return cls._largest_free_block_mb_from_text(buddyinfo, page_size_kb=page_size_kb)

    @staticmethod
    def _read_meminfo_fields() -> dict[str, float]:
        wanted = {
            "MemTotal:",
            "MemAvailable:",
            "MemFree:",
            "Cached:",
            "AnonPages:",
            "Shmem:",
            "CmaTotal:",
            "CmaFree:",
            "SwapTotal:",
            "SwapFree:",
        }
        values: dict[str, float] = {}
        try:
            text = Path("/proc/meminfo").read_text(encoding="utf-8")
        except OSError:
            return values
        for raw_line in text.splitlines():
            parts = raw_line.split()
            if len(parts) < 2 or parts[0] not in wanted:
                continue
            try:
                values[parts[0].rstrip(":").lower() + "_mb"] = round(int(parts[1]) / 1024.0, 2)
            except ValueError:
                continue
        return values

    @classmethod
    def _memory_snapshot(cls) -> dict[str, float]:
        snapshot = cls._read_meminfo_fields()
        lfb_mb = cls._largest_free_block_mb()
        if lfb_mb is not None:
            snapshot["largest_free_block_mb"] = round(lfb_mb, 2)
        return snapshot

    @staticmethod
    def _startup_profile_for_lfb(lfb_mb: float | None) -> tuple[int, int, bool, bool]:
        if lfb_mb is not None and lfb_mb < _FRAGMENTED_LFB_MB:
            return (128, 32, True, True)
        return (128, 32, True, False)

    async def _prepare_memory_for_launch(self) -> float | None:
        lfb_mb = self._largest_free_block_mb()
        for _ in range(3):
            if lfb_mb is not None and lfb_mb >= _FRAGMENTED_LFB_MB:
                return lfb_mb
            self._compact_memory()
            await asyncio.sleep(0.4)
            lfb_mb = self._largest_free_block_mb()
        return lfb_mb

    async def start(self) -> None:
        assert_endpoint_allowed(self.base_url, label="the llama.cpp runtime")
        await self._stop_server()
        await self._cleanup_stale_inference_processes()
        self._ensure_jetson_clocks_sudoers()
        self._maximize_gpu_clocks()
        binary = _find_llama_server()
        env = os.environ.copy()
        apply_airgap_env(env, enabled=self.airgapped)
        bin_dir = os.path.dirname(binary)
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        # Use cudaMallocManaged on Jetson unified memory so CUDA doesn't
        # need physically contiguous pages for large allocations.
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
        env.setdefault("CUDA_MODULE_LOADING", "LAZY")

        resolved_device = self._resolve_device()
        if resolved_device == "cuda" and self._is_jetson_platform():
            # Jetson can fail large unified-memory allocations even when total
            # RAM is available. A patched llama.cpp build uses these knobs to
            # back CUDA/VMM buffers with 1 MiB chunks and smaller VA reserves.
            env.setdefault("GGML_CUDA_VMM_CHUNK_MB", _JETSON_VMM_CHUNK_MB)
            env.setdefault("GGML_CUDA_VMM_RESERVE_MB", _JETSON_VMM_RESERVE_MB)
        lfb_mb = await self._prepare_memory_for_launch() if resolved_device == "cuda" else None
        requested_ngl = self.gpu_layers if resolved_device in ("cuda", "vulkan") else 0
        requested_ctx = self.context_window_tokens
        batch, ubatch, fit_off, no_warmup = self._startup_profile_for_lfb(lfb_mb)
        startup_snapshot = self._memory_snapshot() if resolved_device == "cuda" else {}
        self._emit_diagnostic(
            "runtime_llama_starting",
            model=Path(self.model).name,
            resolved_device=resolved_device,
            requested_ctx=requested_ctx,
            requested_ngl=requested_ngl,
            batch=batch,
            ubatch=ubatch,
            fit_off=fit_off,
            no_warmup=no_warmup,
            jetson_platform=self._is_jetson_platform(),
            airgapped=self.airgapped,
            **startup_snapshot,
        )
        await self._start_once(
            binary=binary,
            env=env,
            ngl=requested_ngl,
            ctx=requested_ctx,
            batch=batch,
            ubatch=ubatch,
            fit_off=fit_off,
            no_warmup=no_warmup,
        )
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
        fit_off: bool,
        no_warmup: bool,
    ) -> None:
        # Ensure swap state directory exists for KV cache save/restore.
        slot_save_dir = Path(".openjet/state/swap")
        slot_save_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
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
            "--slot-save-path",
            str(slot_save_dir.resolve()),
        ]
        if fit_off:
            cmd.extend(["--fit", "off"])
        if no_warmup:
            cmd.append("--no-warmup")
        cmd.extend([
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
        ])
        self._emit_diagnostic(
            "runtime_llama_launch_command",
            command=" ".join(cmd),
            ctx=ctx,
            ngl=ngl,
            batch=batch,
            ubatch=ubatch,
            fit_off=fit_off,
            no_warmup=no_warmup,
            cuda_module_loading=env.get("CUDA_MODULE_LOADING", ""),
            ggml_cuda_enable_unified_memory=env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY", ""),
            ggml_cuda_vmm_chunk_mb=env.get("GGML_CUDA_VMM_CHUNK_MB", ""),
            ggml_cuda_vmm_reserve_mb=env.get("GGML_CUDA_VMM_RESERVE_MB", ""),
        )
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_lines: deque[str] = deque(maxlen=120)
        stderr_task = asyncio.create_task(self._capture_startup_stderr(stderr_lines))

        deadline = asyncio.get_event_loop().time() + 120.0
        while asyncio.get_event_loop().time() < deadline:
            if self._proc.returncode is not None:
                await stderr_task
                err = "\n".join(stderr_lines).strip()
                if len(err) > 8000:
                    keep = 4000
                    omitted = len(err) - (keep * 2)
                    err = (
                        f"{err[:keep]}\n"
                        f"... <{omitted} chars omitted> ...\n"
                        f"{err[-keep:]}"
                    )
                detail = err or "no stderr output"
                self._emit_diagnostic(
                    "runtime_llama_start_failed",
                    error=detail,
                    ctx=ctx,
                    ngl=ngl,
                    batch=batch,
                    ubatch=ubatch,
                    fit_off=fit_off,
                    no_warmup=no_warmup,
                    returncode=self._proc.returncode,
                    stderr_line_count=len(stderr_lines),
                    stderr_tail=list(stderr_lines)[-20:],
                    **self._memory_snapshot(),
                )
                raise RuntimeError(
                    f"llama-server exited with code {self._proc.returncode}: {detail}"
                )
            try:
                r = await self._http.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    stderr_task.cancel()
                    try:
                        await stderr_task
                    except asyncio.CancelledError:
                        pass
                    self._emit_diagnostic(
                        "runtime_llama_start_ready",
                        pid=self._proc.pid if self._proc else 0,
                        ctx=ctx,
                        ngl=ngl,
                        stderr_line_count=len(stderr_lines),
                    )
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
        stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass
        self._emit_diagnostic(
            "runtime_llama_start_timeout",
            ctx=ctx,
            ngl=ngl,
            stderr_line_count=len(stderr_lines),
            stderr_tail=list(stderr_lines)[-20:],
            **self._memory_snapshot(),
        )
        raise TimeoutError("llama-server did not become ready")

    async def _capture_startup_stderr(self, sink: deque[str]) -> None:
        if not self._proc or not self._proc.stderr:
            return
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                return
            text = line.decode(errors="replace").rstrip()
            if not text:
                continue
            sink.append(text)
            lowered = text.lower()
            if any(token in lowered for token in ("cuda error", "out of memory", "nvmap", "ggml", "llama_context", "load_tensors")):
                self._emit_diagnostic(
                    "runtime_llama_stderr_line",
                    summary=text[:200],
                )

    async def _cleanup_stale_inference_processes(self) -> None:
        stale_pids = self._find_stale_inference_pids()
        if not stale_pids:
            return
        self._emit_diagnostic(
            "runtime_llama_cleanup_stale_processes",
            stale_pid_count=len(stale_pids),
            stale_pids=stale_pids[:8],
        )

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
        if self.device in {"cuda", "cpu", "vulkan"}:
            return self.device
        if os.path.exists("/usr/local/cuda") or os.path.exists("/dev/nvhost-gpu"):
            return "cuda"
        from .hardware import _detect_vulkan
        if _detect_vulkan():
            return "vulkan"
        return "cpu"

    async def save_kv_cache(self, path: Path) -> bool:
        """Save KV cache for slot 0 to *path* via llama-server slot API.

        The server resolves *filename* relative to ``--slot-save-path``,
        so we pass just the stem (filename without directory).
        """
        try:
            r = await self._http.post(
                f"{self.base_url}/slots/0?action=save",
                json={"filename": path.name},
                timeout=60.0,
            )
            return r.status_code == 200
        except (httpx.HTTPError, Exception) as exc:
            self._emit_diagnostic("runtime_llama_kv_save_failed", error=str(exc))
            return False

    async def restore_kv_cache(self, path: Path) -> bool:
        """Restore KV cache for slot 0 from *path* via llama-server slot API."""
        try:
            r = await self._http.post(
                f"{self.base_url}/slots/0?action=restore",
                json={"filename": path.name},
                timeout=60.0,
            )
            return r.status_code == 200
        except (httpx.HTTPError, Exception) as exc:
            self._emit_diagnostic("runtime_llama_kv_restore_failed", error=str(exc))
            return False

    async def chat_stream(
        self, messages: list[dict], *, use_tools: bool = True
    ) -> AsyncIterator[StreamChunk]:
        assert_endpoint_allowed(self.base_url, label="the llama.cpp runtime")
        extra_body: dict[str, object] | None = None
        if self.reasoning_mode == "on":
            extra_body = {
                "reasoning_format": "auto",
                "chat_template_kwargs": {"enable_thinking": True},
            }
        elif self.reasoning_mode == "off":
            extra_body = {
                "reasoning_format": "none",
                "chat_template_kwargs": {"enable_thinking": False},
            }
        async for chunk in stream_openai_chat(
            self._http,
            base_url=self.base_url,
            model="local",
            messages=messages,
            use_tools=use_tools,
            extra_body=extra_body,
        ):
            yield chunk
