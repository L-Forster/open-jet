"""HTTP client for SGLang's OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shlex
import shutil
import signal
import sys
from pathlib import Path
from urllib.parse import urlsplit
from typing import AsyncIterator

import httpx

from .airgap import apply_airgap_env, assert_endpoint_allowed
from .runtime_protocol import StreamChunk, stream_openai_chat


def _ensure_sglang_module() -> None:
    if importlib.util.find_spec("sglang.launch_server") is None:
        raise FileNotFoundError(
            "SGLang is not installed in this environment. "
            "Install it with `uv pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python&egg=sglang[all]'`."
        )


class SglangServerClient:
    """Connects to SGLang and optionally manages a local server process."""

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        base_url: str | None = None,
        context_window_tokens: int = 8192,
        device: str = "cuda",
        mem_fraction_static: float = 0.8,
        tensor_parallel_size: int = 1,
        dtype: str = "half",
        attention_backend: str | None = None,
        reasoning_parser: str | None = None,
        tool_call_parser: str | None = None,
        trust_remote_code: bool = True,
        language_model_only: bool = False,
        served_model_name: str = "local",
        launch_mode: str = "auto",
        jetson_container_executable: str = "jetson-containers",
        jetson_autotag_executable: str = "autotag",
        jetson_container_image: str | None = None,
        jetson_container_extra_args: list[str] | None = None,
        docker_executable: str = "docker",
        docker_image: str | None = None,
        docker_container_name: str = "open-jet-sglang",
        docker_use_host_network: bool = True,
        docker_runtime: str = "nvidia",
        docker_extra_args: list[str] | None = None,
        airgapped: bool = False,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.context_window_tokens = max(512, int(context_window_tokens))
        self.device = (device or "cuda").strip().lower()
        self.mem_fraction_static = max(0.1, min(0.95, float(mem_fraction_static)))
        self.tensor_parallel_size = max(1, int(tensor_parallel_size))
        self.dtype = (dtype or "half").strip()
        self.attention_backend = (attention_backend or "").strip() or None
        self.reasoning_parser = (reasoning_parser or "").strip() or None
        self.tool_call_parser = (tool_call_parser or "").strip() or None
        self.trust_remote_code = bool(trust_remote_code)
        self.language_model_only = bool(language_model_only)
        self.served_model_name = (served_model_name or "local").strip() or "local"
        normalized_mode = (launch_mode or "auto").strip().lower()
        self.launch_mode = (
            normalized_mode
            if normalized_mode in {"auto", "managed", "external", "docker", "jetson_container"}
            else "auto"
        )
        self.jetson_container_executable = (jetson_container_executable or "jetson-containers").strip() or "jetson-containers"
        self.jetson_autotag_executable = (jetson_autotag_executable or "autotag").strip() or "autotag"
        self.jetson_container_image = (jetson_container_image or "").strip() or None
        self.jetson_container_extra_args = [
            str(arg) for arg in (jetson_container_extra_args or []) if str(arg).strip()
        ]
        self.docker_executable = (docker_executable or "docker").strip() or "docker"
        self.docker_image = (docker_image or "").strip() or None
        self.docker_container_name = (docker_container_name or "open-jet-sglang").strip() or "open-jet-sglang"
        self.docker_use_host_network = bool(docker_use_host_network)
        self.docker_runtime = (docker_runtime or "").strip() or None
        self.docker_extra_args = [str(arg) for arg in (docker_extra_args or []) if str(arg).strip()]
        self.airgapped = bool(airgapped)
        self.gpu_layers = 0
        self.base_url = _normalize_base_url(base_url, host=self.host, port=self.port)
        parsed = urlsplit(self.base_url)
        if parsed.hostname:
            self.host = parsed.hostname
        if parsed.port:
            self.port = parsed.port
        self._http = httpx.AsyncClient(timeout=120.0)
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        assert_endpoint_allowed(self.base_url, label="the SGLang runtime")
        await self._stop_server()
        if self.launch_mode == "external":
            await self._wait_until_ready(deadline_seconds=30.0)
            return
        if self.launch_mode == "jetson_container":
            await self._start_jetson_container_server()
            await self._wait_until_ready(deadline_seconds=180.0)
            return
        if self.launch_mode == "docker":
            await self._start_docker_server()
            await self._wait_until_ready(deadline_seconds=180.0)
            return
        if self.launch_mode == "auto":
            await self._start_auto()
            return

        _ensure_sglang_module()

        cmd = [sys.executable, "-m", "sglang.launch_server", *self._server_args(self.model)]

        env = os.environ.copy()
        apply_airgap_env(env, enabled=self.airgapped)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        await self._wait_until_ready(deadline_seconds=180.0)

    async def close(self) -> None:
        await self._stop_server()
        await self._http.aclose()

    async def reset_kv_cache(self) -> None:
        try:
            resp = await self._http.post(f"{self.base_url}/flush_cache")
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        if self.launch_mode == "external":
            return
        await self._stop_server()
        await self.start()

    async def save_kv_cache(self, path: Path) -> bool:
        del path
        return False

    async def restore_kv_cache(self, path: Path) -> bool:
        del path
        return False

    async def _stop_server(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                self._proc.kill()
        self._proc = None

    async def _wait_until_ready(self, *, deadline_seconds: float) -> None:
        deadline = asyncio.get_event_loop().time() + deadline_seconds
        while asyncio.get_event_loop().time() < deadline:
            if self._proc and self._proc.returncode is not None:
                err = (await self._proc.stderr.read()).decode(errors="replace").strip()
                raise RuntimeError(
                    f"sglang exited with code {self._proc.returncode}: {err or 'no stderr output'}"
                )
            for endpoint in ("/health", "/v1/models"):
                try:
                    resp = await self._http.get(f"{self.base_url}{endpoint}")
                    if resp.status_code == 200:
                        return
                except (
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.RemoteProtocolError,
                    httpx.TimeoutException,
                ):
                    pass
            await asyncio.sleep(1.0)
        if self.launch_mode == "external":
            raise TimeoutError(
                f"SGLang external server did not respond at {self.base_url} within {int(deadline_seconds)}s"
            )
        raise TimeoutError("sglang did not become ready")

    async def chat_stream(
        self, messages: list[dict], *, use_tools: bool = True
    ) -> AsyncIterator[StreamChunk]:
        assert_endpoint_allowed(self.base_url, label="the SGLang runtime")
        async for chunk in stream_openai_chat(
            self._http,
            base_url=self.base_url,
            model=self.served_model_name,
            messages=messages,
            use_tools=use_tools and bool(self.tool_call_parser),
        ):
            yield chunk

    async def _start_docker_server(self) -> None:
        docker_image = await self._resolve_docker_image()
        if not docker_image:
            raise ValueError(
                "No local SGLang Docker image found. Set `OPEN_JET_SGLANG_DOCKER_IMAGE` "
                "or configure `sglang_docker_image`."
            )

        model_path_in_container, mount_args = self._docker_model_mount_args()
        cmd = [self.docker_executable, "run", "--rm", "--name", self.docker_container_name]
        if self.docker_runtime:
            cmd.extend(["--runtime", self.docker_runtime])
        if self.docker_use_host_network:
            cmd.append("--network=host")
        else:
            cmd.extend(["-p", f"{self.port}:{self.port}"])
        cmd.extend(["--ipc=host", *self.docker_extra_args, *mount_args, docker_image])
        cmd.extend(["python", "-m", "sglang.launch_server", *self._server_args(model_path_in_container)])

        env = os.environ.copy()
        apply_airgap_env(env, enabled=self.airgapped)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _start_jetson_container_server(self) -> None:
        runner = shutil.which(self.jetson_container_executable)
        if not runner:
            raise FileNotFoundError(
                f"{self.jetson_container_executable} not found. Install jetson-containers."
            )

        image = self.jetson_container_image
        if not image:
            autotag = shutil.which(self.jetson_autotag_executable)
            if not autotag:
                raise FileNotFoundError(
                    f"{self.jetson_autotag_executable} not found. Install jetson-containers or set `sglang_jetson_container_image`."
                )
            image_expr = f"$({shlex.quote(autotag)} sglang)"
        else:
            image_expr = shlex.quote(image)

        model_path_in_container, mount_args = self._docker_model_mount_args()
        run_args = [
            shlex.quote(runner),
            "run",
            *[shlex.quote(arg) for arg in [*self.jetson_container_extra_args, *mount_args]],
            image_expr,
            *[shlex.quote(arg) for arg in ["python", "-m", "sglang.launch_server", *self._server_args(
                model_path_in_container,
                host_override="0.0.0.0",
            )]],
        ]
        cmd = " ".join(arg for arg in run_args if arg)
        env = os.environ.copy()
        apply_airgap_env(env, enabled=self.airgapped)
        self._proc = await asyncio.create_subprocess_shell(
            cmd,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _start_auto(self) -> None:
        try:
            _ensure_sglang_module()
            cmd = [sys.executable, "-m", "sglang.launch_server", *self._server_args(self.model)]
            env = os.environ.copy()
            apply_airgap_env(env, enabled=self.airgapped)
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            await self._wait_until_ready(deadline_seconds=45.0)
            return
        except Exception as exc:
            if not await self._should_fallback_to_docker(exc):
                raise
            await self._stop_server()
            await self._start_docker_server()
            await self._wait_until_ready(deadline_seconds=180.0)

    async def _should_fallback_to_docker(self, exc: Exception) -> bool:
        detail = str(exc)
        if "No module named 'triton'" not in detail and "SGLang is not installed" not in detail:
            return False
        return bool(await self._resolve_docker_image())

    async def _resolve_docker_image(self) -> str | None:
        if self.docker_image:
            return self.docker_image

        env_image = os.environ.get("OPEN_JET_SGLANG_DOCKER_IMAGE", "").strip()
        if env_image:
            return env_image

        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_executable,
                "images",
                "--format",
                "{{.Repository}}:{{.Tag}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return None

        stdout, _stderr = await proc.communicate()
        if proc.returncode != 0:
            return None

        images = [
            line.strip()
            for line in stdout.decode(errors="replace").splitlines()
            if line.strip() and "<none>" not in line
        ]
        candidates = [image for image in images if "sglang" in image.lower()]
        if not candidates:
            return None
        for preferred in candidates:
            lowered = preferred.lower()
            if "jetson" in lowered or "l4t" in lowered:
                return preferred
        if len(candidates) == 1:
            return candidates[0]
        return candidates[0]

    def _server_args(self, model_path: str, *, host_override: str | None = None) -> list[str]:
        args = [
            "--model-path",
            model_path,
            "--host",
            host_override or self.host,
            "--port",
            str(self.port),
            "--tp-size",
            str(self.tensor_parallel_size),
            "--mem-fraction-static",
            str(self.mem_fraction_static),
            "--context-length",
            str(self.context_window_tokens),
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            "--served-model-name",
            self.served_model_name,
        ]
        if self.attention_backend:
            args.extend(["--attention-backend", self.attention_backend])
        if self.reasoning_parser:
            args.extend(["--reasoning-parser", self.reasoning_parser])
        if self.tool_call_parser:
            args.extend(["--tool-call-parser", self.tool_call_parser])
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        if self.language_model_only:
            args.append("--language-only")
        return args

    def _docker_model_mount_args(self) -> tuple[str, list[str]]:
        model_path = Path(self.model).expanduser()
        if not model_path.exists():
            return self.model, []
        source_dir = model_path.parent
        target_dir = "/models"
        model_in_container = str(Path(target_dir) / model_path.name)
        return model_in_container, ["-v", f"{source_dir}:{target_dir}:ro"]


def _normalize_base_url(base_url: str | None, *, host: str, port: int) -> str:
    src = (base_url or "").strip()
    if src:
        return src.rstrip("/")
    return f"http://{host}:{port}"
