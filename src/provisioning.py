from __future__ import annotations

import asyncio
import errno
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlparse

import httpx

if os.name == "posix":
    import pty
else:
    pty = None  # type: ignore[assignment]

from .app_paths import openjet_install_root
from .config import setup_direct_model_catalog
from .hardware import HardwareInfo, _darwin_sysctl, is_jetson_label, recommended_context_window_tokens_from_total, running_on_jetson
from .setup_memory import recommend_context_window_for_model

def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    return f"{nbytes / (1 << 10):.0f} KB"


def _fmt_mb_size(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024.0:.1f} GB"
    return f"{mb:.0f} MB"


OPENJET_HOME = openjet_install_root()
MODELS_DIR = OPENJET_HOME / "models"
BIN_DIR = OPENJET_HOME / "bin"
LLAMA_CPP_DIR = OPENJET_HOME / "llama.cpp"
LLAMA_SERVER_EXE_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"
LLAMA_SERVER_BIN = BIN_DIR / LLAMA_SERVER_EXE_NAME
LLAMA_CPP_TAG_FILE = BIN_DIR / "llama-server.tag"
LLAMA_CPP_REPO_URL = "https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_CPP_PINNED_REF = "b9442"
UNIFIED_MEMORY_SYSTEM_RESERVE_MB = 4096.0


def _running_under_wsl() -> bool:
    if sys.platform != "linux":
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    for path in (Path("/proc/sys/kernel/osrelease"), Path("/proc/version")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if "microsoft" in text or "wsl" in text:
            return True
    return False


def _path_without_windows_interop_entries(path_value: str) -> str:
    entries = []
    for entry in path_value.split(os.pathsep):
        normalized = entry.replace("\\", "/").lower()
        if re.match(r"^/mnt/[a-z]/", normalized):
            continue
        entries.append(entry)
    return os.pathsep.join(entries)


def _native_tool_path(name: str) -> str | None:
    path_value = os.environ.get("PATH")
    if path_value is not None and _running_under_wsl():
        path_value = _path_without_windows_interop_entries(path_value)
    return shutil.which(name, path=path_value)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    if _running_under_wsl():
        env["PATH"] = _path_without_windows_interop_entries(env.get("PATH", ""))
    return env


def managed_llama_cpp_ref() -> str:
    ref = os.environ.get("OPENJET_LLAMA_CPP_REF", "").strip()
    return ref or LLAMA_CPP_PINNED_REF


def _normalized_llama_cpp_ref(ref: str) -> str:
    stripped = str(ref or "").strip()
    if stripped in {"mtp", "qwen-mtp", "qwen3.6-mtp", "pull/22673/head", "refs/pull/22673/head", "mtp-pr"}:
        return LLAMA_CPP_PINNED_REF
    return stripped


def _is_mtp_llama_cpp_ref(ref: str) -> bool:
    normalized = str(ref or "").strip().lower()
    return normalized in {"pull/22673/head", "refs/pull/22673/head", "mtp-pr", "mtp", "qwen-mtp", "qwen3.6-mtp"}


def _model_path_looks_mtp(value: object) -> bool:
    stem = Path(str(value or "")).name.lower()
    return bool(re.search(r"(?:^|[-_.])mtp(?:[-_.]|$)", stem))


def _setup_uses_mtp_model(setup_result: Mapping[str, Any]) -> bool:
    if bool(setup_result.get("llama_mtp")):
        return True
    for key in ("llama_model", "model_download_path", "filename", "model_profile_name"):
        if _model_path_looks_mtp(setup_result.get(key)):
            return True
    return False


def _llama_cpp_ref_for_setup(setup_result: Mapping[str, Any]) -> str:
    del setup_result
    return managed_llama_cpp_ref()


def cmake_install_command() -> str:
    if sys.platform == "win32":
        return "python -m pip install cmake"
    if sys.platform == "darwin":
        return "brew install cmake"
    if _linux_os_id() == "arch":
        return "sudo pacman -S cmake"
    return "sudo apt install cmake"


def missing_cmake_message() -> str:
    return (
        "cmake not found on PATH. "
        f"Install CMake with `{cmake_install_command()}`, then rerun `openjet setup`."
    )


def _linux_os_id() -> str:
    if sys.platform != "linux":
        return ""
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("ID="):
                return line.split("=", 1)[1].strip().strip('"').lower()
    except OSError:
        return ""
    return ""


def cuda_toolkit_available() -> bool:
    if _native_tool_path("nvcc") is not None:
        return True
    return any(os.environ.get(name) for name in ("CUDA_PATH", "CUDA_HOME"))


def cuda_toolkit_install_command() -> str:
    if sys.platform == "win32":
        return "install the NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads"
    if _linux_os_id() == "arch":
        return "sudo pacman -S cuda"
    return "sudo apt install nvidia-cuda-toolkit"


def missing_cuda_toolkit_message() -> str:
    return (
        "CUDA Toolkit not found. "
        f"{cuda_toolkit_install_command()}, then rerun `openjet setup`."
    )


def _context_window_for_model(
    hardware_info: HardwareInfo,
    model_size_mb: float,
    kv_bytes_per_token: float,
) -> int:
    has_gpu = (
        hardware_info.has_cuda or hardware_info.has_rocm
        or hardware_info.has_vulkan or hardware_info.has_metal
    )
    vram_mb = hardware_info.vram_mb
    if hardware_info.has_metal:
        vram_mb = hardware_info.total_ram_gb * 1024.0
    fallback_tokens = recommended_context_window_tokens_from_total(
        hardware_info.total_ram_gb,
        headless=False,
    )
    device = "cpu"
    if hardware_info.has_cuda:
        device = "cuda"
    elif hardware_info.has_rocm:
        device = "rocm"
    elif hardware_info.has_vulkan:
        device = "vulkan"
    elif hardware_info.has_metal:
        device = "metal"
    if has_gpu and vram_mb > 0 and model_size_mb > 0 and kv_bytes_per_token > 0:
        return recommend_context_window_for_model(
            device=device,
            fallback_tokens=fallback_tokens,
            model_size_mb=model_size_mb,
            kv_bytes_per_token=kv_bytes_per_token,
            total_vram_mb=vram_mb,
        )
    return fallback_tokens


def _is_moe_catalog_row(row: Mapping[str, object]) -> bool:
    if bool(row.get("unified_memory_only")):
        return True
    text = " ".join(str(row.get(key) or "") for key in ("label", "filename"))
    return re.search(r"\bA\d+(?:\.\d+)?B\b", text, flags=re.IGNORECASE) is not None


def _catalog_float(row: Mapping[str, object], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _active_model_size_mb(row: Mapping[str, object]) -> float:
    for key in ("resident_model_size_mb", "active_model_size_mb"):
        value = _catalog_float(row, key)
        if value > 0:
            return value
    text = " ".join(str(row.get(key) or "") for key in ("label", "filename"))
    match = re.search(r"\bA(\d+(?:\.\d+)?)B\b", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) * 1024.0
    return _catalog_float(row, "model_size_mb")


def _select_direct_model(
    model_catalog: tuple[dict[str, object], ...],
    hardware_info: HardwareInfo,
    *,
    unified_memory: bool = False,
) -> dict[str, object]:
    has_gpu = hardware_info.has_cuda or hardware_info.has_rocm or hardware_info.has_vulkan or hardware_info.has_metal
    total_ram_mb = max(hardware_info.total_ram_gb, 0.0) * 1024.0
    model_ram_mb = max(
        0.0,
        total_ram_mb - (UNIFIED_MEMORY_SYSTEM_RESERVE_MB if unified_memory else 0.0),
    )
    vram_mb = total_ram_mb if hardware_info.has_metal else max(hardware_info.vram_mb, 0.0)
    if unified_memory:
        vram_or_ram_budget_mb = model_ram_mb * 0.9
        combined_budget_mb = model_ram_mb * 0.9
    else:
        vram_or_ram_budget_mb = (vram_mb if has_gpu and vram_mb > 0 else total_ram_mb) * 0.9
        combined_budget_mb = (total_ram_mb + (vram_mb if has_gpu and not hardware_info.has_metal else 0.0)) * 0.9

    dense_rows = [row for row in model_catalog if not _is_moe_catalog_row(row)]
    moe_rows = [row for row in model_catalog if _is_moe_catalog_row(row)]

    def largest(rows: list[dict[str, object]]) -> dict[str, object] | None:
        return max(rows, key=lambda row: (bool(row.get("llama_cpu_moe")), _catalog_float(row, "model_size_mb"), _catalog_float(row, "max_ram_gb"))) if rows else None

    if hardware_info.has_metal:
        metal_moe = largest([
            row for row in moe_rows
            if 0 < _active_model_size_mb(row) <= combined_budget_mb
            and _catalog_float(row, "max_ram_gb") <= hardware_info.total_ram_gb
        ])
        if metal_moe:
            return metal_moe

    large_dense = largest([
        row for row in dense_rows
        if _catalog_float(row, "max_ram_gb") >= 24.0
        and 0 < _catalog_float(row, "model_size_mb") <= vram_or_ram_budget_mb
    ])
    if large_dense:
        return large_dense

    moe = largest([
        row for row in moe_rows
        if has_gpu
        and 0 < _catalog_float(row, "model_size_mb") <= combined_budget_mb
        and _active_model_size_mb(row) <= vram_or_ram_budget_mb
    ])
    if moe:
        return moe

    dense_budget_mb = (
        vram_or_ram_budget_mb
        if has_gpu and vram_mb > 0
        else max(vram_or_ram_budget_mb, total_ram_mb * 0.9)
    )
    dense = largest([
        row for row in dense_rows
        if 0 < _catalog_float(row, "model_size_mb") <= dense_budget_mb
    ])
    if dense:
        return dense

    for row in reversed(model_catalog):
        if max(hardware_info.total_ram_gb, 0.0) >= _catalog_float(row, "max_ram_gb"):
            return row
    return model_catalog[0]


def recommend_direct_model(
    hardware_info: HardwareInfo,
    *,
    cfg: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    model_catalog = setup_direct_model_catalog(cfg)
    unified_memory = hardware_info.has_metal or running_on_jetson()
    if not unified_memory:
        model_catalog = tuple(
            row for row in model_catalog if not _is_moe_catalog_row(row)
        )
    selected = _select_direct_model(model_catalog, hardware_info, unified_memory=unified_memory)
    filename = str(selected["filename"])
    model_size_mb = float(selected.get("model_size_mb", 0) or 0)
    kv_bytes_per_token = float(selected.get("kv_bytes_per_token", 0) or 0)
    context_model_size_mb = _active_model_size_mb(selected)
    return {
        "label": str(selected["label"]),
        "filename": filename,
        "url": str(selected["url"]),
        "target_path": str(MODELS_DIR / filename),
        "model_size_mb": model_size_mb,
        "active_model_size_mb": _active_model_size_mb(selected) if _is_moe_catalog_row(selected) else 0.0,
        "kv_bytes_per_token": kv_bytes_per_token,
        "llama_cpu_moe": bool(selected.get("llama_cpu_moe", False)),
        "llama_n_cpu_moe": int(selected.get("llama_n_cpu_moe", 0) or 0),
        "llama_mtp": _model_path_looks_mtp(filename) or bool(selected.get("llama_mtp", False)),
        "unified_memory_only": bool(selected.get("unified_memory_only", False)),
        "context_window_tokens": _context_window_for_model(
            hardware_info,
            context_model_size_mb,
            kv_bytes_per_token,
        ),
    }


def current_llama_server_path() -> str | None:
    found = shutil.which("llama-server")
    if found:
        return found
    fallback = LLAMA_CPP_DIR / "build" / "bin" / LLAMA_SERVER_EXE_NAME
    if fallback.is_file():
        return str(fallback)
    if LLAMA_SERVER_BIN.is_file():
        return str(LLAMA_SERVER_BIN)
    return None


def _configured_llama_server_path(setup_result: Mapping[str, Any]) -> str | None:
    raw = str(setup_result.get("llama_server_path") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_file():
        return str(path)
    return None


async def _run_exec(*args: str, cwd: Path | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        env=_subprocess_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_raw, err_raw = await proc.communicate()
    return proc.returncode or 0, out_raw.decode(errors="replace"), err_raw.decode(errors="replace")


_GIT_NETWORK_OPTIONS: tuple[str, ...] = (
    "-c",
    "http.version=HTTP/1.1",
    "-c",
    "http.postBuffer=524288000",
)
_GIT_RETRY_MARKERS: tuple[str, ...] = (
    "early eof",
    "fetch-pack",
    "invalid index-pack",
    "rpc failed",
    "stream",
    "unexpected disconnect",
)


def _git_network_args(*args: str) -> tuple[str, ...]:
    return ("git", *_GIT_NETWORK_OPTIONS, *args)


def _looks_like_transient_git_network_error(out: str, err: str) -> bool:
    text = f"{out}\n{err}".lower()
    return any(marker in text for marker in _GIT_RETRY_MARKERS)


async def _run_git_network(*args: str, cwd: Path | None = None, attempts: int = 3) -> tuple[int, str, str]:
    last: tuple[int, str, str] = (1, "", "")
    for attempt in range(1, max(1, attempts) + 1):
        last = await _run_exec(*_git_network_args(*args), cwd=cwd)
        rc, out, err = last
        if rc == 0:
            return last
        if attempt >= attempts or not _looks_like_transient_git_network_error(out, err):
            return last
        await asyncio.sleep(float(attempt))
    return last


_BUILD_PROGRESS_NINJA_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")
_BUILD_PROGRESS_MAKE_RE = re.compile(r"\[\s*(\d+)\s*%\s*\]")


async def _run_build_with_progress(
    *args: str,
    cwd: Path,
    set_status: Callable[[str], None],
    log: Any,
) -> tuple[int, str]:
    tail: list[str] = []
    last_logged_pct = -1

    def handle_line(line: str) -> None:
        nonlocal last_logged_pct
        tail.append(line)
        if len(tail) > 200:
            del tail[: len(tail) - 200]
        pct: int | None = None
        detail: str | None = None
        ninja = _BUILD_PROGRESS_NINJA_RE.search(line)
        if ninja:
            n, m = int(ninja.group(1)), int(ninja.group(2))
            if m > 0:
                pct = int(n * 100 / m)
                detail = f"{n}/{m}"
        else:
            make = _BUILD_PROGRESS_MAKE_RE.search(line)
            if make:
                pct = int(make.group(1))
                detail = f"{pct}%"
        if pct is not None:
            set_status(f"building llama-server {detail}")
            if pct - last_logged_pct >= 25 and pct < 100:
                log.write(f"  [dim]Building llama-server {pct}%...[/]")
                last_logged_pct = pct

    if os.name == "posix":
        master_fd, slave_fd = pty.openpty()
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(cwd),
                env=_subprocess_env(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
            )
        finally:
            os.close(slave_fd)
        pending = ""
        try:
            while True:
                try:
                    raw = await asyncio.to_thread(os.read, master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not raw:
                    break
                pending += raw.decode(errors="replace")
                parts = pending.replace("\r", "\n").split("\n")
                pending = parts.pop() if parts else ""
                for part in parts:
                    if part.strip():
                        handle_line(part.rstrip())
            if pending.strip():
                handle_line(pending.rstrip())
        finally:
            os.close(master_fd)
        await proc.wait()
        return proc.returncode or 0, "\n".join(tail)

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd),
        env=_subprocess_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        handle_line(raw.decode(errors="replace").rstrip())
    await proc.wait()
    return proc.returncode or 0, "\n".join(tail)


def _parse_huggingface_resolve_url(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        return None
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    try:
        resolve_index = parts.index("resolve")
    except ValueError:
        return None
    if resolve_index < 2 or len(parts) <= resolve_index + 2:
        return None
    repo_id = "/".join(parts[:resolve_index])
    revision = parts[resolve_index + 1]
    filename = "/".join(parts[resolve_index + 2 :])
    if not repo_id or not revision or not filename:
        return None
    return repo_id, revision, filename


def _hf_cli_command() -> list[str]:
    hf = _native_tool_path("hf")
    if hf:
        return [hf, "download"]
    huggingface_cli = _native_tool_path("huggingface-cli")
    if huggingface_cli:
        return [huggingface_cli, "download"]
    raise RuntimeError(
        "Hugging Face CLI is required for model downloads. Install OpenJet dependencies "
        "or run `pip install 'huggingface_hub[hf_transfer]' hf_transfer`."
    )


def pending_direct_model_download_summary(setup_result: Mapping[str, Any]) -> str | None:
    if str(setup_result.get("model_source", "local")) != "direct":
        return None
    target_raw = str(setup_result.get("model_download_path") or "").strip()
    if not target_raw:
        return None
    target_path = Path(target_raw).expanduser()
    if target_path.is_file():
        return None
    name = target_path.name or "model"
    try:
        model_size_mb = float(setup_result.get("model_size_mb") or 0.0)
    except (TypeError, ValueError):
        model_size_mb = 0.0
    if model_size_mb > 0:
        return f"Requires model download after restart: {name} ({_fmt_mb_size(model_size_mb)})."
    return f"Requires model download after restart: {name}."


def _clear_old_model_files(model_dir: Path, target_path: Path, *, log: Any) -> int:
    if not model_dir.is_dir():
        return 0
    removed = 0
    target_resolved = target_path.resolve(strict=False)
    for candidate in sorted(model_dir.glob("*.gguf")):
        if candidate.resolve(strict=False) == target_resolved:
            continue
        if not candidate.is_file():
            continue
        candidate.unlink()
        removed += 1
    if removed:
        log.write(f"  [dim]Removed {removed} old model file{'s' if removed != 1 else ''} from {model_dir}.[/]")
    return removed


async def _run_hf_cli_download(
    *,
    repo_id: str,
    filename: str,
    revision: str,
    local_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> tuple[int, str, str]:
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    cmd = [
        *_hf_cli_command(),
        repo_id,
        filename,
        "--revision",
        revision,
        "--local-dir",
        str(local_dir),
    ]

    if os.name == "posix":
        master_fd, slave_fd = pty.openpty()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
            )
        finally:
            os.close(slave_fd)

        try:
            output_parts: list[str] = []
            pending = ""
            while True:
                try:
                    raw = await asyncio.to_thread(os.read, master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not raw:
                    break
                text = raw.decode(errors="replace")
                output_parts.append(text)
                if progress is not None:
                    pending += text
                    parts = pending.replace("\r", "\n").split("\n")
                    pending = parts.pop() if parts else ""
                    for part in parts:
                        clean = part.strip()
                        if clean:
                            progress(clean)
            if progress is not None and pending.strip():
                progress(pending.strip())
        finally:
            os.close(master_fd)
        await proc.wait()
        return proc.returncode or 0, "".join(output_parts), ""

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def collect(stream: asyncio.StreamReader | None) -> str:
        if stream is None:
            return ""
        chunks: list[str] = []
        pending = ""
        while True:
            raw = await stream.read(4096)
            if not raw:
                break
            text = raw.decode(errors="replace")
            chunks.append(text)
            if progress is not None:
                pending += text
                parts = pending.replace("\r", "\n").split("\n")
                pending = parts.pop() if parts else ""
                for part in parts:
                    clean = part.strip()
                    if clean:
                        progress(clean)
        if progress is not None and pending.strip():
            progress(pending.strip())
        return "".join(chunks)

    out, err = await asyncio.gather(collect(proc.stdout), collect(proc.stderr))
    await proc.wait()
    return proc.returncode or 0, out, err


def _llama_cmake_args(hardware_info: HardwareInfo, *, device: str | None = None) -> list[str]:
    args = ["cmake", ".."]
    selected_device = (device or "").strip().lower()
    if not selected_device or selected_device == "auto":
        selected_device = "cuda" if hardware_info.has_cuda else "vulkan" if hardware_info.has_vulkan else ""
    if selected_device == "vulkan" and not hardware_info.has_vulkan and hardware_info.has_cuda:
        selected_device = "cuda"
    if selected_device == "cuda" and hardware_info.has_cuda:
        args.append("-DGGML_CUDA=ON")
        args.append("-DGGML_VULKAN=OFF")
        if is_jetson_label(hardware_info.label):
            args.append("-DCMAKE_CUDA_ARCHITECTURES=87")
    elif selected_device == "vulkan" and hardware_info.has_vulkan:
        args.append("-DGGML_VULKAN=ON")
    return args


def _needs_rebuild(hardware_info: HardwareInfo, existing_binary: str, *, device: str | None = None) -> bool:
    """Check if the existing llama-server needs rebuilding for GPU support."""
    desired_device = (device or "").strip().lower()
    if desired_device == "cpu":
        return False
    if not desired_device:
        desired_device = "cuda" if hardware_info.has_cuda else "vulkan" if hardware_info.has_vulkan else "cpu"
    if desired_device not in {"cuda", "vulkan"}:
        return False
    try:
        ldd_output = os.popen(f"ldd {shlex.quote(existing_binary)} 2>/dev/null").read().lower()
    except Exception:
        return False
    if desired_device == "cuda" and "libcuda" not in ldd_output:
        return True
    if desired_device == "vulkan" and "libvulkan" not in ldd_output:
        return True
    return False


def _managed_source_llama_server_path() -> Path:
    build_bin = LLAMA_CPP_DIR / "build" / "bin"
    direct = build_bin / LLAMA_SERVER_EXE_NAME
    if direct.is_file():
        return direct
    for config_name in ("Release", "RelWithDebInfo", "MinSizeRel", "Debug"):
        candidate = build_bin / config_name / LLAMA_SERVER_EXE_NAME
        if candidate.is_file():
            return candidate
    matches = sorted(build_bin.glob(f"**/{LLAMA_SERVER_EXE_NAME}"))
    if matches:
        return matches[0]
    return direct


def _source_build_tag_path() -> Path:
    return LLAMA_CPP_DIR / "build" / "openjet-llama-server.json"


def _source_build_tag() -> dict[str, str]:
    try:
        raw = _source_build_tag_path().read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _write_source_build_tag(*, ref: str, device: str | None) -> None:
    tag_path = _source_build_tag_path()
    tag_path.parent.mkdir(parents=True, exist_ok=True)
    tag_path.write_text(
        json.dumps({"ref": ref, "device": (device or "auto").strip().lower() or "auto"}, sort_keys=True),
        encoding="utf-8",
    )


def _source_build_device(default: str | None) -> str | None:
    tagged = _source_build_tag().get("device", "").strip().lower()
    if tagged in {"cpu", "cuda", "vulkan", "rocm", "metal"}:
        return tagged
    return default


def _source_checkout_ref_matches(required_ref: str) -> bool:
    ref = _normalized_llama_cpp_ref(required_ref)
    if not ref or not (LLAMA_CPP_DIR / ".git").is_dir():
        return False
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LLAMA_CPP_DIR,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=_subprocess_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    return proc.stdout.strip().lower().startswith(ref.lower())


def _source_build_matches(
    *,
    hardware_info: HardwareInfo,
    required_ref: str,
    desired_device: str | None,
) -> bool:
    binary = _managed_source_llama_server_path()
    if not binary.is_file():
        return False
    tag = _source_build_tag()
    tagged_ref = tag.get("ref", "")
    if tagged_ref:
        ref_matches = tagged_ref == required_ref
    else:
        ref_matches = _source_checkout_ref_matches(required_ref)
    if not ref_matches:
        return False
    tagged_device = tag.get("device", "").strip().lower()
    desired = (desired_device or "").strip().lower()
    if tagged_device in {"cpu", "cuda", "vulkan", "rocm", "metal"} and desired and tagged_device != desired:
        return False
    # The managed build tag is written only after a successful source build.
    # Do not opportunistically rebuild a matching ref just because launch-time
    # hardware preferences changed.
    return True


def _llama_build_command(jobs: int) -> list[str]:
    command = ["cmake", "--build", "."]
    if sys.platform == "win32":
        command.extend(["--config", "Release"])
    command.extend(["--target", "llama-server", "--target", "llama-bench", f"-j{jobs}"])
    return command


async def _sync_managed_llama_cpp_checkout(
    *,
    target_ref: str | None = None,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> str:
    target_ref = _normalized_llama_cpp_ref(target_ref or managed_llama_cpp_ref())
    repo_exists = (LLAMA_CPP_DIR / ".git").is_dir()

    if not repo_exists and LLAMA_CPP_DIR.exists():
        shutil.rmtree(LLAMA_CPP_DIR)

    if not repo_exists:
        set_status("initializing llama.cpp checkout")
        log.write("  [dim]Initializing llama.cpp checkout...[/]")
        LLAMA_CPP_DIR.mkdir(parents=True, exist_ok=True)
        rc, out, err = await _run_exec("git", "init", cwd=LLAMA_CPP_DIR)
        if rc != 0:
            clear_status()
            raise RuntimeError((err or out).strip() or "Failed to initialize llama.cpp checkout")
        rc, out, err = await _run_exec("git", "remote", "add", "origin", LLAMA_CPP_REPO_URL, cwd=LLAMA_CPP_DIR)
        if rc != 0:
            clear_status()
            raise RuntimeError((err or out).strip() or "Failed to configure llama.cpp remote")

    if repo_exists:
        await _run_exec("git", "reset", "--hard", "HEAD", cwd=LLAMA_CPP_DIR)
        await _run_exec("git", "clean", "-fd", cwd=LLAMA_CPP_DIR)
        await _run_exec("git", "remote", "set-url", "origin", LLAMA_CPP_REPO_URL, cwd=LLAMA_CPP_DIR)

    set_status("fetching llama.cpp")
    log.write(f"  [dim]Fetching llama.cpp {target_ref[:12]}...[/]")
    rc, out, err = await _run_git_network("fetch", "--depth=1", "origin", target_ref, cwd=LLAMA_CPP_DIR)
    if rc != 0:
        clear_status()
        raise RuntimeError((err or out).strip() or "Failed to fetch llama.cpp")

    rc, out, err = await _run_exec("git", "checkout", "--detach", "FETCH_HEAD", cwd=LLAMA_CPP_DIR)
    if rc != 0:
        clear_status()
        raise RuntimeError((err or out).strip() or f"Failed to checkout llama.cpp ref {target_ref}.")
    return target_ref


def _prebuilt_asset_candidates(hardware_info: HardwareInfo) -> list[str]:
    """Return substrings identifying release assets that match this host.

    The llama.cpp release assets are named like
    `llama-<tag>-bin-<os>-<variant>-<arch>.zip`. We match by substring so the
    logic stays robust to minor naming churn.

    Returns an empty list when no prebuilt covers this host (e.g. Jetson), which
    causes the caller to fall back to a source build.
    """
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        if machine in {"arm64", "aarch64"} or _darwin_sysctl("hw.optional.arm64") == "1":
            return ["bin-macos-arm64.tar.gz"]
        return ["bin-macos-x64.tar.gz"]
    if sys.platform == "win32":
        if machine in {"x86_64", "amd64"}:
            if hardware_info.has_cuda:
                return ["bin-win-cuda-13.1-x64", "bin-win-cuda-12.4-x64"]
            if hardware_info.has_vulkan:
                return ["bin-win-vulkan-x64"]
            return ["bin-win-cpu-x64"]
        if machine in {"arm64", "aarch64"}:
            return ["bin-win-cpu-arm64"]
    if sys.platform.startswith("linux"):
        if machine in {"x86_64", "amd64"}:
            if hardware_info.has_cuda or hardware_info.has_vulkan:
                return ["bin-ubuntu-vulkan-x64"]
            return ["bin-ubuntu-x64"]
        if hardware_info.has_cuda:
            # Jetson/Linux CUDA has no official release asset; build from source.
            return []
        if machine in {"aarch64", "arm64"}:
            return ["bin-ubuntu-arm64"]
    return []


def _prebuilt_runtime_device(hardware_info: HardwareInfo) -> str | None:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"} and hardware_info.has_cuda:
        return "vulkan"
    if sys.platform == "win32" and hardware_info.has_cuda:
        return "cuda"
    if sys.platform == "darwin" and hardware_info.has_metal:
        return "metal"
    if hardware_info.has_vulkan:
        return "vulkan"
    return None


def _installed_llama_server_tag() -> str | None:
    try:
        return LLAMA_CPP_TAG_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


async def _fetch_release_tag_and_assets(target_ref: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    ref = _normalized_llama_cpp_ref(target_ref or "").strip()
    endpoint = (
        f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{ref}"
        if ref
        else LLAMA_CPP_RELEASES_API
    )
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        headers = {"Accept": "application/vnd.github+json"}
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = await client.get(endpoint, headers=headers)
        response.raise_for_status()
        payload = response.json()
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        raise RuntimeError("llama.cpp release is missing tag_name.")
    assets = payload.get("assets") or []
    if not isinstance(assets, list):
        raise RuntimeError("llama.cpp release returned malformed assets.")
    return tag, assets


_ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")


def _pick_asset(assets: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any] | None:
    for pattern in candidates:
        for asset in assets:
            name = str(asset.get("name") or "")
            if pattern in name and name.endswith(_ARCHIVE_SUFFIXES):
                return asset
    return None


async def _download_to_path(
    url: str,
    target_path: Path,
    *,
    label: str,
    log: Any,
    set_status: Callable[[str], None],
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    set_status(f"downloading {label}")
    log.write(f"  [dim]Downloading {label}...[/]")
    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0
            last_log_pct = -10
            last_log_time = time.monotonic()
            with target_path.open("wb") as fh:
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = max(0, min(100, int(downloaded * 100 / total_bytes)))
                        now = time.monotonic()
                        set_status(f"downloading {label} {pct}%")
                        if pct - last_log_pct >= 20 and now - last_log_time >= 1.0:
                            log.write(f"  [dim]{pct}% ({_fmt_size(downloaded)} / {_fmt_size(total_bytes)})[/]")
                            last_log_pct = pct
                            last_log_time = now


def _extract_archive(archive_path: Path, dest: Path) -> None:
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive_path.name}")


def _install_from_archive(archive_path: Path) -> Path:
    """Extract a llama.cpp release archive, copy binaries into BIN_DIR, return llama-server path."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    for entry in BIN_DIR.glob("llama-*"):
        if entry.is_file():
            entry.unlink()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _extract_archive(archive_path, tmp)
        server_candidates = list(tmp.rglob("llama-server")) + list(tmp.rglob("llama-server.exe"))
        server_candidates = [p for p in server_candidates if p.is_file()]
        if not server_candidates:
            raise RuntimeError("llama-server binary not found in downloaded archive.")
        source_bin_dir = server_candidates[0].parent
        for entry in source_bin_dir.iterdir():
            if not entry.is_file():
                continue
            dest = BIN_DIR / entry.name
            shutil.copy2(entry, dest)
            if entry.name.startswith("llama-") and not entry.suffix:
                dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    installed = LLAMA_SERVER_BIN
    if not installed.is_file():
        raise RuntimeError("llama-server was not installed correctly.")
    if sys.platform != "win32":
        installed.chmod(installed.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if sys.platform == "darwin":
        # Strip quarantine so Gatekeeper doesn't block first launch.
        os.system(f"xattr -dr com.apple.quarantine {shlex.quote(str(BIN_DIR))} 2>/dev/null")
    return installed


async def _install_prebuilt_llama_server(
    hardware_info: HardwareInfo,
    *,
    target_ref: str | None = None,
    log: Any,
    set_status: Callable[[str], None],
) -> tuple[Path, str, str | None] | None:
    candidates = _prebuilt_asset_candidates(hardware_info)
    if not candidates:
        return None
    try:
        tag, assets = await _fetch_release_tag_and_assets(target_ref)
    except Exception as exc:
        log.write(f"  [dim]Could not reach llama.cpp releases API ({exc}).[/]")
        return None
    asset = _pick_asset(assets, candidates)
    if asset is None:
        log.write(f"  [dim]No prebuilt asset matches this host for {tag}.[/]")
        return None
    url = str(asset.get("browser_download_url") or "")
    if not url:
        return None
    set_status(f"downloading llama-server {tag}")
    log.write(f"[bold bright_white]Downloading prebuilt llama-server {tag}...[/]")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / str(asset.get("name") or "llama-cpp.zip")
        try:
            await _download_to_path(
                url,
                archive,
                label=str(asset.get("name") or "llama.cpp release"),
                log=log,
                set_status=set_status,
            )
        except Exception as exc:
            log.write(f"  [dim]Download failed ({exc}).[/]")
            return None
        try:
            installed = _install_from_archive(archive)
        except Exception as exc:
            log.write(f"  [dim]Install failed ({exc}).[/]")
            return None
    LLAMA_CPP_TAG_FILE.write_text(tag, encoding="utf-8")
    log.write(f"[bold bright_white]llama-server {tag} installed.[/]")
    return installed, tag, _prebuilt_runtime_device(hardware_info)


async def _build_llama_server_from_source(
    *,
    hardware_info: HardwareInfo,
    device: str | None = None,
    target_ref: str | None = None,
    rebuilding: bool,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> tuple[Path, str]:
    if _native_tool_path("cmake") is None:
        raise RuntimeError(missing_cmake_message())
    if rebuilding:
        set_status("rebuilding llama-server for GPU support")
        log.write("[bold bright_white]Rebuilding llama-server for GPU support...[/]")
    else:
        set_status("provisioning llama-server")
        log.write("[bold bright_white]Provisioning llama-server...[/]")
    synced_ref = await _sync_managed_llama_cpp_checkout(
        target_ref=target_ref,
        log=log,
        set_status=set_status,
        clear_status=clear_status,
    )

    build_dir = LLAMA_CPP_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    set_status("configuring llama.cpp")
    log.write("  [dim]Configuring build...[/]")
    rc, out, err = await _run_exec(*_llama_cmake_args(hardware_info, device=device), cwd=build_dir)
    if rc != 0:
        clear_status()
        raise RuntimeError((err or out).strip() or "Failed to configure llama.cpp")
    set_status("building llama-server and llama-bench (this may take a few minutes)")
    log.write("  [dim]Building llama-server and llama-bench (this may take a few minutes)...[/]")
    jobs = os.cpu_count() or 4
    rc, tail = await _run_build_with_progress(
        *_llama_build_command(jobs),
        cwd=build_dir,
        set_status=set_status,
        log=log,
    )
    clear_status()
    if rc != 0:
        built = _managed_source_llama_server_path()
        if not built.is_file():
            raise RuntimeError(tail.strip() or "Failed to build llama-server")
    log.write("[bold bright_white]llama-server built successfully.[/]")

    built = _managed_source_llama_server_path()
    if not built.is_file():
        raise RuntimeError("llama-server build completed but binary was not found.")
    _write_source_build_tag(ref=synced_ref, device=device)
    return built, synced_ref


async def ensure_llama_server(
    setup_result: dict[str, Any],
    *,
    hardware_info: HardwareInfo,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> dict[str, Any]:
    prebuilt_device = _prebuilt_runtime_device(hardware_info)
    required_ref = _llama_cpp_ref_for_setup(setup_result)
    requires_mtp_runtime = _setup_uses_mtp_model(setup_result) or _is_mtp_llama_cpp_ref(required_ref)
    configured_device = str(setup_result.get("device") or "").strip().lower() or None
    desired_device = configured_device or None
    if prebuilt_device:
        desired_device = prebuilt_device
    configured_server = _configured_llama_server_path(setup_result)
    if configured_server:
        merged = dict(setup_result)
        merged["llama_server_path"] = configured_server
        merged["setup_missing_runtime"] = False
        merged["llama_cpp_ref"] = required_ref
        if configured_device:
            merged["device"] = configured_device
        return merged

    # Cache hit: managed binary + tag file present and binary doesn't need a GPU rebuild.
    if (
        LLAMA_SERVER_BIN.is_file()
        and not _needs_rebuild(hardware_info, str(LLAMA_SERVER_BIN), device=desired_device)
    ):
        cached_tag = _installed_llama_server_tag()
        if cached_tag == required_ref or not requires_mtp_runtime:
            merged = dict(setup_result)
            merged["llama_server_path"] = str(LLAMA_SERVER_BIN)
            merged["setup_missing_runtime"] = False
            merged["llama_cpp_ref"] = cached_tag
            if prebuilt_device:
                merged["device"] = prebuilt_device
            return merged

    # Legacy source-built binary on PATH or at the old location that still works.
    existing = current_llama_server_path()
    managed_source_binary = _managed_source_llama_server_path()
    existing_is_source_managed = (
        existing is not None and Path(existing).resolve() == managed_source_binary.resolve()
    )
    if (
        not requires_mtp_runtime
        and existing
        and not existing_is_source_managed
        and not _needs_rebuild(hardware_info, existing, device=desired_device)
    ):
        merged = dict(setup_result)
        merged["llama_server_path"] = existing
        merged["setup_missing_runtime"] = False
        if prebuilt_device:
            merged["device"] = prebuilt_device
        return merged

    rebuilding = existing is not None

    prebuilt = await _install_prebuilt_llama_server(
        hardware_info,
        target_ref=required_ref,
        log=log,
        set_status=set_status,
    )
    if prebuilt is not None:
        clear_status()
        if len(prebuilt) == 2:
            installed_path, tag = prebuilt
            runtime_device = prebuilt_device
        else:
            installed_path, tag, runtime_device = prebuilt
        merged = dict(setup_result)
        merged["llama_server_path"] = str(installed_path)
        merged["setup_missing_runtime"] = False
        merged["llama_cpp_ref"] = tag
        if runtime_device:
            merged["device"] = runtime_device
        return merged

    if _source_build_matches(
        hardware_info=hardware_info,
        required_ref=required_ref,
        desired_device=desired_device,
    ):
        source_device = _source_build_device(configured_device or desired_device)
        merged = dict(setup_result)
        merged["llama_server_path"] = str(_managed_source_llama_server_path())
        merged["setup_missing_runtime"] = False
        merged["llama_cpp_ref"] = required_ref
        if source_device:
            merged["device"] = source_device
        return merged

    if sys.platform == "darwin":
        raise RuntimeError("Failed to install the macOS prebuilt llama-server.")

    built, synced_ref = await _build_llama_server_from_source(
        hardware_info=hardware_info,
        device=desired_device,
        target_ref=required_ref,
        rebuilding=rebuilding,
        log=log,
        set_status=set_status,
        clear_status=clear_status,
    )
    merged = dict(setup_result)
    merged["llama_server_path"] = str(built)
    merged["setup_missing_runtime"] = False
    merged["llama_cpp_ref"] = synced_ref
    if desired_device:
        merged["device"] = desired_device
    return merged


async def ensure_direct_model(
    setup_result: dict[str, Any],
    *,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> dict[str, Any]:
    if str(setup_result.get("model_source", "local")) != "direct":
        return setup_result

    url = str(setup_result.get("model_download_url") or "").strip()
    target_raw = str(setup_result.get("model_download_path") or "").strip()
    target_path = Path(target_raw).expanduser()
    if not url or not target_raw:
        raise RuntimeError("Direct model provisioning is missing a download URL or target path.")
    configured_model_raw = str(setup_result.get("llama_model") or "").strip()
    configured_model = Path(configured_model_raw).expanduser() if configured_model_raw else None
    existing_path = target_path if target_path.is_file() else configured_model if configured_model and configured_model.is_file() else None
    if existing_path is not None:
        merged = dict(setup_result)
        merged["llama_model"] = str(existing_path)
        merged["model_download_path"] = str(existing_path)
        merged["setup_missing_model"] = False
        merged.pop("setup_update_model", None)
        merged.pop("model_update_target", None)
        return merged

    parsed = _parse_huggingface_resolve_url(url)
    if parsed is None:
        raise RuntimeError(
            "Direct model provisioning now uses the Hugging Face CLI fast-transfer path "
            "and requires a huggingface.co `/resolve/<revision>/...` model URL."
        )
    repo_id, revision, filename = parsed

    target_path.parent.mkdir(parents=True, exist_ok=True)
    download_dir = target_path.parent
    set_status(f"downloading {target_path.name} from {repo_id}")
    log.write(f"[bold bright_white]Downloading {target_path.name} from {repo_id} with Hugging Face fast transfer...[/]")
    last_progress_pct = -1
    last_progress_text = ""

    def report_progress(text: str) -> None:
        nonlocal last_progress_pct, last_progress_text
        text = " ".join(text.split())
        if not text or text == last_progress_text:
            return
        last_progress_text = text
        set_status(f"downloading {target_path.name}: {text[-80:]}")
        if "unauthenticated requests" in text.lower():
            log.write(f"  [yellow]{text}[/]")
            return

        pct_match = re.search(r"(\d+)%", text)
        if pct_match:
            pct = int(pct_match.group(1))
            if pct < last_progress_pct + 5:
                return
            last_progress_pct = pct
            log.write(f"  [dim]{text[-160:]}[/]")

    rc, out, err = await _run_hf_cli_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=download_dir,
        progress=report_progress,
    )
    if rc != 0:
        clear_status()
        detail = (err or out).strip()
        raise RuntimeError(detail or "Hugging Face CLI model download failed.")

    downloaded_path = download_dir / filename
    if downloaded_path.is_file():
        if downloaded_path != target_path:
            downloaded_path.replace(target_path)
        parent = downloaded_path.parent
        while parent != target_path.parent:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    if not target_path.is_file():
        clear_status()
        raise RuntimeError(f"Hugging Face CLI completed but did not create {target_path}.")

    downloaded = target_path.stat().st_size
    _clear_old_model_files(target_path.parent, target_path, log=log)
    log.write(f"[bold bright_white]Download complete: {_fmt_size(downloaded)}[/]")
    clear_status()
    merged = dict(setup_result)
    merged["llama_model"] = str(target_path)
    merged["setup_missing_model"] = False
    merged.pop("setup_update_model", None)
    merged.pop("model_update_target", None)
    return merged


async def provision_setup_artifacts(
    setup_result: dict[str, Any],
    *,
    hardware_info: HardwareInfo,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> dict[str, Any]:
    resolved = await ensure_direct_model(
        dict(setup_result),
        log=log,
        set_status=set_status,
        clear_status=clear_status,
    )
    resolved = await ensure_llama_server(
        resolved,
        hardware_info=hardware_info,
        log=log,
        set_status=set_status,
        clear_status=clear_status,
    )
    clear_status()
    return resolved
