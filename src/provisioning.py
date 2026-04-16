from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx

from .app_paths import openjet_install_root
from .config import setup_direct_model_catalog
from .hardware import HardwareInfo, is_jetson_label, recommended_context_window_tokens_from_total
from .setup_memory import recommend_context_window_for_model

def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    return f"{nbytes / (1 << 10):.0f} KB"


OPENJET_HOME = openjet_install_root()
MODELS_DIR = OPENJET_HOME / "models"
BIN_DIR = OPENJET_HOME / "bin"
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
LLAMA_SERVER_BIN = BIN_DIR / "llama-server"
LLAMA_CPP_REPO_URL = "https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_PINNED_REF = "64ac9ab6"


def managed_llama_cpp_ref() -> str:
    ref = os.environ.get("OPENJET_LLAMA_CPP_REF", "").strip()
    return ref or LLAMA_CPP_PINNED_REF

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


def recommend_direct_model(
    hardware_info: HardwareInfo,
    *,
    cfg: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    has_gpu = hardware_info.has_cuda or hardware_info.has_rocm or hardware_info.has_vulkan
    if hardware_info.has_metal:
        effective_memory_mb = max(hardware_info.total_ram_gb, 0.0) * 1024.0
    elif has_gpu and hardware_info.vram_mb > 0:
        effective_memory_mb = hardware_info.vram_mb
    else:
        effective_memory_mb = max(hardware_info.total_ram_gb, 0.0) * 1024.0
    effective_gb = effective_memory_mb / 1024.0
    model_catalog = setup_direct_model_catalog(cfg)
    if not hardware_info.has_metal:
        model_catalog = tuple(
            row for row in model_catalog if not bool(row.get("unified_memory_only"))
        )
    fit_budget_mb = effective_memory_mb * 0.9
    fitting_models = [
        row
        for row in model_catalog
        if float(row.get("model_size_mb", 0) or 0) > 0
        and float(row.get("model_size_mb", 0) or 0) <= fit_budget_mb
    ]
    if fitting_models:
        selected = fitting_models[-1]
    else:
        selected = None
        for row in reversed(model_catalog):
            if effective_gb >= float(row["max_ram_gb"]):
                selected = row
                break
        if selected is None:
            selected = model_catalog[0]
    filename = str(selected["filename"])
    model_size_mb = float(selected.get("model_size_mb", 0) or 0)
    kv_bytes_per_token = float(selected.get("kv_bytes_per_token", 0) or 0)
    return {
        "label": str(selected["label"]),
        "filename": filename,
        "url": str(selected["url"]),
        "target_path": str(MODELS_DIR / filename),
        "model_size_mb": model_size_mb,
        "kv_bytes_per_token": kv_bytes_per_token,
        "context_window_tokens": _context_window_for_model(
            hardware_info,
            model_size_mb,
            kv_bytes_per_token,
        ),
    }


def current_llama_server_path() -> str | None:
    found = shutil.which("llama-server")
    if found:
        return found
    fallback = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
    if fallback.is_file():
        return str(fallback)
    if LLAMA_SERVER_BIN.is_file():
        return str(LLAMA_SERVER_BIN)
    return None


async def _run_exec(*args: str, cwd: Path | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_raw, err_raw = await proc.communicate()
    return proc.returncode or 0, out_raw.decode(errors="replace"), err_raw.decode(errors="replace")


def _llama_cmake_args(hardware_info: HardwareInfo) -> list[str]:
    args = ["cmake", ".."]
    if hardware_info.has_cuda:
        args.extend(["-DGGML_CUDA=ON", "-DGGML_CUDA_FA_ALL_QUANTS=ON"])
        if is_jetson_label(hardware_info.label):
            args.append("-DCMAKE_CUDA_ARCHITECTURES=87")
    elif hardware_info.has_vulkan:
        args.append("-DGGML_VULKAN=ON")
    return args


def _needs_rebuild(hardware_info: HardwareInfo, existing_binary: str) -> bool:
    """Check if the existing llama-server needs rebuilding for GPU support."""
    if not hardware_info.has_vulkan and not hardware_info.has_cuda:
        return False
    try:
        ldd_output = os.popen(f"ldd {shlex.quote(existing_binary)} 2>/dev/null").read().lower()
    except Exception:
        return False
    if hardware_info.has_cuda and "libcuda" not in ldd_output:
        return True
    if hardware_info.has_vulkan and not hardware_info.has_cuda and "libvulkan" not in ldd_output:
        return True
    return False


async def _sync_managed_llama_cpp_checkout(
    *,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> str:
    target_ref = managed_llama_cpp_ref()
    repo_exists = (LLAMA_CPP_DIR / ".git").is_dir()

    if not repo_exists and LLAMA_CPP_DIR.exists():
        raise RuntimeError(f"{LLAMA_CPP_DIR} exists but is not a git checkout.")

    if not repo_exists:
        set_status("cloning llama.cpp")
        log.write("  [dim]Cloning llama.cpp...[/]")
        rc, out, err = await _run_exec(
            "git",
            "clone",
            LLAMA_CPP_REPO_URL,
            str(LLAMA_CPP_DIR),
        )
        if rc != 0:
            clear_status()
            raise RuntimeError((err or out).strip() or "Failed to clone llama.cpp")
    else:
        rc, out, err = await _run_exec("git", "status", "--porcelain", cwd=LLAMA_CPP_DIR)
        if rc != 0:
            clear_status()
            raise RuntimeError((err or out).strip() or "Failed to inspect llama.cpp checkout.")
        if out.strip():
            clear_status()
            raise RuntimeError("Cannot update managed llama.cpp checkout with local changes. Commit or stash them first.")

    set_status("fetching llama.cpp")
    log.write("  [dim]Fetching llama.cpp refs...[/]")
    rc, out, err = await _run_exec("git", "fetch", "--tags", "--prune", "origin", cwd=LLAMA_CPP_DIR)
    if rc != 0:
        clear_status()
        raise RuntimeError((err or out).strip() or "Failed to fetch llama.cpp")

    rc, out, err = await _run_exec("git", "checkout", "--detach", target_ref, cwd=LLAMA_CPP_DIR)
    if rc != 0:
        clear_status()
        raise RuntimeError((err or out).strip() or f"Failed to checkout llama.cpp ref {target_ref}.")
    return target_ref


async def ensure_llama_server(
    setup_result: dict[str, Any],
    *,
    hardware_info: HardwareInfo,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> dict[str, Any]:
    existing = current_llama_server_path()
    managed_binary = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"
    existing_is_managed = existing is not None and Path(existing).resolve() == managed_binary.resolve()

    target_ref = managed_llama_cpp_ref()
    current_managed_ref = None
    if (LLAMA_CPP_DIR / ".git").is_dir():
        rc, out, _err = await _run_exec("git", "rev-parse", "HEAD", cwd=LLAMA_CPP_DIR)
        if rc == 0:
            current_managed_ref = out.strip()

    managed_ref_mismatch = existing_is_managed and current_managed_ref not in {None, target_ref}

    if existing and not _needs_rebuild(hardware_info, existing) and not managed_ref_mismatch:
        merged = dict(setup_result)
        merged["llama_server_path"] = existing
        merged["setup_missing_runtime"] = False
        return merged

    rebuilding = existing is not None
    if rebuilding:
        set_status("rebuilding llama-server for GPU support")
        log.write("[bold bright_white]Rebuilding llama-server for GPU support...[/]")
    else:
        set_status("provisioning llama-server")
        log.write("[bold bright_white]Provisioning llama-server...[/]")
    synced_ref = await _sync_managed_llama_cpp_checkout(
        log=log,
        set_status=set_status,
        clear_status=clear_status,
    )

    build_dir = LLAMA_CPP_DIR / "build"
    if rebuilding and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    set_status("configuring llama.cpp")
    log.write("  [dim]Configuring build...[/]")
    rc, out, err = await _run_exec(*_llama_cmake_args(hardware_info), cwd=build_dir)
    if rc != 0:
        clear_status()
        raise RuntimeError((err or out).strip() or "Failed to configure llama.cpp")
    set_status("building llama-server (this may take a few minutes)")
    log.write("  [dim]Building llama-server (this may take a few minutes)...[/]")
    rc, out, err = await _run_exec("cmake", "--build", ".", "--target", "llama-server", "-j4", cwd=build_dir)
    clear_status()
    if rc != 0:
        raise RuntimeError((err or out).strip() or "Failed to build llama-server")
    log.write("[bold bright_white]llama-server built successfully.[/]")

    built = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"
    if not built.is_file():
        raise RuntimeError("llama-server build completed but binary was not found.")
    merged = dict(setup_result)
    merged["llama_server_path"] = str(built)
    merged["setup_missing_runtime"] = False
    merged["llama_cpp_ref"] = synced_ref
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
    target_path = Path(str(setup_result.get("model_download_path") or "").strip()).expanduser()
    if not url or not str(target_path):
        raise RuntimeError("Direct model provisioning is missing a download URL or target path.")
    if target_path.is_file():
        merged = dict(setup_result)
        merged["llama_model"] = str(target_path)
        merged["setup_missing_model"] = False
        return merged

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    downloaded = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else None
    set_status(f"downloading {target_path.name}")
    size_hint = ""
    if downloaded > 0:
        size_hint = f" (resuming from {_fmt_size(downloaded)})"
    log.write(f"[bold bright_white]Downloading {target_path.name}{size_hint}...[/]")

    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        async with client.stream("GET", url, headers=headers) as response:
            if response.status_code not in {200, 206}:
                clear_status()
                raise RuntimeError(f"Model download failed with HTTP {response.status_code}.")
            total = response.headers.get("Content-Length")
            total_bytes = int(total) + downloaded if total and total.isdigit() else None
            mode = "ab" if downloaded > 0 else "wb"
            last_log_pct = -10
            last_log_time = time.monotonic()
            with temp_path.open(mode) as fh:
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = max(0, min(100, int(downloaded * 100 / total_bytes)))
                        now = time.monotonic()
                        set_status(f"downloading {target_path.name} {pct}%")
                        if pct - last_log_pct >= 10 and now - last_log_time >= 2.0:
                            log.write(f"  [dim]{pct}% ({_fmt_size(downloaded)} / {_fmt_size(total_bytes)})[/]")
                            last_log_pct = pct
                            last_log_time = now

    temp_path.replace(target_path)
    log.write(f"[bold bright_white]Download complete: {_fmt_size(downloaded)}[/]")
    clear_status()
    merged = dict(setup_result)
    merged["llama_model"] = str(target_path)
    merged["setup_missing_model"] = False
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
