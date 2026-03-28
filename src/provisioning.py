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

from .config import setup_direct_model_catalog
from .hardware import HardwareInfo, is_jetson_label

def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    return f"{nbytes / (1 << 10):.0f} KB"


OPENJET_HOME = Path.home() / ".openjet"
MODELS_DIR = OPENJET_HOME / "models"
BIN_DIR = OPENJET_HOME / "bin"
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
LLAMA_SERVER_BIN = BIN_DIR / "llama-server"

def recommend_direct_model(
    hardware_info: HardwareInfo,
    *,
    cfg: Mapping[str, object] | None = None,
) -> dict[str, str]:
    total_ram_gb = max(hardware_info.total_ram_gb, 0.0)
    model_catalog = setup_direct_model_catalog(cfg)
    for row in model_catalog:
        if total_ram_gb <= float(row["max_ram_gb"]):
            filename = str(row["filename"])
            return {
                "label": str(row["label"]),
                "filename": filename,
                "url": str(row["url"]),
                "target_path": str(MODELS_DIR / filename),
            }
    row = model_catalog[-1]
    filename = str(row["filename"])
    return {
        "label": str(row["label"]),
        "filename": filename,
        "url": str(row["url"]),
        "target_path": str(MODELS_DIR / filename),
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


async def ensure_llama_server(
    setup_result: dict[str, Any],
    *,
    hardware_info: HardwareInfo,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> dict[str, Any]:
    existing = current_llama_server_path()
    if existing and not _needs_rebuild(hardware_info, existing):
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
    if not LLAMA_CPP_DIR.exists():
        set_status("cloning llama.cpp")
        log.write("  [dim]Cloning llama.cpp...[/]")
        rc, out, err = await _run_exec(
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(LLAMA_CPP_DIR),
        )
        if rc != 0:
            clear_status()
            raise RuntimeError((err or out).strip() or "Failed to clone llama.cpp")

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
