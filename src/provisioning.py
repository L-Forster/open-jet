from __future__ import annotations

import asyncio
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Callable

import httpx

from .hardware import HardwareInfo, is_jetson_label

OPENJET_HOME = Path.home() / ".openjet"
MODELS_DIR = OPENJET_HOME / "models"
BIN_DIR = OPENJET_HOME / "bin"
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
LLAMA_SERVER_BIN = BIN_DIR / "llama-server"

_MODEL_CATALOG: tuple[dict[str, object], ...] = (
    {
        "max_ram_gb": 6.0,
        "label": "Qwen2.5 Coder 1.5B",
        "filename": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf?download=true",
    },
    {
        "max_ram_gb": 24.0,
        "label": "Qwen2.5 Coder 7B",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf?download=true",
    },
    {
        "max_ram_gb": 10_000.0,
        "label": "Qwen2.5 Coder 14B",
        "filename": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf?download=true",
    },
)


def recommend_direct_model(hardware_info: HardwareInfo) -> dict[str, str]:
    total_ram_gb = max(hardware_info.total_ram_gb, 0.0)
    for row in _MODEL_CATALOG:
        if total_ram_gb <= float(row["max_ram_gb"]):
            filename = str(row["filename"])
            return {
                "label": str(row["label"]),
                "filename": filename,
                "url": str(row["url"]),
                "target_path": str(MODELS_DIR / filename),
            }
    row = _MODEL_CATALOG[-1]
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
    return args


async def ensure_llama_server(
    setup_result: dict[str, Any],
    *,
    hardware_info: HardwareInfo,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> dict[str, Any]:
    if str(setup_result.get("runtime", "llama_cpp")) != "llama_cpp":
        return setup_result
    existing = current_llama_server_path()
    if existing:
        merged = dict(setup_result)
        merged["llama_server_path"] = existing
        merged["setup_missing_runtime"] = False
        return merged

    set_status("provisioning llama-server")
    log.write("[bold bright_white]Provisioning llama-server...[/]")
    if not LLAMA_CPP_DIR.exists():
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
    build_dir.mkdir(parents=True, exist_ok=True)
    rc, out, err = await _run_exec(*_llama_cmake_args(hardware_info), cwd=build_dir)
    if rc != 0:
        clear_status()
        raise RuntimeError((err or out).strip() or "Failed to configure llama.cpp")
    rc, out, err = await _run_exec("cmake", "--build", ".", "--target", "llama-server", "-j4", cwd=build_dir)
    clear_status()
    if rc != 0:
        raise RuntimeError((err or out).strip() or "Failed to build llama-server")

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
        merged["model"] = str(target_path)
        merged["llama_model"] = str(target_path)
        merged["setup_missing_model"] = False
        return merged

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")
    downloaded = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else None
    set_status(f"downloading {target_path.name}")
    log.write(f"[bold bright_white]Downloading {target_path.name}...[/]")

    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        async with client.stream("GET", url, headers=headers) as response:
            if response.status_code not in {200, 206}:
                clear_status()
                raise RuntimeError(f"Model download failed with HTTP {response.status_code}.")
            total = response.headers.get("Content-Length")
            total_bytes = int(total) + downloaded if total and total.isdigit() else None
            mode = "ab" if downloaded > 0 else "wb"
            with temp_path.open(mode) as fh:
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = max(0, min(100, int(downloaded * 100 / total_bytes)))
                        set_status(f"downloading {target_path.name} {pct}%")

    temp_path.replace(target_path)
    clear_status()
    merged = dict(setup_result)
    merged["model"] = str(target_path)
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
