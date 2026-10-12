"""Build the Qwen3.8-27B megakernel during `openjet setup`.

The wheel ships the megakernel source (<openjet>/megakernel). It is compiled on the user's machine,
for their GPU, against the llama.cpp checkout that setup builds for CUDA (it needs llama.cpp's headers
and libllama for tokenization). Any failure leaves the setup on llama-server; it never fails setup.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from rich.markup import escape

from .llama_server import (
    _megakernel_dir,
    _megakernel_gpu_info,
    _megakernel_library,
    _megakernel_model_supported,
    _megakernel_root,
    _megakernel_variant,
)
from .provisioning import LLAMA_CPP_DIR, _native_tool_path, _run_exec

_BUILD_TAG = ".openjet-build.json"
_SOURCES = ("Makefile", "qwen38_mk.cu", "qwen38_mk.h", "prefill.cuh", "prefill_gemm.cuh", "gguf.h", "capi.cpp")


def _nvcc_path() -> str | None:
    found = _native_tool_path("nvcc")
    if found:
        return found
    for var in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(var, "").strip()
        if base and (Path(base) / "bin" / "nvcc").is_file():
            return str(Path(base) / "bin" / "nvcc")
    default = Path("/usr/local/cuda/bin/nvcc")
    return str(default) if default.is_file() else None


def _llama_cpp_ready(llama_dir: Path) -> bool:
    return (llama_dir / "include" / "llama.h").is_file() and (llama_dir / "build" / "bin" / "libllama.so").is_file()


def _arch_for_cap(cap: str) -> str:
    return "sm_" + cap.replace(".", "")


def _build_is_current(root: Path, arch: str, model: str = "") -> bool:
    library = _megakernel_library(model)
    variant = _megakernel_variant(model)
    if _megakernel_dir(model) is None:
        return False
    try:
        tag = json.loads((root / (_BUILD_TAG if variant == "native" else ".openjet-build-mixed.json")).read_text())
    except (OSError, ValueError):
        return False
    if tag.get("arch") != arch:
        return False
    built = (root / library).stat().st_mtime
    sources = _SOURCES if variant == "native" else tuple(
        {"qwen38_mk.cu": "qwen38_mixed.cu", "prefill_gemm.cuh": "prefill_gemm_mixed.cuh"}.get(name, name)
        for name in _SOURCES
    )
    return all(not (root / name).is_file() or (root / name).stat().st_mtime <= built for name in sources)


def megakernel_build_command(root: Path, *, nvcc: str, arch: str, llama_dir: Path, model: str = "") -> list[str]:
    # -B: the object file may be from another GPU architecture or an older source.
    return ["make", "-B", "-C", str(root), "-j1", f"NVCC={nvcc}", f"ARCH={arch}", f"LLAMA={llama_dir}", _megakernel_library(model)]


async def ensure_megakernel(
    setup_result: dict[str, Any],
    *,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> dict[str, Any]:
    """Compile the megakernel when this setup can use it; otherwise leave everything as is."""
    if not sys.platform.startswith("linux"):
        return setup_result
    if str(setup_result.get("llama_backend") or "auto").strip().lower() == "llama":
        return setup_result
    if not _megakernel_model_supported(str(setup_result.get("llama_model") or "")):
        return setup_result
    model = str(setup_result.get("llama_model") or "")
    if not Path(model).is_file():
        log.write("[yellow]Skipping the megakernel: download the model before building its engine.[/]")
        return setup_result
    variant = _megakernel_variant(model)
    if variant is None:
        log.write("[yellow]Skipping the megakernel: unsupported or unreadable GGUF tensor types; using llama-server.[/]")
        return setup_result
    library = _megakernel_library(model)
    gpu = _megakernel_gpu_info()
    if gpu is None:
        return setup_result
    root = _megakernel_root()
    if not (root / "Makefile").is_file():
        return setup_result
    arch = _arch_for_cap(gpu[1])
    if _build_is_current(root, arch, model):
        return setup_result

    skip = "[yellow]Skipping the Qwen3.8-27B megakernel ({}); using llama-server.[/]"
    nvcc = _nvcc_path()
    if nvcc is None:
        log.write(skip.format("CUDA toolkit not found: install it for nvcc, then rerun `openjet setup`"))
        return setup_result
    if not _llama_cpp_ready(LLAMA_CPP_DIR):
        log.write(skip.format(f"no llama.cpp source build in {escape(str(LLAMA_CPP_DIR))}"))
        return setup_result
    if not os.access(root, os.W_OK):
        log.write(skip.format(f"{escape(str(root))} is not writable"))
        return setup_result

    set_status(f"building the Qwen3.8-27B {variant} megakernel for {arch} (a few minutes)")
    log.write(f"[bold bright_white]Building the Qwen3.8-27B {variant} megakernel for {arch} (a few minutes)...[/]")
    rc, out, err = await _run_exec(*megakernel_build_command(root, nvcc=nvcc, arch=arch, llama_dir=LLAMA_CPP_DIR, model=model))
    clear_status()
    if rc != 0 or not (root / library).is_file():
        tail = "\n".join((err or out).strip().splitlines()[-15:])
        log.write(skip.format("build failed"))
        if tail:
            log.write(f"[dim]{escape(tail)}[/]")
        return setup_result
    (root / (_BUILD_TAG if variant == "native" else ".openjet-build-mixed.json")).write_text(json.dumps({"arch": arch}) + "\n")
    log.write("[bold bright_white]Megakernel built: OpenJet will serve Qwen3.8-27B with it.[/]")
    return setup_result
