"""Install OpenJet's Qwen3.8-27B engine (the megakernel) during `openjet setup`.

Each GitHub release carries the engine prebuilt for every supported GPU architecture
(.github/workflows/engine.yml), so an installed wheel downloads the archive for this GPU and needs only
the NVIDIA driver. A source checkout, or a release without that archive, compiles the source the wheel
ships (<openjet>/megakernel) for this GPU against the llama.cpp checkout that setup builds for CUDA (it
needs llama.cpp's headers and libllama for tokenization). Any failure leaves the setup on llama-server;
it never fails setup.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from rich.markup import escape

from .llama_server import (
    _MEGAKERNEL_TURING_CAP,
    _megakernel_dir,
    _megakernel_gpu_info,
    _megakernel_library,
    _megakernel_model_supported,
    _megakernel_root,
    _megakernel_variant,
)
from .provisioning import LLAMA_CPP_DIR, _download_to_path, _extract_archive, _native_tool_path, _run_exec

_BUILD_TAG = ".openjet-build.json"
_MIXED_BUILD_TAG = ".openjet-build-mixed.json"
_RELEASE_ASSET_URL = "https://github.com/L-Forster/open-jet/releases/download/v{version}/{asset}"
_SOURCES = ("Makefile", "qwen38_mk.cu", "qwen38_mk.h", "prefill.cuh", "prefill_gemm.cuh",
            "cuda_compat.cuh", "gguf.h", "capi.cpp")


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
        tag = json.loads((root / (_BUILD_TAG if variant == "native" else _MIXED_BUILD_TAG)).read_text())
    except (OSError, ValueError):
        return False
    if tag.get("arch") != arch:
        return False
    if "prebuilt" in tag:
        return tag["prebuilt"] == _openjet_version()
    built = (root / library).stat().st_mtime
    sources = _SOURCES if variant == "native" else tuple(
        {"qwen38_mk.cu": "qwen38_mixed.cu", "prefill_gemm.cuh": "prefill_gemm_mixed.cuh"}.get(name, name)
        for name in _SOURCES
    )
    return all(not (root / name).is_file() or (root / name).stat().st_mtime <= built for name in sources)


def _openjet_version() -> str | None:
    try:
        return importlib.metadata.version("open-jet")
    except importlib.metadata.PackageNotFoundError:
        return None


def prebuilt_asset_name(arch: str) -> str | None:
    """Release asset holding the engine for *arch* (sm_86) on this OS, or None when none is published."""
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return None
    sm = arch.replace("_", "")
    if sys.platform.startswith("linux"):
        return f"openjet-engine-linux-x64-{sm}.tar.gz"
    if sys.platform == "win32":
        return f"openjet-engine-windows-x64-{sm}.zip"
    return None


def _is_source_checkout(root: Path) -> bool:
    # a git checkout compiles its own (possibly edited) engine source
    return (root.parent / ".git").exists()


async def install_prebuilt_megakernel(root: Path, arch: str, *, log: Any, set_status: Callable[[str], None]) -> bool:
    """Download this release's engine for *arch* into *root*; False when there is none or it fails."""
    version, asset = _openjet_version(), prebuilt_asset_name(arch)
    if version is None or asset is None:
        return False
    url = _RELEASE_ASSET_URL.format(version=version, asset=asset)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / asset
        try:
            await _download_to_path(url, archive, label=f"the OpenJet engine for {arch}", log=log, set_status=set_status)
            unpacked = Path(tmp) / "engine"
            _extract_archive(archive, unpacked)
        except Exception as exc:  # network, HTTP 404 for a release without engines, a bad archive
            log.write(f"  [dim]No prebuilt engine for {arch} in v{version} ({escape((str(exc) or type(exc).__name__).splitlines()[0])}).[/]")
            return False
        for entry in unpacked.iterdir():
            dest = root / entry.name
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            shutil.copy2(entry, dest, follow_symlinks=False)
    tag = json.dumps({"arch": arch, "prebuilt": version}) + "\n"
    for stem, tag_name in (("qwen38mk", _BUILD_TAG), ("qwen38mk_mixed", _MIXED_BUILD_TAG)):
        if (root / f"lib{stem}.so").is_file() or (root / f"{stem}.dll").is_file():
            (root / tag_name).write_text(tag)
    return True


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
    """Install or compile the megakernel when this setup can use it; otherwise leave everything as is."""
    if not (sys.platform.startswith("linux") or sys.platform == "win32"):
        return setup_result
    if str(setup_result.get("llama_backend") or "auto").strip().lower() == "llama":
        return setup_result
    if not _megakernel_model_supported(str(setup_result.get("llama_model") or "")):
        return setup_result
    model = str(setup_result.get("llama_model") or "")
    if not Path(model).is_file():
        log.write("[yellow]Skipping OpenJet's Qwen3.8-27B engine: download the model first.[/]")
        return setup_result
    variant = _megakernel_variant(model)
    if variant is None:
        log.write("[yellow]Skipping OpenJet's Qwen3.8-27B engine: unsupported or unreadable GGUF tensor types; using llama-server.[/]")
        return setup_result
    library = _megakernel_library(model)
    gpu = _megakernel_gpu_info()
    if gpu is None:
        return setup_result
    if variant == "mixed" and gpu[1] == _MEGAKERNEL_TURING_CAP:
        log.write("[yellow]Skipping OpenJet's Qwen3.8-27B engine: its mixed-quant build does not support Turing GPUs; "
                  "using llama-server.[/]")
        return setup_result
    root = _megakernel_root()
    if not (root / "Makefile").is_file():
        return setup_result
    arch = _arch_for_cap(gpu[1])
    if _build_is_current(root, arch, model):
        return setup_result

    skip = "[yellow]Skipping OpenJet's Qwen3.8-27B engine ({}); using llama-server.[/]"
    if not _is_source_checkout(root) and os.access(root, os.W_OK):
        installed = await install_prebuilt_megakernel(root, arch, log=log, set_status=set_status)
        clear_status()
        if installed and _build_is_current(root, arch, model):
            log.write("[bold bright_white]OpenJet engine installed: OpenJet will serve Qwen3.8-27B with it.[/]")
            return setup_result
    if sys.platform == "win32":  # Windows builds only come prebuilt
        log.write(skip.format(f"no prebuilt engine for {arch} in this release"))
        return setup_result
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

    set_status(f"building OpenJet's Qwen3.8-27B {variant} engine for {arch} (a few minutes)")
    log.write(f"[bold bright_white]Building OpenJet's Qwen3.8-27B {variant} engine for {arch} (a few minutes)...[/]")
    rc, out, err = await _run_exec(*megakernel_build_command(root, nvcc=nvcc, arch=arch, llama_dir=LLAMA_CPP_DIR, model=model))
    clear_status()
    if rc != 0 or not (root / library).is_file():
        tail = "\n".join((err or out).strip().splitlines()[-15:])
        log.write(skip.format("build failed"))
        if tail:
            log.write(f"[dim]{escape(tail)}[/]")
        return setup_result
    (root / (_BUILD_TAG if variant == "native" else _MIXED_BUILD_TAG)).write_text(json.dumps({"arch": arch}) + "\n")
    log.write("[bold bright_white]OpenJet engine built: OpenJet will serve Qwen3.8-27B with it.[/]")
    return setup_result
