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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from rich.markup import escape

from .llama_server import (
    _MEGAKERNEL_LEGACY_DIR,
    _MEGAKERNEL_TURING_CAP,
    _megakernel_dir,
    _megakernel_gpu_info,
    _megakernel_min_vram_mib,
    _megakernel_library,
    _megakernel_model_supported,
    _megakernel_root,
    _megakernel_variant,
)
from .provisioning import LLAMA_CPP_DIR, _download_to_path, _extract_archive, _native_tool_path, _run_exec
from .runtime_limits import read_memory_snapshot

_BUILD_TAG = ".openjet-build.json"
_MIXED_BUILD_TAG = ".openjet-build-mixed.json"
_RELEASE_ASSET_URL = "https://github.com/L-Forster/open-jet/releases/download/v{version}/{asset}"
_SOURCES = ("Makefile", "build.ps1", "qwen38_mk.cu", "qwen38_mk.h", "mk_shared.cuh", "mk_launch.h", "engine_impl.h", "gguf.h",
            "loader.cuh", "prefill.cuh", "prefill_attn.cuh", "prefill_gemm.cuh", "wv.cuh", "wv_dev.cuh", "wv_host.cuh",
            "engine.cu", "ck_pool.cu", "loader.cu", "prefill_host.cu", "state_files.cu", "host_store.cu", "lanes.cu",
            "capi.cpp", "compat_win32.cpp", "compat_win32.h")
# the legacy engine (megakernel/legacy): its native build serves Turing, its mixed-quant build UD-Q3_K_XL and cards
# under 24 GB. Both compile in a few minutes and under 12 GiB of RAM.
_LEGACY_SOURCES = ("Makefile", "qwen38_mk.cu", "qwen38_mk.h", "kv_paged.cuh", "kv_lowbit.cuh", "kv_spec.h", "prefill.cuh",
                   "prefill_attn.cuh", "prefill_gemm.cuh", "cuda_compat.cuh", "gguf.h", "capi.cpp", "bmm.cuh")
# the mixed build's i-quant codebooks and load-time block layouts
_MIXED_ONLY_SOURCES = ("iq_tables.cuh", "iq_repack.h")
# ptxas peaks at about 21 GiB compiling the megakernel (megakernel/Makefile); a machine with less RAM than this would
# run out of memory, so it gets the prebuilt engine or stays on llama-server
_BUILD_MIN_RAM_MIB = 24 * 1024


def _nvcc_path() -> str | None:
    found = _native_tool_path("nvcc")
    if found:
        return found
    exe = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    for var in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(var, "").strip()
        if base and (Path(base) / "bin" / exe).is_file():
            return str(Path(base) / "bin" / exe)
    default = Path("/usr/local/cuda/bin/nvcc")
    return str(default) if default.is_file() else None


# llama.cpp's import libraries, which the engine DLL links against; the CMake build puts each under its
# own configuration directory, so they are located by name
_WINDOWS_IMPORT_LIBS = ("llama.lib", "ggml-base.lib")


def _windows_import_libs(llama_dir: Path) -> list[str]:
    """Paths of llama.cpp's import libraries in a Windows build tree, or [] when any of them is missing."""
    found: list[str] = []
    for name in _WINDOWS_IMPORT_LIBS:
        hit = next(iter(sorted((llama_dir / "build").rglob(name))), None)
        if hit is None:
            return []
        found.append(str(hit))
    return found


def _copy_windows_runtime(llama_dir: Path, root: Path) -> None:
    """Put llama.cpp's DLLs beside the engine.

    Windows has no RPATH, so the loader resolves the engine's dependencies from its own directory; the
    prebuilt archives ship them adjacent for the same reason.
    """
    for dll in sorted((llama_dir / "build" / "bin").rglob("*.dll")):
        dest = root / dll.name
        if not dest.is_file() or dest.stat().st_mtime < dll.stat().st_mtime:
            shutil.copy2(dll, dest)


def _llama_cpp_ready(llama_dir: Path) -> bool:
    if not (llama_dir / "include" / "llama.h").is_file():
        return False
    if sys.platform == "win32":
        return bool(_windows_import_libs(llama_dir))
    return (llama_dir / "build" / "bin" / "libllama.so").is_file()


def _vcvars_candidates() -> list[Path]:
    """vcvars64.bat of every Visual Studio install found, newest first; an install may be incomplete."""
    roots: list[Path] = []
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        try:
            out = subprocess.run([str(vswhere), "-products", "*", "-prerelease", "-property", "installationPath"],
                                 capture_output=True, text=True, timeout=60)
            roots += [Path(line.strip()) for line in out.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            roots += sorted((Path(base) / "Microsoft Visual Studio").glob("20*/*"), reverse=True)
    seen: set[str] = set()
    found: list[Path] = []
    for root in roots:
        bat = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        key = str(bat).lower()
        if bat.is_file() and key not in seen:
            seen.add(key)
            found.append(bat)
    return found


def _msvc_env() -> dict[str, str] | None:
    """The environment vcvars64.bat exports, so nvcc finds cl.exe, the CRT headers and the Windows SDK.

    Every install is tried: one can be registered but too incomplete to set up a working toolchain, and
    another alongside it (Build Tools next to a broken Community, say) still builds.
    """
    # an unbalanced quote anywhere in PATH breaks cmd's parser before vcvars even runs
    start = dict(os.environ)
    start["PATH"] = os.pathsep.join(
        entry.replace(chr(34), "") for entry in start.get("PATH", "").split(os.pathsep) if entry.strip()
    )
    for bat in _vcvars_candidates():
        # through a script file: cmd mangles the quoting that `cmd /c call "<path>"` needs for a path
        # with spaces, and vcvars lives under "Program Files (x86)" whose parentheses it also splits on
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "openjet_vcvars.bat"
            script.write_bytes(('@echo off\r\ncall "' + str(bat) + '" >nul\r\nset\r\n').encode("mbcs", "replace"))
            try:
                out = subprocess.run(["cmd", "/c", str(script)],
                                     capture_output=True, text=True, timeout=300, env=start)
            except (OSError, subprocess.SubprocessError):
                continue
        if out.returncode != 0:
            continue
        env = dict(start)
        for line in out.stdout.splitlines():
            key, sep, value = line.partition("=")
            if sep:
                env[key] = value
        if env.get("VCToolsInstallDir") and shutil.which("cl", path=env.get("PATH", "")):
            return env
    return None


def _arch_for_cap(cap: str) -> str:
    return "sm_" + cap.replace(".", "")


def _sources(variant: str | None) -> tuple[str, ...]:
    if variant == "legacy":
        return _LEGACY_SOURCES
    if variant == "mixed":
        return tuple({"qwen38_mk.cu": "qwen38_mixed.cu", "prefill_gemm.cuh": "prefill_gemm_mixed.cuh"}.get(name, name)
                     for name in _LEGACY_SOURCES) + _MIXED_ONLY_SOURCES
    return _SOURCES


def _build_is_current(root: Path, arch: str, model: str = "") -> bool:
    """Whether the engine for *model* in *root* (megakernel/, or megakernel/legacy) is built for *arch* from this source."""
    variant = _megakernel_variant(model)
    if _megakernel_dir(model) is None:
        return False
    try:
        tag = json.loads((root / (_MIXED_BUILD_TAG if variant == "mixed" else _BUILD_TAG)).read_text())
    except (OSError, ValueError):
        return False
    if tag.get("arch") != arch:
        return False
    if "prebuilt" in tag:
        return tag["prebuilt"] == _openjet_version()
    built = (root / _megakernel_library(model)).stat().st_mtime
    return all(not (root / name).is_file() or (root / name).stat().st_mtime <= built for name in _sources(variant))


def _openjet_version() -> str | None:
    try:
        return importlib.metadata.version("open-jet")
    except importlib.metadata.PackageNotFoundError:
        return None


def prebuilt_asset_name(arch: str, *, legacy: bool = False) -> str | None:
    """Release asset holding the engine (or the previous one, *legacy*) for *arch* (sm_86) on this OS, or None when
    none is published."""
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return None
    sm = arch.replace("_", "")
    stem = "openjet-engine-legacy" if legacy else "openjet-engine"
    if sys.platform.startswith("linux"):
        return f"{stem}-linux-x64-{sm}.tar.gz"
    if sys.platform == "win32":
        return f"{stem}-windows-x64-{sm}.zip"
    return None


def _is_source_checkout(root: Path) -> bool:
    # a git checkout compiles its own (possibly edited) engine source
    return (_megakernel_root().parent / ".git").exists()


async def install_prebuilt_megakernel(root: Path, arch: str, *, log: Any, set_status: Callable[[str], None],
                                      legacy: bool = False) -> bool:
    """Download this release's engine for *arch* into *root*; False when there is none or it fails."""
    version, asset = _openjet_version(), prebuilt_asset_name(arch, legacy=legacy)
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


def megakernel_build_commands(root: Path, *, nvcc: str, arch: str, llama_dir: Path, model: str = "") -> list[list[str]]:
    """The commands that build the engine library for *model* in *root*, to be run in order.

    The current engine builds through its Makefile on Linux and through build.ps1 on Windows, which sets up MSVC
    2022 itself. The legacy engine (megakernel/legacy) builds through its Makefile on Linux; Windows has no POSIX
    make, so its two nvcc steps (compile the engine, then link it with capi.cpp) are issued directly.
    """
    variant = _megakernel_variant(model)
    library = _megakernel_library(model)
    if sys.platform != "win32":
        # -B: the object files may be from another GPU architecture or an older source.
        return [["make", "-B", "-C", str(root), "-j1", f"NVCC={nvcc}", f"ARCH={arch}", f"LLAMA={llama_dir}", library]]
    if variant not in ("legacy", "mixed"):
        return [["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / "build.ps1"),
                 "-Targets", "engine", "-Arch", arch, "-Llama", str(llama_dir), "-Cuda", str(Path(nvcc).parent.parent)]]
    source = "qwen38_mixed.cu" if variant == "mixed" else "qwen38_mk.cu"
    obj = str(root / ("qwen38_mixed.obj" if variant == "mixed" else "qwen38_mk.obj"))
    flags = ["-O3", "-std=c++17", f"-arch={arch}", "-lineinfo"]
    if arch == "sm_75":
        flags.append("-DQWEN_MK_SM75=1")
    ggml_include = str(llama_dir / "ggml" / "include")
    return [
        [nvcc, *flags, "-Xptxas", "-dlcm=cg", "-I", ggml_include, "-c", str(root / source), "-o", obj],
        [nvcc, *flags, "-shared", "-I", str(llama_dir / "include"), "-I", ggml_include,
         str(root / "capi.cpp"), obj, *_windows_import_libs(llama_dir), "-o", str(root / library)],
    ]


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
    legacy = variant in ("legacy", "mixed")
    if variant is None:
        log.write("[yellow]Skipping OpenJet's Qwen3.8-27B engine: unsupported or unreadable GGUF tensor types; using llama-server.[/]")
        return setup_result
    library = _megakernel_library(model)
    gpu = _megakernel_gpu_info(_megakernel_min_vram_mib(model))
    if gpu is None:
        return setup_result
    if variant == "mixed" and gpu[1] == _MEGAKERNEL_TURING_CAP:
        log.write("[yellow]Skipping OpenJet's Qwen3.8-27B engine: its mixed-quant build does not support Turing GPUs; "
                  "using llama-server.[/]")
        return setup_result
    root = _megakernel_root() / _MEGAKERNEL_LEGACY_DIR if legacy else _megakernel_root()
    if not (root / "Makefile").is_file():
        return setup_result
    arch = _arch_for_cap(gpu[1])
    if _build_is_current(root, arch, model):
        return setup_result

    skip = "[yellow]Skipping OpenJet's Qwen3.8-27B engine ({}); using llama-server.[/]"
    if not _is_source_checkout(root) and os.access(root, os.W_OK):
        installed = await install_prebuilt_megakernel(root, arch, log=log, set_status=set_status, legacy=legacy)
        clear_status()
        if installed and _build_is_current(root, arch, model):
            log.write("[bold bright_white]OpenJet engine installed: OpenJet will serve Qwen3.8-27B with it.[/]")
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
    env = None
    if not legacy:
        memory = read_memory_snapshot()
        if memory is not None and memory.total_mb < _BUILD_MIN_RAM_MIB:
            log.write(skip.format(f"compiling it needs about 22 GiB of RAM and this machine has {memory.total_mb / 1024:.0f} GiB"))
            return setup_result
    elif sys.platform == "win32":
        env = _msvc_env()
        if env is None:
            log.write(skip.format("no Visual Studio C++ build tools found: nvcc needs cl.exe, then rerun `openjet setup`"))
            return setup_result

    what, wait = (f"{variant} engine", "a few minutes") if legacy else ("engine", "about 15 minutes")
    set_status(f"building OpenJet's Qwen3.8-27B {what} for {arch} ({wait})")
    log.write(f"[bold bright_white]Building OpenJet's Qwen3.8-27B {what} for {arch} ({wait})...[/]")
    rc, out, err = 0, "", ""
    for command in megakernel_build_commands(root, nvcc=nvcc, arch=arch, llama_dir=LLAMA_CPP_DIR, model=model):
        rc, out, err = await _run_exec(*command, **({"env": env} if env else {}))
        if rc != 0:
            break
    clear_status()
    if rc != 0 or not (root / library).is_file():
        tail = "\n".join((err or out).strip().splitlines()[-15:])
        log.write(skip.format("build failed"))
        if tail:
            log.write(f"[dim]{escape(tail)}[/]")
        return setup_result
    if sys.platform == "win32":
        _copy_windows_runtime(LLAMA_CPP_DIR, root)
    (root / (_MIXED_BUILD_TAG if variant == "mixed" else _BUILD_TAG)).write_text(json.dumps({"arch": arch}) + "\n")
    log.write("[bold bright_white]OpenJet engine built: OpenJet will serve Qwen3.8-27B with it.[/]")
    return setup_result
