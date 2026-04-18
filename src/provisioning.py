from __future__ import annotations

import asyncio
import errno
import os
import platform
import pty
import re
import shlex
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlparse

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
LLAMA_CPP_TAG_FILE = BIN_DIR / "llama-server.tag"
LLAMA_CPP_REPO_URL = "https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
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
) -> dict[str, object]:
    has_gpu = hardware_info.has_cuda or hardware_info.has_rocm or hardware_info.has_vulkan or hardware_info.has_metal
    total_ram_mb = max(hardware_info.total_ram_gb, 0.0) * 1024.0
    vram_mb = total_ram_mb if hardware_info.has_metal else max(hardware_info.vram_mb, 0.0)
    vram_or_ram_budget_mb = (vram_mb if has_gpu and vram_mb > 0 else total_ram_mb) * 0.9
    combined_budget_mb = (total_ram_mb + (vram_mb if has_gpu and not hardware_info.has_metal else 0.0)) * 0.9

    dense_rows = [row for row in model_catalog if not _is_moe_catalog_row(row)]
    moe_rows = [row for row in model_catalog if _is_moe_catalog_row(row)]

    def largest(rows: list[dict[str, object]]) -> dict[str, object] | None:
        return max(rows, key=lambda row: (_catalog_float(row, "model_size_mb"), _catalog_float(row, "max_ram_gb"))) if rows else None

    if hardware_info.has_metal:
        metal_moe = largest([
            row for row in moe_rows
            if 0 < _catalog_float(row, "model_size_mb") <= combined_budget_mb
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

    dense = largest([
        row for row in dense_rows
        if 0 < _catalog_float(row, "model_size_mb") <= max(vram_or_ram_budget_mb, total_ram_mb * 0.9)
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
    has_gpu = hardware_info.has_cuda or hardware_info.has_rocm or hardware_info.has_vulkan
    if not has_gpu and not hardware_info.has_metal:
        model_catalog = tuple(
            row for row in model_catalog if not _is_moe_catalog_row(row)
        )
    selected = _select_direct_model(model_catalog, hardware_info)
    filename = str(selected["filename"])
    model_size_mb = float(selected.get("model_size_mb", 0) or 0)
    kv_bytes_per_token = float(selected.get("kv_bytes_per_token", 0) or 0)
    context_model_size_mb = _active_model_size_mb(selected) if _is_moe_catalog_row(selected) else model_size_mb
    return {
        "label": str(selected["label"]),
        "filename": filename,
        "url": str(selected["url"]),
        "target_path": str(MODELS_DIR / filename),
        "model_size_mb": model_size_mb,
        "active_model_size_mb": _active_model_size_mb(selected) if _is_moe_catalog_row(selected) else 0.0,
        "kv_bytes_per_token": kv_bytes_per_token,
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
    hf = shutil.which("hf")
    if hf:
        return [hf, "download"]
    huggingface_cli = shutil.which("huggingface-cli")
    if huggingface_cli:
        return [huggingface_cli, "download"]
    raise RuntimeError(
        "Hugging Face CLI is required for model downloads. Install OpenJet dependencies "
        "or run `pip install 'huggingface_hub[hf_transfer]' hf_transfer`."
    )


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


def _llama_cmake_args(hardware_info: HardwareInfo) -> list[str]:
    args = ["cmake", ".."]
    if hardware_info.has_cuda:
        args.append("-DGGML_CUDA=ON")
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


def _prebuilt_asset_candidates(hardware_info: HardwareInfo) -> list[str]:
    """Return substrings identifying release assets that match this host.

    The llama.cpp release assets are named like
    `llama-<tag>-bin-<os>-<variant>-<arch>.zip`. We match by substring so the
    logic stays robust to minor naming churn.

    Returns an empty list when no prebuilt covers this host (e.g. Linux CUDA or
    Jetson), which causes the caller to fall back to a source build.
    """
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        if hardware_info.has_metal:
            return ["bin-macos-arm64"]
        return ["bin-macos-x64"]
    if sys.platform.startswith("linux"):
        if hardware_info.has_cuda:
            # Linux CUDA is not distributed as a release asset; build from source.
            return []
        if machine in {"x86_64", "amd64"}:
            if hardware_info.has_vulkan:
                return ["bin-ubuntu-vulkan-x64"]
            return ["bin-ubuntu-x64"]
        if machine in {"aarch64", "arm64"}:
            return ["bin-ubuntu-arm64"]
    return []


def _installed_llama_server_tag() -> str | None:
    try:
        return LLAMA_CPP_TAG_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


async def _fetch_latest_release_tag_and_assets() -> tuple[str, list[dict[str, Any]]]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        headers = {"Accept": "application/vnd.github+json"}
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = await client.get(LLAMA_CPP_RELEASES_API, headers=headers)
        response.raise_for_status()
        payload = response.json()
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        raise RuntimeError("llama.cpp latest release is missing tag_name.")
    assets = payload.get("assets") or []
    if not isinstance(assets, list):
        raise RuntimeError("llama.cpp latest release returned malformed assets.")
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
    installed = BIN_DIR / "llama-server"
    if not installed.is_file():
        raise RuntimeError("llama-server was not installed correctly.")
    installed.chmod(installed.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if sys.platform == "darwin":
        # Strip quarantine so Gatekeeper doesn't block first launch.
        os.system(f"xattr -dr com.apple.quarantine {shlex.quote(str(BIN_DIR))} 2>/dev/null")
    return installed


async def _install_prebuilt_llama_server(
    hardware_info: HardwareInfo,
    *,
    log: Any,
    set_status: Callable[[str], None],
) -> tuple[Path, str] | None:
    candidates = _prebuilt_asset_candidates(hardware_info)
    if not candidates:
        return None
    try:
        tag, assets = await _fetch_latest_release_tag_and_assets()
    except Exception as exc:
        log.write(f"  [dim]Could not reach llama.cpp releases API ({exc}); will build from source.[/]")
        return None
    asset = _pick_asset(assets, candidates)
    if asset is None:
        log.write(f"  [dim]No prebuilt asset matches this host for {tag}; will build from source.[/]")
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
            log.write(f"  [dim]Download failed ({exc}); will build from source.[/]")
            return None
        try:
            installed = _install_from_archive(archive)
        except Exception as exc:
            log.write(f"  [dim]Install failed ({exc}); will build from source.[/]")
            return None
    LLAMA_CPP_TAG_FILE.write_text(tag, encoding="utf-8")
    log.write(f"[bold bright_white]llama-server {tag} installed.[/]")
    return installed, tag


async def _build_llama_server_from_source(
    *,
    hardware_info: HardwareInfo,
    rebuilding: bool,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> tuple[Path, str]:
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
    jobs = os.cpu_count() or 4
    rc, tail = await _run_build_with_progress(
        "cmake", "--build", ".", "--target", "llama-server", f"-j{jobs}",
        cwd=build_dir,
        set_status=set_status,
        log=log,
    )
    clear_status()
    if rc != 0:
        raise RuntimeError(tail.strip() or "Failed to build llama-server")
    log.write("[bold bright_white]llama-server built successfully.[/]")

    built = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"
    if not built.is_file():
        raise RuntimeError("llama-server build completed but binary was not found.")
    return built, synced_ref


async def ensure_llama_server(
    setup_result: dict[str, Any],
    *,
    hardware_info: HardwareInfo,
    log: Any,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> dict[str, Any]:
    # Cache hit: managed binary + tag file present and binary doesn't need a GPU rebuild.
    if LLAMA_SERVER_BIN.is_file() and not _needs_rebuild(hardware_info, str(LLAMA_SERVER_BIN)):
        cached_tag = _installed_llama_server_tag()
        if cached_tag:
            merged = dict(setup_result)
            merged["llama_server_path"] = str(LLAMA_SERVER_BIN)
            merged["setup_missing_runtime"] = False
            merged["llama_cpp_ref"] = cached_tag
            return merged

    # Legacy source-built binary on PATH or at the old location that still works.
    existing = current_llama_server_path()
    managed_source_binary = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"
    existing_is_source_managed = (
        existing is not None and Path(existing).resolve() == managed_source_binary.resolve()
    )
    if existing and not existing_is_source_managed and not _needs_rebuild(hardware_info, existing):
        merged = dict(setup_result)
        merged["llama_server_path"] = existing
        merged["setup_missing_runtime"] = False
        return merged

    rebuilding = existing is not None

    prebuilt = await _install_prebuilt_llama_server(
        hardware_info,
        log=log,
        set_status=set_status,
    )
    if prebuilt is not None:
        clear_status()
        installed_path, tag = prebuilt
        merged = dict(setup_result)
        merged["llama_server_path"] = str(installed_path)
        merged["setup_missing_runtime"] = False
        merged["llama_cpp_ref"] = tag
        return merged

    built, synced_ref = await _build_llama_server_from_source(
        hardware_info=hardware_info,
        rebuilding=rebuilding,
        log=log,
        set_status=set_status,
        clear_status=clear_status,
    )
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

    parsed = _parse_huggingface_resolve_url(url)
    if parsed is None:
        raise RuntimeError(
            "Direct model provisioning now uses the Hugging Face CLI fast-transfer path "
            "and requires a huggingface.co `/resolve/<revision>/...` model URL."
        )
    repo_id, revision, filename = parsed

    target_path.parent.mkdir(parents=True, exist_ok=True)
    set_status(f"downloading {target_path.name}")
    log.write(f"[bold bright_white]Downloading {target_path.name} with Hugging Face fast transfer...[/]")
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
        local_dir=target_path.parent,
        progress=report_progress,
    )
    if rc != 0:
        clear_status()
        detail = (err or out).strip()
        raise RuntimeError(detail or "Hugging Face CLI model download failed.")

    downloaded_path = target_path.parent / filename
    if not target_path.is_file() and downloaded_path.is_file():
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
