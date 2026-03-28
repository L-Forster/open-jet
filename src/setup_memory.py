from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .runtime_limits import read_memory_snapshot


_CTX_OPTIONS = (1024, 2048, 4096, 6144, 8192, 12288, 16384, 32768)
_MODEL_MB_PER_B_PARAM = 720.0
_MODEL_BASE_OVERHEAD_MB = 256.0
_CTX_KV_MB_PER_TOKEN = 0.55
_CTX_RUNTIME_RESERVE_MB = 512.0


def estimate_model_params_b_from_text(text: str) -> float | None:
    src = str(text or "").strip().lower()
    if not src:
        return None
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*([bm])(?!\w)", src)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if match.group(2) == "b" else value / 1000.0


def estimate_model_memory_mb(*refs: object) -> float | None:
    candidates = [str(ref).strip() for ref in refs if str(ref).strip()]
    for ref in candidates:
        path = Path(ref).expanduser()
        if not path.is_file():
            continue
        try:
            return round(path.stat().st_size / (1024 * 1024), 2)
        except OSError:
            continue
    for ref in candidates:
        params_b = estimate_model_params_b_from_text(ref)
        if params_b is None:
            continue
        return round((params_b * _MODEL_MB_PER_B_PARAM) + _MODEL_BASE_OVERHEAD_MB, 2)
    return None


def detect_free_accelerator_memory_mb(device: str) -> float | None:
    normalized = str(device or "").strip().lower()
    if normalized == "cpu" and sys.platform == "darwin":
        mem = read_memory_snapshot()
        if mem is not None:
            return float(mem.available_mb)
        return None

    if normalized not in {"cuda", "vulkan"}:
        return None

    if normalized == "cuda":
        nvidia_free_mb = _detect_nvidia_free_vram_mb()
        if nvidia_free_mb is not None:
            return nvidia_free_mb
        # Jetson uses unified memory, so available system memory is the best proxy.
        if Path("/dev/nvhost-gpu").exists():
            mem = read_memory_snapshot()
            if mem is not None:
                return float(mem.available_mb)
        return None

    amd_free_mb = _detect_amd_free_vram_mb()
    if amd_free_mb is not None:
        return amd_free_mb
    return None


def _detect_nvidia_free_vram_mb() -> float | None:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return None
    try:
        proc = subprocess.run(
            [
                tool,
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return None
    if proc.returncode != 0:
        return None
    free_values: list[float] = []
    for raw_line in proc.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 4:
            continue
        try:
            free_values.append(float(parts[3]))
        except ValueError:
            continue
    return max(free_values) if free_values else None


def _detect_amd_free_vram_mb() -> float | None:
    sysfs_free = _detect_amd_free_vram_mb_from_sysfs()
    if sysfs_free is not None:
        return sysfs_free
    return _detect_amd_free_vram_mb_from_rocm_smi()


def _detect_amd_free_vram_mb_from_sysfs() -> float | None:
    free_values: list[float] = []
    for total_path in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
        used_path = total_path.with_name("mem_info_vram_used")
        if not used_path.is_file():
            continue
        try:
            total_bytes = int(total_path.read_text(encoding="utf-8").strip())
            used_bytes = int(used_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if total_bytes <= 0 or used_bytes < 0:
            continue
        free_values.append(max(0.0, (total_bytes - used_bytes) / (1024 * 1024)))
    return max(free_values) if free_values else None


def _detect_amd_free_vram_mb_from_rocm_smi() -> float | None:
    tool = shutil.which("rocm-smi")
    if not tool:
        return None
    try:
        proc = subprocess.run(
            [tool, "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    free_values: list[float] = []
    if isinstance(payload, dict):
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            total_raw = (
                value.get("VRAM Total Memory (B)")
                or value.get("vram_total")
                or value.get("total")
            )
            used_raw = (
                value.get("VRAM Total Used Memory (B)")
                or value.get("vram_used")
                or value.get("used")
            )
            try:
                total_bytes = float(str(total_raw).strip().replace(",", ""))
                used_bytes = float(str(used_raw).strip().replace(",", ""))
            except ValueError:
                continue
            free_values.append(max(0.0, (total_bytes - used_bytes) / (1024 * 1024)))
    return max(free_values) if free_values else None


def recommend_context_window_from_remaining_vram_mb(remaining_vram_mb: float | None) -> int:
    if remaining_vram_mb is None:
        return _CTX_OPTIONS[0]
    usable_mb = max(0.0, float(remaining_vram_mb) - _CTX_RUNTIME_RESERVE_MB)
    token_budget = int(usable_mb / _CTX_KV_MB_PER_TOKEN) if usable_mb > 0 else 0
    recommended = _CTX_OPTIONS[0]
    for option in _CTX_OPTIONS:
        if token_budget < option:
            break
        recommended = option
    return recommended


def recommend_setup_context_window(
    *,
    runtime: str,
    device: str,
    fallback_tokens: int,
    model_refs: Iterable[object],
) -> int:
    fallback = max(_CTX_OPTIONS[0], int(fallback_tokens))
    if str(runtime or "").strip().lower() != "llama_cpp":
        return fallback

    free_vram_mb = detect_free_accelerator_memory_mb(device)
    if free_vram_mb is None:
        return fallback

    model_mb = estimate_model_memory_mb(*list(model_refs))
    if model_mb is None:
        return fallback

    return recommend_context_window_from_remaining_vram_mb(free_vram_mb - model_mb)
