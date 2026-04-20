from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import HARDWARE_OVERRIDE_OPTIONS, RECOMMENDED_LLM_BANDS
from .runtime_limits import read_memory_snapshot


@dataclass(frozen=True)
class HardwareInfo:
    label: str
    total_ram_gb: float
    has_cuda: bool
    has_vulkan: bool = False
    has_rocm: bool = False
    has_metal: bool = False
    vram_mb: float = 0.0


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_cuda() -> bool:
    if sys.platform == "linux":
        if Path("/usr/local/cuda").exists() or Path("/dev/nvhost-gpu").exists():
            return True
    if shutil.which("nvidia-smi"):
        return True
    return False


def _detect_rocm() -> bool:
    if shutil.which("rocm-smi"):
        return True
    if Path("/opt/rocm").is_dir():
        return True
    return False


def _detect_vulkan() -> bool:
    if shutil.which("vulkaninfo"):
        return True
    for candidate in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
        p = Path(candidate)
        if p.is_dir() and any(p.iterdir()):
            return True
    return False


def _darwin_sysctl(name: str) -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _detect_metal() -> bool:
    if sys.platform != "darwin":
        return False
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return True
    # Under Rosetta, Python can report x86_64 even on Apple Silicon.
    return _darwin_sysctl("hw.optional.arm64") == "1"


# ---------------------------------------------------------------------------
# VRAM detection
# ---------------------------------------------------------------------------

def _read_nvidia_vram_mb() -> float:
    """Total VRAM across all NVIDIA GPUs via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total = 0.0
    for line in out.strip().splitlines():
        try:
            total += float(line.strip())
        except ValueError:
            pass
    return total


def _read_rocm_vram_mb() -> float:
    """Total VRAM across all AMD GPUs via sysfs or rocm-smi."""
    total = 0.0
    # Try sysfs first (no dependency on rocm-smi output format)
    drm = Path("/sys/class/drm")
    if drm.is_dir():
        for card in drm.glob("card[0-9]*/device/mem_info_vram_total"):
            try:
                total += float(card.read_text().strip()) / (1024.0 * 1024.0)
            except (OSError, ValueError):
                pass
    if total > 0:
        return total
    # Fall back to rocm-smi
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram"], text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    for line in out.splitlines():
        if "Total" in line and "Memory" in line:
            for part in line.split():
                try:
                    total += float(part) / (1024.0 * 1024.0)
                    break
                except ValueError:
                    continue
    return total


def _parse_vulkan_heap_size_mb(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value) / (1024.0 * 1024.0)
    text = str(value or "").strip()
    if not text:
        return 0.0
    lowered = text.lower()
    if lowered.startswith("0x"):
        try:
            return int(lowered, 16) / (1024.0 * 1024.0)
        except ValueError:
            return 0.0
    cleaned = lowered.replace(",", "").replace("_", "")
    for suffix, scale in (
        ("gib", 1024.0),
        ("gb", 1000.0),
        ("mib", 1.0),
        ("mb", 1000.0 / 1024.0),
        ("kib", 1.0 / 1024.0),
        ("kb", 1000.0 / (1024.0 * 1024.0)),
        ("b", 1.0 / (1024.0 * 1024.0)),
    ):
        if cleaned.endswith(suffix):
            try:
                return float(cleaned.removesuffix(suffix).strip()) * scale
            except ValueError:
                return 0.0
    try:
        return float(cleaned) / (1024.0 * 1024.0)
    except ValueError:
        return 0.0


def _collect_vulkan_device_local_heap_mb(payload: object) -> float:
    total = 0.0
    if isinstance(payload, dict):
        heaps = payload.get("memoryHeaps")
        if isinstance(heaps, list):
            for heap in heaps:
                if not isinstance(heap, dict):
                    continue
                flags = str(heap.get("flags") or heap.get("Flags") or "")
                if "DEVICE_LOCAL" not in flags.upper():
                    continue
                size = (
                    heap.get("size")
                    or heap.get("Size")
                    or heap.get("sizeBytes")
                    or heap.get("heapSize")
                )
                total += _parse_vulkan_heap_size_mb(size)
        for value in payload.values():
            total += _collect_vulkan_device_local_heap_mb(value)
    elif isinstance(payload, list):
        for item in payload:
            total += _collect_vulkan_device_local_heap_mb(item)
    return total


def _parse_vulkaninfo_text_total_vram_mb(text: str) -> float:
    total = 0.0
    current_size = 0.0
    current_device_local = False

    def flush() -> None:
        nonlocal total, current_size, current_device_local
        if current_device_local and current_size > 0:
            total += current_size
        current_size = 0.0
        current_device_local = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"memoryHeaps\[\d+\]:", line):
            flush()
            continue
        if "MEMORY_HEAP_DEVICE_LOCAL_BIT" in line:
            current_device_local = True
            continue
        if line.startswith("size"):
            match = re.search(r"=\s*(\d+)", line)
            if match:
                current_size = int(match.group(1)) / (1024.0 * 1024.0)
    flush()
    return total


def _read_vulkan_vram_mb() -> float:
    """Total device-local VRAM via vulkaninfo JSON or text output."""
    tool = shutil.which("vulkaninfo")
    if not tool:
        return 0.0
    shell = os.environ.get("SHELL") or "/bin/bash"
    for args in (["--json"], ["--summary"], []):
        try:
            proc = subprocess.run(
                [shell, "-lc", " ".join([tool, *args])],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
        if "--json" in args:
            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                payload = None
            if payload is not None:
                total = _collect_vulkan_device_local_heap_mb(payload)
                if total > 0:
                    return total
        total = _parse_vulkaninfo_text_total_vram_mb(output)
        if total > 0:
            return total
    return 0.0


def _read_metal_vram_mb() -> float:
    """Apple Silicon uses unified memory — VRAM = system RAM."""
    mem = read_memory_snapshot()
    return mem.total_mb if mem else 0.0


def _detect_vram_mb(has_cuda: bool, has_rocm: bool, has_metal: bool, has_vulkan: bool) -> float:
    if has_cuda:
        vram = _read_nvidia_vram_mb()
        if vram > 0:
            return vram
    if has_rocm:
        vram = _read_rocm_vram_mb()
        if vram > 0:
            return vram
    if has_vulkan:
        vram = _read_vulkan_vram_mb()
        if vram > 0:
            return vram
    if has_metal:
        return _read_metal_vram_mb()
    return 0.0


# ---------------------------------------------------------------------------
# Device recommendation
# ---------------------------------------------------------------------------

def recommended_device() -> str:
    if _detect_cuda():
        return "cuda"
    if _detect_rocm():
        return "rocm"
    if _detect_metal():
        return "metal"
    if _detect_vulkan():
        return "vulkan"
    return "cpu"


# ---------------------------------------------------------------------------
# Board / platform label
# ---------------------------------------------------------------------------

def read_device_model() -> str | None:
    if sys.platform == "darwin":
        chip = _darwin_sysctl("machdep.cpu.brand_string")
        if chip:
            return chip
        return f"Apple {platform.machine()}"
    try:
        raw = Path("/proc/device-tree/model").read_bytes()
    except OSError:
        return None
    text = raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    return text or None


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------

def detect_hardware_info() -> HardwareInfo:
    mem = read_memory_snapshot()
    total_ram_gb = (mem.total_mb / 1024.0) if mem else 0.0
    has_cuda = _detect_cuda()
    has_rocm = _detect_rocm()
    has_vulkan = _detect_vulkan()
    has_metal = _detect_metal()
    vram_mb = _detect_vram_mb(has_cuda, has_rocm, has_metal, has_vulkan)
    board = read_device_model()
    if board:
        label = board
    elif has_metal:
        label = f"Apple Silicon ({total_ram_gb:.0f} GB unified)"
    elif has_cuda:
        label = "CUDA-capable device"
    elif has_rocm:
        label = "ROCm-capable device"
    elif has_vulkan:
        label = "Vulkan-capable device"
    else:
        label = "CPU-only device"
    return HardwareInfo(
        label=label,
        total_ram_gb=total_ram_gb,
        has_cuda=has_cuda,
        has_vulkan=has_vulkan,
        has_rocm=has_rocm,
        has_metal=has_metal,
        vram_mb=vram_mb,
    )


def effective_hardware_info(profile: str, detected: HardwareInfo, override_key: str | None = None) -> HardwareInfo:
    if profile != "other":
        return detected
    for key, label, ram_gb, has_cuda in HARDWARE_OVERRIDE_OPTIONS:
        if key == override_key:
            clean_label = label.split(" (", 1)[0]
            return HardwareInfo(label=clean_label, total_ram_gb=ram_gb, has_cuda=has_cuda)
    return detected


def recommended_device_for_hardware(profile: str, detected: HardwareInfo, override_key: str | None = None) -> str:
    hw = effective_hardware_info(profile, detected, override_key)
    if hw.has_cuda:
        return "cuda"
    if hw.has_rocm:
        return "rocm"
    if hw.has_metal:
        return "metal"
    if hw.has_vulkan:
        return "vulkan"
    return "cpu"


def recommended_param_budget_b(profile: str, detected: HardwareInfo, override_key: str | None = None) -> float:
    hw = effective_hardware_info(profile, detected, override_key)
    # On unified memory (Metal) or when VRAM is known, use the larger of RAM and VRAM
    effective_gb = hw.total_ram_gb
    if hw.vram_mb > 0 and not hw.has_metal:
        effective_gb = max(effective_gb, hw.vram_mb / 1024.0)
    total_gb = effective_gb
    if total_gb < 6:
        cap = 2.0
    elif total_gb < 12:
        cap = 4.0
    elif total_gb < 24:
        cap = 8.0
    elif total_gb < 48:
        cap = 14.0
    else:
        cap = 32.0
    has_gpu = hw.has_cuda or hw.has_vulkan or hw.has_rocm or hw.has_metal
    if not has_gpu:
        cap = min(cap, 8.0)
    return cap


def recommended_llm_models(max_params_b: float) -> list[tuple[str, str]]:
    for band_limit, models in RECOMMENDED_LLM_BANDS:
        if max_params_b <= band_limit:
            return [(f"{title} ({params:g}B params)", tag) for tag, params, title in models]
    models = RECOMMENDED_LLM_BANDS[-1][1]
    return [(f"{title} ({params:g}B params)", tag) for tag, params, title in models]


def recommended_context_window_tokens() -> int:
    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    mem = read_memory_snapshot()
    if not mem:
        return 4096 if headless else 2048
    total_gb = mem.total_mb / 1024.0
    return recommended_context_window_tokens_from_total(
        total_gb,
        headless=headless,
        available_mb=mem.available_mb,
    )


def recommended_context_window_tokens_from_total(
    total_gb: float,
    *,
    headless: bool,
    available_mb: float | None = None,
) -> int:
    if total_gb >= 48:
        rec = 12288
    elif total_gb >= 24:
        rec = 8192
    elif total_gb >= 12:
        rec = 6144
    elif total_gb >= 7:
        rec = 4096 if headless else 3072
    elif total_gb >= 4:
        rec = 2048
    else:
        rec = 1024

    if available_mb is not None and available_mb < 1200:
        rec = min(rec, 2048)
    return rec


def recommended_gpu_layers(device: str, total_ram_gb: float | None = None) -> int:
    if device in ("cuda", "vulkan", "rocm", "metal"):
        return 99
    return 0


def is_jetson_label(label: str | None) -> bool:
    if not label:
        return False
    return "jetson" in label.lower()


def running_on_jetson() -> bool:
    if is_jetson_label(read_device_model()):
        return True
    return Path("/etc/nv_tegra_release").exists()
