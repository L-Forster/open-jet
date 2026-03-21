from __future__ import annotations

import os
import shutil
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


def _detect_vulkan() -> bool:
    if shutil.which("vulkaninfo"):
        return True
    for candidate in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
        p = Path(candidate)
        if p.is_dir() and any(p.iterdir()):
            return True
    return False


def recommended_device() -> str:
    if Path("/usr/local/cuda").exists() or Path("/dev/nvhost-gpu").exists():
        return "cuda"
    if _detect_vulkan():
        return "vulkan"
    return "cpu"


def read_device_model() -> str | None:
    try:
        raw = Path("/proc/device-tree/model").read_bytes()
    except OSError:
        return None
    text = raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    return text or None


def detect_hardware_info() -> HardwareInfo:
    mem = read_memory_snapshot()
    total_ram_gb = (mem.total_mb / 1024.0) if mem else 0.0
    has_cuda = bool(Path("/usr/local/cuda").exists() or Path("/dev/nvhost-gpu").exists())
    has_vulkan = _detect_vulkan()
    board = read_device_model()
    if board:
        label = board
    elif has_cuda:
        label = "CUDA-capable device"
    elif has_vulkan:
        label = "Vulkan-capable device"
    else:
        label = "CPU-only device"
    return HardwareInfo(label=label, total_ram_gb=total_ram_gb, has_cuda=has_cuda, has_vulkan=has_vulkan)


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
    if hw.has_vulkan:
        return "vulkan"
    return "cpu"


def recommended_param_budget_b(profile: str, detected: HardwareInfo, override_key: str | None = None) -> float:
    hw = effective_hardware_info(profile, detected, override_key)
    total_gb = hw.total_ram_gb
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
    if not hw.has_cuda and not hw.has_vulkan:
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
    if device in ("cuda", "vulkan"):
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
