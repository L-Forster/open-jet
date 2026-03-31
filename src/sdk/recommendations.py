from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

from ..config import setup_direct_model_catalog
from ..hardware import (
    HardwareInfo,
    effective_hardware_info,
    recommended_context_window_tokens_from_total,
    recommended_device_for_hardware,
    recommended_gpu_layers,
    recommended_param_budget_b,
)


@dataclass(frozen=True)
class HardwareRecommendationInput:
    total_ram_gb: float
    gpu: str = ""
    label: str = ""
    vram_mb: float = 0.0
    hardware_profile: str = "auto"
    hardware_override: str = ""


@dataclass(frozen=True)
class RecommendedModel:
    label: str
    filename: str
    url: str
    target_path: str


@dataclass(frozen=True)
class RecommendedLlamaConfig:
    device: str
    gpu_layers: int
    context_window_tokens: int


@dataclass(frozen=True)
class HardwareRecommendation:
    detected_hardware: HardwareInfo
    effective_hardware: HardwareInfo
    model: RecommendedModel
    llama: RecommendedLlamaConfig
    hardware_profile: str
    hardware_override: str


def recommend_hardware_config(
    hardware: HardwareInfo | HardwareRecommendationInput | Mapping[str, object],
    *,
    cfg: Mapping[str, object] | None = None,
) -> HardwareRecommendation:
    detected = _coerce_hardware_info(hardware)
    current_cfg = _recommendation_cfg(hardware, cfg=cfg)
    hardware_profile = "other" if str(current_cfg.get("hardware_profile") or "") == "other" else "auto"
    hardware_override = str(current_cfg.get("hardware_override") or "") if hardware_profile == "other" else ""
    effective = effective_hardware_info(
        hardware_profile,
        detected,
        hardware_override or None,
    )

    device = recommended_device_for_hardware(
        hardware_profile,
        detected,
        hardware_override or None,
    )
    model = _recommend_model(effective, cfg=current_cfg)
    llama = RecommendedLlamaConfig(
        device=device,
        gpu_layers=recommended_gpu_layers(device, effective.total_ram_gb),
        context_window_tokens=_recommend_context_window_tokens(effective),
    )

    return HardwareRecommendation(
        detected_hardware=detected,
        effective_hardware=effective,
        model=model,
        llama=llama,
        hardware_profile=hardware_profile,
        hardware_override=hardware_override,
    )


def _coerce_hardware_info(
    hardware: HardwareInfo | HardwareRecommendationInput | Mapping[str, object],
) -> HardwareInfo:
    if isinstance(hardware, HardwareInfo):
        return hardware
    if isinstance(hardware, HardwareRecommendationInput):
        return _hardware_info_from_values(
            total_ram_gb=hardware.total_ram_gb,
            gpu=hardware.gpu,
            label=hardware.label,
            vram_mb=hardware.vram_mb,
        )
    if isinstance(hardware, Mapping):
        return _hardware_info_from_values(
            total_ram_gb=float(hardware.get("total_ram_gb", 0.0) or 0.0),
            gpu=str(hardware.get("gpu") or hardware.get("device") or ""),
            label=str(hardware.get("label") or ""),
            vram_mb=float(hardware.get("vram_mb", 0.0) or 0.0),
        )
    raise TypeError("hardware must be HardwareInfo, HardwareRecommendationInput, or a mapping")


def _hardware_info_from_values(
    *,
    total_ram_gb: float,
    gpu: str,
    label: str,
    vram_mb: float,
) -> HardwareInfo:
    normalized_gpu = gpu.strip().lower()
    has_cuda = normalized_gpu == "cuda"
    has_vulkan = normalized_gpu == "vulkan"
    has_rocm = normalized_gpu == "rocm"
    has_metal = normalized_gpu == "metal"
    resolved_label = label.strip()
    if not resolved_label:
        if normalized_gpu:
            resolved_label = f"{normalized_gpu.upper()} device"
        else:
            resolved_label = "CPU-only device"
    return HardwareInfo(
        label=resolved_label,
        total_ram_gb=max(0.0, float(total_ram_gb)),
        has_cuda=has_cuda,
        has_vulkan=has_vulkan,
        has_rocm=has_rocm,
        has_metal=has_metal,
        vram_mb=max(0.0, float(vram_mb)),
    )


def _recommendation_cfg(
    hardware: HardwareInfo | HardwareRecommendationInput | Mapping[str, object],
    *,
    cfg: Mapping[str, object] | None,
) -> dict[str, object]:
    merged = dict(cfg or {})
    if isinstance(hardware, HardwareRecommendationInput):
        if hardware.hardware_profile:
            merged["hardware_profile"] = hardware.hardware_profile
        if hardware.hardware_override:
            merged["hardware_override"] = hardware.hardware_override
        return merged
    if isinstance(hardware, Mapping):
        if hardware.get("hardware_profile") is not None:
            merged["hardware_profile"] = str(hardware.get("hardware_profile") or "")
        if hardware.get("hardware_override") is not None:
            merged["hardware_override"] = str(hardware.get("hardware_override") or "")
    return merged


def _recommend_model(
    hardware: HardwareInfo,
    *,
    cfg: Mapping[str, object] | None,
) -> RecommendedModel:
    catalog = setup_direct_model_catalog(cfg)
    if not catalog:
        return RecommendedModel(label="", filename="", url="", target_path="")

    sizing_hw = _model_sizing_hardware(hardware)
    budget_b = recommended_param_budget_b("auto", sizing_hw)

    parsed_catalog: list[tuple[Mapping[str, object], float | None]] = []
    for row in catalog:
        size_b = _parse_model_size_b(str(row.get("label") or ""))
        parsed_catalog.append((row, size_b))

    candidates = [item for item in parsed_catalog if item[1] is not None]
    if candidates:
        row, _size_b = min(
            candidates,
            key=lambda item: (
                abs(float(item[1]) - budget_b),
                0 if float(item[1]) <= budget_b else 1,
                float(item[1]),
            ),
        )
    else:
        row = catalog[-1]

    filename = str(row.get("filename") or "")
    target_path = str(Path.home() / ".openjet" / "models" / filename) if filename else ""
    return RecommendedModel(
        label=str(row.get("label") or _model_label_from_target(target_path)),
        filename=filename,
        url=str(row.get("url") or ""),
        target_path=target_path,
    )


def _model_sizing_hardware(hardware: HardwareInfo) -> HardwareInfo:
    total_ram_gb = max(0.0, float(hardware.total_ram_gb))
    if hardware.vram_mb > 0 and not hardware.has_metal and (hardware.has_cuda or hardware.has_rocm or hardware.has_vulkan):
        total_ram_gb += max(0.0, float(hardware.vram_mb)) / 1024.0
    return HardwareInfo(
        label=hardware.label,
        total_ram_gb=total_ram_gb,
        has_cuda=hardware.has_cuda,
        has_vulkan=hardware.has_vulkan,
        has_rocm=hardware.has_rocm,
        has_metal=hardware.has_metal,
        vram_mb=hardware.vram_mb,
    )


def _recommend_context_window_tokens(hardware: HardwareInfo) -> int:
    sizing_hw = _model_sizing_hardware(hardware)
    return recommended_context_window_tokens_from_total(
        sizing_hw.total_ram_gb,
        headless=False,
    )


def _parse_model_size_b(label: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", label)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _model_label_from_target(target: str) -> str:
    filename = target.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip()
    if not filename:
        return ""
    return filename[:-5] if filename.lower().endswith(".gguf") else filename
