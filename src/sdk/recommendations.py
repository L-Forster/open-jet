from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..hardware import (
    HardwareInfo,
    effective_hardware_info,
    recommended_device_for_hardware,
    recommended_gpu_layers,
)
from ..provisioning import recommend_direct_model


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
    direct = recommend_direct_model(effective, cfg=current_cfg)
    model = RecommendedModel(
        label=str(direct.get("label") or ""),
        filename=str(direct.get("filename") or ""),
        url=str(direct.get("url") or ""),
        target_path=str(direct.get("target_path") or ""),
    )
    llama = RecommendedLlamaConfig(
        device=device,
        gpu_layers=recommended_gpu_layers(device, effective.total_ram_gb),
        context_window_tokens=int(direct.get("context_window_tokens", 1024) or 1024),
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


