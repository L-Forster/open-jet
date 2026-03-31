from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..hardware import HardwareInfo, effective_hardware_info
from ..provisioning import recommend_direct_model
from ..setup import build_recommended_payload


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
    payload = build_recommended_payload(
        hardware_info=detected,
        recommended_ctx=0,
        current_cfg=current_cfg,
    )
    hardware_profile = str(payload.get("hardware_profile") or "auto")
    hardware_override = str(payload.get("hardware_override") or "")
    effective = effective_hardware_info(
        hardware_profile,
        detected,
        hardware_override or None,
    )
    direct_model = recommend_direct_model(effective, cfg=current_cfg)
    target_path = str(payload.get("model_download_path") or payload.get("llama_model") or "")
    direct_target = str(direct_model.get("target_path") or "")
    model = RecommendedModel(
        label=str(direct_model.get("label") or _model_label_from_target(target_path)),
        filename=_filename_from_payload(payload),
        url=str(payload.get("model_download_url") or (direct_model.get("url") if target_path == direct_target else "") or ""),
        target_path=target_path,
    )
    llama = RecommendedLlamaConfig(
        device=str(payload.get("device") or "cpu"),
        gpu_layers=int(payload.get("gpu_layers", 0) or 0),
        context_window_tokens=int(payload.get("context_window_tokens", 1024) or 1024),
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


def _filename_from_payload(payload: Mapping[str, object]) -> str:
    target = str(payload.get("model_download_path") or payload.get("llama_model") or "").strip()
    if not target:
        return ""
    return target.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _model_label_from_target(target: str) -> str:
    filename = target.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].strip()
    if not filename:
        return ""
    return filename[:-5] if filename.lower().endswith(".gguf") else filename
