from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Mapping

from ..hardware import HardwareInfo, effective_hardware_info, recommended_device_for_hardware
from ..provisioning import recommend_direct_model

_BYTES_PER_MB = 1024.0 * 1024.0
_KV_CACHE_Q8_0_BYTES_PER_ELEMENT = 34.0 / 32.0


@dataclass(frozen=True)
class HardwarePerformanceProfile:
    key: str
    label: str
    device: str
    memory_gb: float
    memory_bandwidth_gbps: float
    compute_tflops: float


@dataclass(frozen=True)
class ModelPerformanceProfile:
    key: str
    label: str
    filename: str
    parameter_count_billion: float
    model_size_mb: float
    kv_bytes_per_token: float
    decode_context_tokens: int


@dataclass(frozen=True)
class TokenGenerationEstimate:
    hardware_key: str
    hardware_label: str
    model_key: str
    model_label: str
    context_window_tokens: int
    model_size_mb: float
    context_cache_mb: float
    total_memory_required_mb: float
    hardware_memory_mb: float
    compute_bound_tokens_per_second: float | None
    bandwidth_bound_tokens_per_second: float
    estimated_tokens_per_second: float
    limiting_factor: str


@dataclass(frozen=True)
class TokenGenerationWorkload:
    hardware_key: str
    hardware_label: str
    model_key: str
    model_label: str
    context_window_tokens: int
    model_size_mb: float
    context_cache_mb: float
    total_memory_required_mb: float
    hardware_memory_mb: float
    weight_bytes_per_token: float
    kv_cache_read_bytes_per_token: float
    kv_cache_write_bytes_per_token: float
    total_bandwidth_bytes_per_token: float
    dense_flops_per_token: float
    attention_flops_per_token: float
    total_flops_per_token: float


@dataclass(frozen=True)
class TokenGenerationModelEstimate:
    hardware_key: str
    hardware_label: str
    model_key: str
    model_label: str
    fits_in_memory: bool
    error: str
    estimate: TokenGenerationEstimate | None


def list_hardware_performance_profiles(*, device: str = "") -> list[HardwarePerformanceProfile]:
    normalized_device = _normalize_key(device)
    profiles = [_hardware_profile_from_row(row) for row in _load_hardware_registry()]
    if not normalized_device:
        return profiles
    return [
        profile
        for profile in profiles
        if _normalize_key(profile.device) == normalized_device
    ]


def list_model_performance_profiles() -> list[ModelPerformanceProfile]:
    return [_model_profile_from_row(row) for row in _load_model_registry()]


def build_token_generation_workload(
    *,
    hardware: HardwarePerformanceProfile,
    model: ModelPerformanceProfile,
    context_window_tokens: int | None = None,
    hardware_memory_gb: float | None = None,
) -> TokenGenerationWorkload:
    context_tokens = max(1, int(context_window_tokens or model.decode_context_tokens))
    context_cache_mb = (context_tokens * model.kv_bytes_per_token) / _BYTES_PER_MB
    total_memory_required_mb = model.model_size_mb + context_cache_mb
    resolved_memory_gb = float(hardware_memory_gb) if hardware_memory_gb is not None else hardware.memory_gb
    hardware_memory_mb = resolved_memory_gb * 1024.0
    if total_memory_required_mb > hardware_memory_mb:
        raise ValueError(
            f"{model.label} at context {context_tokens} requires {total_memory_required_mb:.2f}MB, "
            f"which exceeds {hardware.label} memory budget {hardware_memory_mb:.2f}MB."
        )

    weight_bytes_per_token = model.model_size_mb * _BYTES_PER_MB
    kv_cache_read_bytes_per_token = context_tokens * model.kv_bytes_per_token
    kv_cache_write_bytes_per_token = model.kv_bytes_per_token
    total_bandwidth_bytes_per_token = (
        weight_bytes_per_token
        + kv_cache_read_bytes_per_token
        + kv_cache_write_bytes_per_token
    )
    dense_flops_per_token = model.parameter_count_billion * 1_000_000_000.0 * 2.0
    kv_cache_elements_per_token = model.kv_bytes_per_token / _KV_CACHE_Q8_0_BYTES_PER_ELEMENT
    attention_flops_per_token = context_tokens * kv_cache_elements_per_token * 2.0
    total_flops_per_token = dense_flops_per_token + attention_flops_per_token

    return TokenGenerationWorkload(
        hardware_key=hardware.key,
        hardware_label=hardware.label,
        model_key=model.key,
        model_label=model.label,
        context_window_tokens=context_tokens,
        model_size_mb=model.model_size_mb,
        context_cache_mb=round(context_cache_mb, 2),
        total_memory_required_mb=round(total_memory_required_mb, 2),
        hardware_memory_mb=round(hardware_memory_mb, 2),
        weight_bytes_per_token=weight_bytes_per_token,
        kv_cache_read_bytes_per_token=kv_cache_read_bytes_per_token,
        kv_cache_write_bytes_per_token=kv_cache_write_bytes_per_token,
        total_bandwidth_bytes_per_token=total_bandwidth_bytes_per_token,
        dense_flops_per_token=dense_flops_per_token,
        attention_flops_per_token=attention_flops_per_token,
        total_flops_per_token=total_flops_per_token,
    )


def estimate_token_generation_speed(
    *,
    hardware_key: str,
    model_key: str,
    context_window_tokens: int | None = None,
    hardware_memory_gb: float | None = None,
) -> TokenGenerationEstimate:
    hardware = get_hardware_performance_profile(hardware_key)
    model = get_model_performance_profile(model_key)
    return estimate_token_generation_speed_for_profiles(
        hardware=hardware,
        model=model,
        context_window_tokens=context_window_tokens,
        hardware_memory_gb=hardware_memory_gb,
    )


def estimate_token_generation_speed_for_profiles(
    *,
    hardware: HardwarePerformanceProfile,
    model: ModelPerformanceProfile,
    context_window_tokens: int | None = None,
    hardware_memory_gb: float | None = None,
) -> TokenGenerationEstimate:
    workload = build_token_generation_workload(
        hardware=hardware,
        model=model,
        context_window_tokens=context_window_tokens,
        hardware_memory_gb=hardware_memory_gb,
    )
    compute_bound: float | None = None
    if hardware.compute_tflops > 0:
        compute_bound = (hardware.compute_tflops * 1_000_000_000_000.0) / workload.total_flops_per_token
    bandwidth_bound = (hardware.memory_bandwidth_gbps * 1_000_000_000.0) / workload.total_bandwidth_bytes_per_token
    if compute_bound is None:
        limiting_factor = "memory"
        estimated = bandwidth_bound
    else:
        limiting_factor = "compute" if compute_bound <= bandwidth_bound else "memory"
        estimated = min(compute_bound, bandwidth_bound)

    return TokenGenerationEstimate(
        hardware_key=workload.hardware_key,
        hardware_label=workload.hardware_label,
        model_key=workload.model_key,
        model_label=workload.model_label,
        context_window_tokens=workload.context_window_tokens,
        model_size_mb=workload.model_size_mb,
        context_cache_mb=workload.context_cache_mb,
        total_memory_required_mb=workload.total_memory_required_mb,
        hardware_memory_mb=workload.hardware_memory_mb,
        compute_bound_tokens_per_second=round(compute_bound, 2) if compute_bound is not None else None,
        bandwidth_bound_tokens_per_second=round(bandwidth_bound, 2),
        estimated_tokens_per_second=round(estimated, 2),
        limiting_factor=limiting_factor,
    )


def estimate_token_generation_speeds_for_hardware(
    *,
    hardware_key: str,
    hardware_memory_gb: float | None = None,
    context_window_tokens: int | None = None,
) -> list[TokenGenerationModelEstimate]:
    hardware = get_hardware_performance_profile(hardware_key)
    results: list[TokenGenerationModelEstimate] = []
    for row in _load_model_registry():
        model = _model_profile_from_row(row)
        try:
            estimate = estimate_token_generation_speed(
                hardware_key=hardware_key,
                model_key=model.key,
                context_window_tokens=context_window_tokens,
                hardware_memory_gb=hardware_memory_gb,
            )
        except ValueError as exc:
            results.append(
                TokenGenerationModelEstimate(
                    hardware_key=hardware.key,
                    hardware_label=hardware.label,
                    model_key=model.key,
                    model_label=model.label,
                    fits_in_memory=False,
                    error=str(exc),
                    estimate=None,
                )
            )
            continue
        results.append(
            TokenGenerationModelEstimate(
                hardware_key=hardware.key,
                hardware_label=hardware.label,
                model_key=model.key,
                model_label=model.label,
                fits_in_memory=True,
                error="",
                estimate=estimate,
            )
        )
    return results


def estimate_recommended_token_generation_speed(
    hardware: HardwareInfo | Mapping[str, object],
    *,
    hardware_profile: str = "auto",
    hardware_override: str = "",
    cfg: Mapping[str, object] | None = None,
) -> TokenGenerationEstimate:
    detected = _coerce_hardware_info(hardware)
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
    direct = recommend_direct_model(effective, cfg=cfg)
    hardware_key = resolve_hardware_profile_key(
        detected=detected,
        effective=effective,
        hardware_profile=hardware_profile,
        hardware_override=hardware_override,
    )
    model_key = resolve_model_profile_key(
        label=str(direct.get("label") or ""),
        filename=str(direct.get("filename") or ""),
    )
    return estimate_token_generation_speed(
        hardware_key=hardware_key,
        model_key=model_key,
        context_window_tokens=int(direct.get("context_window_tokens", 0) or 0) or None,
        hardware_memory_gb=_device_memory_budget_gb(effective, device=device),
    )


def get_hardware_performance_profile(key: str) -> HardwarePerformanceProfile:
    normalized = _normalize_key(key)
    for row in _load_hardware_registry():
        if _normalize_key(str(row["key"])) == normalized:
            return _hardware_profile_from_row(row)
    raise ValueError(f"Unknown hardware performance profile: {key}")


def get_model_performance_profile(key: str) -> ModelPerformanceProfile:
    normalized = _normalize_key(key)
    for row in _load_model_registry():
        if _normalize_key(str(row["key"])) == normalized:
            return _model_profile_from_row(row)
    raise ValueError(f"Unknown model performance profile: {key}")


def resolve_hardware_profile_key(
    *,
    detected: HardwareInfo,
    effective: HardwareInfo,
    hardware_profile: str,
    hardware_override: str,
) -> str:
    if hardware_profile == "other":
        normalized_override = _normalize_key(hardware_override)
        if normalized_override.startswith("desktop"):
            return "intel_core_i9_14900k"
        for row in _load_hardware_registry():
            if _normalize_key(str(row["key"])) == normalized_override:
                return str(row["key"])
        raise ValueError(f"No token estimator hardware profile for override: {hardware_override}")

    device = recommended_device_for_hardware(hardware_profile, detected, hardware_override or None)
    return _resolve_hardware_key_by_label_and_memory(
        label=effective.label,
        memory_gb=effective.total_ram_gb if device == "cpu" else effective.vram_mb / 1024.0,
        device=device,
    )


def resolve_model_profile_key(*, label: str, filename: str) -> str:
    normalized_label = _normalize_key(label)
    normalized_filename = _normalize_key(filename)
    for row in _load_model_registry():
        if _normalize_key(str(row["label"])) == normalized_label:
            return str(row["key"])
        if _normalize_key(str(row["filename"])) == normalized_filename:
            return str(row["key"])
        for alias in row.get("aliases", []):
            if _normalize_key(str(alias)) in {normalized_label, normalized_filename}:
                return str(row["key"])
    raise ValueError(f"No token estimator model profile for label={label!r} filename={filename!r}")


def _resolve_hardware_key_by_label_and_memory(*, label: str, memory_gb: float, device: str) -> str:
    normalized_label = _normalize_key(label)
    normalized_device = _normalize_key(device)
    matched: list[dict[str, object]] = []
    for row in _load_hardware_registry():
        if _normalize_key(str(row["device"])) != normalized_device:
            continue
        labels = [_normalize_key(str(row["label"]))]
        labels.extend(_normalize_key(str(alias)) for alias in row.get("aliases", []))
        if normalized_label in labels:
            matched.append(row)
    if not matched:
        raise ValueError(f"No token estimator hardware profile for label={label!r} device={device!r}")
    exact = [
        row for row in matched
        if abs(float(row.get("memory_gb", 0.0) or 0.0) - float(memory_gb)) < 0.01
    ]
    if len(exact) == 1:
        return str(exact[0]["key"])
    if len(matched) == 1:
        return str(matched[0]["key"])
    raise ValueError(
        f"Ambiguous token estimator hardware profile for label={label!r} device={device!r} memory_gb={memory_gb!r}"
    )


def _coerce_hardware_info(hardware: HardwareInfo | Mapping[str, object]) -> HardwareInfo:
    if isinstance(hardware, HardwareInfo):
        return hardware
    if isinstance(hardware, Mapping):
        normalized_gpu = str(hardware.get("gpu") or hardware.get("device") or "").strip().lower()
        return HardwareInfo(
            label=str(hardware.get("label") or "").strip() or "CPU-only device",
            total_ram_gb=max(0.0, float(hardware.get("total_ram_gb", 0.0) or 0.0)),
            has_cuda=normalized_gpu == "cuda",
            has_vulkan=normalized_gpu == "vulkan",
            has_rocm=normalized_gpu == "rocm",
            has_metal=normalized_gpu == "metal",
            vram_mb=max(0.0, float(hardware.get("vram_mb", 0.0) or 0.0)),
        )
    raise TypeError("hardware must be HardwareInfo or a mapping")


def _device_memory_budget_gb(hardware: HardwareInfo, *, device: str) -> float | None:
    normalized_device = (device or "").strip().lower()
    if normalized_device in {"cuda", "rocm", "vulkan"} and hardware.vram_mb > 0:
        return hardware.vram_mb / 1024.0
    if normalized_device in {"cpu", "metal"} and hardware.total_ram_gb > 0:
        return hardware.total_ram_gb
    if hardware.vram_mb > 0:
        return hardware.vram_mb / 1024.0
    if hardware.total_ram_gb > 0:
        return hardware.total_ram_gb
    return None


def _hardware_profile_from_row(row: Mapping[str, object]) -> HardwarePerformanceProfile:
    return HardwarePerformanceProfile(
        key=str(row["key"]),
        label=str(row["label"]),
        device=str(row["device"]),
        memory_gb=float(row["memory_gb"]),
        memory_bandwidth_gbps=float(row["memory_bandwidth_gbps"]),
        compute_tflops=float(row["compute_tflops"]),
    )


def _model_profile_from_row(row: Mapping[str, object]) -> ModelPerformanceProfile:
    return ModelPerformanceProfile(
        key=str(row["key"]),
        label=str(row["label"]),
        filename=str(row["filename"]),
        parameter_count_billion=float(row["parameter_count_billion"]),
        model_size_mb=float(row["model_size_mb"]),
        kv_bytes_per_token=float(row["kv_bytes_per_token"]),
        decode_context_tokens=int(row["decode_context_tokens"]),
    )


def _normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


@lru_cache(maxsize=1)
def _load_hardware_registry() -> tuple[dict[str, object], ...]:
    path = files("src").joinpath("data/token_estimator_hardware.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(dict(row) for row in payload)


@lru_cache(maxsize=1)
def _load_model_registry() -> tuple[dict[str, object], ...]:
    path = files("src").joinpath("data/token_estimator_models.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(dict(row) for row in payload)
