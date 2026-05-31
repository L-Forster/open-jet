from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .runtime_registry import CODEX_RUNTIME, DEFAULT_RUNTIME, LITELLM_RUNTIME, RUNTIME_LABEL, active_model_ref, active_runtime

PROFILE_KEYS: tuple[str, ...] = (
    "runtime",
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "codex_base_url",
    "reasoning_effort",
    "reasoning_summary",
    "text_verbosity",
    "llama_model",
    "model_download_url",
    "model_download_path",
    "llama_server_path",
    "model_source",
    "setup_missing_model",
    "setup_update_model",
    "model_update_target",
    "model_update_applied",
    "device",
    "context_window_tokens",
    "gpu_layers",
    "llama_cpu_moe",
    "llama_n_cpu_moe",
    "llama_cpp_ref",
    "llama_mtp",
    "model_size_mb",
    "active_model_size_mb",
    "resident_model_size_mb",
    "kv_bytes_per_token",
    "unified_memory_only",
    "hardware_profile",
    "hardware_override",
    "setup_complete",
)


def profile_model_ref(profile: Mapping[str, object]) -> str:
    return active_model_ref(dict(profile))


def default_profile_name(source: Mapping[str, object]) -> str:
    model_ref = profile_model_ref(source)
    if active_runtime(dict(source)) in {CODEX_RUNTIME, LITELLM_RUNTIME}:
        return model_ref or "OpenAI Codex"
    if not model_ref:
        return RUNTIME_LABEL
    name = Path(model_ref).name or model_ref
    if name.lower().endswith(".gguf"):
        name = name[:-5]
    return name.strip() or RUNTIME_LABEL


def build_model_profile(source: Mapping[str, object], *, name: str | None = None) -> dict[str, Any] | None:
    model_ref = profile_model_ref(source)
    if not model_ref:
        return None

    profile_name = (name or str(source.get("active_model_profile") or "")).strip() or default_profile_name(source)
    runtime = active_runtime(dict(source))
    profile: dict[str, Any] = {"name": profile_name, "runtime": runtime}
    for key in PROFILE_KEYS:
        value = source.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        profile[key] = value
    if runtime in {CODEX_RUNTIME, LITELLM_RUNTIME}:
        profile["model"] = model_ref
        if runtime == CODEX_RUNTIME:
            profile.setdefault("provider", "openai-codex")
    else:
        profile["llama_model"] = model_ref
        profile["runtime"] = DEFAULT_RUNTIME
    return profile


def list_model_profiles(cfg: Mapping[str, object], *, include_active_fallback: bool = True) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_profiles = cfg.get("model_profiles")
    if isinstance(raw_profiles, list):
        for item in raw_profiles:
            if not isinstance(item, Mapping):
                continue
            built = build_model_profile(item, name=str(item.get("name") or "").strip() or None)
            if not built:
                continue
            key = built["name"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            profiles.append(built)

    if include_active_fallback:
        active_name = str(cfg.get("active_model_profile") or "").strip() or None
        active = build_model_profile(cfg, name=active_name)
        if active:
            active_key = active["name"].strip().lower()
            if active_key not in seen:
                profiles.insert(0, active)
    return profiles


def get_model_profile(cfg: Mapping[str, object], name: str) -> dict[str, Any] | None:
    needle = name.strip().lower()
    if not needle:
        return None
    for profile in list_model_profiles(cfg):
        if profile["name"].strip().lower() == needle:
            return dict(profile)
    return None


def replace_model_profile(
    cfg: dict[str, Any],
    profile: Mapping[str, object],
    *,
    previous_name: str | None = None,
) -> dict[str, Any]:
    built = build_model_profile(profile, name=str(profile.get("name") or "").strip() or None)
    if not built:
        raise ValueError("Model profile requires a configured model reference.")

    previous_key = (previous_name or "").strip().lower()
    target_key = built["name"].strip().lower()
    updated: list[dict[str, Any]] = [built]
    for existing in list_model_profiles(cfg, include_active_fallback=False):
        existing_key = existing["name"].strip().lower()
        if existing_key in {previous_key, target_key}:
            continue
        updated.append(existing)
    cfg["model_profiles"] = updated
    return built


def apply_model_profile(cfg: dict[str, Any], profile: Mapping[str, object]) -> dict[str, Any]:
    built = build_model_profile(profile, name=str(profile.get("name") or "").strip() or None)
    if not built:
        raise ValueError("Model profile requires a configured model reference.")

    for key in PROFILE_KEYS:
        if key in built:
            cfg[key] = built[key]
        elif key in cfg:
            cfg.pop(key, None)
    cfg["active_model_profile"] = built["name"]
    return built


def sync_active_model_profile(cfg: dict[str, Any], *, preferred_name: str | None = None) -> dict[str, Any] | None:
    built = build_model_profile(cfg, name=preferred_name)
    if not built:
        return None
    previous_name = preferred_name or str(cfg.get("active_model_profile") or "").strip() or None
    built = replace_model_profile(cfg, built, previous_name=previous_name)
    apply_model_profile(cfg, built)
    return built
