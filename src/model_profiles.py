from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .runtime_registry import DEFAULT_RUNTIME, active_model_ref, normalize_runtime, runtime_spec

PROFILE_KEYS: tuple[str, ...] = (
    "runtime",
    "model_source",
    "model",
    "llama_model",
    "ollama_model",
    "recommended_llm",
    "openai_compatible_model",
    "openai_compatible_base_url",
    "openai_compatible_api_key_env",
    "openai_compatible_headers",
    "openai_compatible_extra_body",
    "openai_compatible_verify_connection",
    "openrouter_model",
    "openrouter_base_url",
    "openrouter_api_key_env",
    "openrouter_headers",
    "openrouter_extra_body",
    "openrouter_site_url",
    "openrouter_app_name",
    "openrouter_verify_connection",
    "device",
    "context_window_tokens",
    "gpu_layers",
    "hardware_profile",
    "hardware_override",
    "setup_complete",
)


def profile_model_ref(profile: Mapping[str, object]) -> str:
    model_ref = active_model_ref(dict(profile))
    if model_ref:
        return model_ref
    if str(profile.get("model_source", "local")).strip().lower() == "ollama":
        return str(profile.get("ollama_model") or "").strip()
    return ""


def default_profile_name(source: Mapping[str, object]) -> str:
    runtime = normalize_runtime(str(source.get("runtime", DEFAULT_RUNTIME)))
    model_ref = profile_model_ref(source)
    if not model_ref:
        return runtime_spec(runtime).label
    name = Path(model_ref).name or model_ref
    if runtime == "llama_cpp" and name.lower().endswith(".gguf"):
        name = name[:-5]
    return name.strip() or runtime_spec(runtime).label


def build_model_profile(source: Mapping[str, object], *, name: str | None = None) -> dict[str, Any] | None:
    runtime = normalize_runtime(str(source.get("runtime", DEFAULT_RUNTIME)))
    model_ref = profile_model_ref(source)
    if not model_ref:
        return None

    profile_name = (name or str(source.get("active_model_profile") or "")).strip() or default_profile_name(source)
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
    profile["runtime"] = runtime
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
    cfg["runtime"] = str(built.get("runtime", DEFAULT_RUNTIME))
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
