from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .model_profiles import get_model_profile, list_model_profiles, replace_model_profile
from .runtime_registry import LITELLM_RUNTIME

OPENROUTER_PROVIDER = "openrouter"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Curated OpenRouter models. LiteLLM model ids use the openrouter/ prefix.
OPENROUTER_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "name": "ox-alpha",
        "label": "Ox Alpha (free)",
        "model": "openrouter/stealth/ox-alpha",
        "context_window_tokens": 1048576,
        "max_tokens": 131072,
        "free": True,
        "featured": True,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    },
    {
        "name": "openrouter-free",
        "label": "Free router",
        "model": "openrouter/openrouter/free",
        "context_window_tokens": 200000,
        "max_tokens": 4096,
        "free": True,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    },
    {
        "name": "claude-opus-openrouter",
        "label": "Claude Opus 4.8",
        "model": "openrouter/anthropic/claude-opus-4.8",
        "context_window_tokens": 1000000,
        "max_tokens": 128000,
        "cost": {"input": 5, "output": 25, "cacheRead": 0.5, "cacheWrite": 6.25},
    },
    {
        "name": "gemini-openrouter",
        "label": "Gemini 3.1 Pro",
        "model": "openrouter/google/gemini-3.1-pro-preview",
        "context_window_tokens": 1048576,
        "max_tokens": 65536,
        "cost": {"input": 2, "output": 12, "cacheRead": 0.2, "cacheWrite": 0.375},
    },
    {
        "name": "grok-openrouter",
        "label": "Grok 4.20",
        "model": "openrouter/x-ai/grok-4.20",
        "context_window_tokens": 2000000,
        "max_tokens": 4096,
        "cost": {"input": 1.25, "output": 2.5, "cacheRead": 0.2, "cacheWrite": 0},
    },
    {
        "name": "deepseek-v4-openrouter",
        "label": "DeepSeek V4 Pro",
        "model": "openrouter/deepseek/deepseek-v4-pro",
        "context_window_tokens": 1048576,
        "max_tokens": 131072,
        "cost": {"input": 1.168, "output": 2.336, "cacheRead": 0.09855, "cacheWrite": 0},
    },
    {
        "name": "glm-openrouter",
        "label": "GLM 5.1",
        "model": "openrouter/z-ai/glm-5.1",
        "context_window_tokens": 202752,
        "max_tokens": 131072,
        "cost": {"input": 1.4, "output": 4.4, "cacheRead": 0.26, "cacheWrite": 0},
    },
    {
        "name": "kimi-openrouter",
        "label": "Kimi K2.5",
        "model": "openrouter/moonshotai/kimi-k2.5",
        "context_window_tokens": 262144,
        "max_tokens": 4096,
        "cost": {"input": 0.41, "output": 2.06, "cacheRead": 0.07, "cacheWrite": 0},
    },
)


def featured_openrouter_model() -> str:
    for entry in OPENROUTER_CATALOG:
        if entry.get("featured"):
            return str(entry["model"])
    return str(OPENROUTER_CATALOG[0]["model"])


def catalog_entry_for_model(model: str) -> dict[str, Any] | None:
    needle = str(model or "").strip()
    if not needle:
        return None
    bare = needle.removeprefix("openrouter/")
    for entry in OPENROUTER_CATALOG:
        full = str(entry.get("model") or "")
        if full == needle or full.removeprefix("openrouter/") == bare:
            return dict(entry)
    return None


def openrouter_picker_models() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in OPENROUTER_CATALOG:
        cost = dict(entry.get("cost") or {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0})
        full = str(entry["model"])
        rows.append(
            {
                "id": full.removeprefix("openrouter/"),
                "name": str(entry.get("label") or entry.get("name") or full),
                "contextWindow": int(entry.get("context_window_tokens") or 128000),
                "maxTokens": int(entry.get("max_tokens") or 8192),
                "reasoning": True,
                "featured": bool(entry.get("featured")),
                "cost": cost,
            }
        )
    return rows


def openrouter_model_choices() -> list[tuple[str, str]]:
    return [(f"{entry['label']} · {entry['model']}", str(entry["model"])) for entry in OPENROUTER_CATALOG]


def openrouter_model_option_ids() -> list[str]:
    return [str(entry["model"]) for entry in OPENROUTER_CATALOG]


def openrouter_profile_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": str(entry.get("name") or "").strip(),
        "runtime": LITELLM_RUNTIME,
        "provider": OPENROUTER_PROVIDER,
        "model": str(entry.get("model") or "").strip(),
        "api_key_env": OPENROUTER_API_KEY_ENV,
        "context_window_tokens": int(entry.get("context_window_tokens") or 128000),
    }


def ensure_openrouter_model_profiles(cfg: dict[str, Any]) -> bool:
    """Add curated OpenRouter profiles when missing. Returns True if any profile was added."""
    added = False
    for entry in OPENROUTER_CATALOG:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        if get_model_profile(cfg, name) is not None:
            continue
        replace_model_profile(cfg, openrouter_profile_payload(entry))
        added = True
    return added


def normalize_openrouter_model_id(model: str) -> str:
    value = str(model or "").strip()
    if not value:
        return featured_openrouter_model()
    if value.startswith("openrouter/"):
        return value
    return f"openrouter/{value}"


def openrouter_profile_name_for_model(model: str) -> str:
    litellm_id = normalize_openrouter_model_id(model)
    entry = catalog_entry_for_model(litellm_id)
    if entry is not None:
        return str(entry["name"])
    return litellm_id.removeprefix("openrouter/").replace("/", "-")


def upsert_openrouter_profile(
    cfg: dict[str, Any],
    model: str,
    *,
    context_window_tokens: int | None = None,
) -> dict[str, Any]:
    litellm_id = normalize_openrouter_model_id(model)
    name = openrouter_profile_name_for_model(litellm_id)
    existing = get_model_profile(cfg, name)
    payload = openrouter_profile_payload(
        catalog_entry_for_model(litellm_id)
        or {
            "name": name,
            "model": litellm_id,
            "context_window_tokens": context_window_tokens or 128000,
        }
    )
    payload["name"] = name
    payload["model"] = litellm_id
    if existing:
        payload = {**existing, **payload}
    if context_window_tokens:
        payload["context_window_tokens"] = int(context_window_tokens)
    return replace_model_profile(cfg, payload, previous_name=str(existing.get("name") or "") if existing else None)


def openrouter_catalog_ts_path() -> Path:
    return Path(__file__).resolve().parent.parent / "ui" / "src" / "openrouter-models.generated.ts"


def render_openrouter_catalog_ts() -> str:
    body = json.dumps(openrouter_picker_models(), indent=2)
    return (
        "// Generated from src/openrouter_catalog.py. Do not edit.\n"
        f"export const CURATED_OPENROUTER_MODELS = {body} as const;\n"
    )


def write_openrouter_catalog_ts(path: Path | None = None) -> Path:
    destination = path or openrouter_catalog_ts_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_openrouter_catalog_ts(), encoding="utf-8")
    return destination


def list_openrouter_profiles(cfg: Mapping[str, object]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for profile in list_model_profiles(cfg):
        if str(profile.get("provider") or "").strip().lower().replace("_", "-") != OPENROUTER_PROVIDER:
            continue
        profiles.append(dict(profile))
    return profiles
