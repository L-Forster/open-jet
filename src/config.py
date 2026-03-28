from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEFAULT_SESSION_STATE_PATH = ".openjet/state/session_state.json"
DEFAULT_LOG_DIRECTORY = ".openjet/state/sessions"
LEGACY_SESSION_STATE_PATH = "session_state.json"
LEGACY_LOG_DIRECTORY = "session_logs"

# Curated, size-banded shortlist used by setup recommendations.
# Band limit = max param budget in billions for that RAM tier.
# Disk/RAM sizes (Q4_K_M): 0.6B~523MB, 1.7B~1.4GB, 4B~2.6GB, 8B~5.2GB,
# 14B~9.3GB, 27B~17GB, 32B~20GB, 35B-A3B~24GB, 122B-A10B~81GB.
RECOMMENDED_LLM_BANDS: tuple[tuple[float, tuple[tuple[str, float, str], ...]], ...] = (
    (
        2.0,
        (
            ("qwen3:1.7b", 1.7, "Qwen3 1.7B"),
            ("qwen3:0.6b", 0.6, "Qwen3 0.6B"),
            ("deepseek-r1:1.5b", 1.5, "DeepSeek R1 1.5B"),
        ),
    ),
    (
        4.0,
        (
            ("qwen3:4b", 4.0, "Qwen3 4B"),
            ("qwen3:1.7b", 1.7, "Qwen3 1.7B"),
            ("gemma2:2b", 2.0, "Gemma 2 2B"),
        ),
    ),
    (
        8.0,
        (
            ("qwen3:8b", 8.0, "Qwen3 8B"),
            ("qwen3:4b", 4.0, "Qwen3 4B"),
            ("deepseek-r1:7b", 7.0, "DeepSeek R1 7B"),
        ),
    ),
    (
        14.0,
        (
            ("qwen3:14b", 14.0, "Qwen3 14B"),
            ("qwen3:8b", 8.0, "Qwen3 8B"),
            ("deepseek-r1:14b", 14.0, "DeepSeek R1 14B"),
        ),
    ),
    (
        32.0,
        (
            ("qwen3.5:27b", 27.0, "Qwen3.5 27B"),
            ("qwen3.5:35b-a3b", 35.0, "Qwen3.5 35B MoE"),
            ("qwen3:32b", 32.0, "Qwen3 32B"),
        ),
    ),
)

DEFAULT_DIRECT_MODEL_CATALOG: tuple[dict[str, object], ...] = (
    {
        "max_ram_gb": 6.0,
        "label": "Qwen3.5 4B",
        "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true",
    },
    {
        "max_ram_gb": 12.0,
        "label": "Qwen3.5 9B",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true",
    },
    {
        "max_ram_gb": 24.0,
        "label": "Qwen3.5 27B",
        "filename": "Qwen3.5-27B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf?download=true",
    },
)

HARDWARE_OVERRIDE_OPTIONS: tuple[tuple[str, str, float, bool], ...] = (
    ("desktop_8", "Desktop / Laptop (8GB RAM)", 8.0, False),
    ("desktop_16", "Desktop / Laptop (16GB RAM)", 16.0, False),
    ("desktop_32", "Desktop / Laptop (32GB RAM)", 32.0, False),
    ("desktop_64", "Desktop / Laptop (64GB RAM)", 64.0, False),
    ("jetson_nano_4", "Jetson Nano (4GB RAM)", 4.0, True),
    ("jetson_xavier_nx_8", "Jetson Xavier NX (8GB RAM)", 8.0, True),
    ("jetson_orin_nano_8", "Jetson Orin Nano (8GB RAM)", 8.0, True),
    ("jetson_orin_nx_16", "Jetson Orin NX (16GB RAM)", 16.0, True),
    ("jetson_agx_orin_32", "Jetson AGX Orin (32GB RAM)", 32.0, True),
    ("jetson_agx_orin_64", "Jetson AGX Orin (64GB RAM)", 64.0, True),
)


def load_config() -> dict:
    for candidate in [Path("config.yaml"), CONFIG_PATH]:
        if candidate.exists():
            raw = yaml.safe_load(candidate.read_text()) or {}
            return normalize_config(raw)
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False))


def normalize_config(cfg: dict) -> dict:
    normalized = dict(cfg or {})
    legacy_exact_keys = {"runtime", "model", "ollama_model", "recommended_llm"}
    legacy_prefixes = ("openai_compatible_", "openrouter_")
    for key in list(normalized):
        if key in legacy_exact_keys or any(key.startswith(prefix) for prefix in legacy_prefixes):
            normalized.pop(key, None)

    state_cfg = dict(normalized.get("state") or {})
    state_path = str(state_cfg.get("path", "")).strip()
    if not state_path or state_path == LEGACY_SESSION_STATE_PATH:
        state_cfg["path"] = DEFAULT_SESSION_STATE_PATH
    if state_cfg:
        normalized["state"] = state_cfg

    log_cfg = dict(normalized.get("logging") or {})
    log_dir = str(log_cfg.get("directory", "")).strip()
    if not log_dir or log_dir == LEGACY_LOG_DIRECTORY:
        log_cfg["directory"] = DEFAULT_LOG_DIRECTORY
    if log_cfg:
        normalized["logging"] = log_cfg

    return normalized


def setup_direct_model_catalog(cfg: Mapping[str, object] | None = None) -> tuple[dict[str, object], ...]:
    if not isinstance(cfg, Mapping):
        return DEFAULT_DIRECT_MODEL_CATALOG

    setup_cfg = cfg.get("setup_recommendations")
    if not isinstance(setup_cfg, Mapping):
        return DEFAULT_DIRECT_MODEL_CATALOG

    raw_models = setup_cfg.get("direct_models")
    if not isinstance(raw_models, list):
        return DEFAULT_DIRECT_MODEL_CATALOG

    parsed: list[dict[str, Any]] = []
    for item in raw_models:
        if not isinstance(item, Mapping):
            continue
        label = str(item.get("label") or "").strip()
        filename = str(item.get("filename") or "").strip()
        url = str(item.get("url") or "").strip()
        try:
            max_ram_gb = float(item.get("max_ram_gb"))
        except (TypeError, ValueError):
            continue
        if max_ram_gb <= 0 or not label or not filename or not url:
            continue
        parsed.append(
            {
                "max_ram_gb": max_ram_gb,
                "label": label,
                "filename": filename,
                "url": url,
            }
        )

    if not parsed:
        return DEFAULT_DIRECT_MODEL_CATALOG
    parsed.sort(key=lambda row: float(row["max_ram_gb"]))
    return tuple(parsed)
