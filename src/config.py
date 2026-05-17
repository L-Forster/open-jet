from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEFAULT_SESSION_STATE_PATH = ".openjet/state/session_state.yaml"
DEFAULT_LOG_DIRECTORY = ".openjet/state/sessions"
ROOT_LOG_DIRECTORY = "session_logs"
QWEN36_27B_MTP_REPO = "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF"
QWEN36_35B_A3B_MTP_REPO = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
QWEN36_27B_MTP_SOURCE_FILENAME = "Qwen3.6-27B-Q4_K_M.gguf"
QWEN36_27B_MTP_FILENAME = "Qwen3.6-27B-Q4_K_M-MTP.gguf"
QWEN36_27B_LEGACY_MTP_FILENAME = "Qwen3.6-27B-Q4_K_M-mtp.gguf"
QWEN36_27B_MTP_URL = (
    f"{QWEN36_27B_MTP_REPO}/resolve/main/"
    f"{QWEN36_27B_MTP_SOURCE_FILENAME}?download=true"
)
QWEN36_27B_MTP_LLAMA_CPP_REF = "b9189"
QWEN36_27B_OLD_MTP_REPO = "https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF/"
QWEN36_27B_NON_MTP_REPO = "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/"
QWEN36_27B_MTP_UPDATE_ID = "qwen36-27b-mtp-unsloth-b9189"
QWEN36_27B_PREFIX = "qwen3.6-27b-"
QWEN36_35B_A3B_NON_MTP_REPO = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/"
QWEN36_35B_A3B_MTP_UPDATE_ID = "qwen36-35b-a3b-mtp-unsloth-b9189"


def _qwen36_27b_mtp_url(filename: str) -> str:
    return f"{QWEN36_27B_MTP_REPO}/resolve/main/{filename}?download=true"


def _qwen36_35b_a3b_mtp_url(filename: str) -> str:
    return f"{QWEN36_35B_A3B_MTP_REPO}/resolve/main/{filename}?download=true"


def _qwen36_mtp_local_filename(filename: str) -> str:
    path = Path(filename)
    suffix = path.suffix or ".gguf"
    stem = path.stem
    if stem.lower().endswith("-mtp"):
        stem = stem[:-4]
    return f"{stem}-MTP{suffix}"


def _qwen36_mtp_source_filename(filename: str) -> str:
    path = Path(filename)
    suffix = path.suffix or ".gguf"
    stem = path.stem
    if stem.lower().endswith("-mtp"):
        return f"{stem[:-4]}{suffix}"
    return path.name

# Curated, size-banded shortlist used by setup recommendations.
# Band limit = max param budget in billions for that RAM tier.
# Disk/RAM sizes (Q4_K_M unless noted): 0.6B~523MB, 1.7B~1.4GB, 4B~2.6GB,
# 8B~5.2GB, 14B~9.3GB, 27B UD-IQ2_XXS~9.4GB, 27B UD-IQ3_XXS~12.0GB,
# 27B Q4_K_M~16.5GB, 32B~20GB, 35B-A3B UD-Q3_K_XL~16.8GB,
# 35B-A3B UD-Q4_K_M~22.1GB, 122B-A10B~81GB.
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
            ("qwen3.6:27b", 27.0, "Qwen3.6 27B"),
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
        "model_size_mb": 2806,
        "kv_bytes_per_token": 17408,
    },
    {
        "max_ram_gb": 12.0,
        "label": "Qwen3.5 9B",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true",
        "model_size_mb": 5816,
        "kv_bytes_per_token": 17408,
    },
    {
        "max_ram_gb": 12.0,
        "label": "Qwen3.6 27B UD-IQ2_XXS MTP",
        "filename": _qwen36_mtp_local_filename("Qwen3.6-27B-UD-IQ2_XXS.gguf"),
        "url": _qwen36_27b_mtp_url("Qwen3.6-27B-UD-IQ2_XXS.gguf"),
        "llama_cpp_ref": QWEN36_27B_MTP_LLAMA_CPP_REF,
        "llama_mtp": True,
        "model_size_mb": 9626,
        "kv_bytes_per_token": 34816,
    },
    {
        "max_ram_gb": 16.0,
        "label": "Qwen3.6 27B UD-IQ3_XXS MTP",
        "filename": _qwen36_mtp_local_filename("Qwen3.6-27B-UD-IQ3_XXS.gguf"),
        "url": _qwen36_27b_mtp_url("Qwen3.6-27B-UD-IQ3_XXS.gguf"),
        "llama_cpp_ref": QWEN36_27B_MTP_LLAMA_CPP_REF,
        "llama_mtp": True,
        "model_size_mb": 12288,
        "kv_bytes_per_token": 34816,
    },
    {
        "max_ram_gb": 20.0,
        "label": "Qwen3.6 27B Q4_K_M MTP",
        "filename": QWEN36_27B_MTP_FILENAME,
        "url": QWEN36_27B_MTP_URL,
        "llama_cpp_ref": QWEN36_27B_MTP_LLAMA_CPP_REF,
        "llama_mtp": True,
        "model_size_mb": 16817,
        "resident_model_size_mb": 16896,
        "kv_bytes_per_token": 34816,
    },
    {
        "max_ram_gb": 24.0,
        "label": "Gemma 4 26B A4B",
        "filename": "gemma-4-26B-A4B.i1-Q4_K_S.gguf",
        "url": "https://huggingface.co/mradermacher/gemma-4-26B-A4B-i1-GGUF/resolve/main/gemma-4-26B-A4B.i1-Q4_K_S.gguf?download=true",
        "model_size_mb": 15974,
        "active_model_size_mb": 4096,
        "kv_bytes_per_token": 16384,
        "unified_memory_only": True,
    },
    {
        "max_ram_gb": 24.0,
        "label": "Qwen3.6 35B A3B UD-IQ2_XXS MTP",
        "filename": _qwen36_mtp_local_filename("Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"),
        "url": _qwen36_35b_a3b_mtp_url("Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf"),
        "llama_cpp_ref": QWEN36_27B_MTP_LLAMA_CPP_REF,
        "llama_mtp": True,
        "model_size_mb": 12288,
        "active_model_size_mb": 3072,
        "kv_bytes_per_token": 24576,
        "unified_memory_only": True,
        "llama_cpu_moe": True,
        "llama_n_cpu_moe": 0,
    },
    {
        "max_ram_gb": 32.0,
        "label": "Qwen3.6 35B A3B UD-Q3_K_XL MTP",
        "filename": _qwen36_mtp_local_filename("Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"),
        "url": _qwen36_35b_a3b_mtp_url("Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"),
        "llama_cpp_ref": QWEN36_27B_MTP_LLAMA_CPP_REF,
        "llama_mtp": True,
        "model_size_mb": 17203,
        "active_model_size_mb": 3072,
        "kv_bytes_per_token": 24576,
        "unified_memory_only": True,
        "llama_cpu_moe": True,
        "llama_n_cpu_moe": 0,
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
    removed_exact_keys = {"runtime", "model", "ollama_model", "recommended_llm"}
    removed_prefixes = ("openai_compatible_", "openrouter_")
    for key in list(normalized):
        if key in removed_exact_keys or any(key.startswith(prefix) for prefix in removed_prefixes):
            normalized.pop(key, None)

    state_cfg = dict(normalized.get("state") or {})
    state_path = str(state_cfg.get("path", "")).strip()
    if not state_path:
        state_cfg["path"] = DEFAULT_SESSION_STATE_PATH
    elif Path(state_path).suffix.lower() == ".json":
        state_cfg["path"] = str(Path(state_path).with_suffix(".yaml")).replace("\\", "/")
    if state_cfg:
        normalized["state"] = state_cfg

    log_cfg = dict(normalized.get("logging") or {})
    log_dir = str(log_cfg.get("directory", "")).strip()
    if not log_dir or log_dir == ROOT_LOG_DIRECTORY:
        log_cfg["directory"] = DEFAULT_LOG_DIRECTORY
    if log_cfg:
        normalized["logging"] = log_cfg

    migrate_config_for_current_release(normalized)

    return normalized


def migrate_config_for_current_release(cfg: dict[str, Any]) -> bool:
    changed = False

    def resolve_model_path(value: object) -> str | None:
        path = Path(str(value or "").strip())
        if path.name != QWEN36_27B_LEGACY_MTP_FILENAME:
            return None
        return str(path.with_name(QWEN36_27B_MTP_FILENAME))

    def looks_like_qwen_mtp_update_target(row: dict[str, Any]) -> bool:
        values = [
            row.get("llama_model"),
            row.get("model_download_path"),
            row.get("model_download_url"),
            row.get("filename"),
            row.get("model_profile_name"),
            row.get("name"),
        ]
        text = "\n".join(str(value or "") for value in values)
        lowered = text.lower()
        return (
            QWEN36_27B_OLD_MTP_REPO.lower() in lowered
            or QWEN36_27B_NON_MTP_REPO.lower() in lowered
            or (
                QWEN36_27B_PREFIX in lowered
                and bool(row.get("llama_mtp"))
                and str(row.get("model_update_applied") or "") != QWEN36_27B_MTP_UPDATE_ID
            )
            or QWEN36_35B_A3B_NON_MTP_REPO.lower() in lowered
            or (
                "qwen3.6-35b-a3b-" in lowered
                and bool(row.get("llama_mtp"))
                and str(row.get("model_update_applied") or "") != QWEN36_35B_A3B_MTP_UPDATE_ID
            )
        )

    def qwen_mtp_update_details(row: dict[str, Any], replacement: str | None) -> tuple[str, str]:
        values = [
            row.get("llama_model"),
            row.get("model_download_path"),
            row.get("model_download_url"),
            row.get("filename"),
            row.get("model_profile_name"),
            row.get("name"),
            replacement,
        ]
        lowered = "\n".join(str(value or "") for value in values).lower()
        if "qwen3.6-35b-a3b-" in lowered or QWEN36_35B_A3B_NON_MTP_REPO.lower() in lowered:
            filename = Path(str(replacement or row.get("model_download_path") or row.get("llama_model") or row.get("filename") or "")).name
            if not filename:
                filename = "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
            filename = _qwen36_mtp_source_filename(filename)
            return _qwen36_35b_a3b_mtp_url(filename), QWEN36_35B_A3B_MTP_UPDATE_ID
        filename = Path(str(replacement or row.get("model_download_path") or row.get("llama_model") or row.get("filename") or "")).name
        if not filename:
            filename = QWEN36_27B_MTP_SOURCE_FILENAME
        filename = _qwen36_mtp_source_filename(filename)
        return _qwen36_27b_mtp_url(filename), QWEN36_27B_MTP_UPDATE_ID

    def apply_qwen_mtp_update(row: dict[str, Any], *, replacement: str | None) -> None:
        nonlocal changed

        def set_value(key: str, value: object) -> None:
            nonlocal changed
            if row.get(key) != value:
                row[key] = value
                changed = True

        model_download_url, model_update_target = qwen_mtp_update_details(row, replacement)
        set_value("model_source", "direct")
        set_value("model_download_url", model_download_url)
        set_value("llama_mtp", True)
        set_value("llama_cpp_ref", QWEN36_27B_MTP_LLAMA_CPP_REF)
        set_value("setup_missing_model", True)
        set_value("setup_update_model", True)
        set_value("model_update_target", model_update_target)
        if replacement:
            set_value("llama_model", replacement)
            set_value("model_download_path", replacement)

    def normalize_row(row: dict[str, Any]) -> None:
        nonlocal changed
        replacement = resolve_model_path(row.get("llama_model"))
        replacement = replacement or resolve_model_path(row.get("model_download_path"))
        if not replacement and looks_like_qwen_mtp_update_target(row):
            raw_path = str(row.get("model_download_path") or row.get("llama_model") or "").strip()
            if raw_path:
                raw_name = Path(raw_path).name.lower()
                if "qwen3.6-35b-a3b-" in raw_name or QWEN36_27B_PREFIX in raw_name:
                    replacement = str(Path(raw_path).with_name(_qwen36_mtp_local_filename(Path(raw_path).name)))
                else:
                    replacement = str(Path(raw_path).with_name(QWEN36_27B_MTP_FILENAME))
        if replacement or looks_like_qwen_mtp_update_target(row):
            apply_qwen_mtp_update(row, replacement=replacement)

    normalize_row(cfg)
    raw_profiles = cfg.get("model_profiles")
    if isinstance(raw_profiles, list):
        for item in raw_profiles:
            if isinstance(item, dict):
                normalize_row(item)
    return changed


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
        row: dict[str, Any] = {
            "max_ram_gb": max_ram_gb,
            "label": label,
            "filename": filename,
            "url": url,
        }
        for key in ("model_size_mb", "kv_bytes_per_token", "resident_model_size_mb", "active_model_size_mb"):
            try:
                value = float(item.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                row[key] = value
        if "unified_memory_only" in item:
            row["unified_memory_only"] = bool(item.get("unified_memory_only"))
        if "llama_cpu_moe" in item:
            row["llama_cpu_moe"] = bool(item.get("llama_cpu_moe"))
        if "llama_cpp_ref" in item:
            llama_cpp_ref = str(item.get("llama_cpp_ref") or "").strip()
            if llama_cpp_ref:
                row["llama_cpp_ref"] = llama_cpp_ref
        if "llama_mtp" in item:
            row["llama_mtp"] = bool(item.get("llama_mtp"))
        try:
            llama_n_cpu_moe = int(item.get("llama_n_cpu_moe"))
        except (TypeError, ValueError):
            pass
        else:
            if llama_n_cpu_moe >= 0:
                row["llama_n_cpu_moe"] = llama_n_cpu_moe
        parsed.append(row)

    if not parsed:
        return DEFAULT_DIRECT_MODEL_CATALOG
    parsed.sort(key=lambda row: float(row["max_ram_gb"]))
    return tuple(parsed)
