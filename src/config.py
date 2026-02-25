from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

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

JETSON_OVERRIDE_OPTIONS: tuple[tuple[str, str, float], ...] = (
    ("jetson_nano_4", "Jetson Nano (4GB RAM)", 4.0),
    ("jetson_xavier_nx_8", "Jetson Xavier NX (8GB RAM)", 8.0),
    ("jetson_orin_nano_8", "Jetson Orin Nano (8GB RAM)", 8.0),
    ("jetson_orin_nx_16", "Jetson Orin NX (16GB RAM)", 16.0),
    ("jetson_agx_orin_32", "Jetson AGX Orin (32GB RAM)", 32.0),
    ("jetson_agx_orin_64", "Jetson AGX Orin (64GB RAM)", 64.0),
)


def load_config() -> dict:
    for candidate in [Path("config.yaml"), CONFIG_PATH]:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text()) or {}
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False))
