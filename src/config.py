from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Curated, size-banded shortlist used by setup recommendations.
RECOMMENDED_LLM_BANDS: tuple[tuple[float, tuple[tuple[str, float, str], ...]], ...] = (
    (
        2.0,
        (
            ("qwen2.5:1.5b", 1.5, "Qwen2.5 1.5B"),
            ("deepseek-r1:1.5b", 1.5, "DeepSeek R1 1.5B"),
            ("gemma2:2b", 2.0, "Gemma 2 2B"),
        ),
    ),
    (
        4.0,
        (
            ("qwen2.5:3b", 3.0, "Qwen2.5 3B"),
            ("qwen2.5:3b-instruct", 3.0, "Qwen2.5 3B Instruct"),
            ("gemma2:2b", 2.0, "Gemma 2 2B"),
        ),
    ),
    (
        8.0,
        (
            ("qwen2.5:7b", 7.0, "Qwen2.5 7B"),
            ("mistral:7b", 7.0, "Mistral 7B"),
            ("deepseek-r1:7b", 7.0, "DeepSeek R1 7B"),
        ),
    ),
    (
        14.0,
        (
            ("qwen2.5:14b", 14.0, "Qwen2.5 14B"),
            ("deepseek-r1:14b", 14.0, "DeepSeek R1 14B"),
            ("gemma2:9b", 9.0, "Gemma 2 9B"),
        ),
    ),
    (
        32.0,
        (
            ("qwen2.5:32b", 32.0, "Qwen2.5 32B"),
            ("qwen2.5-coder:32b", 32.0, "Qwen2.5 Coder 32B"),
            ("gemma2:27b", 27.0, "Gemma 2 27B"),
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
