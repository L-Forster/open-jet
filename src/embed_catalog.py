"""Model catalog for embedded/SDK use.

Deliberately separate from the coding-agent catalog in `config.py`. That catalog is
banded on *detected* RAM and curated for coding capability, where aggressive quant on a
large model is a good trade. Embedded models are chosen against a *declared* target
device and a budget the host application is willing to give up, where latency and a
quant floor matter more than raw capability.
"""

from __future__ import annotations

from typing import Any, Mapping

from .config import (
    QWEN38_27B_FILENAME,
    QWEN38_27B_URL,
    _qwen36_27b_mtp_url,
    _qwen36_mtp_local_filename,
)

MB_PER_GB = 1024.0

# Use cases an embedded model is provisioned for. `min_quant_rank` is the quant floor:
# damage from aggressive quant shows up much faster on a small model than a 27B, and
# structured output degrades before free-form chat does.
EMBED_USE_CASES: tuple[dict[str, object], ...] = (
    {
        "id": "chat",
        "label": "In-app chat / assistant",
        "detail": "Free-form conversation inside your application.",
        "min_quant_rank": 3,
        "max_ttft_ms": 1200,
    },
    {
        "id": "dialogue",
        "label": "Real-time dialogue (game NPC, voice)",
        "detail": "Latency-critical. Must answer in-scene.",
        "min_quant_rank": 3,
        "max_ttft_ms": 350,
    },
    {
        "id": "extract",
        "label": "Structured output / extraction",
        "detail": "JSON and schema-constrained decoding.",
        "min_quant_rank": 4,
        "max_ttft_ms": 1500,
    },
    {
        "id": "classify",
        "label": "Classification / routing",
        "detail": "Short inputs, small label space, high throughput.",
        "min_quant_rank": 4,
        "max_ttft_ms": 400,
    },
    {
        "id": "summarize",
        "label": "Summarization / RAG answers",
        "detail": "Long inputs. KV cache dominates the budget.",
        "min_quant_rank": 3,
        "max_ttft_ms": 2500,
    },
)

# Target devices are declared, not detected: the machine running `openjet project` is
# the developer's, not the one the application ships to. `budget_gb` is the default
# slice the model may take, not the device total — the host application owns the rest.
TARGET_DEVICE_PRESETS: tuple[dict[str, object], ...] = (
    {"id": "handheld", "label": "Handheld / Steam Deck class", "total_gb": 16.0, "budget_gb": 4.0, "gpu": True},
    {"id": "laptop_8", "label": "Laptop, 8GB, integrated GPU", "total_gb": 8.0, "budget_gb": 3.0, "gpu": False},
    {"id": "laptop_16", "label": "Laptop, 16GB, integrated GPU", "total_gb": 16.0, "budget_gb": 6.0, "gpu": False},
    {"id": "desktop_gpu_8", "label": "Desktop, 8GB VRAM discrete GPU", "total_gb": 8.0, "budget_gb": 6.0, "gpu": True},
    {"id": "desktop_gpu_16", "label": "Desktop, 16GB VRAM discrete GPU", "total_gb": 16.0, "budget_gb": 12.0, "gpu": True},
    {"id": "desktop_gpu_24", "label": "Desktop, 24GB VRAM discrete GPU", "total_gb": 24.0, "budget_gb": 20.0, "gpu": True},
    {"id": "server", "label": "Server / no co-tenant", "total_gb": 64.0, "budget_gb": 48.0, "gpu": True},
)

# Quant rank orders the quant floor check. Higher is less lossy.
QUANT_RANKS: dict[str, int] = {
    "UD-IQ2_XXS": 1,
    "UD-IQ3_XXS": 2,
    "Q3_K_XL": 2,
    "Q4_K_S": 3,
    "Q4_K_M": 4,
    "Q5_K_M": 5,
    "Q6_K": 6,
    "Q8_0": 7,
}

# Only URLs already exercised by this repo's direct catalog appear here. Rows are
# ordered weakest to strongest; selection walks from the end so the strongest model
# that fits the budget wins.
#
# `ttft_ms` and `tokens_per_sec` are planning estimates scaled off active parameter
# count on the target's class of accelerator, not measured numbers. Run
# `open-jet benchmark` on real target hardware before committing to a latency budget.
EMBED_MODEL_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "qwen35-4b-q4km",
        "label": "Qwen3.5 4B",
        "quant": "Q4_K_M",
        "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true",
        "model_size_mb": 2806,
        "active_params_b": 4.0,
        "kv_bytes_per_token": 17408,
        "ttft_ms": 180,
        "tokens_per_sec": 55,
        "use_cases": ("chat", "dialogue", "extract", "classify", "summarize"),
    },
    {
        "id": "qwen35-9b-q4km",
        "label": "Qwen3.5 9B",
        "quant": "Q4_K_M",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true",
        "model_size_mb": 5816,
        "active_params_b": 9.0,
        "kv_bytes_per_token": 17408,
        "ttft_ms": 400,
        "tokens_per_sec": 32,
        "use_cases": ("chat", "extract", "summarize"),
    },
    {
        "id": "qwen36-27b-iq3",
        "label": "Qwen3.6 27B UD-IQ3_XXS MTP",
        "quant": "UD-IQ3_XXS",
        "filename": _qwen36_mtp_local_filename("Qwen3.6-27B-UD-IQ3_XXS.gguf"),
        "url": _qwen36_27b_mtp_url("Qwen3.6-27B-UD-IQ3_XXS.gguf"),
        "model_size_mb": 12288,
        "active_params_b": 27.0,
        "kv_bytes_per_token": 34816,
        "ttft_ms": 900,
        "tokens_per_sec": 18,
        "llama_mtp": True,
        "use_cases": ("chat", "summarize"),
    },
    {
        "id": "qwen38-27b-q4km",
        "label": "Qwen3.8 27B Q4_K_M MTP",
        "quant": "Q4_K_M",
        "filename": QWEN38_27B_FILENAME,
        "url": QWEN38_27B_URL,
        "model_size_mb": 16817,
        "resident_model_size_mb": 16896,
        "active_params_b": 27.0,
        "kv_bytes_per_token": 34816,
        "ttft_ms": 1100,
        "tokens_per_sec": 15,
        "llama_mtp": True,
        "use_cases": ("chat", "summarize", "extract"),
    },
)

DEFAULT_EMBED_CONTEXT_TOKENS = 4096


def use_case(use_case_id: str) -> dict[str, object]:
    for row in EMBED_USE_CASES:
        if row["id"] == use_case_id:
            return dict(row)
    raise ValueError(
        f"Unknown use case {use_case_id!r}. Expected one of: "
        + ", ".join(str(row["id"]) for row in EMBED_USE_CASES)
    )


def target_preset(target_id: str) -> dict[str, object]:
    for row in TARGET_DEVICE_PRESETS:
        if row["id"] == target_id:
            return dict(row)
    raise ValueError(
        f"Unknown target device {target_id!r}. Expected one of: "
        + ", ".join(str(row["id"]) for row in TARGET_DEVICE_PRESETS)
    )


def resident_mb(row: Mapping[str, Any], context_tokens: int) -> float:
    """Weights plus KV cache for the requested context, in MB."""
    weights = float(row.get("resident_model_size_mb") or row.get("model_size_mb") or 0.0)
    kv = float(row.get("kv_bytes_per_token") or 0.0) * context_tokens / (1024.0 * 1024.0)
    return weights + kv


def fits_budget(row: Mapping[str, Any], budget_gb: float, context_tokens: int) -> bool:
    return resident_mb(row, context_tokens) <= budget_gb * MB_PER_GB


def meets_use_case(row: Mapping[str, Any], case: Mapping[str, Any]) -> bool:
    if case["id"] not in row.get("use_cases", ()):
        return False
    if QUANT_RANKS.get(str(row.get("quant")), 0) < int(case["min_quant_rank"]):
        return False
    return int(row.get("ttft_ms") or 0) <= int(case["max_ttft_ms"])


def recommend_embed_model(
    *,
    use_case_id: str,
    budget_gb: float,
    context_tokens: int = DEFAULT_EMBED_CONTEXT_TOKENS,
) -> dict[str, Any] | None:
    """Strongest catalog row that fits the budget and satisfies the use case.

    Returns None when nothing fits, which is a real outcome: a 350ms dialogue budget on
    3GB genuinely excludes every row here, and reporting that beats shipping a model
    that misses the latency target on the target device.
    """
    case = use_case(use_case_id)
    for row in reversed(EMBED_MODEL_CATALOG):
        if meets_use_case(row, case) and fits_budget(row, budget_gb, context_tokens):
            return dict(row)
    return None


def rejection_reasons(
    *,
    use_case_id: str,
    budget_gb: float,
    context_tokens: int = DEFAULT_EMBED_CONTEXT_TOKENS,
) -> list[str]:
    """Per-row explanation of why nothing was selected."""
    case = use_case(use_case_id)
    reasons: list[str] = []
    for row in EMBED_MODEL_CATALOG:
        if case["id"] not in row.get("use_cases", ()):
            reasons.append(f"{row['label']}: not suited to {case['label'].lower()}")
        elif QUANT_RANKS.get(str(row.get("quant")), 0) < int(case["min_quant_rank"]):
            reasons.append(f"{row['label']}: {row['quant']} is below the quant floor for this use case")
        elif int(row.get("ttft_ms") or 0) > int(case["max_ttft_ms"]):
            reasons.append(
                f"{row['label']}: ~{row['ttft_ms']}ms first token exceeds the {case['max_ttft_ms']}ms budget"
            )
        else:
            needed = resident_mb(row, context_tokens) / MB_PER_GB
            reasons.append(f"{row['label']}: needs {needed:.1f}GB, budget is {budget_gb:.1f}GB")
    return reasons


def describe_row(row: Mapping[str, Any], context_tokens: int) -> str:
    footprint = resident_mb(row, context_tokens) / MB_PER_GB
    return (
        f"{row['label']} ({row['quant']}) — {footprint:.1f}GB resident at {context_tokens} ctx, "
        f"~{row['ttft_ms']}ms first token, ~{row['tokens_per_sec']} tok/s"
    )
