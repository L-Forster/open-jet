from __future__ import annotations

from typing import Any, Callable

from .airgap import airgapped_from_cfg
from .llama_server import LlamaServerClient
from .runtime_client import RuntimeClient


DEFAULT_RUNTIME = "llama_cpp"
RUNTIME_LABEL = "Local model: llama.cpp (GGUF)"


def runtime_options() -> list[tuple[str, str]]:
    return [(RUNTIME_LABEL, DEFAULT_RUNTIME)]


def active_model_ref(cfg: dict) -> str:
    return str(cfg.get("llama_model") or "").strip()


def create_runtime_client(
    cfg: dict,
    *,
    diagnostics_hook: Callable[[str, dict[str, Any]], None] | None = None,
) -> RuntimeClient:
    model = active_model_ref(cfg)
    if not model:
        raise ValueError("Missing model for llama.cpp runtime (`llama_model`).")
    return LlamaServerClient(
        model=model,
        context_window_tokens=int(cfg.get("context_window_tokens", 2048)),
        device=str(cfg.get("device", "auto")),
        gpu_layers=int(cfg.get("gpu_layers", 99)),
        cpu_moe=bool(cfg.get("llama_cpu_moe", False)),
        n_cpu_moe=int(cfg.get("llama_n_cpu_moe", 0)),
        airgapped=airgapped_from_cfg(cfg),
        diagnostics_hook=diagnostics_hook,
    )
