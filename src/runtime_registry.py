from __future__ import annotations

from typing import Any, Callable

from .airgap import AirgapViolationError, airgapped_from_cfg, endpoint_is_loopback
from .llama_server import LlamaServerClient
from .runtime_client import RuntimeClient


DEFAULT_RUNTIME = "llama_cpp"
CODEX_RUNTIME = "openai_codex"
LITELLM_RUNTIME = "litellm"
RUNTIME_LABEL = "Local model: llama.cpp (GGUF)"
CODEX_RUNTIME_LABEL = "OpenAI Codex OAuth"
LITELLM_RUNTIME_LABEL = "API model: LiteLLM"


def runtime_options() -> list[tuple[str, str]]:
    return [
        (RUNTIME_LABEL, DEFAULT_RUNTIME),
        (CODEX_RUNTIME_LABEL, CODEX_RUNTIME),
        (LITELLM_RUNTIME_LABEL, LITELLM_RUNTIME),
    ]


def active_runtime(cfg: dict) -> str:
    runtime = str(cfg.get("runtime") or DEFAULT_RUNTIME).strip().lower()
    return runtime or DEFAULT_RUNTIME


def active_model_ref(cfg: dict) -> str:
    if active_runtime(cfg) in {CODEX_RUNTIME, LITELLM_RUNTIME}:
        return str(cfg.get("model") or "").strip()
    return str(cfg.get("llama_model") or "").strip()


def create_runtime_client(
    cfg: dict,
    *,
    diagnostics_hook: Callable[[str, dict[str, Any]], None] | None = None,
) -> RuntimeClient:
    runtime = active_runtime(cfg)
    if runtime == CODEX_RUNTIME:
        if airgapped_from_cfg(cfg):
            raise ValueError("Cloud model profiles are disabled in air-gapped mode.")
        from .openai_codex_client import OpenAICodexClient

        model = active_model_ref(cfg)
        if not model:
            raise ValueError("Missing model for OpenAI Codex runtime (`model`).")
        return OpenAICodexClient(
            model=model,
            base_url=str(cfg.get("codex_base_url") or cfg.get("base_url") or "https://chatgpt.com/backend-api/codex"),
            context_window_tokens=int(cfg.get("context_window_tokens", 272000)),
            reasoning_effort=str(cfg.get("reasoning_effort") or "medium"),
            reasoning_summary=str(cfg.get("reasoning_summary") or "auto"),
            text_verbosity=str(cfg.get("text_verbosity") or "medium"),
        )
    if runtime == LITELLM_RUNTIME:
        if airgapped_from_cfg(cfg):
            base_url = str(cfg.get("base_url") or "").strip()
            if not endpoint_is_loopback(base_url):
                detail = base_url or "<provider default endpoint>"
                raise AirgapViolationError(f"Air-gapped mode blocks LiteLLM provider: {detail}")
        from .litellm_client import LiteLLMClient

        model = active_model_ref(cfg)
        if not model:
            raise ValueError("Missing model for LiteLLM runtime (`model`).")
        return LiteLLMClient(
            model=model,
            provider=str(cfg.get("provider") or ""),
            base_url=str(cfg.get("base_url") or ""),
            api_key_env=str(cfg.get("api_key_env") or ""),
            context_window_tokens=int(cfg.get("context_window_tokens", 128000)),
        )
    if runtime != DEFAULT_RUNTIME:
        raise ValueError(f"Unknown runtime `{runtime}`.")
    model = active_model_ref(cfg)
    if not model:
        raise ValueError("Missing model for llama.cpp runtime (`llama_model`).")
    return LlamaServerClient(
        model=model,
        binary=str(cfg.get("llama_server_path") or "") or None,
        context_window_tokens=int(cfg.get("context_window_tokens", 2048)),
        device=str(cfg.get("device", "auto")),
        gpu_layers=int(cfg.get("gpu_layers", 99)),
        llama_cpu_moe=bool(cfg.get("llama_cpu_moe", False)),
        llama_n_cpu_moe=int(cfg.get("llama_n_cpu_moe", 0) or 0),
        llama_mtp=bool(cfg.get("llama_mtp", False)),
        airgapped=airgapped_from_cfg(cfg),
        diagnostics_hook=diagnostics_hook,
    )
