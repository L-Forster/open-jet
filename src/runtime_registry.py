from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

from .airgap import AirgapViolationError, airgapped_from_cfg, assert_endpoint_allowed
from .llama_server import LlamaServerClient
from .openai_compatible import OpenAICompatibleClient, _normalize_base_url
from .runtime_client import RuntimeClient
from .sglang_server import SglangServerClient
from .trtllm_server import TrtllmServerClient


@dataclass(frozen=True)
class RuntimeSpec:
    key: str
    label: str
    model_config_key: str
    uses_gguf: bool = False
    uses_gpu_layers: bool = False
    supports_ollama: bool = False
    show_in_setup: bool = True


RUNTIME_SPECS: dict[str, RuntimeSpec] = {
    "llama_cpp": RuntimeSpec(
        key="llama_cpp",
        label="llama.cpp (GGUF)",
        model_config_key="llama_model",
        uses_gguf=True,
        uses_gpu_layers=True,
        supports_ollama=True,
    ),
    "sglang": RuntimeSpec(
        key="sglang",
        label="SGLang",
        model_config_key="sglang_model",
    ),
    "trtllm_pytorch": RuntimeSpec(
        key="trtllm_pytorch",
        label="TensorRT-LLM (PyTorch runtime)",
        model_config_key="trtllm_model",
    ),
    "openai_compatible": RuntimeSpec(
        key="openai_compatible",
        label="OpenAI-compatible API",
        model_config_key="openai_compatible_model",
        show_in_setup=False,
    ),
    "openrouter": RuntimeSpec(
        key="openrouter",
        label="OpenRouter",
        model_config_key="openrouter_model",
        show_in_setup=False,
    ),
}

DEFAULT_RUNTIME = "llama_cpp"
RUNTIME_ALIASES = {
    "openai": "openai_compatible",
    "unified_api": "openrouter",
}


def normalize_runtime(value: str | None) -> str:
    runtime = (value or DEFAULT_RUNTIME).strip().lower()
    runtime = RUNTIME_ALIASES.get(runtime, runtime)
    return runtime if runtime in RUNTIME_SPECS else DEFAULT_RUNTIME


def runtime_options() -> list[tuple[str, str]]:
    return [(spec.label, spec.key) for spec in RUNTIME_SPECS.values() if spec.show_in_setup]


def runtime_spec(value: str | None) -> RuntimeSpec:
    return RUNTIME_SPECS[normalize_runtime(value)]


def active_model_ref(cfg: dict) -> str:
    spec = runtime_spec(str(cfg.get("runtime", DEFAULT_RUNTIME)))
    return str(cfg.get(spec.model_config_key) or cfg.get("model") or "").strip()


def create_runtime_client(
    cfg: dict,
    *,
    diagnostics_hook: Callable[[str, dict[str, Any]], None] | None = None,
) -> RuntimeClient:
    runtime = normalize_runtime(str(cfg.get("runtime", DEFAULT_RUNTIME)))
    context_window_tokens = int(cfg.get("context_window_tokens", 2048))
    airgapped = airgapped_from_cfg(cfg)
    if runtime == "openrouter":
        if airgapped:
            raise AirgapViolationError("Air-gapped mode does not allow the OpenRouter runtime.")
        model = str(cfg.get("openrouter_model") or cfg.get("model") or "").strip()
        extra_headers = _openrouter_headers(cfg)
        extra_body = _mapping(cfg.get("openrouter_extra_body"))
        return OpenAICompatibleClient(
            model=model,
            base_url=str(cfg.get("openrouter_base_url") or "https://openrouter.ai/api/v1").strip(),
            api_key=str(cfg.get("openrouter_api_key") or "").strip() or None,
            api_key_env=str(cfg.get("openrouter_api_key_env", "OPENROUTER_API_KEY")),
            context_window_tokens=context_window_tokens,
            extra_headers=extra_headers or None,
            extra_body=extra_body or None,
            verify_connection=bool(cfg.get("openrouter_verify_connection", False)),
            airgapped=airgapped,
        )
    if runtime == "openai_compatible":
        model = str(cfg.get("openai_compatible_model") or cfg.get("model") or "").strip()
        base_url = str(cfg.get("openai_compatible_base_url") or "").strip() or None
        if airgapped:
            assert_endpoint_allowed(
                _normalize_base_url(base_url),
                label="the OpenAI-compatible runtime",
            )
        extra_headers = _string_dict(cfg.get("openai_compatible_headers"))
        extra_body = _mapping(cfg.get("openai_compatible_extra_body"))
        return OpenAICompatibleClient(
            model=model,
            base_url=base_url,
            api_key=str(cfg.get("openai_compatible_api_key") or "").strip() or None,
            api_key_env=str(cfg.get("openai_compatible_api_key_env", "OPENAI_API_KEY")),
            context_window_tokens=context_window_tokens,
            extra_headers=extra_headers or None,
            extra_body=extra_body or None,
            verify_connection=bool(cfg.get("openai_compatible_verify_connection", False)),
            airgapped=airgapped,
        )
    if runtime == "trtllm_pytorch":
        model = str(cfg.get("trtllm_model") or cfg.get("model") or "").strip()
        if not model:
            raise ValueError("Missing model for trtllm runtime (`trtllm_model` or `model`).")
        return TrtllmServerClient(
            model=model,
            context_window_tokens=context_window_tokens,
            backend=str(cfg.get("trtllm_backend", "pytorch")),
            config_path=str(cfg.get("trtllm_config_path", "")).strip() or None,
            trust_remote_code=bool(cfg.get("trtllm_trust_remote_code", True)),
            airgapped=airgapped,
        )
    if runtime == "sglang":
        model = str(cfg.get("sglang_model") or cfg.get("model") or "").strip()
        if not model:
            raise ValueError("Missing model for sglang runtime (`sglang_model` or `model`).")
        sglang_base_url = str(cfg.get("sglang_base_url", "")).strip() or None
        sglang_host = str(cfg.get("sglang_host", "127.0.0.1"))
        sglang_port = int(cfg.get("sglang_port", 8080))
        if airgapped:
            assert_endpoint_allowed(
                sglang_base_url or f"http://{sglang_host}:{sglang_port}",
                label="the SGLang runtime",
            )
        return SglangServerClient(
            model=model,
            host=sglang_host,
            port=sglang_port,
            base_url=sglang_base_url,
            context_window_tokens=context_window_tokens,
            device=str(cfg.get("device", "cuda")),
            mem_fraction_static=float(cfg.get("sglang_mem_fraction_static", 0.8)),
            tensor_parallel_size=int(cfg.get("sglang_tensor_parallel_size", 1)),
            dtype=str(cfg.get("sglang_dtype", "half")),
            attention_backend=str(cfg.get("sglang_attention_backend", "")).strip() or None,
            reasoning_parser=str(cfg.get("sglang_reasoning_parser", "")).strip() or None,
            tool_call_parser=str(cfg.get("sglang_tool_call_parser", "")).strip() or None,
            trust_remote_code=bool(cfg.get("sglang_trust_remote_code", True)),
            language_model_only=bool(cfg.get("sglang_language_model_only", False)),
            served_model_name=str(cfg.get("sglang_served_model_name", "local")),
            launch_mode=str(cfg.get("sglang_launch_mode", "managed")),
            jetson_container_executable=str(cfg.get("sglang_jetson_container_executable", "jetson-containers")),
            jetson_autotag_executable=str(cfg.get("sglang_jetson_autotag_executable", "autotag")),
            jetson_container_image=str(cfg.get("sglang_jetson_container_image", "")).strip() or None,
            jetson_container_extra_args=_string_list(cfg.get("sglang_jetson_container_extra_args")),
            docker_executable=str(cfg.get("sglang_docker_executable", "docker")),
            docker_image=str(cfg.get("sglang_docker_image", "")).strip() or None,
            docker_container_name=str(cfg.get("sglang_docker_container_name", "open-jet-sglang")),
            docker_use_host_network=bool(cfg.get("sglang_docker_use_host_network", True)),
            docker_runtime=str(cfg.get("sglang_docker_runtime", "nvidia")).strip() or None,
            docker_extra_args=_string_list(cfg.get("sglang_docker_extra_args")),
            airgapped=airgapped,
        )
    model = str(cfg.get("llama_model") or cfg.get("model") or "").strip()
    if not model:
        raise ValueError("Missing model for llama.cpp runtime (`llama_model` or `model`).")
    return LlamaServerClient(
        model=model,
        context_window_tokens=context_window_tokens,
        device=str(cfg.get("device", "auto")),
        gpu_layers=int(cfg.get("gpu_layers", 99)),
        airgapped=airgapped,
        diagnostics_hook=diagnostics_hook,
    )


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _string_dict(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    payload: dict[str, str] = {}
    for key, item in value.items():
        str_key = str(key).strip()
        str_value = str(item).strip()
        if str_key and str_value:
            payload[str_key] = str_value
    return payload


def _mapping(value: object) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _openrouter_headers(cfg: dict) -> dict[str, str]:
    headers = _string_dict(cfg.get("openrouter_headers"))
    referer = str(cfg.get("openrouter_site_url") or "").strip() or str(os.environ.get("OPENROUTER_SITE_URL", "")).strip()
    title = str(cfg.get("openrouter_app_name") or "").strip() or str(os.environ.get("OPENROUTER_APP_NAME", "")).strip()
    if referer and "HTTP-Referer" not in headers:
        headers["HTTP-Referer"] = referer
    if title and "X-Title" not in headers:
        headers["X-Title"] = title
    return headers
