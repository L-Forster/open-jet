from __future__ import annotations

from dataclasses import dataclass

from .llama_server import LlamaServerClient
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
}

DEFAULT_RUNTIME = "llama_cpp"


def normalize_runtime(value: str | None) -> str:
    runtime = (value or DEFAULT_RUNTIME).strip().lower()
    return runtime if runtime in RUNTIME_SPECS else DEFAULT_RUNTIME


def runtime_options() -> list[tuple[str, str]]:
    return [(spec.label, spec.key) for spec in RUNTIME_SPECS.values()]


def runtime_spec(value: str | None) -> RuntimeSpec:
    return RUNTIME_SPECS[normalize_runtime(value)]


def active_model_ref(cfg: dict) -> str:
    spec = runtime_spec(str(cfg.get("runtime", DEFAULT_RUNTIME)))
    return str(cfg.get(spec.model_config_key) or cfg.get("model") or "").strip()


def create_runtime_client(cfg: dict) -> RuntimeClient:
    runtime = normalize_runtime(str(cfg.get("runtime", DEFAULT_RUNTIME)))
    context_window_tokens = int(cfg.get("context_window_tokens", 2048))
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
        )
    if runtime == "sglang":
        model = str(cfg.get("sglang_model") or cfg.get("model") or "").strip()
        if not model:
            raise ValueError("Missing model for sglang runtime (`sglang_model` or `model`).")
        return SglangServerClient(
            model=model,
            host=str(cfg.get("sglang_host", "127.0.0.1")),
            port=int(cfg.get("sglang_port", 8080)),
            base_url=str(cfg.get("sglang_base_url", "")).strip() or None,
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
        )
    model = str(cfg.get("llama_model") or cfg.get("model") or "").strip()
    if not model:
        raise ValueError("Missing model for llama.cpp runtime (`llama_model` or `model`).")
    return LlamaServerClient(
        model=model,
        context_window_tokens=context_window_tokens,
        device=str(cfg.get("device", "auto")),
        gpu_layers=int(cfg.get("gpu_layers", 99)),
    )


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []
