"""Runtime-layer facade.

This layer owns protocol shaping, runtime selection, and transport clients.
"""

from ..llama_server import LlamaServerClient
from ..openai_compatible import OpenAICompatibleClient
from ..runtime_client import RuntimeClient
from ..runtime_protocol import StreamChunk, ToolCall, stream_openai_chat
from ..runtime_registry import (
    DEFAULT_RUNTIME,
    RUNTIME_SPECS,
    RuntimeSpec,
    active_model_ref,
    create_runtime_client,
    normalize_runtime,
    runtime_options,
    runtime_spec,
)
from ..sglang_server import SglangServerClient
from ..trtllm_server import TrtllmServerClient

__all__ = [
    "DEFAULT_RUNTIME",
    "LlamaServerClient",
    "OpenAICompatibleClient",
    "RUNTIME_SPECS",
    "RuntimeClient",
    "RuntimeSpec",
    "SglangServerClient",
    "StreamChunk",
    "ToolCall",
    "TrtllmServerClient",
    "active_model_ref",
    "create_runtime_client",
    "normalize_runtime",
    "runtime_options",
    "runtime_spec",
    "stream_openai_chat",
]
