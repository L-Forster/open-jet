"""Runtime-layer facade.

This layer owns protocol shaping, runtime selection, and transport clients.
"""

from ..llama_server import LlamaServerClient
from ..runtime_client import RuntimeClient
from ..runtime_protocol import StreamChunk, ToolCall, stream_openai_chat, stream_openai_responses
from ..runtime_registry import (
    CODEX_RUNTIME,
    DEFAULT_RUNTIME,
    LITELLM_RUNTIME,
    active_model_ref,
    active_runtime,
    create_runtime_client,
    runtime_options,
)

__all__ = [
    "CODEX_RUNTIME",
    "DEFAULT_RUNTIME",
    "LITELLM_RUNTIME",
    "LlamaServerClient",
    "RuntimeClient",
    "StreamChunk",
    "ToolCall",
    "active_model_ref",
    "active_runtime",
    "create_runtime_client",
    "runtime_options",
    "stream_openai_chat",
    "stream_openai_responses",
]
