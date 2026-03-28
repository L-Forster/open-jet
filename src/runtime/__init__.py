"""Runtime-layer facade.

This layer owns protocol shaping, runtime selection, and transport clients.
"""

from ..llama_server import LlamaServerClient
from ..runtime_client import RuntimeClient
from ..runtime_protocol import StreamChunk, ToolCall, stream_openai_chat
from ..runtime_registry import (
    DEFAULT_RUNTIME,
    active_model_ref,
    create_runtime_client,
    runtime_options,
)

__all__ = [
    "DEFAULT_RUNTIME",
    "LlamaServerClient",
    "RuntimeClient",
    "StreamChunk",
    "ToolCall",
    "active_model_ref",
    "create_runtime_client",
    "runtime_options",
    "stream_openai_chat",
]
