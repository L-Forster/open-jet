from __future__ import annotations

from pathlib import Path
from typing import Protocol


class RuntimeClient(Protocol):
    model: str
    context_window_tokens: int
    gpu_layers: int

    async def start(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def reset_kv_cache(self) -> None:
        ...

    async def chat_stream(self, messages: list[dict], *, use_tools: bool = True):
        ...

    async def save_kv_cache(self, path: Path) -> bool:
        """Save KV cache state to *path*. Returns True on success."""
        ...

    async def restore_kv_cache(self, path: Path) -> bool:
        """Restore KV cache state from *path*. Returns True on success."""
        ...
