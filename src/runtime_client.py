from __future__ import annotations

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
