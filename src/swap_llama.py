"""Swap backend implementation for llama.cpp (llama-server).

Uses:
- KV cache save/restore via llama-server slot API
- Server stop/start via ``LlamaServerClient``
- Memory reading via ``/proc/meminfo``
- State dir: ``.openjet/state/swap/`` (messages.json + kv_cache.bin)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from .llama_server import LlamaServerClient
from .runtime_limits import read_memory_snapshot

log = logging.getLogger(__name__)


class LlamaSwapPlugin:
    """Swap backend backed by a running ``LlamaServerClient``."""

    def __init__(
        self,
        client: LlamaServerClient,
        messages: list[dict] | None = None,
    ) -> None:
        self._client = client
        self._messages = messages if messages is not None else []
        # Rough estimate — actual varies by model size / quant.
        self._estimated_model_mb: float = 0.0

    def set_messages(self, messages: list[dict]) -> None:
        """Keep a reference to the live conversation so we can persist it."""
        self._messages = messages

    def set_estimated_model_mb(self, mb: float) -> None:
        """Override the default model-size estimate used for swap planning."""
        self._estimated_model_mb = mb

    async def save_state(self, state_dir: Path) -> bool:
        """Persist the live message list and best-effort KV cache state."""
        state_dir.mkdir(parents=True, exist_ok=True)

        # 1. Persist conversation messages
        msgs_path = state_dir / "messages.json"
        try:
            msgs_path.write_text(
                json.dumps(self._messages, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("Failed to save messages: %s", exc)
            return False

        # 2. Save KV cache via slot API
        cache_path = state_dir / "kv_cache.bin"
        try:
            saved = await self._client.save_kv_cache(cache_path)
            if not saved:
                log.info("KV cache save returned False (server may not support slots)")
        except Exception as exc:
            log.warning("KV cache save failed: %s", exc)
            # Messages are saved — we can still restore via re-prompt.

        return True

    async def unload(self) -> None:
        """Stop the llama.cpp server so its memory can be reused."""
        await self._client._stop_server()

    async def reload(self) -> None:
        """Start the llama.cpp server again after the shell command finishes."""
        await self._client.start()

    async def restore_state(self, state_dir: Path) -> bool:
        """Reload saved messages and restore the KV cache when possible."""
        # 1. Restore conversation messages
        msgs_path = state_dir / "messages.json"
        if msgs_path.exists():
            try:
                loaded = json.loads(msgs_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    self._messages.clear()
                    self._messages.extend(loaded)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Failed to restore messages: %s", exc)
                return False

        # 2. Restore KV cache
        cache_path = state_dir / "kv_cache.bin"
        if cache_path.exists():
            try:
                restored = await self._client.restore_kv_cache(cache_path)
                if restored:
                    return True
                log.info("KV cache restore returned False — will re-prompt")
            except Exception as exc:
                log.warning("KV cache restore failed: %s — will re-prompt", exc)

        # Messages restored but KV cache not — caller should re-prompt.
        return False

    def available_memory_mb(self) -> float:
        """Read the current available RAM from the system snapshot."""
        snapshot = read_memory_snapshot()
        if snapshot and snapshot.available_mb is not None:
            return float(snapshot.available_mb)
        return 0.0

    def model_memory_mb(self) -> float:
        """Return the planned model-memory estimate used by ``SwapManager``."""
        if self._estimated_model_mb > 0:
            return self._estimated_model_mb
        # Rough heuristic: check memory drop after unload would give real
        # numbers, but we haven't unloaded yet.  Use a conservative default.
        return 2500.0
