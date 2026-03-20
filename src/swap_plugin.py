"""Plugin interface for model swap (unload/reload) lifecycle.

Any harness (llama.cpp, Ollama, vLLM, remote API) implements ``SwapPlugin``
so that ``SwapManager`` can orchestrate memory-freeing unload/reload cycles
without knowing runtime internals.

This module is importable standalone — it has no open-jet internal dependencies.
"""

from __future__ import annotations

from pathlib import Path


class SwapPlugin:
    """Interface that any harness implements to participate in model swap."""

    async def save_state(self, state_dir: Path) -> bool:
        """Save model state (KV cache, conversation, etc.) to *state_dir*.

        Returns ``True`` on success.  The directory is created by the caller.
        """
        raise NotImplementedError

    async def unload(self) -> None:
        """Unload model from RAM (kill server, free GPU memory)."""
        raise NotImplementedError

    async def reload(self) -> None:
        """Reload model into RAM (restart server)."""
        raise NotImplementedError

    async def restore_state(self, state_dir: Path) -> bool:
        """Restore previously saved state.

        Returns ``False`` if restore failed — caller falls back to
        re-prompting the full conversation history.
        """
        raise NotImplementedError

    def available_memory_mb(self) -> float:
        """How much RAM is free right now."""
        raise NotImplementedError

    def model_memory_mb(self) -> float:
        """Approximate RAM the loaded model uses."""
        raise NotImplementedError
