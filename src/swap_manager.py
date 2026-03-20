"""Orchestrates model unload/reload cycles using a registered SwapPlugin.

Usage::

    manager = SwapManager(plugin, state_dir=Path(".openjet/state/swap"))
    if manager.should_unload(estimated_need_mb=4000):
        result = await manager.run_with_swap("make -j4", timeout=300)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .executor import ExecResult, run_shell
from .swap_plugin import SwapPlugin

log = logging.getLogger(__name__)

# Minimum free MB headroom to keep even when model is unloaded
_MIN_HEADROOM_MB = 200.0


@dataclass
class SwapResult:
    """Result of a swap-guarded command execution."""

    exec_result: ExecResult
    unloaded: bool = False
    reload_ok: bool = True
    restore_ok: bool = True
    swap_duration_s: float = 0.0


class SwapManager:
    """Orchestrates the unload → run → reload → restore cycle."""

    def __init__(
        self,
        plugin: SwapPlugin,
        *,
        state_dir: Path | None = None,
        status_hook: Callable[[str], None] | None = None,
    ) -> None:
        self._plugin = plugin
        self._state_dir = state_dir or Path(".openjet/state/swap")
        self._status_hook = status_hook

    @property
    def plugin(self) -> SwapPlugin:
        return self._plugin

    def should_unload(self, estimated_need_mb: float) -> bool:
        """Return True if we need to free the model to run the task."""
        available = self._plugin.available_memory_mb()
        if available >= estimated_need_mb + _MIN_HEADROOM_MB:
            return False
        # Would freeing the model give us enough?
        model_mb = self._plugin.model_memory_mb()
        projected = available + model_mb
        return projected >= estimated_need_mb + _MIN_HEADROOM_MB

    async def run_with_swap(
        self,
        command: str,
        timeout: int = 120,
    ) -> SwapResult:
        """Full cycle: save → unload → run → reload → restore → return."""
        t0 = time.monotonic()
        self._emit("Saving model state...")

        save_ok = await self._plugin.save_state(self._state_dir)
        if not save_ok:
            log.warning("State save failed — running without swap")
            result = await run_shell(command, timeout_seconds=timeout)
            return SwapResult(exec_result=result, unloaded=False)

        self._emit("Unloading model to free memory...")
        await self._plugin.unload()

        self._emit(f"Running: {command[:80]}...")
        exec_result = await run_shell(command, timeout_seconds=timeout)

        self._emit("Reloading model...")
        reload_ok = True
        try:
            await self._plugin.reload()
        except Exception as exc:
            log.error("Model reload failed: %s", exc)
            reload_ok = False

        restore_ok = False
        if reload_ok:
            self._emit("Restoring model state...")
            try:
                restore_ok = await self._plugin.restore_state(self._state_dir)
            except Exception as exc:
                log.warning("State restore failed: %s", exc)

        if not restore_ok:
            self._emit("KV cache restore failed — conversation will be re-prompted")

        swap_duration = time.monotonic() - t0
        self._emit("")
        return SwapResult(
            exec_result=exec_result,
            unloaded=True,
            reload_ok=reload_ok,
            restore_ok=restore_ok,
            swap_duration_s=round(swap_duration, 2),
        )

    def _emit(self, text: str) -> None:
        if self._status_hook:
            try:
                self._status_hook(text)
            except Exception:
                pass
