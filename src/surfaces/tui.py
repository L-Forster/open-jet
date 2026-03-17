"""Lazy TUI loading helpers."""

from __future__ import annotations

import asyncio


def create_tui_app(*, force_setup: bool = False):
    from ..app import OpenJetApp

    return OpenJetApp(force_setup=force_setup)


def launch_tui(*, force_setup: bool = False) -> None:
    asyncio.run(create_tui_app(force_setup=force_setup).run_async())
