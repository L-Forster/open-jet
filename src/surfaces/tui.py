"""TypeScript TUI launch surface."""

from __future__ import annotations

from ..tui_launcher import TuiLaunchError, launch_tui as _launch_tui


def launch_tui(*, force_setup: bool = False) -> None:
    try:
        _launch_tui(force_setup=force_setup)
    except TuiLaunchError as exc:
        raise SystemExit(str(exc)) from exc

__all__ = ["launch_tui"]
