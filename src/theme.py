"""Shared visual theme for prompt-toolkit and Rich output."""

from __future__ import annotations

from rich.markup import escape
from rich.theme import Theme


RICH_THEME = Theme(
    {
        "brand": "bold #86efac",
        "user": "bold #67e8f9",
        "assistant": "bold #a7f3d0",
        "command": "bold #5eead4",
        "tool": "bold #67e8f9",
        "tool_output": "#d1fae5",
        "code": "#cbd5e1 on #111827",
        "status": "bold #86efac",
        "muted": "#cbd5e1",
        "dim": "#94a3b8",
        "warning": "bold #7dd3fc",
        "error": "bold #f87171",
        "success": "bold #86efac",
    }
)


PROMPT_STYLE = {
    "prompt": "#e5e7eb bold",
    "brand": "#86efac bold",
    "prompt-airgapped": "#ffffff bold",
    "brand-airgapped": "#ffffff bold",
    "prompt-status": "#a7f3d0 bold",
    "prompt-command": "#5eead4 bold",
    "prompt-splash-text": "#d9f99d bold",
    "prompt-splash-block-1": "#166534 bold",
    "prompt-splash-block-2": "#16a34a bold",
    "prompt-splash-block-3": "#4ade80 bold",
    "prompt-splash-block-4": "#bbf7d0 bold",
    "prompt-warning": "#ffffff bold",
    "bottom-toolbar": "#ffffff",
    "toolbar-label": "#94a3b8 bold",
    "toolbar-value": "#ffffff",
    "toolbar-accent": "#86efac bold",
    "toolbar-warning": "#ffffff bold",
    "toolbar-danger": "#f87171 bold",
    "completion-menu.completion": "bg:#111827 #cbd5e1",
    "completion-menu.completion.current": "bg:#243244 #f8fafc bold",
    "scrollbar.background": "bg:#0b1220",
    "scrollbar.button": "bg:#243244",
}


def rich_text(text: str, style: str) -> str:
    return f"[{style}]{escape(text)}[/]"
