"""Shared visual theme for prompt-toolkit and Rich output."""

from __future__ import annotations

from rich.markup import escape
from rich.theme import Theme


RICH_THEME = Theme(
    {
        "brand": "bold #8bd450",
        "user": "bold #67e8f9",
        "assistant": "bold #93c5fd",
        "command": "bold #fbbf24",
        "tool": "bold #fb7185",
        "tool_output": "#dbeafe",
        "code": "#cbd5e1 on #111827",
        "status": "bold #8bd450",
        "muted": "#cbd5e1",
        "dim": "#94a3b8",
        "warning": "bold #f59e0b",
        "error": "bold #f87171",
        "success": "bold #8bd450",
    }
)


PROMPT_STYLE = {
    "prompt": "#e5e7eb bold",
    "brand": "#8bd450 bold",
    "prompt-airgapped": "#ffedd5 bold",
    "brand-airgapped": "#fdba74 bold",
    "prompt-status": "#93c5fd bold",
    "prompt-command": "#fbbf24 bold",
    "prompt-warning": "#f59e0b bold",
    "bottom-toolbar": "bg:#0b1220 #cbd5e1",
    "toolbar-label": "#94a3b8 bold",
    "toolbar-value": "#e5e7eb",
    "toolbar-accent": "#8bd450 bold",
    "toolbar-warning": "#f59e0b bold",
    "toolbar-danger": "#f87171 bold",
    "completion-menu.completion": "bg:#111827 #cbd5e1",
    "completion-menu.completion.current": "bg:#243244 #f8fafc bold",
    "scrollbar.background": "bg:#0b1220",
    "scrollbar.button": "bg:#243244",
}


def rich_text(text: str, style: str) -> str:
    return f"[{style}]{escape(text)}[/]"
