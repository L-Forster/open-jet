"""Shared visual theme for prompt-toolkit and Rich output."""

from __future__ import annotations

from rich.markup import escape
from rich.theme import Theme


RICH_THEME = Theme(
    {
        "brand": "bold #d8f3dc",
        "user": "bold #8bd3ff",
        "assistant": "bold #d8f3dc",
        "command": "bold #63e6be",
        "tool": "bold #8bd3ff",
        "tool_output": "#e6f4ea",
        "code": "#dbe7ee on #0f1720",
        "status": "bold #b7efc5",
        "muted": "#cbd5dc",
        "dim": "#8b98a2",
        "warning": "bold #fef3c7",
        "error": "bold #fca5a5",
        "success": "bold #b7efc5",
        "approve_selected": "bold black on #86efac",
        "approve_idle": "bold #86efac",
        "deny_selected": "bold white on #ef4444",
        "deny_idle": "bold #fca5a5",
    }
)


PROMPT_STYLE = {
    "prompt": "#eef6f1 bold",
    "brand": "#86efac bold",
    "prompt-airgapped": "#ffffff bold",
    "brand-airgapped": "#ffffff bold",
    "brand-chip": "bg:#173026 #d8f3dc bold",
    "brand-chip-airgapped": "bg:#4c0519 #fff1f2 bold",
    "prompt-divider": "#63e6be",
    "prompt-divider-airgapped": "#fecdd3",
    "prompt-status": "#b7efc5 bold",
    "prompt-tip": "#9fb7aa italic",
    "prompt-status-label": "bg:#103126 #9ae6b4 bold",
    "prompt-command": "#8bd3ff bold",
    "prompt-splash-text": "#dcfce7 bold",
    "prompt-splash-block-1": "#164e3a bold",
    "prompt-splash-block-2": "#1f7a56 bold",
    "prompt-splash-block-3": "#4fb286 bold",
    "prompt-splash-block-4": "#c7f9cc bold",
    "prompt-warning": "#ffffff bold",
    "bottom-toolbar": "bg:#08130f #dce6df",
    "toolbar-chip": "bg:#173026 #d9f99d bold",
    "toolbar-label": "#94a3b8 bold",
    "toolbar-value": "#eef6f1",
    "toolbar-accent": "#b7efc5 bold",
    "toolbar-warning": "#fef3c7 bold",
    "toolbar-note": "#a9b8b0",
    "toolbar-danger": "#f87171 bold",
    "completion-menu.completion": "bg:#111827 #cbd5e1",
    "completion-menu.completion.current": "bg:#243244 #f8fafc bold",
    "scrollbar.background": "bg:#0b1220",
    "scrollbar.button": "bg:#243244",
}


def rich_text(text: str, style: str) -> str:
    return f"[{style}]{escape(text)}[/]"
