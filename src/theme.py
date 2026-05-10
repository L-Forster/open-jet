"""Shared visual theme for prompt-toolkit and Rich output."""

from __future__ import annotations

from rich.markup import escape
from rich.theme import Theme


RICH_THEME = Theme(
    {
        "brand": "bold #48d17a",
        "user": "bold #48d17a",
        "assistant": "bold #f8fafc",
        "command": "bold #9be7b2",
        "tool": "bold #48d17a",
        "tool_output": "#e6f4ea",
        "code": "#dbe7ee on #0f1720",
        "status": "bold #b7efc5",
        "muted": "#d1d5db",
        "dim": "#8a928c",
        "warning": "bold #facc15",
        "error": "bold #fca5a5",
        "success": "bold white",
        "diff_add": "#7ee09a",
        "diff_remove": "#ef4444",
        "diff_add_body": "#dcfce7 on #14361e",
        "diff_remove_body": "#fee2e2 on #4a0f12",
        "diff_add_marker": "bold #86efac on #14361e",
        "diff_remove_marker": "bold #fca5a5 on #4a0f12",
        "diff_lineno_add": "bold #86efac",
        "diff_lineno_remove": "bold #fca5a5",
        "diff_context": "#cbd5dc",
        "diff_lineno_context": "#6b7280",
        "approval_border": "#facc15",
        "approval_title": "bold #facc15",
        "approval_meta": "#e5e7eb",
        "approve_selected": "bold #7ee09a",
        "approve_idle": "#d1d5db",
        "deny_selected": "bold #ef4444",
        "deny_idle": "#d1d5db",
        "chrome_brand": "bold #0a0f0d on #48d17a",
        "chrome_brand_text": "bold #48d17a",
        "chrome_border": "#4a514d",
        "chrome_label": "bold #a3aaa5",
        "chrome_value": "#e5e7eb",
        "chrome_accent": "bold white",
        "chrome_warning": "bold #fde68a",
    }
)


PROMPT_STYLE = {
    "prompt": "#f3f7f4 bold",
    "brand": "#9be7b2 bold",
    "prompt-airgapped": "#ffffff bold",
    "brand-airgapped": "#ffffff bold",
    "brand-chip": "bg:#121816 #9be7b2 bold",
    "brand-chip-airgapped": "bg:#221316 #ffd5dc bold",
    "prompt-divider": "#7ee09a",
    "prompt-divider-airgapped": "#fecdd3",
    "prompt-border": "#3f4642",
    "frame.border": "#4a514d",
    "frame.label": "#4a514d",
    "prompt-placeholder": "#7c8781",
    "prompt-status": "#9be7b2 bold",
    "prompt-tip": "#9fb7aa italic",
    "prompt-status-label": "bg:#121816 #9be7b2 bold",
    "prompt-command": "#e5e7eb bold",
    "prompt-splash-text": "#dcfce7 bold",
    "prompt-splash-block-1": "#164e3a bold",
    "prompt-splash-block-2": "#1f7a56 bold",
    "prompt-splash-block-3": "#4fb286 bold",
    "prompt-splash-block-4": "#c7f9cc bold",
    "prompt-warning": "#ffffff bold",
    "bottom-toolbar": "noreverse bg:#0b0f0d #9ca3af",
    "toolbar-brand": "bg:#121816 #9be7b2 bold",
    "toolbar-cell": "bg:#101715 #d1d5db",
    "toolbar-chip": "bg:#101715 #9ca3af bold",
    "toolbar-label": "#9ca3af bold",
    "toolbar-value": "#eef6f1",
    "toolbar-accent": "#9be7b2 bold",
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
