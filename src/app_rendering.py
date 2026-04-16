"""Rendering, formatting, and markdown helpers for the TUI.

Extracted from app.py to reduce its size. These are pure functions that
handle text formatting, syntax detection, and approval display formatting.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

DIFF_CONTEXT_LINES = 1

from rich.syntax import Syntax
from rich.text import Text

from .agent import ToolCall
from .theme import rich_text
from .tool_executor import format_tool_args


def _read_file_or_empty(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace") if path else ""
    except OSError:
        return ""


def _diff_preview(old: str, new: str, *, context: int = DIFF_CONTEXT_LINES) -> list[str]:
    if old == new:
        return []
    out: list[str] = []
    old_no = new_no = 0
    for raw in difflib.unified_diff(old.splitlines(), new.splitlines(), n=context, lineterm=""):
        if raw.startswith(("--- ", "+++ ")):
            continue
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
        if m:
            old_no, new_no = int(m.group(1)), int(m.group(2))
            continue
        prefix, body = raw[:1], raw[1:]
        if prefix == "+":
            out.append(f"+{new_no:>4} + {body}"); new_no += 1
        elif prefix == "-":
            out.append(f"-{old_no:>4} - {body}"); old_no += 1
        else:
            out.append(f" {new_no:>4}   {body}"); old_no += 1; new_no += 1
    return out


def format_assistant_output_line(line: str, *, in_code_block: bool) -> tuple[object, bool]:
    stripped = line.strip()
    if stripped.startswith("```"):
        return rich_text(stripped or "```", "tool"), not in_code_block
    if in_code_block:
        return rich_text(line, "code"), in_code_block
    if re.fullmatch(r"\s{0,3}#{1,6}\s+.+", line):
        return Text(re.sub(r"^\s{0,3}#{1,6}\s+", "", line), style="bold"), in_code_block
    return render_markdown_inline_segments(line), in_code_block


def format_tool_output_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    first = stripped.split(" ", 1)[0]
    if stripped.startswith("$") or stripped.startswith("/") or first in {
        "git",
        "python",
        "python3",
        "pytest",
        "bash",
        "sh",
        "npm",
        "pnpm",
        "cargo",
        "make",
        "rg",
    }:
        return rich_text(line, "command")
    return rich_text(line, "tool_output")


def render_markdown_inline_segments(line: str) -> Text:
    if "`" not in line and "**" not in line and "__" not in line:
        return Text(line)
    text = Text()
    parts = re.split(r"(`[^`]+`)", line)
    for part in parts:
        if not part:
            continue
        if len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            text.append(part[1:-1], style="code")
        else:
            _append_markdown_emphasis(text, part)
    return text


def _append_markdown_emphasis(text: Text, segment: str) -> None:
    cursor = 0
    for match in re.finditer(r"(\*\*.+?\*\*|__.+?__)", segment):
        start, end = match.span()
        if start > cursor:
            text.append(segment[cursor:start])
        inner = match.group(0)[2:-2]
        if inner:
            text.append(inner, style="bold")
        cursor = end
    if cursor < len(segment):
        text.append(segment[cursor:])


def tool_result_syntax(lines: list[str], *, tool_call: ToolCall | None = None) -> Syntax | None:
    if not lines:
        return None
    lexer = tool_result_lexer(lines, tool_call=tool_call)
    if not lexer:
        return None
    code = "\n".join(lines)
    return Syntax(code, lexer, theme="monokai", word_wrap=True, line_numbers=False, background_color="default")


def tool_result_lexer(lines: list[str], *, tool_call: ToolCall | None = None) -> str | None:
    if tool_call is not None:
        if tool_call.name in {"list_directory", "grep", "glob"}:
            return "text"
        if tool_call.name in {"read_file", "load_file"}:
            path = str(tool_call.arguments.get("path", "")).strip() if isinstance(tool_call.arguments, dict) else ""
            return lexer_for_path(path) or "text"
    sample = "\n".join(lines)
    if re.search(r"^[\w./-]+:\d+:", sample, flags=re.MULTILINE):
        return "text"
    if re.search(r"^\s*(def |class |from |import )", sample, flags=re.MULTILINE):
        return "python"
    if re.search(r"^\s*[{\[]", sample):
        return "json"
    return None


def lexer_for_path(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".yml", ".yaml"}:
        return "yaml"
    if suffix in {".toml"}:
        return "toml"
    if suffix in {".sh", ".bash"}:
        return "bash"
    if suffix:
        return "text"
    return None


def format_command_status_label(command: str, max_len: int = 72) -> str:
    compact = " ".join(command.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def approval_summary_text(tc: ToolCall) -> str:
    if tc.name == "write_file":
        path = str(tc.arguments.get("path", "")).strip()
        content = str(tc.arguments.get("content", ""))
        return f"write_file -> {path} ({len(content)} bytes)"
    if tc.name == "edit_file":
        return f"edit_file -> {str(tc.arguments.get('path', '')).strip()}"
    if tc.name == "memory":
        return f"memory -> {str(tc.arguments.get('action', '')).strip()} {str(tc.arguments.get('scope', '')).strip()}".strip()
    if tc.name == "shell":
        command = str(tc.arguments.get("command", "")).strip()
        resource_mode = str(tc.arguments.get("resource_mode", "normal") or "normal").strip()
        reload_delay = tc.arguments.get("reload_delay_seconds")
        if len(command) > 120:
            command = command[:117] + "..."
        suffix = f" [{resource_mode}]"
        if isinstance(reload_delay, int) and reload_delay > 0:
            suffix += f" delay={reload_delay}s"
        return f"shell{suffix} -> {command}"
    if tc.name == "system_info":
        scope = str(tc.arguments.get("scope", "summary") or "summary").strip()
        return f"system_info -> {scope}"
    if tc.name == "exit_plan_mode":
        summary = str(tc.arguments.get("plan_summary", "")).strip()
        if len(summary) > 120:
            summary = summary[:117] + "..."
        return f"exit_plan_mode -> {summary or 'plan summary'}"
    return f"{tc.name} -> {format_tool_args(tc)}"


def tool_preview_lines(tc: ToolCall) -> list[str]:
    if tc.name == "shell":
        command = str(tc.arguments.get("command", "")).strip()
        timeout_seconds = tc.arguments.get("timeout_seconds")
        resource_mode = str(tc.arguments.get("resource_mode", "normal") or "normal").strip()
        estimated_ram_mb = tc.arguments.get("estimated_ram_mb")
        estimated_vram_mb = tc.arguments.get("estimated_vram_mb")
        reload_delay_seconds = tc.arguments.get("reload_delay_seconds")
        lines = [f"command: {command[:200] + ('...' if len(command) > 200 else '')}"]
        if isinstance(timeout_seconds, int):
            lines.append(f"timeout_seconds: {timeout_seconds}")
        lines.append(f"resource_mode: {resource_mode}")
        if isinstance(estimated_ram_mb, int):
            lines.append(f"estimated_ram_mb: {estimated_ram_mb}")
        if isinstance(estimated_vram_mb, int):
            lines.append(f"estimated_vram_mb: {estimated_vram_mb}")
        if isinstance(reload_delay_seconds, int) and reload_delay_seconds > 0:
            lines.append(f"reload_delay_seconds: {reload_delay_seconds}")
        return lines
    if tc.name == "system_info":
        return [f"scope: {str(tc.arguments.get('scope', 'summary') or 'summary').strip()}"]
    if tc.name == "write_file":
        path = str(tc.arguments.get("path", "")).strip()
        new = str(tc.arguments.get("content", ""))
        old = _read_file_or_empty(path)
        return [f"path: {path}", f"bytes: {len(new)}", *_diff_preview(old, new)]
    if tc.name == "edit_file":
        path = str(tc.arguments.get("path", "")).strip()
        old_str = tc.arguments.get("old_string") or ""
        new_str = tc.arguments.get("new_string") or ""
        src = _read_file_or_empty(path)
        new_src = src.replace(old_str, new_str, 1) if isinstance(old_str, str) and old_str else src
        return [f"path: {path}", *_diff_preview(src, new_src)]
    if tc.name == "memory":
        return [f"scope: {str(tc.arguments.get('scope', '')).strip()}", f"action: {str(tc.arguments.get('action', '')).strip()}"]
    if tc.name == "exit_plan_mode":
        return [f"plan_summary: {str(tc.arguments.get('plan_summary', '')).strip()}"]
    return [str(format_tool_args(tc))]
