"""Tools-layer facade.

This package keeps imports lazy so runtime schema helpers can import
`src.tools.registry` without pulling in the executor implementation.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "EditFileResult",
    "ExecResult",
    "LoadFileResult",
    "TOOL_REGISTRY",
    "ToolExecutionResult",
    "ToolSpec",
    "edit_file",
    "execute_tool",
    "format_tool_args",
    "glob_files",
    "grep_files",
    "list_directory",
    "load_file",
    "read_file",
    "run_shell",
    "write_file",
]


def __getattr__(name: str):
    if name in {"TOOL_REGISTRY", "ToolSpec"}:
        module = import_module(".registry", __name__)
        return getattr(module, name)
    if name in {"ToolExecutionResult", "execute_tool", "format_tool_args"}:
        module = import_module("..tool_executor", __name__)
        return getattr(module, name)
    if name in {
        "EditFileResult",
        "ExecResult",
        "LoadFileResult",
        "edit_file",
        "glob_files",
        "grep_files",
        "list_directory",
        "load_file",
        "read_file",
        "run_shell",
        "write_file",
    }:
        module = import_module("..executor", __name__)
        return getattr(module, name)
    raise AttributeError(name)
