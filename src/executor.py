"""Guarded executor: runs shell commands only after explicit approval."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .runtime_limits import (
    MIN_TOKEN_BUDGET,
    derive_file_token_budget,
    estimate_tokens,
    read_memory_snapshot,
)


@dataclass
class ExecResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    timeout_seconds: int | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def summary(self) -> str:
        out = self.stdout.strip()
        err = self.stderr.strip()
        parts: list[str] = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr] {err}")
        if not parts:
            return f"(exit {self.exit_code}, no output)"
        return "\n".join(parts)


@dataclass
class LoadFileResult:
    ok: bool
    path: str
    content: str = ""
    detail: str = ""
    estimated_tokens: int = 0
    returned_tokens: int = 0
    token_budget: int = 0
    truncated: bool = False
    mem_available_mb: float | None = None

    @property
    def summary(self) -> str:
        if not self.ok:
            return self.detail
        trunc = "yes" if self.truncated else "no"
        mem_str = (
            f"{self.mem_available_mb:.0f}MB"
            if self.mem_available_mb is not None
            else "unknown"
        )
        return (
            f"Loaded {self.path} "
            f"(tokens~{self.returned_tokens}/{self.token_budget}, "
            f"full~{self.estimated_tokens}, "
            f"truncated={trunc}, mem_available={mem_str})"
        )


_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".xml", ".html", ".css", ".scss",
    ".sh", ".bash", ".zsh", ".fish", ".sql",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".go", ".rs", ".java", ".kt", ".swift",
    ".rb", ".php", ".lua", ".pl",
    ".lock",
    ".dockerfile", ".makefile",
}
_TEXT_FILENAMES = {
    "dockerfile",
    "makefile",
    ".env",
    ".envrc",
    ".gitignore",
    ".gitattributes",
    ".dockerignore",
    ".editorconfig",
    ".prettierignore",
    ".eslintignore",
    ".npmrc",
    ".nvmrc",
    ".pylintrc",
    ".flake8",
    ".tool-versions",
}
_MAX_READ_BYTES = 2 * 1024 * 1024
DEFAULT_SHELL_TIMEOUT_SECONDS = 60
SHELL_TIMEOUT_EXIT_CODE = 124


# -- Shell executor ----------------------------------------------------------


async def run_shell(command: str, timeout_seconds: int = DEFAULT_SHELL_TIMEOUT_SECONDS) -> ExecResult:
    """Run a shell command and capture output. Caller must gate on approval first."""
    if timeout_seconds <= 0:
        timeout_seconds = DEFAULT_SHELL_TIMEOUT_SECONDS
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        stdout_bytes, stderr_bytes = await proc.communicate()
        timeout_note = (
            f"Command timed out after {timeout_seconds}s and was terminated."
        ).encode()
        if stderr_bytes:
            stderr_bytes = stderr_bytes + b"\n" + timeout_note
        else:
            stderr_bytes = timeout_note
    exit_code = proc.returncode if proc.returncode is not None else 0
    if timed_out:
        exit_code = SHELL_TIMEOUT_EXIT_CODE
    return ExecResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
        timed_out=timed_out,
        timeout_seconds=timeout_seconds,
    )


# -- File tools (no shell needed) -------------------------------------------


async def read_file(path: str) -> str:
    """Read and return file contents."""
    p = _normalize_tool_path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if p.is_dir():
        return f"Error: path is a directory: {path}"
    try:
        raw = p.read_bytes()
        clipped = False
        if len(raw) > _MAX_READ_BYTES:
            raw = raw[:_MAX_READ_BYTES]
            clipped = True
        text = raw.decode("utf-8", errors="replace")
        if clipped:
            text = f"{text}\n\n...[truncated at {_MAX_READ_BYTES} bytes for safety]"
        return text
    except Exception as e:
        return f"Error reading {path}: {e}"


async def write_file(path: str, content: str) -> str:
    """Write content to a file. Caller must gate on approval first."""
    raw_path = path.strip()
    if not raw_path:
        return "Error writing file: path is empty."

    p = _normalize_tool_path(raw_path)
    try:
        # Reject directory targets like "." or existing directories.
        if p.exists() and p.is_dir():
            return f"Error writing file: path is a directory: {path}"
        if str(p) in {".", ""}:
            return f"Error writing file: path is a directory: {path}"

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


async def load_file(path: str, max_tokens: int | None = None) -> LoadFileResult:
    """Load a text/code file for prompt context with RAM-aware truncation."""
    raw_path = path.strip()
    if not raw_path:
        return LoadFileResult(ok=False, path=path, detail="Error: path is empty.")

    p = _normalize_tool_path(raw_path)
    if not p.exists():
        return LoadFileResult(ok=False, path=path, detail=f"Error: file not found: {path}")
    if p.is_dir():
        return LoadFileResult(ok=False, path=path, detail=f"Error: path is a directory: {path}")
    if not _is_supported_text_file(p):
        return LoadFileResult(
            ok=False,
            path=path,
            detail="Error: only text/code files are supported for context loading.",
        )

    try:
        raw = p.read_bytes()
    except Exception as e:
        return LoadFileResult(ok=False, path=path, detail=f"Error reading {path}: {e}")

    if b"\x00" in raw[:4096]:
        return LoadFileResult(
            ok=False,
            path=path,
            detail="Error: binary file detected; only text/code files are supported.",
        )

    clipped_to_max_read = False
    if len(raw) > _MAX_READ_BYTES:
        raw = raw[:_MAX_READ_BYTES]
        clipped_to_max_read = True

    text = raw.decode("utf-8", errors="replace")
    mem = read_memory_snapshot()
    mem_available_mb = mem.available_mb if mem else None
    budget_from_ram = derive_file_token_budget(mem_available_mb)
    requested_budget = max_tokens if max_tokens is not None else budget_from_ram
    token_budget = max(MIN_TOKEN_BUDGET, min(budget_from_ram, int(requested_budget)))
    max_chars = token_budget * 4

    estimated_tokens = estimate_tokens(text)
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    if clipped_to_max_read:
        truncated = True
    if truncated:
        text = f"{text}\n\n...[truncated for context safety]"
    returned_tokens = estimate_tokens(text)

    return LoadFileResult(
        ok=True,
        path=str(p),
        content=text,
        estimated_tokens=estimated_tokens,
        returned_tokens=returned_tokens,
        token_budget=token_budget,
        truncated=truncated,
        mem_available_mb=mem_available_mb,
    )


# -- Edit file tool ----------------------------------------------------------


async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace exact string occurrences in a file."""
    raw_path = path.strip()
    if not raw_path:
        return "Error: path is empty."

    p = _normalize_tool_path(raw_path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if p.is_dir():
        return f"Error: path is a directory: {path}"

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {path}: {e}"

    if old_string not in content:
        return f"Error: old_string not found in {path}"

    if not replace_all:
        count = content.count(old_string)
        if count > 1:
            return (
                f"Error: old_string appears {count} times in {path}. "
                "Provide more surrounding context to make it unique, or set replace_all=true."
            )
        new_content = content.replace(old_string, new_string, 1)
    else:
        new_content = content.replace(old_string, new_string)

    try:
        p.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return f"Error writing {path}: {e}"

    replacements = content.count(old_string) if replace_all else 1
    return f"Edited {path}: {replacements} replacement(s) made."


# -- Glob tool ---------------------------------------------------------------

_DEFAULT_GLOB_IGNORE = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs", "*.egg-info",
}

_MAX_GLOB_RESULTS = 200


async def glob_files(pattern: str, path: str | None = None) -> str:
    """Find files matching a glob pattern."""
    base = Path(path).expanduser() if path else Path.cwd()
    if not base.exists():
        return f"Error: directory not found: {base}"
    if not base.is_dir():
        return f"Error: not a directory: {base}"

    matches: list[str] = []
    try:
        for match in sorted(base.glob(pattern)):
            # Skip ignored directories
            parts = match.relative_to(base).parts
            if any(part in _DEFAULT_GLOB_IGNORE for part in parts):
                continue
            if any(fnmatch.fnmatch(part, p) for part in parts for p in _DEFAULT_GLOB_IGNORE):
                continue
            matches.append(str(match))
            if len(matches) >= _MAX_GLOB_RESULTS:
                break
    except Exception as e:
        return f"Error: {e}"

    if not matches:
        return f"No files matched pattern: {pattern}"

    result = "\n".join(matches)
    if len(matches) >= _MAX_GLOB_RESULTS:
        result += f"\n... (truncated at {_MAX_GLOB_RESULTS} results)"
    return result


# -- Grep tool ---------------------------------------------------------------

_MAX_GREP_MATCHES = 100
_MAX_GREP_LINE_LEN = 500


async def grep_files(
    pattern: str,
    path: str | None = None,
    glob_filter: str | None = None,
    ignore_case: bool = False,
) -> str:
    """Search file contents with regex."""
    base = Path(path).expanduser() if path else Path.cwd()
    if not base.exists():
        return f"Error: path not found: {base}"

    try:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: invalid regex: {e}"

    matches: list[str] = []

    if base.is_file():
        files = [base]
    else:
        files = sorted(base.rglob(glob_filter or "*"))

    for fp in files:
        if not fp.is_file():
            continue
        # Skip ignored directories
        try:
            rel_parts = fp.relative_to(base).parts if base.is_dir() else ()
        except ValueError:
            rel_parts = ()
        if any(part in _DEFAULT_GLOB_IGNORE for part in rel_parts):
            continue
        if any(fnmatch.fnmatch(part, p) for part in rel_parts for p in _DEFAULT_GLOB_IGNORE):
            continue

        # Skip binary files
        try:
            head = fp.read_bytes()[:4096]
            if b"\x00" in head:
                continue
            text = fp.read_text(encoding="utf-8", errors="replace")
        except (PermissionError, OSError):
            continue

        for line_no, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                display_line = line[:_MAX_GREP_LINE_LEN]
                if len(line) > _MAX_GREP_LINE_LEN:
                    display_line += "..."
                matches.append(f"{fp}:{line_no}: {display_line}")
                if len(matches) >= _MAX_GREP_MATCHES:
                    break
        if len(matches) >= _MAX_GREP_MATCHES:
            break

    if not matches:
        return f"No matches for pattern: {pattern}"

    result = "\n".join(matches)
    if len(matches) >= _MAX_GREP_MATCHES:
        result += f"\n... (truncated at {_MAX_GREP_MATCHES} matches)"
    return result


# -- List directory tool -----------------------------------------------------

_MAX_LS_ENTRIES = 200


async def list_directory(path: str | None = None) -> str:
    """List directory contents with file types and sizes."""
    base = Path(path).expanduser() if path else Path.cwd()
    if not base.exists():
        return f"Error: path not found: {base}"
    if not base.is_dir():
        return f"Error: not a directory: {base}"

    entries: list[str] = []
    try:
        for item in sorted(base.iterdir()):
            if item.name.startswith(".") and item.name in {".git"}:
                continue
            try:
                if item.is_dir():
                    entries.append(f"  {item.name}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    entries.append(f"  {item.name}  ({size_str})")
            except OSError:
                entries.append(f"  {item.name}  (error)")
            if len(entries) >= _MAX_LS_ENTRIES:
                entries.append(f"  ... (truncated at {_MAX_LS_ENTRIES} entries)")
                break
    except PermissionError:
        return f"Error: permission denied: {base}"

    if not entries:
        return f"{base}/  (empty)"

    return f"{base}/\n" + "\n".join(entries)


# -- Helpers -----------------------------------------------------------------


def _normalize_tool_path(path: str) -> Path:
    raw = path.strip()
    # LLMs often emit "/home/user" as a generic Linux placeholder path.
    # Normalize that exact placeholder to this machine's real home directory
    # so tool calls remain usable without leaking hardcoded usernames.
    if raw == "/home/user":
        raw = str(Path.home())
    elif raw.startswith("/home/user/"):
        raw = str(Path.home() / raw.removeprefix("/home/user/"))
    return Path(raw).expanduser()


def _is_supported_text_file(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in _TEXT_FILENAMES:
        return True
    if name.startswith(".env"):
        return True
    return suffix in _TEXT_EXTENSIONS
