"""Guarded executor: runs shell commands only after explicit approval."""

from __future__ import annotations

import asyncio
import ast
import difflib
import fnmatch
import os
import re
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path

from .runtime_limits import (
    MIN_TOKEN_BUDGET,
    derive_file_token_budget,
    estimate_tokens,
    read_memory_snapshot,
)
from .shell_targets import resolve_shell_target, run_shell_via_target


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


@dataclass(frozen=True)
class EditFileResult:
    ok: bool
    output: str
    internal_retry: bool = False
    replacements: int = 0
    match_strategy: str | None = None
    validation_error: str | None = None


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
_SEARCH_BLOCK_HEADER = "<<<<<<< SEARCH"
_SEARCH_BLOCK_DIVIDER = "======="
_SEARCH_BLOCK_FOOTER = ">>>>>>> REPLACE"
_FUZZY_MATCH_THRESHOLD = 0.92
_FUZZY_AMBIGUITY_DELTA = 0.02


# -- Shell executor ----------------------------------------------------------


async def run_shell(
    command: str,
    timeout_seconds: int = DEFAULT_SHELL_TIMEOUT_SECONDS,
    *,
    target: str | None = None,
) -> ExecResult:
    """Run a shell command and capture output. Caller must gate on approval first."""
    if timeout_seconds <= 0:
        timeout_seconds = DEFAULT_SHELL_TIMEOUT_SECONDS
    shell_target = resolve_shell_target(target)
    if shell_target is not None:
        exit_code, stdout, stderr, timed_out = await run_shell_via_target(
            command,
            target=shell_target,
            timeout_seconds=timeout_seconds,
        )
        if timed_out:
            exit_code = SHELL_TIMEOUT_EXIT_CODE
        return ExecResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            timeout_seconds=timeout_seconds,
        )
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


async def read_system_info(scope: str | None = None) -> str:
    """Return a compact local system summary for agent planning."""
    normalized_scope = (scope or "summary").strip().lower() or "summary"
    if normalized_scope not in {"summary", "memory", "gpu", "disk", "all"}:
        return "Error: invalid arguments for system_info (scope must be summary, memory, gpu, disk, or all)"
    lines = ["SYSTEM INFO", f"cwd: {Path.cwd()}"]

    if normalized_scope in {"summary", "memory", "all"}:
        mem = read_memory_snapshot()
        if mem is None:
            lines.append("memory: unavailable")
        else:
            lines.append(
                "memory: "
                f"available={mem.available_mb:.0f}MB total={mem.total_mb:.0f}MB used={mem.used_percent:.1f}%"
            )

    if normalized_scope in {"summary", "disk", "all"}:
        try:
            usage = shutil.disk_usage(Path.cwd())
            gb = 1024 ** 3
            lines.append(
                "disk: "
                f"free={usage.free / gb:.1f}GB total={usage.total / gb:.1f}GB"
            )
        except OSError:
            lines.append("disk: unavailable")

    if normalized_scope in {"summary", "all"} and hasattr(os, "getloadavg"):
        try:
            load1, load5, load15 = os.getloadavg()
            lines.append(f"load: 1m={load1:.2f} 5m={load5:.2f} 15m={load15:.2f}")
        except OSError:
            lines.append("load: unavailable")

    if normalized_scope in {"summary", "gpu", "all"}:
        lines.extend(await _gpu_info_lines())

    return "\n".join(lines)


async def _gpu_info_lines() -> list[str]:
    tool = shutil.which("nvidia-smi")
    if not tool:
        if Path("/dev/nvhost-gpu").exists():
            return ["gpu: detected (Jetson/unified memory), dedicated VRAM query unavailable"]
        return ["gpu: unavailable"]

    try:
        proc = await asyncio.create_subprocess_exec(
            tool,
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=3)
    except (OSError, asyncio.TimeoutError):
        return ["gpu: query failed"]

    if proc.returncode != 0:
        stderr = stderr_bytes.decode(errors="replace").strip()
        detail = stderr or f"exit {proc.returncode}"
        return [f"gpu: query failed ({detail})"]

    lines = [line.strip() for line in stdout_bytes.decode(errors="replace").splitlines() if line.strip()]
    if not lines:
        return ["gpu: unavailable"]

    rendered: list[str] = []
    for index, line in enumerate(lines, start=1):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            rendered.append(f"gpu[{index}]: {line}")
            continue
        name, total_mb, used_mb, free_mb = parts
        rendered.append(
            f"gpu[{index}]: {name} total={total_mb}MB used={used_mb}MB free={free_mb}MB"
        )
    return rendered


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


async def edit_file(
    path: str,
    old_string: str | None = None,
    new_string: str | None = None,
    replace_all: bool = False,
    *,
    patch: str | None = None,
    return_result: bool = False,
) -> EditFileResult | str:
    """Apply a SEARCH/REPLACE patch or legacy exact string replacement."""
    raw_path = path.strip()
    if not raw_path:
        result = EditFileResult(ok=False, output="Error: path is empty.")
        return result if return_result else result.output

    p = _normalize_tool_path(raw_path)
    if not p.exists():
        result = EditFileResult(ok=False, output=f"Error: file not found: {path}")
        return result if return_result else result.output
    if p.is_dir():
        result = EditFileResult(ok=False, output=f"Error: path is a directory: {path}")
        return result if return_result else result.output

    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        result = EditFileResult(ok=False, output=f"Error reading {path}: {e}")
        return result if return_result else result.output

    try:
        if patch is not None:
            new_content, replacements, match_strategy = _apply_search_replace_blocks(content, patch)
        else:
            if old_string is None or new_string is None:
                result = EditFileResult(
                    ok=False,
                    output="Error: edit_file requires either patch or old_string/new_string.",
                )
                return result if return_result else result.output
            new_content, replacements, match_strategy = _apply_legacy_string_edit(
                content,
                old_string,
                new_string,
                replace_all=replace_all,
                path=path,
            )
    except ValueError as exc:
        result = EditFileResult(ok=False, output=f"Error: {exc}")
        return result if return_result else result.output

    validation_error = _validate_edited_content(p, new_content)
    if validation_error:
        result = EditFileResult(
            ok=False,
            output=validation_error,
            internal_retry=True,
            replacements=replacements,
            match_strategy=match_strategy,
            validation_error=validation_error,
        )
        return result if return_result else result.output

    try:
        p.write_text(new_content, encoding="utf-8")
    except Exception as e:
        result = EditFileResult(ok=False, output=f"Error writing {path}: {e}")
        return result if return_result else result.output

    strategy_suffix = f" via {match_strategy}" if match_strategy else ""
    result = EditFileResult(
        ok=True,
        output=f"Edited {path}: {replacements} replacement(s) made{strategy_suffix}.",
        replacements=replacements,
        match_strategy=match_strategy,
    )
    return result if return_result else result.output


def _apply_legacy_string_edit(
    content: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool,
    path: str,
) -> tuple[str, int, str]:
    if old_string not in content:
        raise ValueError(f"old_string not found in {path}")

    if not replace_all:
        count = content.count(old_string)
        if count > 1:
            raise ValueError(
                f"old_string appears {count} times in {path}. "
                "Provide more surrounding context to make it unique, or set replace_all=true."
            )
        return content.replace(old_string, new_string, 1), 1, "exact"

    replacements = content.count(old_string)
    return content.replace(old_string, new_string), replacements, "exact"


def _apply_search_replace_blocks(content: str, patch: str) -> tuple[str, int, str]:
    blocks = _parse_search_replace_blocks(patch)
    new_content = content
    strategies: list[str] = []

    for search_text, replace_text in blocks:
        new_content, strategy = _replace_search_block(new_content, search_text, replace_text)
        strategies.append(strategy)

    match_strategy = "exact"
    if any(strategy == "fuzzy" for strategy in strategies):
        match_strategy = "fuzzy"
    elif any(strategy == "line-normalized" for strategy in strategies):
        match_strategy = "line-normalized"
    return new_content, len(blocks), match_strategy


def _parse_search_replace_blocks(patch: str) -> list[tuple[str, str]]:
    lines = patch.splitlines(keepends=True)
    blocks: list[tuple[str, str]] = []
    idx = 0

    while idx < len(lines):
        if not lines[idx].strip():
            idx += 1
            continue
        if lines[idx].rstrip("\r\n") != _SEARCH_BLOCK_HEADER:
            raise ValueError(
                "patch must use strict SEARCH/REPLACE blocks with "
                f"{_SEARCH_BLOCK_HEADER}, {_SEARCH_BLOCK_DIVIDER}, {_SEARCH_BLOCK_FOOTER}."
            )
        idx += 1
        search_lines: list[str] = []
        while idx < len(lines) and lines[idx].rstrip("\r\n") != _SEARCH_BLOCK_DIVIDER:
            search_lines.append(lines[idx])
            idx += 1
        if idx >= len(lines):
            raise ValueError("patch is missing the ======= divider.")
        idx += 1
        replace_lines: list[str] = []
        while idx < len(lines) and lines[idx].rstrip("\r\n") != _SEARCH_BLOCK_FOOTER:
            replace_lines.append(lines[idx])
            idx += 1
        if idx >= len(lines):
            raise ValueError("patch is missing the >>>>>>> REPLACE footer.")
        idx += 1
        search_text = "".join(search_lines)
        if not search_text:
            raise ValueError("SEARCH blocks must not be empty.")
        blocks.append((search_text, "".join(replace_lines)))

    if not blocks:
        raise ValueError("patch did not contain any SEARCH/REPLACE blocks.")
    return blocks


def _replace_search_block(content: str, search_text: str, replace_text: str) -> tuple[str, str]:
    exact_count = content.count(search_text)
    if exact_count == 1:
        return content.replace(search_text, replace_text, 1), "exact"
    if exact_count > 1:
        raise ValueError(
            "SEARCH block matched multiple locations exactly. Add more surrounding context to make it unique."
        )

    line_match = _find_line_normalized_match(content, search_text)
    if line_match is not None:
        start, end = line_match
        return content[:start] + replace_text + content[end:], "line-normalized"

    fuzzy_match = _find_fuzzy_line_match(content, search_text)
    if fuzzy_match is not None:
        start, end = fuzzy_match
        return content[:start] + replace_text + content[end:], "fuzzy"

    raise ValueError("SEARCH block did not match the target file.")


def _find_line_normalized_match(content: str, search_text: str) -> tuple[int, int] | None:
    content_lines = content.splitlines(keepends=True)
    search_lines = search_text.splitlines(keepends=True)
    if not content_lines or not search_lines or len(search_lines) > len(content_lines):
        return None

    wanted = [_normalize_match_line(line) for line in search_lines]
    matches: list[tuple[int, int]] = []
    for start_idx in range(len(content_lines) - len(search_lines) + 1):
        window = content_lines[start_idx:start_idx + len(search_lines)]
        if [_normalize_match_line(line) for line in window] != wanted:
            continue
        matches.append((_line_offset(content_lines, start_idx), _line_offset(content_lines, start_idx + len(search_lines))))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "SEARCH block matched multiple locations after line normalization. Add more surrounding context."
        )
    return None


def _find_fuzzy_line_match(content: str, search_text: str) -> tuple[int, int] | None:
    content_lines = content.splitlines(keepends=True)
    search_lines = search_text.splitlines(keepends=True)
    if not content_lines or not search_lines:
        return None

    target = _normalize_match_block(search_text)
    preferred_len = len(search_lines)
    candidate_lengths = sorted({max(1, preferred_len - 1), preferred_len, preferred_len + 1})
    scored: list[tuple[float, int, int]] = []

    for window_len in candidate_lengths:
        if window_len > len(content_lines):
            continue
        for start_idx in range(len(content_lines) - window_len + 1):
            window_text = "".join(content_lines[start_idx:start_idx + window_len])
            ratio = difflib.SequenceMatcher(None, _normalize_match_block(window_text), target).ratio()
            scored.append((ratio, start_idx, window_len))

    if not scored:
        return None

    scored.sort(reverse=True)
    best_ratio, best_start, best_len = scored[0]
    if best_ratio < _FUZZY_MATCH_THRESHOLD:
        return None
    if len(scored) > 1 and (best_ratio - scored[1][0]) < _FUZZY_AMBIGUITY_DELTA:
        raise ValueError(
            "SEARCH block fuzzy-matched multiple locations. Add more surrounding context to make it unique."
        )

    return (
        _line_offset(content_lines, best_start),
        _line_offset(content_lines, best_start + best_len),
    )


def _line_offset(lines: list[str], line_index: int) -> int:
    return sum(len(line) for line in lines[:line_index])


def _normalize_match_line(line: str) -> str:
    return line.rstrip().replace("\r\n", "\n")


def _normalize_match_block(text: str) -> str:
    return "\n".join(part.rstrip() for part in text.replace("\r\n", "\n").strip("\n").split("\n"))


def _validate_edited_content(path: Path, content: str) -> str | None:
    if path.suffix.lower() != ".py":
        return None

    try:
        ast.parse(content, filename=str(path))
    except SyntaxError as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return (
            f"Edit validation failed for {path}.\n"
            "Python AST validation failed; the file was not written.\n"
            "Fix the SEARCH/REPLACE block and try again.\n"
            f"Traceback:\n{tb}"
        )
    return None


# -- Glob tool ---------------------------------------------------------------

_DEFAULT_GLOB_IGNORE = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs", "*.egg-info",
}
_DEFAULT_TOOL_STATE_DIRS = {".openjet", "session_logs"}
_DEFAULT_TOOL_STATE_FILES = {"session_state.json"}

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
            if _should_skip_tool_path(match, base):
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
        if _should_skip_tool_path(fp, base):
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
            if _should_skip_tool_path(item, base):
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


def _path_points_inside_tool_state(path: Path) -> bool:
    return any(part in _DEFAULT_TOOL_STATE_DIRS for part in path.parts) or path.name in _DEFAULT_TOOL_STATE_FILES


def _should_skip_tool_path(path: Path, base: Path) -> bool:
    if _path_points_inside_tool_state(base):
        return False

    try:
        rel_parts = path.relative_to(base).parts if base.is_dir() else (path.name,)
    except ValueError:
        rel_parts = path.parts

    if any(part in _DEFAULT_GLOB_IGNORE for part in rel_parts):
        return True
    if any(fnmatch.fnmatch(part, pattern) for part in rel_parts for pattern in _DEFAULT_GLOB_IGNORE):
        return True
    if any(part in _DEFAULT_TOOL_STATE_DIRS for part in rel_parts):
        return True
    if path.name in _DEFAULT_TOOL_STATE_FILES:
        return True
    return False


def _is_supported_text_file(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in _TEXT_FILENAMES:
        return True
    if name.startswith(".env"):
        return True
    return suffix in _TEXT_EXTENSIONS
