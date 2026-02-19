"""Guarded executor: runs shell commands only after explicit approval."""

from __future__ import annotations

import asyncio
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
