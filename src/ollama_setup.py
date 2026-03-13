from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from rich.markup import escape

from .hardware import running_on_jetson

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_PULL_PERCENT_RE = re.compile(r"(?P<pct>\d{1,3})%")
_PULL_SIZE_RE = re.compile(
    r"(?P<done>\d+(?:\.\d+)?)\s*(?P<done_unit>[KMGTP]B)\s*/\s*(?P<total>\d+(?:\.\d+)?)\s*(?P<total_unit>[KMGTP]B)",
    flags=re.IGNORECASE,
)
_PULL_SPEED_RE = re.compile(r"(?P<speed>\d+(?:\.\d+)?)\s*(?P<speed_unit>[KMGTP]B/s)", flags=re.IGNORECASE)
_PULL_ETA_RE = re.compile(r"(?P<eta>(?:\d+h)?(?:\d+m)?(?:\d+s)|\d+ms)$", flags=re.IGNORECASE)


def find_ollama_cli() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    # Common install locations when PATH isn't inherited by TUI launchers.
    for candidate in (
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        "/opt/homebrew/bin/ollama",
        "/snap/bin/ollama",
    ):
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def discover_installed_ollama_models() -> list[str]:
    ollama_cli = find_ollama_cli()
    if not ollama_cli:
        return []

    try:
        run_result = subprocess.run(
            [ollama_cli, "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if run_result.returncode != 0:
        return []

    found: set[str] = set()
    for raw_line in run_result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("NAME") or upper.startswith("MODEL"):
            continue
        tag = line.split()[0].strip()
        if ":" in tag:
            found.add(tag)
    return sorted(found)


async def run_command_capture(*args: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_raw, err_raw = await proc.communicate()
    return (proc.returncode or 0, out_raw.decode("utf-8", errors="ignore"), err_raw.decode("utf-8", errors="ignore"))


async def run_command_stream(
    *args: str,
    on_chunk: Callable[[str, bool], None] | None = None,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_chunks: list[str] = []
    err_chunks: list[str] = []

    async def _pump(stream: asyncio.StreamReader | None, sink: list[str], is_stderr: bool) -> None:
        if stream is None:
            return
        while True:
            raw = await stream.read(4096)
            if not raw:
                break
            text = raw.decode("utf-8", errors="ignore")
            sink.append(text)
            if on_chunk:
                on_chunk(text, is_stderr)

    await asyncio.gather(
        _pump(proc.stdout, out_chunks, False),
        _pump(proc.stderr, err_chunks, True),
    )
    rc = await proc.wait()
    return rc, "".join(out_chunks), "".join(err_chunks)


async def resolve_ollama_model_file(ollama_model: str) -> str:
    ollama_cli = find_ollama_cli()
    if not ollama_cli:
        raise RuntimeError("`ollama` CLI is not installed or not discoverable.")
    rc, out, err = await run_command_capture(ollama_cli, "show", ollama_model, "--modelfile")
    if rc != 0:
        detail = (err or out).strip()[:500]
        raise RuntimeError(f"Unable to inspect pulled model '{ollama_model}': {detail or 'unknown error'}")

    from_ref = ""
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            from_ref = stripped.split(maxsplit=1)[1].strip()
            break
    if not from_ref:
        raise RuntimeError("Could not resolve Ollama model file path from modelfile output.")

    candidates: list[Path] = []
    ref_path = Path(from_ref).expanduser()
    if ref_path.is_absolute():
        candidates.append(ref_path)
    if from_ref.startswith("sha256:"):
        candidates.append(Path.home() / ".ollama" / "models" / "blobs" / from_ref.replace("sha256:", "sha256-"))
        candidates.append(Path("/usr/share/ollama/.ollama/models/blobs") / from_ref.replace("sha256:", "sha256-"))
    if from_ref.startswith("sha256-"):
        candidates.append(Path.home() / ".ollama" / "models" / "blobs" / from_ref)
        candidates.append(Path("/usr/share/ollama/.ollama/models/blobs") / from_ref)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise RuntimeError(
        "Pulled model was resolved by Ollama, but no local GGUF/blob path could be found for llama-server."
    )


async def read_ollama_model_details(ollama_cli: str, ollama_model: str) -> dict[str, str]:
    rc, out, err = await run_command_capture(ollama_cli, "show", ollama_model, "--json")
    if rc != 0:
        detail = (err or out).strip()
        if "unknown flag: --json" in detail.lower():
            return {}
        short = detail[:400]
        raise RuntimeError(
            f"Unable to inspect pulled model metadata for '{ollama_model}': {short or 'unknown error'}"
        )
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return {}
    details = payload.get("details")
    if not isinstance(details, dict):
        return {}
    extracted: dict[str, str] = {}
    for key in ("format", "family", "parameter_size", "quantization_level"):
        val = details.get(key)
        if isinstance(val, str):
            extracted[key] = val.strip()
    return extracted


def file_is_gguf(path: str) -> bool:
    try:
        with Path(path).open("rb") as fh:
            return fh.read(4) == b"GGUF"
    except OSError:
        return False


def jetson_constraints_required(setup_result: dict) -> bool:
    profile = str(setup_result.get("hardware_profile", "")).strip()
    override = str(setup_result.get("hardware_override", "")).strip()
    if profile == "other" and override.startswith("jetson_"):
        return True
    return running_on_jetson()


async def validate_jetson_ollama_model(
    *,
    setup_result: dict,
    ollama_cli: str,
    ollama_model: str,
    resolved_model: str,
) -> None:
    if not jetson_constraints_required(setup_result):
        return

    if not file_is_gguf(resolved_model):
        raise RuntimeError(
            "Jetson setup requires a quantized GGUF model, but pulled artifact is not GGUF."
        )

    details = await read_ollama_model_details(ollama_cli, ollama_model)
    model_format = details.get("format", "").lower()
    quant = details.get("quantization_level", "").upper()

    if model_format and model_format != "gguf":
        raise RuntimeError(
            f"Jetson setup requires GGUF format, but Ollama reports format '{details.get('format', 'unknown')}'."
        )
    if quant and quant.startswith(("F16", "F32", "BF16")):
        raise RuntimeError(
            "Jetson setup requires a quantized Ollama model (e.g. Q4/Q5 GGUF), not a base/unquantized variant."
        )


def extract_pull_progress(line: str) -> dict[str, str] | None:
    pct_match = _PULL_PERCENT_RE.search(line)
    if not pct_match:
        return None
    pct = max(0, min(100, int(pct_match.group("pct"))))

    size_match = _PULL_SIZE_RE.search(line)
    speed_match = _PULL_SPEED_RE.search(line)
    eta_match = _PULL_ETA_RE.search(line.strip())

    payload: dict[str, str] = {"pct": str(pct)}
    if size_match:
        payload["done"] = f"{size_match.group('done')} {size_match.group('done_unit').upper()}"
        payload["total"] = f"{size_match.group('total')} {size_match.group('total_unit').upper()}"
    if speed_match:
        payload["speed"] = f"{speed_match.group('speed')} {speed_match.group('speed_unit').upper()}"
    if eta_match:
        payload["eta"] = eta_match.group("eta")
    return payload


def render_progress_bar(percent: int, width: int = 24) -> str:
    clamped = max(0, min(100, percent))
    filled = int(round((clamped / 100) * width))
    return "#" * filled + "-" * (width - filled)


async def materialize_setup_model(
    setup_result: dict,
    log: Any,
    *,
    set_status: Callable[[str], None],
    clear_status: Callable[[], None],
) -> dict:
    if str(setup_result.get("model_source", "local")) != "ollama":
        return setup_result

    ollama_model = str(setup_result.get("ollama_model", "")).strip()
    if not ollama_model:
        raise RuntimeError("Ollama model tag is missing.")
    ollama_cli = find_ollama_cli()
    if not ollama_cli:
        raise RuntimeError("`ollama` CLI is not installed or not discoverable. Install it and retry setup.")

    try:
        resolved_model = await resolve_ollama_model_file(ollama_model)
        await validate_jetson_ollama_model(
            setup_result=setup_result,
            ollama_cli=ollama_cli,
            ollama_model=ollama_model,
            resolved_model=resolved_model,
        )
        log.write(f"[bold bright_white]Using installed Ollama model {escape(ollama_model)}.[/]")
        merged = dict(setup_result)
        merged["model"] = resolved_model
        return merged
    except Exception:
        pass

    log.write(
        f"[bold bright_white]Pulling {escape(ollama_model)} from Ollama...[/] "
        "[dim](this can take several minutes for larger models)[/]"
    )

    progress_buffer = ""
    last_percent = -1
    last_rendered = ""
    last_emit_t = 0.0
    last_status = ""

    def _set_pull_status(text: str) -> None:
        set_status(f"[bold #88D83F]{escape(text)}[/]")

    def _emit_progress(text: str, _is_stderr: bool) -> None:
        nonlocal progress_buffer, last_percent, last_rendered, last_emit_t, last_status
        progress_buffer += _ANSI_ESCAPE_RE.sub("", text)
        chunks = re.split(r"[\r\n]+", progress_buffer)
        progress_buffer = chunks.pop() if chunks else ""

        for raw_line in chunks:
            line = raw_line.strip()
            if not line:
                continue

            parsed = extract_pull_progress(line)
            now = time.monotonic()
            if parsed:
                pct = int(parsed["pct"])
                if pct < last_percent and pct < 5:
                    last_percent = -1
                if pct <= last_percent and (now - last_emit_t) < 1.0:
                    continue
                last_percent = pct
                bar = render_progress_bar(pct)
                detail = f"{parsed.get('done', '?')}/{parsed.get('total', '?')}"
                speed = parsed.get("speed", "?")
                eta = parsed.get("eta", "?")
                rendered = f"pull {pct:3d}% |{bar}| {detail} {speed} ETA {eta}"
                if rendered != last_rendered or (now - last_emit_t) >= 1.0:
                    _set_pull_status(rendered)
                    last_rendered = rendered
                    last_emit_t = now
                continue

            line_low = line.lower()
            if (
                line_low.startswith("pulling")
                or line_low.startswith("verifying")
                or line_low.startswith("processing")
                or line_low.startswith("writing")
                or line_low.startswith("success")
            ) and line != last_status:
                _set_pull_status(line)
                last_status = line
                last_emit_t = now

    try:
        rc, out, err = await run_command_stream(ollama_cli, "pull", ollama_model, on_chunk=_emit_progress)
        if rc != 0:
            detail = (err or out).strip()[:700]
            raise RuntimeError(f"Ollama pull failed for '{ollama_model}': {detail or 'unknown error'}")
        resolved_model = await resolve_ollama_model_file(ollama_model)
        await validate_jetson_ollama_model(
            setup_result=setup_result,
            ollama_cli=ollama_cli,
            ollama_model=ollama_model,
            resolved_model=resolved_model,
        )
        log.write(f"[bold bright_white]Pulled {escape(ollama_model)} and resolved local model file.[/]")
    finally:
        clear_status()

    merged = dict(setup_result)
    merged["model"] = resolved_model
    return merged
