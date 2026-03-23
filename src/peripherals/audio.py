from __future__ import annotations

import signal
import subprocess
import tempfile
from pathlib import Path

from .system import resolve_binary, run_command
from .types import (
    CommandResult,
    CommandRunner,
    Observation,
    ObservationModality,
    PeripheralDevice,
    PeripheralKind,
    PeripheralTransport,
    WhichResolver,
)


def record_clip(
    device: PeripheralDevice,
    *,
    duration_seconds: int = 5,
    output_path: str | Path | None = None,
    runner: CommandRunner | None = None,
    which: WhichResolver | None = None,
) -> Observation:
    if device.kind is not PeripheralKind.MICROPHONE:
        raise ValueError("record_clip requires a microphone device")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")

    runner = runner or _run
    which = which or resolve_binary
    target = Path(output_path) if output_path is not None else _default_recording_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    command, backend = _record_command(device, target=target, duration_seconds=duration_seconds, which=which)
    result = _run_record_command(
        command,
        backend=backend,
        duration_seconds=duration_seconds,
        target=target,
        runner=runner,
    )
    if not result.ok:
        detail = result.stderr.strip()
        raise RuntimeError(f"audio capture failed: {detail or 'unknown error'}")

    return Observation(
        source_id=device.id,
        source_type=device.kind.value,
        transport=device.transport.value,
        modality=ObservationModality.AUDIO_CLIP,
        summary=f"Recorded {duration_seconds}s audio clip from {device.label}",
        payload_ref=str(target),
        metadata={"backend": backend, "duration_seconds": duration_seconds, "device_path": device.path},
    )


def _record_command(
    device: PeripheralDevice,
    *,
    target: Path,
    duration_seconds: int,
    which: WhichResolver,
) -> tuple[tuple[str, ...], str]:
    arecord = which("arecord")
    if device.transport is PeripheralTransport.ALSA and arecord:
        return (
            (
                arecord,
                "-q",
                "-d",
                str(duration_seconds),
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-D",
                device.path or "default",
                str(target),
            ),
            "arecord",
        )
    parecord = which("parecord")
    if parecord and device.transport is PeripheralTransport.AUDIO_SERVER:
        return (
            (
                parecord,
                "--device",
                device.path or "default",
                "--file-format=wav",
                str(target),
            ),
            "parecord",
        )
    ffmpeg = which("ffmpeg")
    if ffmpeg and device.transport in {PeripheralTransport.AUDIO_SERVER, PeripheralTransport.ALSA}:
        input_format = "pulse" if device.transport is PeripheralTransport.AUDIO_SERVER else "alsa"
        return (
            (
                ffmpeg,
                "-y",
                "-f",
                input_format,
                "-t",
                str(duration_seconds),
                "-i",
                device.path or "default",
                str(target),
            ),
            "ffmpeg",
        )
    raise RuntimeError("no supported audio capture backend found")


def _run_record_command(
    command: tuple[str, ...],
    *,
    backend: str,
    duration_seconds: int,
    target: Path,
    runner: CommandRunner,
) -> CommandResult:
    if backend == "parecord" and runner is _run:
        return _run_timed_capture(command, duration_seconds=duration_seconds, target=target)
    return runner(command)


def _run_timed_capture(
    args: tuple[str, ...],
    *,
    duration_seconds: int,
    target: Path,
) -> CommandResult:
    proc = subprocess.Popen(
        list(args),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr = ""
    returncode = 0
    try:
        proc.wait(timeout=duration_seconds)
        returncode = proc.returncode or 0
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        returncode = proc.returncode or 0
    finally:
        if proc.stderr is not None:
            stderr = proc.stderr.read() or ""
            proc.stderr.close()
    if returncode == 0 and target.is_file() and target.stat().st_size <= 44:
        returncode = 1
        if not stderr.strip():
            stderr = "capture produced an empty audio file"
    return CommandResult(args=args, returncode=returncode, stderr=stderr)


def _default_recording_path() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="openjet-audio-", suffix=".wav", delete=False)
    handle.close()
    return Path(handle.name)


def _run(args):
    return run_command(args, timeout_seconds=15)
