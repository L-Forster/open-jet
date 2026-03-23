from __future__ import annotations

import tempfile
from pathlib import Path

from .system import resolve_binary, run_command
from .types import (
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
    result = runner(command)
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


def _default_recording_path() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="openjet-audio-", suffix=".wav", delete=False)
    handle.close()
    return Path(handle.name)


def _run(args):
    return run_command(args, timeout_seconds=15)
