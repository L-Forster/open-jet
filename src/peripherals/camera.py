from __future__ import annotations

import tempfile
from pathlib import Path

from .system import resolve_binary, run_command
from .types import CommandRunner, Observation, ObservationModality, PeripheralDevice, PeripheralKind, WhichResolver


def capture_snapshot(
    device: PeripheralDevice,
    *,
    output_path: str | Path | None = None,
    runner: CommandRunner | None = None,
    which: WhichResolver | None = None,
) -> Observation:
    if device.kind is not PeripheralKind.CAMERA:
        raise ValueError("capture_snapshot requires a camera device")
    if not device.path:
        raise ValueError("camera device is missing a capture path")

    runner = runner or _run
    which = which or resolve_binary
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for camera capture")

    target = Path(output_path) if output_path is not None else _default_snapshot_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    result = runner(
        (
            ffmpeg,
            "-y",
            "-f",
            "video4linux2",
            "-i",
            device.path,
            "-frames:v",
            "1",
            str(target),
        )
    )
    if not result.ok:
        detail = result.stderr.strip()
        raise RuntimeError(f"camera capture failed: {detail or 'unknown error'}")

    return Observation(
        source_id=device.id,
        source_type=device.kind.value,
        transport=device.transport.value,
        modality=ObservationModality.IMAGE,
        summary=f"Captured image from {device.label}",
        payload_ref=str(target),
        metadata={"device_path": device.path, "backend": "ffmpeg"},
    )


def _default_snapshot_path() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="openjet-camera-", suffix=".jpg", delete=False)
    handle.close()
    return Path(handle.name)


def _run(args):
    return run_command(args, timeout_seconds=15)
