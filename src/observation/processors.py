from __future__ import annotations

import math
import wave
from importlib import import_module
from pathlib import Path
from typing import Mapping

from ..hardware import detect_hardware_info
from ..peripherals.system import resolve_binary, run_command
from ..peripherals.types import CommandRunner, Observation, ObservationModality, WhichResolver
from .store import ObservationStore


_FASTER_WHISPER_MODELS: dict[tuple[str, str, str], object] = {}
_FASTER_WHISPER_DOWNLOAD_ROOT = (Path.home() / ".openjet" / "models" / "faster-whisper").expanduser()


def save_frame_observation(observation: Observation, *, store: ObservationStore) -> Observation:
    if observation.modality is not ObservationModality.IMAGE:
        raise ValueError("save_frame_observation requires an image observation")
    if not observation.payload_ref:
        raise ValueError("image observation is missing a payload_ref")
    return store.persist(observation, copy_payload=True)


def detect_speech_activity(
    observation: Observation,
    *,
    energy_threshold: int = 500,
    min_active_ratio: float = 0.05,
    window_ms: int = 30,
    store: ObservationStore | None = None,
) -> Observation:
    if observation.modality is not ObservationModality.AUDIO_CLIP:
        raise ValueError("detect_speech_activity requires an audio_clip observation")
    if not observation.payload_ref:
        raise ValueError("audio observation is missing a payload_ref")

    active_ratio, peak_rms = _analyze_wav_activity(
        Path(observation.payload_ref),
        energy_threshold=max(1, int(energy_threshold)),
        window_ms=max(10, int(window_ms)),
    )
    speech_detected = active_ratio >= max(0.0, float(min_active_ratio))
    summary = (
        f"Speech detected on {observation.source_id} "
        f"(active_ratio={active_ratio:.2f}, peak_rms={peak_rms})"
        if speech_detected
        else f"No speech detected on {observation.source_id} (peak_rms={peak_rms})"
    )
    processed = Observation(
        source_id=observation.source_id,
        source_type=observation.source_type,
        transport=observation.transport,
        modality=ObservationModality.TEXT,
        summary=summary,
        timestamp=observation.timestamp,
        payload_ref=observation.payload_ref,
        metadata={
            **dict(observation.metadata),
            "speech_detected": speech_detected,
            "active_ratio": round(active_ratio, 4),
            "peak_rms": peak_rms,
            "source_modality": observation.modality.value,
        },
        changed=speech_detected,
    )
    return store.persist(processed) if store else processed


def process_audio_observation(
    observation: Observation,
    *,
    store: ObservationStore | None = None,
    transcription_cfg: Mapping[str, object] | None = None,
    runner: CommandRunner | None = None,
    which: WhichResolver | None = None,
) -> Observation:
    transcribed = transcribe_audio_clip(
        observation,
        store=store,
        transcription_cfg=transcription_cfg,
        runner=runner,
        which=which,
    )
    if transcribed is not None:
        return transcribed
    return detect_speech_activity(observation, store=store)


def provision_default_faster_whisper_model() -> bool:
    try:
        module = import_module("faster_whisper")
        model_cls = getattr(module, "WhisperModel")
    except Exception:
        return False

    settings = _default_transcription_settings()
    loaded = _load_faster_whisper_model(
        model_cls,
        model_name=str(settings["model"]),
        device=str(settings["device"]),
        compute_type=str(settings["compute_type"]),
    )
    return loaded is not None


def transcribe_audio_clip(
    observation: Observation,
    *,
    store: ObservationStore | None = None,
    transcription_cfg: Mapping[str, object] | None = None,
    runner: CommandRunner | None = None,
    which: WhichResolver | None = None,
) -> Observation | None:
    if observation.modality is not ObservationModality.AUDIO_CLIP:
        raise ValueError("transcribe_audio_clip requires an audio_clip observation")
    if not observation.payload_ref:
        raise ValueError("audio observation is missing a payload_ref")

    settings = _normalize_transcription_settings(transcription_cfg)
    if not settings["enabled"]:
        return None

    transcript = _transcribe_with_preferred_backend(
        observation,
        settings=settings,
        runner=runner,
        which=which,
    )
    if not transcript:
        return None

    payload_ref = None
    if store is not None:
        payload_ref = str(
            store.append_text_buffer(
                observation.source_id,
                f"{observation.timestamp.isoformat()} | {transcript}",
                buffer_name="microphone-transcript.txt",
                max_lines=400,
            )
        )

    excerpt = transcript[:140]
    if len(transcript) > 140:
        excerpt = f"{excerpt.rstrip()}..."
    processed = Observation(
        source_id=observation.source_id,
        source_type=observation.source_type,
        transport=observation.transport,
        modality=ObservationModality.TEXT,
        summary=f"Transcript from {observation.source_id}: {excerpt}",
        timestamp=observation.timestamp,
        payload_ref=payload_ref,
        metadata={
            **dict(observation.metadata),
            "speech_detected": True,
            "transcription_backend": str(settings["resolved_backend"]),
            "transcript_text": transcript,
            "audio_payload_ref": observation.payload_ref,
            "source_modality": observation.modality.value,
        },
        changed=True,
    )
    return store.persist(processed) if store else processed


def append_gpio_text_buffer(
    observation: Observation,
    *,
    store: ObservationStore,
    buffer_name: str = "gpio-buffer.txt",
    max_lines: int = 200,
) -> Observation:
    if observation.modality is not ObservationModality.STRUCTURED_STATE:
        raise ValueError("append_gpio_text_buffer requires a structured_state observation")
    values = observation.metadata.get("values")
    if not isinstance(values, dict):
        raise ValueError("structured_state observation is missing metadata.values")

    line = f"{observation.timestamp.isoformat()} | {observation.summary}"
    buffer_path = store.append_text_buffer(
        observation.source_id,
        line,
        buffer_name=buffer_name,
        max_lines=max_lines,
    )
    processed = Observation(
        source_id=observation.source_id,
        source_type=observation.source_type,
        transport=observation.transport,
        modality=ObservationModality.TEXT,
        summary=f"GPIO buffer updated for {observation.source_id}",
        timestamp=observation.timestamp,
        payload_ref=str(buffer_path),
        metadata={
            **dict(observation.metadata),
            "buffer_path": str(buffer_path),
            "buffer_name": buffer_name,
            "source_modality": observation.modality.value,
        },
        changed=observation.changed,
    )
    return store.persist(processed)


def _analyze_wav_activity(path: Path, *, energy_threshold: int, window_ms: int) -> tuple[float, int]:
    with wave.open(str(path), "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frames_per_window = max(1, int(frame_rate * (window_ms / 1000.0)))
        bytes_per_window = frames_per_window * sample_width * channels
        active_windows = 0
        total_windows = 0
        peak_rms = 0

        while True:
            chunk = wav_file.readframes(frames_per_window)
            if not chunk:
                break
            total_windows += 1
            rms = _pcm_rms(chunk, sample_width=sample_width)
            peak_rms = max(peak_rms, int(rms))
            if rms >= energy_threshold:
                active_windows += 1

    if total_windows == 0:
        return 0.0, 0
    return active_windows / total_windows, peak_rms


def _pcm_rms(chunk: bytes, *, sample_width: int) -> int:
    if sample_width <= 0:
        raise ValueError("sample_width must be positive")
    sample_count = len(chunk) // sample_width
    if sample_count == 0:
        return 0

    total_energy = 0
    for index in range(0, sample_count * sample_width, sample_width):
        sample = _decode_pcm_sample(chunk[index : index + sample_width], sample_width=sample_width)
        total_energy += sample * sample
    return int(math.sqrt(total_energy / sample_count))


def _decode_pcm_sample(sample_bytes: bytes, *, sample_width: int) -> int:
    if sample_width == 1:
        return int(sample_bytes[0]) - 128
    return int.from_bytes(sample_bytes, byteorder="little", signed=True)


def _normalize_transcription_settings(
    transcription_cfg: Mapping[str, object] | None,
) -> dict[str, object]:
    raw = dict(transcription_cfg or {})
    defaults = _default_transcription_settings()
    extra_args = raw.get("extra_args")
    normalized_extra_args = extra_args if isinstance(extra_args, list) else []
    language = str(raw.get("language", "auto") or "auto").strip() or "auto"
    return {
        "enabled": bool(raw.get("enabled", True)),
        "backend": str(raw.get("backend", defaults["backend"]) or defaults["backend"]).strip() or defaults["backend"],
        "command": str(raw.get("command", "whisper-cli") or "whisper-cli").strip() or "whisper-cli",
        "model_path": str(raw.get("model_path", "") or "").strip(),
        "model": str(raw.get("model", defaults["model"]) or defaults["model"]).strip() or defaults["model"],
        "device": str(raw.get("device", defaults["device"]) or defaults["device"]).strip() or defaults["device"],
        "compute_type": str(raw.get("compute_type", defaults["compute_type"]) or defaults["compute_type"]).strip() or defaults["compute_type"],
        "beam_size": int(raw.get("beam_size", 1) or 1),
        "vad_filter": bool(raw.get("vad_filter", True)),
        "language": language,
        "translate": bool(raw.get("translate", False)),
        "extra_args": normalized_extra_args,
        "resolved_backend": "",
    }


def _normalize_transcript_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines).strip()


def _run(args: tuple[str, ...]):
    return run_command(args, timeout_seconds=120)


def _default_transcription_settings() -> dict[str, str]:
    hardware = detect_hardware_info()
    if hardware.has_cuda:
        return {
            "backend": "faster_whisper",
            "model": "base",
            "device": "cuda",
            "compute_type": "int8_float16",
        }
    return {
        "backend": "faster_whisper",
        "model": "tiny",
        "device": "cpu",
        "compute_type": "int8",
    }


def _transcribe_with_preferred_backend(
    observation: Observation,
    *,
    settings: dict[str, object],
    runner: CommandRunner | None,
    which: WhichResolver | None,
) -> str | None:
    backend = str(settings["backend"]).strip().lower() or "faster_whisper"
    if backend in {"auto", "faster_whisper"}:
        transcript = _transcribe_with_faster_whisper(observation, settings=settings)
        if transcript:
            settings["resolved_backend"] = "faster_whisper"
            return transcript
        if backend == "faster_whisper":
            return None
    if backend in {"auto", "whisper_cpp"}:
        transcript = _transcribe_with_whisper_cpp(
            observation,
            settings=settings,
            runner=runner,
            which=which,
        )
        if transcript:
            settings["resolved_backend"] = "whisper_cpp"
            return transcript
    return None


def _transcribe_with_faster_whisper(
    observation: Observation,
    *,
    settings: Mapping[str, object],
) -> str | None:
    try:
        module = import_module("faster_whisper")
        model_cls = getattr(module, "WhisperModel")
    except Exception:
        return None

    model_name = str(settings["model"]).strip()
    device = str(settings["device"]).strip()
    compute_type = str(settings["compute_type"]).strip()
    model = _load_faster_whisper_model(
        model_cls,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    if model is None:
        return None

    kwargs: dict[str, object] = {
        "beam_size": max(1, int(settings["beam_size"])),
        "vad_filter": bool(settings["vad_filter"]),
        "condition_on_previous_text": False,
    }
    language = str(settings["language"]).strip().lower()
    if language and language != "auto":
        kwargs["language"] = language
    if bool(settings["translate"]):
        kwargs["task"] = "translate"

    try:
        segments, _info = model.transcribe(str(observation.payload_ref), **kwargs)
    except Exception:
        return None

    text_parts = [str(getattr(segment, "text", "") or "").strip() for segment in segments]
    return _normalize_transcript_text("\n".join(part for part in text_parts if part))


def _load_faster_whisper_model(
    model_cls,
    *,
    model_name: str,
    device: str,
    compute_type: str,
):
    key = (model_name, device, compute_type)
    cached = _FASTER_WHISPER_MODELS.get(key)
    if cached is not None:
        return cached

    try:
        loaded = _instantiate_faster_whisper_model(
            model_cls,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )
    except Exception:
        if device != "cuda":
            return None
        try:
            fallback_key = (model_name, "cpu", "int8")
            fallback = _FASTER_WHISPER_MODELS.get(fallback_key)
            if fallback is None:
                fallback = _instantiate_faster_whisper_model(
                    model_cls,
                    model_name=model_name,
                    device="cpu",
                    compute_type="int8",
                )
                _FASTER_WHISPER_MODELS[fallback_key] = fallback
            return fallback
        except Exception:
            return None

    _FASTER_WHISPER_MODELS[key] = loaded
    return loaded


def _instantiate_faster_whisper_model(
    model_cls,
    *,
    model_name: str,
    device: str,
    compute_type: str,
):
    kwargs = _faster_whisper_init_kwargs(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )
    try:
        return model_cls(model_name, **kwargs)
    except TypeError:
        kwargs.pop("download_root", None)
        return model_cls(model_name, **kwargs)


def _faster_whisper_init_kwargs(
    *,
    model_name: str,
    device: str,
    compute_type: str,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "device": device,
        "compute_type": compute_type,
    }
    if _looks_like_filesystem_path(model_name):
        return kwargs
    _FASTER_WHISPER_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    kwargs["download_root"] = str(_FASTER_WHISPER_DOWNLOAD_ROOT)
    return kwargs


def _looks_like_filesystem_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    return text.startswith(("/", ".", "~")) or "/" in text or "\\" in text


def _transcribe_with_whisper_cpp(
    observation: Observation,
    *,
    settings: Mapping[str, object],
    runner: CommandRunner | None,
    which: WhichResolver | None,
) -> str | None:
    model_path = Path(str(settings["model_path"]))
    if not model_path.is_file():
        return None

    runner = runner or _run
    which = which or resolve_binary
    command_ref = str(settings["command"]).strip() or "whisper-cli"
    command_path = which(command_ref)
    if not command_path and Path(command_ref).is_file():
        command_path = command_ref
    if not command_path:
        return None

    command = [
        command_path,
        "-m",
        str(model_path),
        "-f",
        str(observation.payload_ref),
        "-nt",
        "-np",
        "-l",
        str(settings["language"]),
    ]
    if bool(settings["translate"]):
        command.append("-tr")
    for arg in settings["extra_args"]:
        command.append(str(arg))

    result = runner(tuple(command))
    if not result.ok:
        return None
    return _normalize_transcript_text(result.stdout)
