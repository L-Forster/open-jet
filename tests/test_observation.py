from __future__ import annotations

import math
import sys
import struct
import tempfile
import types
import unittest
import wave
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.observation import (
    ObservationStore,
    append_gpio_text_buffer,
    detect_speech_activity,
    observation_to_agent_content,
    observations_to_agent_content,
    process_audio_observation,
    provision_default_faster_whisper_model,
    save_frame_observation,
)
from src.peripherals import CommandResult
from src.peripherals import Observation, ObservationModality, PeripheralDevice, PeripheralKind, PeripheralTransport, build_sensor_observation


class ObservationStoreTests(unittest.TestCase):
    def test_save_frame_observation_copies_payload_and_records_event(self) -> None:
        timestamp = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "frame.jpg"
            original.write_bytes(b"jpeg-bytes")
            store = ObservationStore(root / "state")
            observation = Observation(
                source_id="camera:/dev/video0",
                source_type="camera",
                transport="v4l2",
                modality=ObservationModality.IMAGE,
                summary="Captured image from camera",
                timestamp=timestamp,
                payload_ref=str(original),
            )

            stored = save_frame_observation(observation, store=store)

            self.assertNotEqual(stored.payload_ref, str(original))
            self.assertTrue(Path(stored.payload_ref or "").is_file())
            latest = store.source_dir(observation.source_id) / "latest.json"
            events = store.source_dir(observation.source_id) / "events.jsonl"
            self.assertTrue(latest.is_file())
            self.assertTrue(events.is_file())
            self.assertIn("Captured image from camera", latest.read_text(encoding="utf-8"))


class SpeechDetectionTests(unittest.TestCase):
    def test_detect_speech_activity_marks_active_audio_as_text_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "speech.wav"
            _write_wav_tone(clip, amplitude=12000)
            observation = Observation(
                source_id="microphone:hw:2,0",
                source_type="microphone",
                transport="alsa",
                modality=ObservationModality.AUDIO_CLIP,
                summary="Recorded 1s clip",
                payload_ref=str(clip),
            )

            processed = detect_speech_activity(observation, energy_threshold=1000)

            self.assertEqual(processed.modality, ObservationModality.TEXT)
            self.assertEqual(processed.metadata["speech_detected"], True)
            self.assertIn("Speech detected", processed.summary)

    def test_detect_speech_activity_marks_silence_as_no_speech(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "silence.wav"
            _write_wav_tone(clip, amplitude=0)
            observation = Observation(
                source_id="microphone:hw:2,0",
                source_type="microphone",
                transport="alsa",
                modality=ObservationModality.AUDIO_CLIP,
                summary="Recorded 1s clip",
                payload_ref=str(clip),
            )

            processed = detect_speech_activity(observation, energy_threshold=1000)

            self.assertEqual(processed.metadata["speech_detected"], False)
            self.assertIn("No speech detected", processed.summary)

    def test_process_audio_observation_uses_local_transcription_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "speech.wav"
            model = root / "ggml-base.en.bin"
            _write_wav_tone(clip, amplitude=12000)
            model.write_bytes(b"model")
            store = ObservationStore(root / "state")
            observation = Observation(
                source_id="microphone:hw:2,0",
                source_type="microphone",
                transport="alsa",
                modality=ObservationModality.AUDIO_CLIP,
                summary="Recorded 1s clip",
                payload_ref=str(clip),
            )

            processed = process_audio_observation(
                observation,
                store=store,
                transcription_cfg={
                    "enabled": True,
                    "backend": "whisper_cpp",
                    "command": "whisper-cli",
                    "model_path": str(model),
                    "language": "en",
                },
                runner=lambda args: CommandResult(tuple(args), 0, stdout="hello from the mic\n"),
                which=lambda name: "/usr/bin/whisper-cli" if name == "whisper-cli" else None,
            )

            self.assertEqual(processed.modality, ObservationModality.TEXT)
            self.assertEqual(processed.metadata["transcription_backend"], "whisper_cpp")
            self.assertIn("hello from the mic", processed.metadata["transcript_text"])
            self.assertTrue(Path(processed.payload_ref or "").is_file())
            self.assertIn("hello from the mic", Path(processed.payload_ref or "").read_text(encoding="utf-8"))

    def test_process_audio_observation_uses_packaged_faster_whisper_by_default(self) -> None:
        class _Segment:
            def __init__(self, text: str) -> None:
                self.text = text

        class _FakeWhisperModel:
            def __init__(self, model_name: str, *, device: str, compute_type: str) -> None:
                self.model_name = model_name
                self.device = device
                self.compute_type = compute_type

            def transcribe(self, _path: str, **_kwargs):
                return ([_Segment("packaged transcript"), _Segment("from faster whisper")], object())

        fake_module = types.SimpleNamespace(WhisperModel=_FakeWhisperModel)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"faster_whisper": fake_module}):
            root = Path(tmp)
            clip = root / "speech.wav"
            store = ObservationStore(root / "state")
            _write_wav_tone(clip, amplitude=12000)
            observation = Observation(
                source_id="microphone:hw:2,0",
                source_type="microphone",
                transport="alsa",
                modality=ObservationModality.AUDIO_CLIP,
                summary="Recorded 1s clip",
                payload_ref=str(clip),
            )

            with patch("src.observation.processors.detect_hardware_info", return_value=types.SimpleNamespace(has_cuda=False)):
                processed = process_audio_observation(observation, store=store)

            self.assertEqual(processed.modality, ObservationModality.TEXT)
            self.assertEqual(processed.metadata["transcription_backend"], "faster_whisper")
            self.assertIn("packaged transcript", processed.metadata["transcript_text"])
            self.assertTrue(Path(processed.payload_ref or "").is_file())

    def test_process_audio_observation_falls_back_to_speech_detection_when_transcriber_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "speech.wav"
            _write_wav_tone(clip, amplitude=12000)
            observation = Observation(
                source_id="microphone:hw:2,0",
                source_type="microphone",
                transport="alsa",
                modality=ObservationModality.AUDIO_CLIP,
                summary="Recorded 1s clip",
                payload_ref=str(clip),
            )

            processed = process_audio_observation(
                observation,
                transcription_cfg={"enabled": True, "backend": "faster_whisper"},
            )

            self.assertEqual(processed.modality, ObservationModality.TEXT)
            self.assertIn("Speech detected", processed.summary)
            self.assertNotIn("transcription_backend", processed.metadata)

    def test_provision_default_faster_whisper_model_uses_openjet_cache_root(self) -> None:
        calls: list[tuple[str, str, str, str | None]] = []

        class _FakeWhisperModel:
            def __init__(self, model_name: str, *, device: str, compute_type: str, download_root: str | None = None) -> None:
                calls.append((model_name, device, compute_type, download_root))

        fake_module = types.SimpleNamespace(WhisperModel=_FakeWhisperModel)

        with patch.dict(sys.modules, {"faster_whisper": fake_module}), patch(
            "src.observation.processors.detect_hardware_info",
            return_value=types.SimpleNamespace(has_cuda=False),
        ), patch.dict("src.observation.processors._FASTER_WHISPER_MODELS", {}, clear=True):
            self.assertTrue(provision_default_faster_whisper_model())

        self.assertEqual(calls[0][0], "tiny")
        self.assertEqual(calls[0][1], "cpu")
        self.assertEqual(calls[0][2], "int8")
        self.assertIn(".openjet", calls[0][3] or "")


class GpioBufferTests(unittest.TestCase):
    def test_append_gpio_text_buffer_writes_readings_to_buffer_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "state")
            device = PeripheralDevice(
                id="sensor:/dev/gpiochip0",
                kind=PeripheralKind.SENSOR,
                transport=PeripheralTransport.GPIO,
                label="GPIO chip /dev/gpiochip0",
                path="/dev/gpiochip0",
                metadata={"interface": "gpiochip"},
            )
            observation = build_sensor_observation(
                device,
                {"pin_17": 1, "pin_18": 0},
                summary="pin_17=1, pin_18=0",
            )

            buffered = append_gpio_text_buffer(observation, store=store)

            self.assertEqual(buffered.modality, ObservationModality.TEXT)
            self.assertTrue(Path(buffered.payload_ref or "").is_file())
            buffer_text = Path(buffered.payload_ref or "").read_text(encoding="utf-8")
            self.assertIn("pin_17=1, pin_18=0", buffer_text)


class AgentBridgeTests(unittest.TestCase):
    def test_observation_to_agent_content_uses_image_blocks_for_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.jpg"
            image.write_bytes(b"jpeg-bytes")
            observation = Observation(
                source_id="camera:/dev/video0",
                source_type="camera",
                transport="v4l2",
                modality=ObservationModality.IMAGE,
                summary="Front camera frame",
                payload_ref=str(image),
            )

            content = observation_to_agent_content(observation, prompt_text="Describe this frame")

            self.assertIsInstance(content, list)
            self.assertEqual(content[1]["type"], "input_image")
            self.assertEqual(content[1]["path"], str(image))

    def test_observation_to_agent_content_reads_gpio_buffer_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ObservationStore(Path(tmp) / "state")
            buffer_path = store.append_text_buffer(
                "sensor:/dev/gpiochip0",
                "2026-03-21T12:00:00Z | pin_17=1, pin_18=0",
                buffer_name="gpio-buffer.txt",
            )
            observation = Observation(
                source_id="sensor:/dev/gpiochip0",
                source_type="sensor",
                transport="gpio",
                modality=ObservationModality.TEXT,
                summary="GPIO buffer updated for sensor:/dev/gpiochip0",
                payload_ref=str(buffer_path),
            )

            content = observation_to_agent_content(observation, store=store)

            self.assertIsInstance(content, str)
            self.assertIn("pin_17=1, pin_18=0", content)

    def test_observations_to_agent_content_combines_text_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.jpg"
            image.write_bytes(b"jpeg-bytes")
            observations = [
                Observation(
                    source_id="camera:/dev/video0",
                    source_type="camera",
                    transport="v4l2",
                    modality=ObservationModality.IMAGE,
                    summary="Front camera frame",
                    payload_ref=str(image),
                ),
                Observation(
                    source_id="microphone:hw:2,0",
                    source_type="microphone",
                    transport="alsa",
                    modality=ObservationModality.TEXT,
                    summary="Speech detected on microphone:hw:2,0",
                ),
            ]

            content = observations_to_agent_content(observations, prompt_text="Summarize recent inputs")

            self.assertIsInstance(content, list)
            self.assertIn("Summarize recent inputs", content[0]["text"])
            self.assertEqual(content[1]["type"], "input_image")


def _write_wav_tone(path: Path, *, amplitude: int, frame_rate: int = 16000, duration_seconds: float = 1.0) -> None:
    total_frames = int(frame_rate * duration_seconds)
    frames: list[bytes] = []
    for index in range(total_frames):
        sample = int(amplitude * math.sin(2.0 * math.pi * 440.0 * (index / frame_rate))) if amplitude else 0
        frames.append(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(b"".join(frames))
