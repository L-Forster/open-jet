from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from src.device_sources import DeviceSource
from src.peripherals import Observation, ObservationModality, PeripheralDevice, PeripheralKind, PeripheralTransport
from src.runtime_protocol import ToolCall
from src.tool_executor import execute_tool


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class DeviceToolExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_device_list_tool_renders_detected_sources(self) -> None:
        source = DeviceSource(
            primary_ref="front",
            refs=("front", "camera0"),
            device=PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            ),
            enabled=True,
        )
        with patch("src.tool_executor.load_config", return_value={}), patch(
            "src.tool_executor.list_device_sources",
            return_value=[source],
        ), patch(
            "src.tool_executor.sync_devices_registry",
            return_value=Path("/tmp/devices.md"),
        ):
            result = await execute_tool(ToolCall(name="device_list", arguments={"kind": "camera"}))

        self.assertTrue(result.ok)
        self.assertIn("@front", result.output)
        self.assertIn("Front Camera", result.output)
        self.assertEqual(str(result.meta["registry_path"]).replace("\\", "/"), "/tmp/devices.md")

    async def test_device_list_tool_includes_wsl_hint_when_no_sources_detected(self) -> None:
        with patch("src.tool_executor.load_config", return_value={}), patch(
            "src.tool_executor.list_device_sources",
            return_value=[],
        ), patch(
            "src.tool_executor.device_discovery_hint",
            return_value="Running inside WSL2.",
        ):
            result = await execute_tool(ToolCall(name="device_list", arguments={}))

        self.assertTrue(result.ok)
        self.assertIn("No device sources detected.", result.output)
        self.assertIn("Running inside WSL2.", result.output)

    async def test_camera_snapshot_tool_returns_multimodal_context(self) -> None:
        source = DeviceSource(
            primary_ref="front",
            refs=("front", "camera0"),
            device=PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            ),
            enabled=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.jpg"
            image.write_bytes(b"jpeg-bytes")
            observation = Observation(
                source_id=source.device.id,
                source_type=source.device.kind.value,
                transport=source.device.transport.value,
                modality=ObservationModality.IMAGE,
                summary="Captured image from Front Camera",
                payload_ref=str(image),
            )

            with patch("src.tool_executor.load_config", return_value={}), patch(
                "src.tool_executor.list_device_sources",
                return_value=[source],
            ), patch(
                "src.tool_executor.collect_device_observation",
                return_value=observation,
            ), patch(
                "src.tool_executor.sync_devices_registry",
                return_value=Path("/tmp/devices.md"),
            ):
                result = await execute_tool(ToolCall(name="camera_snapshot", arguments={}))

        self.assertTrue(result.ok)
        self.assertIsInstance(result.context_content, list)
        self.assertIn("Captured image from Front Camera", result.output)
        self.assertTrue(any(block.get("type") == "input_image" for block in result.context_content))
        self.assertEqual(str(result.meta["registry_path"]).replace("\\", "/"), "/tmp/devices.md")

    async def test_microphone_record_tool_uses_duration_and_returns_text_context(self) -> None:
        source = DeviceSource(
            primary_ref="room-mic",
            refs=("room-mic", "mic0"),
            device=PeripheralDevice(
                id="microphone:hw:2,0",
                kind=PeripheralKind.MICROPHONE,
                transport=PeripheralTransport.ALSA,
                label="Room Mic",
                path="hw:2,0",
            ),
            enabled=True,
        )
        observation = Observation(
            source_id=source.device.id,
            source_type=source.device.kind.value,
            transport=source.device.transport.value,
            modality=ObservationModality.TEXT,
            summary="Speech detected on microphone:hw:2,0",
            metadata={"speech_detected": True},
        )

        with patch("src.tool_executor.load_config", return_value={}), patch(
            "src.tool_executor.list_device_sources",
            return_value=[source],
        ), patch(
            "src.tool_executor.collect_device_observation",
            return_value=observation,
        ) as collect_observation:
            result = await execute_tool(
                ToolCall(name="microphone_record", arguments={"duration_seconds": 3})
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.context_content, "Speech detected on microphone:hw:2,0")
        self.assertIn("Speech detected", result.output)
        collect_observation.assert_called_once()
        self.assertEqual(collect_observation.call_args.kwargs["duration_seconds"], 3)

    async def test_microphone_record_returns_error_when_source_is_disabled(self) -> None:
        source = DeviceSource(
            primary_ref="room-mic",
            refs=("room-mic", "mic0"),
            device=PeripheralDevice(
                id="microphone:hw:2,0",
                kind=PeripheralKind.MICROPHONE,
                transport=PeripheralTransport.ALSA,
                label="Room Mic",
                path="hw:2,0",
            ),
            enabled=False,
        )

        with patch("src.tool_executor.load_config", return_value={}), patch(
            "src.tool_executor.list_device_sources",
            return_value=[source],
        ), patch(
            "src.tool_executor.collect_device_observation",
            side_effect=RuntimeError("source @room-mic is disabled"),
        ):
            result = await execute_tool(ToolCall(name="microphone_record", arguments={}))

        self.assertFalse(result.ok)
        self.assertIn("disabled", result.output.lower())

    async def test_microphone_set_enabled_persists_state(self) -> None:
        source = DeviceSource(
            primary_ref="room-mic",
            refs=("room-mic", "mic0"),
            device=PeripheralDevice(
                id="microphone:hw:2,0",
                kind=PeripheralKind.MICROPHONE,
                transport=PeripheralTransport.ALSA,
                label="Room Mic",
                path="hw:2,0",
            ),
            enabled=False,
        )
        cfg: dict[str, object] = {}

        with patch("src.tool_executor.load_config", return_value=cfg), patch(
            "src.tool_executor._select_device_source",
            return_value=source,
        ), patch(
            "src.tool_executor.set_device_enabled",
            return_value=source,
        ) as set_enabled, patch("src.tool_executor.save_config") as save_config, patch(
            "src.tool_executor.sync_devices_registry",
            return_value=Path("/tmp/devices.md"),
        ):
            result = await execute_tool(
                ToolCall(name="microphone_set_enabled", arguments={"source": "room-mic", "enabled": False})
            )

        self.assertTrue(result.ok)
        self.assertIn("disabled", result.output)
        set_enabled.assert_called_once()
        save_config.assert_called_once_with(cfg)
        self.assertEqual(str(result.meta["registry_path"]).replace("\\", "/"), "/tmp/devices.md")

    async def test_gpio_read_rejects_non_gpio_sources(self) -> None:
        source = DeviceSource(
            primary_ref="i2c0",
            refs=("i2c0",),
            device=PeripheralDevice(
                id="sensor:/dev/i2c-1",
                kind=PeripheralKind.SENSOR,
                transport=PeripheralTransport.I2C,
                label="I2C bus /dev/i2c-1",
                path="/dev/i2c-1",
            ),
            enabled=True,
        )

        with patch("src.tool_executor.load_config", return_value={}), patch(
            "src.tool_executor.list_device_sources",
            return_value=[source],
        ):
            result = await execute_tool(ToolCall(name="gpio_read", arguments={}))

        self.assertFalse(result.ok)
        self.assertIn("only gpio is supported right now", result.output.lower())

    async def test_camera_snapshot_tool_updates_registry_with_latest_payload(self) -> None:
        device = PeripheralDevice(
            id="camera:/dev/video0",
            kind=PeripheralKind.CAMERA,
            transport=PeripheralTransport.V4L2,
            label="Front Camera",
            path="/dev/video0",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_image = root / "frame.jpg"
            raw_image.write_bytes(b"jpeg-bytes")
            captured = Observation(
                source_id=device.id,
                source_type=device.kind.value,
                transport=device.transport.value,
                modality=ObservationModality.IMAGE,
                summary="Captured image from Front Camera",
                payload_ref=str(raw_image),
            )
            with pushd(root), patch("src.tool_executor.load_config", return_value={}), patch(
                "src.device_sources.discover_peripherals",
                return_value=[device],
            ), patch(
                "src.device_sources.capture_snapshot",
                return_value=captured,
            ):
                result = await execute_tool(ToolCall(name="camera_snapshot", arguments={"source": "camera0"}))
            registry = root / ".openjet" / "state" / "devices.md"
            registry_exists = registry.is_file()
            rendered = registry.read_text(encoding="utf-8") if registry_exists else ""
            payload_path = Path(result.meta["payload_ref"])
            if not payload_path.is_absolute():
                payload_path = root / payload_path
            payload_exists = payload_path.is_file()

        self.assertTrue(result.ok)
        self.assertTrue(registry_exists)
        self.assertIn("latest_payload_file: `", rendered)
        self.assertNotIn("latest_payload_file: `none`", rendered)
        self.assertTrue(payload_exists)
        self.assertIn(str(result.meta["payload_ref"]), rendered)

    async def test_microphone_record_tool_updates_registry_with_transcript_payload(self) -> None:
        device = PeripheralDevice(
            id="microphone:hw:2,0",
            kind=PeripheralKind.MICROPHONE,
            transport=PeripheralTransport.ALSA,
            label="Room Mic",
            path="hw:2,0",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip = root / "speech.wav"
            clip.write_bytes(b"RIFFfake-wave")
            recorded = Observation(
                source_id=device.id,
                source_type=device.kind.value,
                transport=device.transport.value,
                modality=ObservationModality.AUDIO_CLIP,
                summary="Recorded 1s clip",
                payload_ref=str(clip),
            )

            def _fake_process(observation, *, store, transcription_cfg=None):
                transcript_path = store.append_text_buffer(
                    observation.source_id,
                    "2026-03-22T12:00:00Z | hello from spoofed mic",
                    buffer_name="microphone-transcript.txt",
                )
                transcript = Observation(
                    source_id=observation.source_id,
                    source_type=observation.source_type,
                    transport=observation.transport,
                    modality=ObservationModality.TEXT,
                    summary="Transcript from microphone: hello from spoofed mic",
                    payload_ref=str(transcript_path),
                    metadata={"transcript_text": "hello from spoofed mic", "speech_detected": True},
                )
                return store.persist(transcript)

            with pushd(root), patch("src.tool_executor.load_config", return_value={}), patch(
                "src.device_sources.discover_peripherals",
                return_value=[device],
            ), patch(
                "src.device_sources.record_clip",
                return_value=recorded,
            ), patch(
                "src.device_sources.process_audio_observation",
                side_effect=_fake_process,
            ):
                result = await execute_tool(ToolCall(name="microphone_record", arguments={"source": "mic0"}))
            registry = root / ".openjet" / "state" / "devices.md"
            registry_exists = registry.is_file()
            rendered = registry.read_text(encoding="utf-8") if registry_exists else ""
            payload_path = Path(result.meta["payload_ref"])
            if not payload_path.is_absolute():
                payload_path = root / payload_path
            payload_exists = payload_path.is_file()

        self.assertTrue(result.ok)
        self.assertTrue(registry_exists)
        self.assertIn("microphone-transcript.txt", rendered)
        self.assertIn(str(result.meta["payload_ref"]), rendered)
        self.assertTrue(payload_exists)

    async def test_gpio_read_tool_updates_registry_with_buffer_payload(self) -> None:
        device = PeripheralDevice(
            id="sensor:/dev/gpiochip0",
            kind=PeripheralKind.SENSOR,
            transport=PeripheralTransport.GPIO,
            label="GPIO chip /dev/gpiochip0",
            path="/dev/gpiochip0",
        )
        gpio_text = (
            "gpiochip0 - 8 lines:\n"
            "line 0: unnamed unused input active-high\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with pushd(root), patch("src.tool_executor.load_config", return_value={}), patch(
                "src.device_sources.discover_peripherals",
                return_value=[device],
            ), patch(
                "src.device_sources.resolve_binary",
                return_value="/usr/bin/gpioinfo",
            ), patch(
                "src.device_sources.run_command",
                return_value=type("Result", (), {"ok": True, "stdout": gpio_text, "stderr": ""})(),
            ):
                result = await execute_tool(ToolCall(name="gpio_read", arguments={"source": "gpio0"}))
            registry = root / ".openjet" / "state" / "devices.md"
            registry_exists = registry.is_file()
            rendered = registry.read_text(encoding="utf-8") if registry_exists else ""
            payload_path = Path(result.meta["payload_ref"])
            if not payload_path.is_absolute():
                payload_path = root / payload_path
            payload_exists = payload_path.is_file()

        self.assertTrue(result.ok)
        self.assertTrue(registry_exists)
        self.assertIn("gpio-buffer.txt", rendered)
        self.assertIn(str(result.meta["payload_ref"]), rendered)
        self.assertTrue(payload_exists)
