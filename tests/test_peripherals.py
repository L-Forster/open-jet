from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.peripherals import (
    AudioDiscoveryAdapter,
    CommandResult,
    ObservationModality,
    PeripheralDevice,
    PeripheralKind,
    PeripheralRegistry,
    PeripheralTransport,
    SensorDiscoveryAdapter,
    VideoDiscoveryAdapter,
    build_sensor_observation,
    capture_snapshot,
    parse_key_value_text,
    record_clip,
)


class _FakeAdapter:
    name = "fake"

    def __init__(self, devices: list[PeripheralDevice]) -> None:
        self._devices = devices

    def discover(self) -> list[PeripheralDevice]:
        return list(self._devices)


class PeripheralRegistryTests(unittest.TestCase):
    def test_registry_accepts_custom_discovery_adapter(self) -> None:
        registry = PeripheralRegistry()
        registry.register(
            _FakeAdapter(
                [
                    PeripheralDevice(
                        id="sensor:fake",
                        kind=PeripheralKind.SENSOR,
                        transport=PeripheralTransport.GPIO,
                        label="Fake sensor",
                    )
                ]
            )
        )

        devices = registry.discover()

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].label, "Fake sensor")


class DiscoveryAdapterTests(unittest.TestCase):
    def test_video_discovery_finds_video_devices_in_numeric_order(self) -> None:
        adapter = VideoDiscoveryAdapter(globber=lambda _pattern: ["/dev/video10", "/dev/video2", "/dev/video0", "/dev/not-video"])

        devices = adapter.discover()

        self.assertEqual([device.path for device in devices], ["/dev/video0", "/dev/video2", "/dev/video10"])
        self.assertTrue(all(device.kind is PeripheralKind.CAMERA for device in devices))

    def test_audio_discovery_prefers_pactl_when_available(self) -> None:
        outputs = {
            ("pactl", "list", "short", "sources"): CommandResult(
                args=("pactl", "list", "short", "sources"),
                returncode=0,
                stdout="1\talsa_input.usb-Mic\tmodule\tstate\n",
            ),
            ("pactl", "list", "short", "sinks"): CommandResult(
                args=("pactl", "list", "short", "sinks"),
                returncode=0,
                stdout="2\talsa_output.usb-Speaker\tmodule\tstate\n",
            ),
        }

        adapter = AudioDiscoveryAdapter(
            runner=lambda args: outputs[tuple(args)],
            which=lambda name: f"/usr/bin/{name}" if name == "pactl" else None,
        )

        devices = adapter.discover()

        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].transport, PeripheralTransport.AUDIO_SERVER)
        self.assertEqual(devices[0].kind, PeripheralKind.MICROPHONE)
        self.assertEqual(devices[1].kind, PeripheralKind.SPEAKER)

    def test_audio_discovery_falls_back_to_alsa(self) -> None:
        outputs = {
            ("arecord", "-l"): CommandResult(
                args=("arecord", "-l"),
                returncode=0,
                stdout="card 2: USBMIC [USB Mic], device 0: USB Audio [USB Audio]\n",
            ),
            ("aplay", "-l"): CommandResult(
                args=("aplay", "-l"),
                returncode=0,
                stdout="card 3: USBDAC [USB DAC], device 0: USB Audio [USB Audio]\n",
            ),
        }

        adapter = AudioDiscoveryAdapter(
            runner=lambda args: outputs[tuple(args)],
            which=lambda name: f"/usr/bin/{name}" if name in {"arecord", "aplay"} else None,
        )

        devices = adapter.discover()

        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].path, "hw:2,0")
        self.assertEqual(devices[1].path, "hw:3,0")
        self.assertTrue(all(device.transport is PeripheralTransport.ALSA for device in devices))

    def test_sensor_discovery_finds_gpio_i2c_and_serial_paths(self) -> None:
        paths = {
            "/dev/gpiochip*": ["/dev/gpiochip0"],
            "/dev/i2c-*": ["/dev/i2c-1"],
            "/dev/ttyUSB*": ["/dev/ttyUSB0"],
            "/dev/ttyACM*": ["/dev/ttyACM1"],
        }
        adapter = SensorDiscoveryAdapter(globber=lambda pattern: paths.get(pattern, []))

        devices = adapter.discover()

        self.assertEqual(len(devices), 4)
        self.assertEqual(
            [device.transport for device in devices],
            [
                PeripheralTransport.GPIO,
                PeripheralTransport.I2C,
                PeripheralTransport.USB_SERIAL,
                PeripheralTransport.USB_SERIAL,
            ],
        )


class AdapterOperationTests(unittest.TestCase):
    def test_capture_snapshot_returns_image_observation(self) -> None:
        device = PeripheralDevice(
            id="camera:/dev/video0",
            kind=PeripheralKind.CAMERA,
            transport=PeripheralTransport.V4L2,
            label="Camera /dev/video0",
            path="/dev/video0",
        )
        calls: list[tuple[str, ...]] = []
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frame.jpg"

            observation = capture_snapshot(
                device,
                output_path=target,
                runner=lambda args: calls.append(tuple(args)) or CommandResult(tuple(args), 0),
                which=lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
            )

        self.assertEqual(observation.modality, ObservationModality.IMAGE)
        self.assertEqual(observation.payload_ref, str(target))
        self.assertIn("/dev/video0", calls[0])

    def test_record_clip_returns_audio_observation(self) -> None:
        device = PeripheralDevice(
            id="microphone:hw:2,0",
            kind=PeripheralKind.MICROPHONE,
            transport=PeripheralTransport.ALSA,
            label="USB Mic / USB Audio",
            path="hw:2,0",
        )
        calls: list[tuple[str, ...]] = []
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clip.wav"

            observation = record_clip(
                device,
                duration_seconds=3,
                output_path=target,
                runner=lambda args: calls.append(tuple(args)) or CommandResult(tuple(args), 0),
                which=lambda name: f"/usr/bin/{name}" if name == "arecord" else None,
            )

        self.assertEqual(observation.modality, ObservationModality.AUDIO_CLIP)
        self.assertEqual(observation.payload_ref, str(target))
        self.assertIn("hw:2,0", calls[0])

    def test_build_sensor_observation_normalizes_values(self) -> None:
        device = PeripheralDevice(
            id="sensor:/dev/i2c-1",
            kind=PeripheralKind.SENSOR,
            transport=PeripheralTransport.I2C,
            label="I2C bus /dev/i2c-1",
            path="/dev/i2c-1",
            metadata={"interface": "i2c"},
        )

        observation = build_sensor_observation(
            device,
            {"humidity_percent": 45.2, "temp_c": 22.125},
        )

        self.assertEqual(observation.modality, ObservationModality.STRUCTURED_STATE)
        self.assertIn("humidity percent=45.20", observation.summary)
        self.assertIn("temp c=22.12", observation.summary)
        self.assertEqual(observation.metadata["values"]["temp_c"], 22.125)

    def test_parse_key_value_text_coerces_numbers_and_booleans(self) -> None:
        parsed = parse_key_value_text("temp_c=22.5 humidity_percent=40 active=true mode=auto")

        self.assertEqual(parsed["temp_c"], 22.5)
        self.assertEqual(parsed["humidity_percent"], 40)
        self.assertEqual(parsed["active"], True)
        self.assertEqual(parsed["mode"], "auto")
