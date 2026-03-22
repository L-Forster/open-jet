from __future__ import annotations

import re
from typing import Callable, Sequence

from .system import glob_paths, resolve_binary, run_command
from .types import (
    CommandResult,
    PeripheralDevice,
    PeripheralKind,
    PeripheralRegistry,
    PeripheralTransport,
    WhichResolver,
)

PathGlob = Callable[[str], Sequence[str]]
Runner = Callable[[Sequence[str]], CommandResult]


class VideoDiscoveryAdapter:
    name = "video"

    def __init__(self, *, globber: PathGlob | None = None) -> None:
        self._glob = globber or glob_paths

    def discover(self) -> list[PeripheralDevice]:
        devices: list[PeripheralDevice] = []
        for path in _sorted_numeric_paths(self._glob("/dev/video*")):
            if not re.fullmatch(r"/dev/video\d+", path):
                continue
            devices.append(
                PeripheralDevice(
                    id=f"camera:{path}",
                    kind=PeripheralKind.CAMERA,
                    transport=PeripheralTransport.V4L2,
                    label=f"Camera {path}",
                    path=path,
                    metadata={"backend": "v4l2"},
                )
            )
        return devices


class AudioDiscoveryAdapter:
    name = "audio"

    def __init__(
        self,
        *,
        runner: Runner | None = None,
        which: WhichResolver | None = None,
    ) -> None:
        self._runner = runner or _run
        self._which = which or resolve_binary

    def discover(self) -> list[PeripheralDevice]:
        pactl_devices = self._discover_pactl_devices()
        if pactl_devices:
            return pactl_devices
        return self._discover_alsa_devices()

    def _discover_pactl_devices(self) -> list[PeripheralDevice]:
        if not self._which("pactl"):
            return []
        devices: list[PeripheralDevice] = []
        source_result = self._runner(("pactl", "list", "short", "sources"))
        sink_result = self._runner(("pactl", "list", "short", "sinks"))
        if source_result.ok:
            devices.extend(_parse_pactl_rows(source_result.stdout, kind=PeripheralKind.MICROPHONE))
        if sink_result.ok:
            devices.extend(_parse_pactl_rows(sink_result.stdout, kind=PeripheralKind.SPEAKER))
        return devices

    def _discover_alsa_devices(self) -> list[PeripheralDevice]:
        devices: list[PeripheralDevice] = []
        if self._which("arecord"):
            result = self._runner(("arecord", "-l"))
            if result.ok:
                devices.extend(_parse_alsa_cards(result.stdout, kind=PeripheralKind.MICROPHONE))
        if self._which("aplay"):
            result = self._runner(("aplay", "-l"))
            if result.ok:
                devices.extend(_parse_alsa_cards(result.stdout, kind=PeripheralKind.SPEAKER))
        return devices


class SensorDiscoveryAdapter:
    name = "sensor"

    def __init__(self, *, globber: PathGlob | None = None) -> None:
        self._glob = globber or glob_paths

    def discover(self) -> list[PeripheralDevice]:
        devices: list[PeripheralDevice] = []
        devices.extend(self._discover_gpio())
        devices.extend(self._discover_i2c())
        devices.extend(self._discover_serial())
        return devices

    def _discover_gpio(self) -> list[PeripheralDevice]:
        devices: list[PeripheralDevice] = []
        for path in _sorted_numeric_paths(self._glob("/dev/gpiochip*")):
            if not re.fullmatch(r"/dev/gpiochip\d+", path):
                continue
            devices.append(
                PeripheralDevice(
                    id=f"sensor:{path}",
                    kind=PeripheralKind.SENSOR,
                    transport=PeripheralTransport.GPIO,
                    label=f"GPIO chip {path}",
                    path=path,
                    metadata={"interface": "gpiochip"},
                )
            )
        return devices

    def _discover_i2c(self) -> list[PeripheralDevice]:
        devices: list[PeripheralDevice] = []
        for path in _sorted_numeric_paths(self._glob("/dev/i2c-*")):
            if not re.fullmatch(r"/dev/i2c-\d+", path):
                continue
            devices.append(
                PeripheralDevice(
                    id=f"sensor:{path}",
                    kind=PeripheralKind.SENSOR,
                    transport=PeripheralTransport.I2C,
                    label=f"I2C bus {path}",
                    path=path,
                    metadata={"interface": "i2c"},
                )
            )
        return devices

    def _discover_serial(self) -> list[PeripheralDevice]:
        devices: list[PeripheralDevice] = []
        serial_paths = list(self._glob("/dev/ttyUSB*")) + list(self._glob("/dev/ttyACM*"))
        for path in _sorted_numeric_paths(serial_paths):
            if not re.fullmatch(r"/dev/(ttyUSB|ttyACM)\d+", path):
                continue
            devices.append(
                PeripheralDevice(
                    id=f"sensor:{path}",
                    kind=PeripheralKind.SENSOR,
                    transport=PeripheralTransport.USB_SERIAL,
                    label=f"Serial sensor bus {path}",
                    path=path,
                    metadata={"interface": "serial"},
                )
            )
        return devices


def build_default_registry(
    *,
    globber: PathGlob | None = None,
    runner: Runner | None = None,
    which: WhichResolver | None = None,
) -> PeripheralRegistry:
    return PeripheralRegistry(
        [
            VideoDiscoveryAdapter(globber=globber),
            AudioDiscoveryAdapter(runner=runner, which=which),
            SensorDiscoveryAdapter(globber=globber),
        ]
    )


def discover_peripherals(
    *,
    globber: PathGlob | None = None,
    runner: Runner | None = None,
    which: WhichResolver | None = None,
) -> list[PeripheralDevice]:
    return build_default_registry(globber=globber, runner=runner, which=which).discover()


def _run(args: Sequence[str]) -> CommandResult:
    return run_command(args)


def _sorted_numeric_paths(paths: Sequence[str]) -> list[str]:
    return sorted(paths, key=_numeric_path_sort_key)


def _numeric_path_sort_key(path: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", path)
    return (re.sub(r"\d+$", "", path), int(match.group(1)) if match else -1)


def _parse_pactl_rows(stdout: str, *, kind: PeripheralKind) -> list[PeripheralDevice]:
    devices: list[PeripheralDevice] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) < 2:
            continue
        index, name = parts[0], parts[1]
        devices.append(
            PeripheralDevice(
                id=f"{kind.value}:{name}",
                kind=kind,
                transport=PeripheralTransport.AUDIO_SERVER,
                label=name,
                path=name,
                metadata={"backend": "pactl", "index": index},
            )
        )
    return devices


_ALSA_CARD_RE = re.compile(
    r"card\s+(?P<card>\d+):\s+(?P<card_id>[^\[]+)\[(?P<card_label>[^\]]+)\],\s+device\s+(?P<device>\d+):\s+(?P<device_id>[^\[]+)\[(?P<device_label>[^\]]+)\]"
)


def _parse_alsa_cards(stdout: str, *, kind: PeripheralKind) -> list[PeripheralDevice]:
    devices: list[PeripheralDevice] = []
    for line in stdout.splitlines():
        match = _ALSA_CARD_RE.search(line)
        if not match:
            continue
        card = match.group("card")
        device = match.group("device")
        card_label = match.group("card_label").strip()
        device_label = match.group("device_label").strip()
        hw = f"hw:{card},{device}"
        label = f"{card_label} / {device_label}"
        devices.append(
            PeripheralDevice(
                id=f"{kind.value}:{hw}",
                kind=kind,
                transport=PeripheralTransport.ALSA,
                label=label,
                path=hw,
                metadata={"backend": "alsa", "card": card, "device": device},
            )
        )
    return devices
