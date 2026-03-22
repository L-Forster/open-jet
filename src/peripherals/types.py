from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Protocol, Sequence


class PeripheralKind(str, Enum):
    CAMERA = "camera"
    MICROPHONE = "microphone"
    SPEAKER = "speaker"
    SENSOR = "sensor"


class PeripheralTransport(str, Enum):
    V4L2 = "v4l2"
    AUDIO_SERVER = "audio_server"
    ALSA = "alsa"
    GPIO = "gpio"
    I2C = "i2c"
    USB_SERIAL = "usb_serial"
    UNKNOWN = "unknown"


class ObservationModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO_CLIP = "audio_clip"
    STRUCTURED_STATE = "structured_state"


@dataclass(frozen=True)
class PeripheralDevice:
    id: str
    kind: PeripheralKind
    transport: PeripheralTransport
    label: str
    path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "transport": self.transport.value,
            "label": self.label,
            "path": self.path,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Observation:
    source_id: str
    source_type: str
    transport: str
    modality: ObservationModality
    summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    payload_ref: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    changed: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "transport": self.transport,
            "modality": self.modality.value,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "payload_ref": self.payload_ref,
            "metadata": dict(self.metadata),
            "changed": self.changed,
        }


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class DeviceDiscoveryAdapter(Protocol):
    name: str

    def discover(self) -> list[PeripheralDevice]:
        ...


CommandRunner = Callable[[Sequence[str]], CommandResult]
WhichResolver = Callable[[str], str | None]


class PeripheralRegistry:
    def __init__(self, adapters: Sequence[DeviceDiscoveryAdapter] | None = None) -> None:
        self._adapters: list[DeviceDiscoveryAdapter] = list(adapters or [])

    def register(self, adapter: DeviceDiscoveryAdapter) -> None:
        self._adapters.append(adapter)

    def discover(self) -> list[PeripheralDevice]:
        merged: list[PeripheralDevice] = []
        seen: set[str] = set()
        for adapter in self._adapters:
            for device in adapter.discover():
                if device.id in seen:
                    continue
                seen.add(device.id)
                merged.append(device)
        return merged
