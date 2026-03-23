"""Peripheral discovery and device-input helpers.

This layer is intentionally small: transport-based discovery, normalized
observation records, and thin adapters that higher-level workflows can compose.
"""

from .audio import record_clip
from .camera import capture_snapshot
from .discovery import (
    AudioDiscoveryAdapter,
    SensorDiscoveryAdapter,
    VideoDiscoveryAdapter,
    build_default_registry,
    discover_peripherals,
)
from .sensors import build_sensor_observation, parse_key_value_text, summarize_sensor_values
from .types import (
    CommandResult,
    Observation,
    ObservationModality,
    PeripheralDevice,
    PeripheralKind,
    PeripheralRegistry,
    PeripheralTransport,
)

__all__ = [
    "AudioDiscoveryAdapter",
    "CommandResult",
    "Observation",
    "ObservationModality",
    "PeripheralDevice",
    "PeripheralKind",
    "PeripheralRegistry",
    "PeripheralTransport",
    "SensorDiscoveryAdapter",
    "VideoDiscoveryAdapter",
    "build_default_registry",
    "build_sensor_observation",
    "capture_snapshot",
    "discover_peripherals",
    "parse_key_value_text",
    "record_clip",
    "summarize_sensor_values",
]
