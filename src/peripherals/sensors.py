from __future__ import annotations

import re
from typing import Mapping

from .types import Observation, ObservationModality, PeripheralDevice, PeripheralKind


def build_sensor_observation(
    device: PeripheralDevice,
    values: Mapping[str, object],
    *,
    changed: bool = True,
    summary: str | None = None,
) -> Observation:
    if device.kind is not PeripheralKind.SENSOR:
        raise ValueError("build_sensor_observation requires a sensor device")
    normalized = {str(key): values[key] for key in sorted(values)}
    rendered_summary = summary or summarize_sensor_values(normalized)
    return Observation(
        source_id=device.id,
        source_type=device.kind.value,
        transport=device.transport.value,
        modality=ObservationModality.STRUCTURED_STATE,
        summary=rendered_summary,
        payload_ref=device.path,
        metadata={"values": normalized, **dict(device.metadata)},
        changed=changed,
    )


def summarize_sensor_values(values: Mapping[str, object]) -> str:
    parts: list[str] = []
    for key, value in values.items():
        parts.append(f"{_humanize_key(key)}={_format_value(value)}")
    return ", ".join(parts) if parts else "no sensor values"


def parse_key_value_text(text: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, raw_value in re.findall(r"([A-Za-z0-9_]+)\s*=\s*([^\s,;]+)", text):
        parsed[key] = _coerce_value(raw_value)
    return parsed


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").strip()


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _coerce_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"-?\d+", raw):
        try:
            return int(raw)
        except ValueError:
            return raw
    if re.fullmatch(r"-?\d+\.\d+", raw):
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw
