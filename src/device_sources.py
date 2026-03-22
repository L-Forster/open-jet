from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .observation import ObservationStore, process_audio_observation, save_frame_observation
from .peripherals import PeripheralDevice, PeripheralKind, PeripheralTransport, capture_snapshot, discover_peripherals, record_clip
from .peripherals.system import resolve_binary, run_command
from .peripherals.types import Observation, ObservationModality

DEFAULT_DEVICES_REGISTRY_PATH = Path(".openjet/state/devices.md")


@dataclass(frozen=True)
class DeviceSource:
    primary_ref: str
    refs: tuple[str, ...]
    device: PeripheralDevice
    enabled: bool = True


def list_device_sources(
    cfg: Mapping[str, object] | None,
) -> list[DeviceSource]:
    devices = discover_peripherals()
    custom_aliases = _custom_aliases(cfg)
    disabled_ids = _disabled_device_ids(cfg)
    generated_counts: dict[str, int] = {}
    sources: list[DeviceSource] = []
    sources.extend(_configured_gpio_sources(cfg, devices, disabled_ids))
    for device in devices:
        default_ref = _default_ref(device, generated_counts)
        refs = [default_ref, device.id]
        basename = _device_basename_alias(device)
        if basename and basename not in refs:
            refs.append(basename)
        for alias, device_id in custom_aliases.items():
            if device_id == device.id and alias not in refs:
                refs.insert(0, alias)
        sources.append(
            DeviceSource(
                primary_ref=refs[0],
                refs=tuple(_dedupe_refs(refs)),
                device=device,
                enabled=device.id not in disabled_ids,
            )
        )
    return sources


def resolve_device_source(reference: str, cfg: Mapping[str, object] | None) -> DeviceSource | None:
    needle = reference.strip().lstrip("@").lower()
    if not needle:
        return None
    for source in list_device_sources(cfg):
        if any(ref.lower() == needle for ref in source.refs):
            return source
    return None


def assign_device_alias(
    cfg: dict[str, object],
    *,
    reference: str,
    alias: str,
) -> DeviceSource:
    normalized = _normalize_alias(alias)
    if not normalized:
        raise ValueError("alias must contain letters, numbers, '.', '_' or '-'")
    source = resolve_device_source(reference, cfg)
    if source is None:
        raise ValueError(f"unknown device reference: {reference}")
    for existing in list_device_sources(cfg):
        if normalized.lower() in {ref.lower() for ref in existing.refs} and existing.device.id != source.device.id:
            raise ValueError(f"alias already in use: {normalized}")
    aliases = dict(_custom_aliases(cfg))
    aliases[normalized] = source.device.id
    cfg["device_aliases"] = aliases
    return resolve_device_source(normalized, cfg) or source


def set_device_enabled(
    cfg: dict[str, object],
    *,
    reference: str,
    enabled: bool,
) -> DeviceSource:
    source = resolve_device_source(reference, cfg)
    if source is None:
        raise ValueError(f"unknown device reference: {reference}")
    disabled_ids = set(_disabled_device_ids(cfg))
    if enabled:
        disabled_ids.discard(source.device.id)
    else:
        disabled_ids.add(source.device.id)
    cfg["disabled_device_ids"] = sorted(disabled_ids)
    return resolve_device_source(source.primary_ref, cfg) or DeviceSource(
        primary_ref=source.primary_ref,
        refs=source.refs,
        device=source.device,
        enabled=enabled,
    )

def write_devices_markdown(
    cfg: Mapping[str, object] | None,
    *,
    store: ObservationStore,
    output_path: str | Path | None = None,
) -> Path:
    target = devices_registry_path(store=store, output_path=output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_devices_markdown(cfg, store=store), encoding="utf-8")
    return target


def devices_registry_path(
    *,
    store: ObservationStore | None = None,
    output_path: str | Path | None = None,
) -> Path:
    if output_path is not None:
        target = Path(output_path)
    elif store is not None:
        target = store.root.parent / "devices.md"
    else:
        target = DEFAULT_DEVICES_REGISTRY_PATH
    return target.expanduser().resolve()


def sync_devices_registry(
    cfg: Mapping[str, object] | None,
    *,
    store: ObservationStore | None = None,
    output_path: str | Path | None = None,
) -> Path:
    active_store = store or ObservationStore()
    return write_devices_markdown(cfg, store=active_store, output_path=output_path)


def render_devices_markdown(
    cfg: Mapping[str, object] | None,
    *,
    store: ObservationStore,
) -> str:
    lines = [
        "# Devices",
        "",
        "This file lists the currently discovered local I/O devices.",
        "Use the plain ids below in chat by prefixing them with `@`.",
        "Loading this registry into context should not preload every device log or payload file.",
        "",
    ]
    sources = sorted(list_device_sources(cfg), key=lambda item: item.primary_ref.lower())
    if not sources:
        lines.append("No devices discovered.")
        return "\n".join(lines).strip() + "\n"

    for source in sources:
        latest_json_path, latest_payload_path = _latest_device_artifacts(source, store=store)
        refs = ", ".join(f"`@{ref}`" for ref in source.refs)
        lines.extend(
            [
                f"## {source.primary_ref}",
                f"- id: `{source.primary_ref}`",
                f"- chat_tag: `@{source.primary_ref}`",
                f"- refs: {refs}",
                f"- device_id: `{source.device.id}`",
                f"- label: {source.device.label}",
                f"- kind: `{source.device.kind.value}`",
                f"- transport: `{source.device.transport.value}`",
                f"- state: `{'enabled' if source.enabled else 'disabled'}`",
                f"- hardware_path: `{source.device.path or 'n/a'}`",
                f"- source_dir: `{store.source_dir(source.device.id)}`",
                f"- latest_observation_file: `{latest_json_path or 'none'}`",
                f"- latest_payload_file: `{latest_payload_path or 'none'}`",
                "- note: Read the specific observation or payload file only when relevant. Device logs and payloads are not preloaded by tagging this id.",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _latest_device_artifacts(
    source: DeviceSource,
    *,
    store: ObservationStore,
) -> tuple[str | None, str | None]:
    latest_json = store.source_dir(source.device.id) / "latest.json"
    if not latest_json.is_file():
        return None, None
    payload = _latest_payload_ref(latest_json)
    return str(latest_json), payload


def _latest_payload_ref(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8")).get("payload_ref")
    except Exception:
        return None
    text = str(payload or "").strip()
    return text or None


def collect_device_observation(
    source: DeviceSource,
    *,
    store: ObservationStore,
    cfg: Mapping[str, object] | None = None,
    duration_seconds: int = 5,
    output_path: str | Path | None = None,
) -> Observation:
    if not source.enabled:
        raise RuntimeError(f"source @{source.primary_ref} is disabled")
    device = source.device
    if device.kind is PeripheralKind.CAMERA:
        captured = capture_snapshot(device, output_path=output_path)
        return save_frame_observation(captured, store=store)
    if device.kind is PeripheralKind.MICROPHONE:
        captured = record_clip(device, duration_seconds=duration_seconds, output_path=output_path)
        transcription_cfg = cfg.get("speech_to_text") if isinstance(cfg, Mapping) else None
        return process_audio_observation(captured, store=store, transcription_cfg=transcription_cfg)
    if device.kind is PeripheralKind.SENSOR and device.transport is PeripheralTransport.GPIO:
        return capture_gpio_text(device, store=store)
    raise RuntimeError(f"source type is not supported for direct capture yet: {device.kind.value}")


def capture_gpio_text(
    device: PeripheralDevice,
    *,
    store: ObservationStore,
) -> Observation:
    if device.kind is not PeripheralKind.SENSOR or device.transport is not PeripheralTransport.GPIO:
        raise ValueError("capture_gpio_text requires a GPIO sensor device")
    text = _read_gpio_text(device)
    buffer_path = store.append_text_buffer(
        device.id,
        text,
        buffer_name="gpio-buffer.txt",
        max_lines=400,
    )
    summary = _summarize_gpio_text(device, text)
    observation = Observation(
        source_id=device.id,
        source_type=device.kind.value,
        transport=device.transport.value,
        modality=ObservationModality.TEXT,
        summary=summary,
        payload_ref=str(buffer_path),
        metadata={
            **dict(device.metadata),
            "buffer_path": str(buffer_path),
            "buffer_name": "gpio-buffer.txt",
            "device_path": device.path,
            "source_modality": "gpio_text",
        },
    )
    return store.persist(observation)


def _read_gpio_text(device: PeripheralDevice) -> str:
    path = str(device.path or "").strip()
    if path:
        candidate = Path(path)
        if candidate.exists() and candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
            return _filter_gpio_text(text, device)
    gpioinfo = resolve_binary("gpioinfo")
    if not gpioinfo:
        raise RuntimeError("gpioinfo is required to inspect GPIO state")
    target = Path(path).name if path else _gpio_target_name(device)
    result = run_command((gpioinfo, target))
    text = result.stdout.strip() or result.stderr.strip()
    if not result.ok and not text:
        raise RuntimeError("gpioinfo failed to read GPIO state")
    if not text:
        raise RuntimeError("GPIO state output was empty")
    return _filter_gpio_text(text, device)


def _summarize_gpio_text(device: PeripheralDevice, text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line:
        return f"GPIO snapshot from {device.label}: {first_line[:140]}"
    return f"GPIO snapshot from {device.label}"


def _custom_aliases(cfg: Mapping[str, object] | None) -> dict[str, str]:
    raw = cfg.get("device_aliases") if isinstance(cfg, Mapping) else None
    if not isinstance(raw, dict):
        return {}
    aliases: dict[str, str] = {}
    for key, value in raw.items():
        alias = _normalize_alias(str(key))
        device_id = str(value or "").strip()
        if alias and device_id:
            aliases[alias] = device_id
    return aliases


def _disabled_device_ids(cfg: Mapping[str, object] | None) -> set[str]:
    raw = cfg.get("disabled_device_ids") if isinstance(cfg, Mapping) else None
    if not isinstance(raw, list):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def _default_ref(device: PeripheralDevice, counts: dict[str, int]) -> str:
    key = _default_ref_prefix(device)
    index = counts.get(key, 0)
    counts[key] = index + 1
    return f"{key}{index}"


def _default_ref_prefix(device: PeripheralDevice) -> str:
    if device.kind is PeripheralKind.CAMERA:
        return "camera"
    if device.kind is PeripheralKind.MICROPHONE:
        return "mic"
    if device.kind is PeripheralKind.SPEAKER:
        return "speaker"
    if device.transport is PeripheralTransport.GPIO:
        return "gpio"
    if device.transport is PeripheralTransport.I2C:
        return "i2c"
    if device.transport is PeripheralTransport.USB_SERIAL:
        return "serial"
    return "sensor"


def _device_basename_alias(device: PeripheralDevice) -> str | None:
    path = str(device.path or "").strip()
    if not path:
        return None
    return _normalize_alias(Path(path).name)


def _normalize_alias(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-")
    return cleaned.lower()


def _dedupe_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for ref in refs:
        normalized = ref.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _configured_gpio_sources(
    cfg: Mapping[str, object] | None,
    devices: list[PeripheralDevice],
    disabled_ids: set[str],
) -> list[DeviceSource]:
    raw = cfg.get("gpio_bindings") if isinstance(cfg, Mapping) else None
    if not isinstance(raw, list):
        return []

    gpio_devices = [device for device in devices if device.transport is PeripheralTransport.GPIO]
    sources: list[DeviceSource] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        name = _normalize_alias(str(entry.get("name", "")))
        if not name:
            continue
        chip_ref = str(entry.get("chip", "") or "").strip()
        chip_device = _match_gpio_chip(gpio_devices, chip_ref)
        if chip_device is None:
            continue
        lines = _normalize_gpio_lines(entry.get("lines"))
        line_names = _normalize_gpio_line_names(entry.get("line_names"))
        label = str(entry.get("label", "") or "").strip() or name
        device_id = f"sensor:gpio-binding:{name}"
        refs = _dedupe_refs([name, device_id])
        device = PeripheralDevice(
            id=device_id,
            kind=PeripheralKind.SENSOR,
            transport=PeripheralTransport.GPIO,
            label=label,
            path=chip_device.path,
            metadata={
                **dict(chip_device.metadata),
                "gpio_binding": True,
                "gpio_chip_id": chip_device.id,
                "gpio_chip_path": chip_device.path,
                "gpio_lines": lines,
                "gpio_line_names": line_names,
            },
        )
        sources.append(
            DeviceSource(
                primary_ref=refs[0],
                refs=tuple(refs),
                device=device,
                enabled=device.id not in disabled_ids,
            )
        )
    return sources


def _match_gpio_chip(devices: list[PeripheralDevice], reference: str) -> PeripheralDevice | None:
    needle = reference.strip().lstrip("@").lower()
    if not needle:
        return None
    for device in devices:
        path = str(device.path or "").strip().lower()
        candidates = {
            device.id.lower(),
            path,
            Path(path).name if path else "",
        }
        if path.startswith("/dev/gpiochip"):
            suffix = path.rsplit("gpiochip", 1)[-1]
            candidates.add(f"gpio{suffix}")
            candidates.add(f"gpiochip{suffix}")
        if needle in candidates:
            return device
    return None


def _normalize_gpio_lines(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    lines: list[int] = []
    for item in value:
        if isinstance(item, int):
            lines.append(item)
            continue
        text = str(item).strip()
        if not text:
            continue
        try:
            lines.append(int(text))
        except ValueError:
            continue
    return lines


def _normalize_gpio_line_names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        text = _normalize_alias(str(item))
        if text:
            names.append(text)
    return names


def _gpio_target_name(device: PeripheralDevice) -> str:
    chip_path = str(device.metadata.get("gpio_chip_path") or device.path or "").strip()
    if chip_path:
        return Path(chip_path).name
    return device.id.split(":", 1)[-1]


def _filter_gpio_text(text: str, device: PeripheralDevice) -> str:
    requested_lines = {
        int(line)
        for line in device.metadata.get("gpio_lines", [])
        if isinstance(line, int)
    }
    requested_names = {
        str(name).strip().lower()
        for name in device.metadata.get("gpio_line_names", [])
        if str(name).strip()
    }
    if not requested_lines and not requested_names:
        return text

    kept: list[str] = []
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if index == 0 and stripped:
            kept.append(line)
            continue
        match = re.search(r"\bline\s+(\d+):", line)
        if match and int(match.group(1)) in requested_lines:
            kept.append(line)
            continue
        normalized = _normalize_alias(line)
        if requested_names and any(name in normalized for name in requested_names):
            kept.append(line)
    return "\n".join(kept).strip() or text
