from __future__ import annotations

import json
from pathlib import Path

from .lockfile import ensure_hub_layout
from .model import HubTap


def load_taps(hub_root: Path | None = None) -> tuple[HubTap, ...]:
    root = ensure_hub_layout(hub_root)
    try:
        payload = json.loads((root / "taps.json").read_text(encoding="utf-8") or "[]")
    except json.JSONDecodeError:
        payload = []
    if not isinstance(payload, list):
        return ()
    return tuple(HubTap.from_dict(item) for item in payload if isinstance(item, dict))


def save_taps(taps: tuple[HubTap, ...], hub_root: Path | None = None) -> Path:
    root = ensure_hub_layout(hub_root)
    path = root / "taps.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps([tap.to_dict() for tap in taps], ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    return path
