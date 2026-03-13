from __future__ import annotations

import json
from pathlib import Path


def write_debug_runtime_messages(
    *,
    root: Path,
    turn_id: str,
    messages: list[dict],
) -> Path:
    debug_dir = root / ".openjet" / "state" / "debug_prompts"
    debug_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(messages, ensure_ascii=False, indent=2)
    target = debug_dir / f"{turn_id}.messages.json"
    target.write_text(payload, encoding="utf-8")

    latest = debug_dir / "latest.messages.json"
    latest.write_text(payload, encoding="utf-8")
    return target


def write_debug_context_snapshot(
    *,
    root: Path,
    turn_id: str,
    snapshot: dict,
) -> Path:
    debug_dir = root / ".openjet" / "state" / "debug_prompts"
    debug_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
    target = debug_dir / f"{turn_id}.context.json"
    target.write_text(payload, encoding="utf-8")

    latest = debug_dir / "latest.context.json"
    latest.write_text(payload, encoding="utf-8")
    return target
