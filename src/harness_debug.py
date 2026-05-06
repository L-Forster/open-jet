from __future__ import annotations

from pathlib import Path

import yaml


class _DebugPromptDumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.nodes.ScalarNode:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_DebugPromptDumper.add_representer(str, _represent_str)


def _dump_yaml(payload: object) -> str:
    return yaml.dump(
        payload,
        Dumper=_DebugPromptDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )


def write_debug_runtime_messages(
    *,
    root: Path,
    turn_id: str,
    messages: list[dict],
) -> Path:
    debug_dir = root / ".openjet" / "state" / "debug_prompts"
    debug_dir.mkdir(parents=True, exist_ok=True)

    payload = _dump_yaml(messages)
    target = debug_dir / f"{turn_id}.messages.yaml"
    target.write_text(payload, encoding="utf-8")

    latest = debug_dir / "latest.messages.yaml"
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

    payload = _dump_yaml(snapshot)
    target = debug_dir / f"{turn_id}.context.yaml"
    target.write_text(payload, encoding="utf-8")

    latest = debug_dir / "latest.context.yaml"
    latest.write_text(payload, encoding="utf-8")
    return target
