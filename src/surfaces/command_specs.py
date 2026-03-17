"""Shared slash-command metadata without UI dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    aliases: tuple[str, ...] = ()


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(name="help", description="Show command help", aliases=("commands", "?")),
    CommandSpec(name="exit", description="Quit the app", aliases=("quit",)),
    CommandSpec(
        name="clear",
        description="Clear chat and restart runtime (flush KV cache)",
        aliases=("reset",),
    ),
    CommandSpec(
        name="clear-chat",
        description="Clear chat only (keep current server/KV state)",
        aliases=("clear_messages",),
    ),
    CommandSpec(name="status", description="Show runtime memory/context status", aliases=("stats",)),
    CommandSpec(name="condense", description="Manually condense older context"),
    CommandSpec(name="load", description="Load file into context: /load <path>", aliases=("add",)),
    CommandSpec(name="memory", description="Inspect or update persistent memory: /memory [show|clear <user|agent>]"),
    CommandSpec(name="reasoning", description="Show or set llama.cpp reasoning mode: /reasoning [status|on|off|default]"),
    CommandSpec(name="air-gapped", description="Show or set air-gapped mode: /air-gapped [status|true|false]", aliases=("airgapped",)),
    CommandSpec(name="resume", description="Load previous session state into chat"),
    CommandSpec(name="setup", description="Open setup wizard and restart runtime"),
    CommandSpec(
        name="model",
        description="Show or switch saved model presets: /model [status|list|<name>]",
        aliases=("models",),
    ),
    CommandSpec(name="edit-model", description="Edit a saved model preset: /edit-model [name]"),
    CommandSpec(name="mode", description="Show or set harness mode: /mode [chat|code|review|debug|status]; shell stays approval-gated in chat"),
    CommandSpec(name="skills", description="Show or clear selected harness skills: /skills [status|list|clear]"),
    CommandSpec(name="skill", description="Pin one or more harness skills: /skill <name[,name...]>"),
    CommandSpec(name="step", description="Inspect or control the active step: /step [status|next|split]"),
    CommandSpec(
        name="util",
        description="Show/hide utilization line: /util [show|hide|toggle|status]",
        aliases=("usage",),
    ),
)
