"""Shared slash-command metadata without UI dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    aliases: tuple[str, ...] = ()
    hidden: bool = False


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
    CommandSpec(
        name="device",
        description="List or configure devices: /device [list|add <existing_id> <new_id>|on <id>|off <id>|help]",
        aliases=("devices", "sources"),
    ),
    CommandSpec(
        name="device-add",
        description="Compatibility alias for /device add <existing_id> <new_id>",
        aliases=("source-name",),
        hidden=True,
    ),
    CommandSpec(
        name="device-on",
        description="Compatibility alias for /device on <id>",
        aliases=("source-on",),
        hidden=True,
    ),
    CommandSpec(
        name="device-off",
        description="Compatibility alias for /device off <id>",
        aliases=("source-off",),
        hidden=True,
    ),
    CommandSpec(name="condense", description="Manually condense older context"),
    CommandSpec(name="load", description="Load file into context: /load <path>", aliases=("add",)),
    CommandSpec(
        name="memory",
        description="Inspect or update persistent memory: /memory [show [global|project] [user|agent]|clear [global] <user|agent>]",
    ),
    CommandSpec(name="reasoning", description="Show or set llama.cpp reasoning mode: /reasoning [status|on|off|default]"),
    CommandSpec(name="air-gapped", description="Show or set air-gapped mode: /air-gapped [status|true|false]", aliases=("airgapped",)),
    CommandSpec(name="resume", description="Pick and load a saved chat back into chat/runtime"),
    CommandSpec(name="setup", description="Open setup wizard and restart runtime"),
    CommandSpec(
        name="model",
        description="Show or switch saved model presets: /model [status|list|<name>]",
        aliases=("models",),
    ),
    CommandSpec(name="edit-model", description="Edit a saved model preset: /edit-model [name]"),
    CommandSpec(name="mode", description="Show or set harness mode: /mode [chat|code|review|debug|status]; shell stays approval-gated in chat"),
    CommandSpec(name="plan", description="Inspect or control plan mode: /plan [status|on|approve|reject]"),
    CommandSpec(name="skills", description="Alias for /skill [status|list|clear|load <name[,name...]>|<name[,name...]>]"),
    CommandSpec(name="skill", description="Inspect, load, or pin harness skills: /skill [status|list|clear|load <name[,name...]>|<name[,name...]>]"),
    CommandSpec(name="todo", description="Inspect or clear the todo ledger: /todo [status|clear]"),
    CommandSpec(
        name="util",
        description="Show/hide utilization line: /util [show|hide|toggle|status]",
        aliases=("usage",),
    ),
)
