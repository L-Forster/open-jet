"""Slash command handling for the open-jet chat UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.widgets import RichLog

if TYPE_CHECKING:
    from .app import OpenJetApp


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    aliases: tuple[str, ...] = ()


class SlashCommandHandler:
    COMMANDS: tuple[CommandSpec, ...] = (
        CommandSpec(name="help", description="Show command help", aliases=("commands", "?")),
        CommandSpec(
            name="clear",
            description="Clear chat and restart llama-server (flush KV cache)",
            aliases=("reset",),
        ),
        CommandSpec(
            name="clear-chat",
            description="Clear chat only (keep current server/KV state)",
            aliases=("clear_messages",),
        ),
        CommandSpec(name="status", description="Show conversation/context status"),
        CommandSpec(name="condense", description="Manually condense older context"),
    )

    def __init__(self, app: OpenJetApp, banner: str) -> None:
        self.app = app
        self.banner = banner

    async def maybe_handle(self, text: str) -> bool:
        if not text.startswith("/"):
            return False

        log = self.app.query_one("#chat-log", RichLog)
        log.write(f"[bold green]> [/]{text}")
        log.write("")
        if self.app.session_logger:
            self.app.session_logger.log_event("slash_command", text=text)

        raw = text[1:].strip()
        if not raw:
            self._render_unknown(log, text)
            return True

        cmd = self.resolve_command(raw.split()[0])
        if cmd == "help":
            self._render_help(log)
            return True
        if cmd == "clear":
            await self._clear(log, reset_kv_cache=True)
            return True
        if cmd == "clear-chat":
            await self._clear(log, reset_kv_cache=False)
            return True
        if cmd == "status":
            self._status(log)
            return True
        if cmd == "condense":
            self._condense(log)
            return True

        self._render_unknown(log, text)
        return True

    def _render_help(self, log: RichLog) -> None:
        lines = ["[bold]Slash commands[/]"]
        for spec in self.COMMANDS:
            aliases = f" (aliases: {', '.join(f'/{a}' for a in spec.aliases)})" if spec.aliases else ""
            lines.append(f"  [green]/{spec.name}[/] - {spec.description}{aliases}")
        for line in lines:
            log.write(line)
        log.write("")

    async def _clear(self, log: RichLog, *, reset_kv_cache: bool) -> None:
        if self.app._awaiting_approval:
            log.write("[yellow]Cannot clear while a tool approval prompt is active.[/]")
            log.write("")
            return
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return
        if not self.app.agent:
            log.write("[bold red]error:[/] Agent not initialized.")
            log.write("")
            return

        self.app.agent.reset_conversation()
        log.clear()
        log.write(self.banner)
        log.write("  [dim]Conversation history cleared.[/]")

        kv_reset_ok = None
        if reset_kv_cache:
            if not self.app.client:
                log.write("  [bold red]KV cache reset skipped: client unavailable.[/]")
                kv_reset_ok = False
            else:
                log.write("  [dim]Resetting llama-server to flush KV cache...[/]")
                try:
                    await self.app.client.reset_kv_cache()
                    log.write("  [dim]KV cache reset complete.[/]")
                    kv_reset_ok = True
                except Exception as exc:
                    log.write(f"  [bold red]KV cache reset failed:[/] {exc}")
                    kv_reset_ok = False
        log.write("")

        if self.app.session_logger:
            self.app.session_logger.log_event(
                "conversation_cleared",
                reset_kv_cache=reset_kv_cache,
                kv_reset_ok=kv_reset_ok,
            )

    def _status(self, log: RichLog) -> None:
        if not self.app.agent:
            log.write("[dim]Agent not initialized.[/]")
            log.write("")
            return

        msg_count = self.app.agent.conversation_message_count()
        thinking = self.app._thinking_timer is not None
        log.write(
            "[dim]"
            f"Context messages (excluding system): {msg_count} | "
            f"generating: {'yes' if thinking else 'no'}"
            "[/]"
        )
        log.write("")

    def _condense(self, log: RichLog) -> None:
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return
        if not self.app.agent:
            log.write("[bold red]error:[/] Agent not initialized.")
            log.write("")
            return

        summary = self.app.agent.condense_context()
        log.write(f"[dim]{summary}[/]")
        log.write("")
        if self.app.session_logger:
            self.app.session_logger.log_event("manual_condense", summary=summary)

    def _render_unknown(self, log: RichLog, text: str) -> None:
        log.write(f"[yellow]Unknown command:[/] {text}")
        log.write("[dim]Run /help to list available commands.[/]")
        log.write("")

    def resolve_command(self, token: str) -> str | None:
        needle = token.strip().lower()
        if not needle:
            return None
        for spec in self.COMMANDS:
            if needle == spec.name or needle in spec.aliases:
                return spec.name
        return None

    def matching_commands(self, prefix: str) -> list[str]:
        needle = prefix.strip().lower()
        matches: list[str] = []
        for spec in self.COMMANDS:
            names = (spec.name, *spec.aliases)
            if any(name.startswith(needle) for name in names):
                matches.append(spec.name)
        return sorted(set(matches))

    def command_description(self, canonical_name: str) -> str:
        for spec in self.COMMANDS:
            if spec.name == canonical_name:
                return spec.description
        return ""
