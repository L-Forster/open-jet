"""Slash command handling for the open-jet chat UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        CommandSpec(name="exit", description="Quit the app", aliases=("quit",)),
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
        CommandSpec(name="status", description="Show runtime memory/context status", aliases=("stats",)),
        CommandSpec(name="condense", description="Manually condense older context"),
        CommandSpec(name="load", description="Load file into context: /load <path>", aliases=("add",)),
        CommandSpec(name="resume", description="Load previous session state into chat"),
        CommandSpec(name="setup", description="Open setup wizard and restart runtime"),
        CommandSpec(
            name="util",
            description="Show/hide utilization line: /util [show|hide|toggle|status]",
            aliases=("usage",),
        ),
    )

    def __init__(self, app: OpenJetApp, banner: str) -> None:
        self.app = app
        self.banner = banner

    async def maybe_handle(self, text: str) -> bool:
        if not text.startswith("/"):
            return False

        log = self.app.query_one("#chat-log")
        log.write(f"[bold green]> [/]{text}")
        log.write("")
        if self.app.session_logger:
            self.app.session_logger.log_event("slash_command", text=text)

        raw = text[1:].strip()
        if not raw:
            self._render_unknown(log, text)
            return True

        parts = raw.split(maxsplit=1)
        cmd = self.resolve_command(parts[0])
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "help":
            self._render_help(log)
            return True
        if cmd == "exit":
            await self.app.action_quit()
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
            await self._condense(log)
            return True
        if cmd == "load":
            await self._load(log, arg)
            return True
        if cmd == "resume":
            await self._resume(log)
            return True
        if cmd == "setup":
            await self._setup(log)
            return True
        if cmd == "util":
            self._util(log, arg)
            return True

        self._render_unknown(log, text)
        return True

    def _render_help(self, log) -> None:
        lines = ["[bold]Slash commands[/]"]
        for spec in self.COMMANDS:
            aliases = f" (aliases: {', '.join(f'/{a}' for a in spec.aliases)})" if spec.aliases else ""
            lines.append(f"  [green]/{spec.name}[/] - {spec.description}{aliases}")
        for line in lines:
            log.write(line)
        log.write("")

    async def _clear(self, log, *, reset_kv_cache: bool) -> None:
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
        self.app.loaded_files.clear()
        log.clear()
        log.write(self.banner)
        log.write("  [bold bright_white]Conversation history cleared.[/]")

        kv_reset_ok = None
        if reset_kv_cache:
            if not self.app.client:
                log.write("  [bold red]KV cache reset skipped: client unavailable.[/]")
                kv_reset_ok = False
            else:
                log.write("  [bold bright_white]Resetting llama-server to flush KV cache...[/]")
                try:
                    await self.app.client.reset_kv_cache()
                    log.write("  [bold bright_white]KV cache reset complete.[/]")
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
        self.app.persist_session_state(reason="clear_command")

    def _status(self, log) -> None:
        snapshot = self.app.runtime_status_snapshot()
        if not snapshot.get("ready"):
            log.write("[bold bright_white]Agent not initialized.[/]")
            log.write("")
            return

        log.write(
            "[bold bright_white]"
            f"Messages: {snapshot['messages']} | "
            f"Generating: {'yes' if snapshot['generating'] else 'no'}"
            "[/]"
        )
        log.write(
            "[bold bright_white]"
            f"Context tokens: {snapshot['context_tokens']}/{snapshot['context_window_tokens']} | "
            f"Prompt budget: {snapshot['prompt_budget_tokens']} | "
            f"Reserve: {snapshot['reserve_tokens']} | "
            f"Remaining: {snapshot['remaining_prompt_tokens']}"
            "[/]"
        )
        total_mb = snapshot.get("memory_total_mb")
        available_mb = snapshot.get("memory_available_mb")
        used_percent = snapshot.get("memory_used_percent")
        if total_mb is not None and available_mb is not None and used_percent is not None:
            log.write(
                "[bold bright_white]"
                f"RAM: {available_mb:.0f}MB free / {total_mb:.0f}MB total "
                f"({used_percent:.1f}% used)"
                "[/]"
            )
        log.write("")

    async def _condense(self, log) -> None:
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return
        if not self.app.agent:
            log.write("[bold red]error:[/] Agent not initialized.")
            log.write("")
            return

        summary = await self.app.agent.condense_context(force=True)
        log.write(f"[bold bright_white]{summary}[/]")
        log.write("")
        self.app.refresh_token_counter()
        self.app.persist_session_state(reason="manual_condense")
        if self.app.session_logger:
            self.app.session_logger.log_event("manual_condense", summary=summary)

    def _render_unknown(self, log, text: str) -> None:
        log.write(f"[yellow]Unknown command:[/] {text}")
        log.write("[bold bright_white]Run /help to list available commands.[/]")
        log.write("")

    async def _load(self, log, raw_arg: str) -> None:
        path = raw_arg.strip()
        if not path:
            log.write("[yellow]Usage:[/] /load <path>")
            log.write("")
            return

        if path.startswith("@[") and path.endswith("]"):
            path = path[2:-1].strip()
        elif path.startswith("@"):
            path = path[1:].strip()

        ok = await self.app.load_context_file(path, log)
        if not ok:
            log.write("[bold bright_white]Use /status to inspect current budget and memory.[/]")
            log.write("")

    async def _setup(self, log) -> None:
        if self.app._awaiting_approval:
            log.write("[yellow]Cannot open setup while a tool approval prompt is active.[/]")
            log.write("")
            return
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return
        self.app.run_setup_command_worker()

    async def _resume(self, log) -> None:
        if self.app._awaiting_approval:
            log.write("[yellow]Cannot resume while a tool approval prompt is active.[/]")
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
        self.app.loaded_files.clear()
        log.clear()
        log.write(self.banner)
        if not self.app._restore_session_state(log):
            log.write("  [bold bright_white]No previous session state found.[/]")
        log.write("")
        self.app.refresh_token_counter()

    def _util(self, log, raw_arg: str) -> None:
        action = raw_arg.strip().lower()
        if not action or action == "toggle":
            visible = self.app.toggle_utilization_visible()
            state = "shown" if visible else "hidden"
            log.write(f"[bold bright_white]Utilization line {state}.[/]")
            log.write("")
            return

        if action == "show":
            self.app.set_utilization_visible(True)
            log.write("[bold bright_white]Utilization line shown.[/]")
            log.write("")
            return

        if action == "hide":
            self.app.set_utilization_visible(False)
            log.write("[bold bright_white]Utilization line hidden.[/]")
            log.write("")
            return

        if action == "status":
            state = "shown" if self.app.is_utilization_visible() else "hidden"
            log.write(f"[bold bright_white]Utilization line is currently {state}.[/]")
            log.write("")
            return

        log.write("[yellow]Usage:[/] /util [show|hide|toggle|status]")
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
        return sorted(spec.name for spec in self.COMMANDS if spec.name.startswith(needle))

    def command_description(self, canonical_name: str) -> str:
        for spec in self.COMMANDS:
            if spec.name == canonical_name:
                return spec.description
        return ""
