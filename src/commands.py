"""Slash command handling for the open-jet chat UI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit.shortcuts import radiolist_dialog

from .config import save_config
from .model_profiles import get_model_profile, list_model_profiles, replace_model_profile
from .persistent_memory import build_system_prompt, load_persistent_memory, update_persistent_memory
from .runtime_registry import runtime_spec
from .setup import _prompt_text
from .surfaces.command_specs import COMMANDS, CommandSpec
from .theme import rich_text

if TYPE_CHECKING:
    from .app import OpenJetApp


class SlashCommandHandler:
    COMMANDS: tuple[CommandSpec, ...] = COMMANDS

    def __init__(self, app: OpenJetApp, banner: str) -> None:
        self.app = app
        self.banner = banner

    async def maybe_handle(self, text: str) -> bool:
        if not text.startswith("/"):
            return False

        log = self.app.query_one("#chat-log")
        log.write(f"{rich_text('> ', 'user')}{rich_text(text, 'command')}")
        log.write("")
        if self.app.session_logger:
            self.app.session_logger.record_slash_command(text)

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
        if cmd == "memory":
            await self._memory(log, arg)
            return True
        if cmd == "reasoning":
            self._reasoning(log, arg)
            return True
        if cmd == "air-gapped":
            self._air_gapped(log, arg)
            return True
        if cmd == "resume":
            await self._resume(log)
            return True
        if cmd == "setup":
            await self._setup(log)
            return True
        if cmd == "model":
            await self._model(log, arg)
            return True
        if cmd == "edit-model":
            await self._edit_model(log, arg)
            return True
        if cmd == "mode":
            self._mode(log, arg)
            return True
        if cmd == "skills":
            self._skills(log, arg)
            return True
        if cmd == "skill":
            self._skill(log, arg)
            return True
        if cmd == "step":
            self._step(log, arg)
            return True
        if cmd == "util":
            self._util(log, arg)
            return True

        self._render_unknown(log, text)
        return True

    def _render_help(self, log: Any) -> None:
        lines = [rich_text("Slash commands", "assistant")]
        for spec in self.COMMANDS:
            aliases = f" (aliases: {', '.join(f'/{a}' for a in spec.aliases)})" if spec.aliases else ""
            lines.append(f"  {rich_text('/' + spec.name, 'command')} {rich_text('- ' + spec.description + aliases, 'muted')}")
        for line in lines:
            log.write(line)
        log.write("")

    async def _clear(self, log: Any, *, reset_kv_cache: bool) -> None:
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

        self.app.agent.system_prompt = await build_system_prompt(
            str(self.app.cfg.get("system_prompt", "")),
            Path.cwd(),
        )
        self.app.agent.reset_conversation()
        self.app.agent.clear_turn_context()
        self.app.loaded_files.clear()
        self.app._pending_image_paths.clear()
        self.app.harness_state = type(self.app.harness_state)()
        self.app._turn_context_docs = []
        self.app._turn_context_tokens = 0
        log.clear()
        log.write(self.banner)
        log.write("  [bold bright_white]Conversation history cleared.[/]")

        kv_reset_ok = None
        if reset_kv_cache:
            if not self.app.client:
                log.write("  [bold red]KV cache reset skipped: client unavailable.[/]")
                kv_reset_ok = False
            else:
                log.write("  [bold bright_white]Resetting runtime to flush KV cache...[/]")
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
        self.app.persist_harness_state()
        self.app.persist_session_state(reason="clear_command")

    def _status(self, log: Any) -> None:
        snapshot = self.app.runtime_status_snapshot()
        if not snapshot.get("ready"):
            log.write("[bold bright_white]Agent not initialized.[/]")
            log.write(
                "[bold bright_white]"
                f"Air-gapped mode: {'true' if snapshot.get('airgapped') else 'false'}"
                "[/]"
            )
            log.write("")
            return

        log.write(
            "[bold bright_white]"
            f"Messages: {snapshot['messages']} | "
            f"Generating: {'yes' if snapshot['generating'] else 'no'}"
            "[/]"
        )
        if snapshot.get("command_in_progress"):
            log.write(
                "[bold bright_white]"
                f"Command: {snapshot.get('active_command') or 'running'}"
                "[/]"
            )
        if snapshot.get("reasoning_mode"):
            log.write(
                "[bold bright_white]"
                f"Reasoning mode: {snapshot.get('reasoning_mode')}"
                "[/]"
            )
        log.write(
            "[bold bright_white]"
            f"Air-gapped mode: {'true' if snapshot.get('airgapped') else 'false'}"
            "[/]"
        )
        log.write(
            "[bold bright_white]"
            f"Context tokens: {snapshot['context_tokens']}/{snapshot['context_window_tokens']} | "
            f"Prompt budget: {snapshot['prompt_budget_tokens']} | "
            f"Reserve: {snapshot['reserve_tokens']} | "
            f"Overhead: {snapshot.get('runtime_overhead_tokens', 0)} | "
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
        if snapshot.get("harness_mode"):
            log.write(
                "[bold bright_white]"
                f"Workflow: active_step={snapshot.get('harness_active_step') or 'n/a'} | "
                f"docs={snapshot.get('harness_doc_tokens', 0)}t | "
                f"state={snapshot.get('harness_state_summary_tokens', 0)}t | "
                f"docs_budget={snapshot.get('turn_docs_budget', 0)}t | "
                f"candidate_docs={snapshot.get('harness_candidate_count', 0)}"
                "[/]"
            )
            docs = snapshot.get("harness_docs") or []
            if docs:
                log.write(
                    "[bold bright_white]"
                    f"Loaded harness docs: {', '.join(str(doc) for doc in docs)}"
                    "[/]"
                )
            alerts = snapshot.get("harness_budget_alerts") or []
            for alert in alerts:
                log.write(f"[bold yellow]Budget alert:[/] {alert}")
        log.write("")

    async def _condense(self, log: Any) -> None:
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
            self.app.session_logger.record_manual_condense(summary)

    def _render_unknown(self, log: Any, text: str) -> None:
        log.write(f"{rich_text('Unknown command:', 'warning')} {rich_text(text, 'command')}")
        log.write(rich_text("Run /help to list available commands.", "muted"))
        log.write("")

    async def _load(self, log: Any, raw_arg: str) -> None:
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

    async def _setup(self, log: Any) -> None:
        if self.app._awaiting_approval:
            log.write("[yellow]Cannot open setup while a tool approval prompt is active.[/]")
            log.write("")
            return
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return
        self.app.run_setup_command_worker()

    async def _memory(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip()
        if not arg or arg == "show":
            snapshot = await load_persistent_memory(Path.cwd())
            log.write("[bold bright_white]Persistent user preferences:[/]")
            log.write(snapshot.user or "(empty)")
            log.write("")
            log.write("[bold bright_white]Persistent agent memory:[/]")
            log.write(snapshot.agent or "(empty)")
            log.write("")
            return

        parts = arg.split(maxsplit=1)
        if parts[0] != "clear" or len(parts) != 2:
            log.write("[yellow]Usage:[/] /memory [show|clear <user|agent>]")
            log.write("")
            return
        try:
            result = await update_persistent_memory(
                Path.cwd(),
                scope=parts[1],
                action="clear",
            )
        except ValueError as exc:
            log.write(f"[yellow]{exc}[/]")
            log.write("")
            return
        log.write(f"[bold bright_white]{result}[/]")
        log.write("")

    async def _model(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip()
        profiles = self.app.model_profiles()
        active = str(self.app.cfg.get("active_model_profile") or "").strip()

        if not arg:
            if not profiles:
                log.write("[yellow]No saved model presets yet. Run /setup to add one.[/]")
                log.write("")
                return
            if self.app._awaiting_approval:
                log.write("[yellow]Cannot switch models while a tool approval prompt is active.[/]")
                log.write("")
                return
            if self.app._thinking_timer:
                log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
                log.write("")
                return

            selected = await radiolist_dialog(
                title="Switch model preset",
                text="Use arrow keys to choose a saved model preset.",
                values=[
                    (
                        profile["name"],
                        f"{profile['name']}"
                        f"{' (active)' if profile['name'] == active else ''}"
                        f" | runtime={profile.get('runtime', 'llama_cpp')}"
                        f" | ctx={profile.get('context_window_tokens', 'n/a')}"
                        f" | gpu={profile.get('gpu_layers', 'n/a')}",
                    )
                    for profile in profiles
                ],
            ).run_async()
            if selected is None:
                log.write("[bold bright_white]Model switch cancelled.[/]")
                log.write("")
                return
            if str(selected).strip().lower() == active.lower():
                log.write(f"[bold bright_white]Model preset '{selected}' is already active.[/]")
                log.write("")
                return
            if not await self.app.activate_model_profile(str(selected), log):
                log.write(f"[yellow]Unknown model preset:[/] {selected}")
                log.write("")
            return

        if arg.lower() in {"status", "list"}:
            if not profiles:
                log.write("[yellow]No saved model presets yet. Run /setup to add one.[/]")
                log.write("")
                return
            log.write(f"[bold bright_white]Active model preset: {active or 'none'}[/]")
            for profile in profiles:
                marker = " (active)" if profile["name"] == active else ""
                model_ref = str(profile.get("ollama_model") or profile.get("model") or profile.get("llama_model") or "")
                log.write(
                    "[bold bright_white]"
                    f"- {profile['name']}{marker}: runtime={profile.get('runtime', 'llama_cpp')} "
                    f"context={profile.get('context_window_tokens', 'n/a')} gpu={profile.get('gpu_layers', 'n/a')} "
                    f"model={model_ref or 'n/a'}"
                    "[/]"
                )
            log.write("")
            return

        if self.app._awaiting_approval:
            log.write("[yellow]Cannot switch models while a tool approval prompt is active.[/]")
            log.write("")
            return
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return
        if not await self.app.activate_model_profile(arg, log):
            log.write(f"[yellow]Unknown model preset:[/] {arg}")
            log.write("")

    async def _edit_model(self, log: Any, raw_arg: str) -> None:
        if self.app._awaiting_approval:
            log.write("[yellow]Cannot edit model presets while a tool approval prompt is active.[/]")
            log.write("")
            return
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return

        target_name = raw_arg.strip() or str(self.app.cfg.get("active_model_profile") or "").strip()
        if not target_name:
            profiles = list_model_profiles(self.app.cfg)
            if len(profiles) == 1:
                target_name = profiles[0]["name"]
            else:
                log.write("[yellow]Usage:[/] /edit-model [name]")
                log.write("")
                return

        profile = get_model_profile(self.app.cfg, target_name)
        if not profile:
            log.write(f"[yellow]Unknown model preset:[/] {target_name}")
            log.write("")
            return

        runtime = str(profile.get("runtime", "llama_cpp"))
        model_source = str(profile.get("model_source", "local"))
        model_key = "ollama_model" if model_source == "ollama" else runtime_spec(runtime).model_config_key
        model_prompt = "ollama model> " if model_source == "ollama" else "model ref> "

        name_value = await _prompt_text(self.app._session, "model name> ", default=str(profile["name"]))
        model_value = await _prompt_text(self.app._session, model_prompt, default=str(profile.get(model_key) or profile.get("model") or ""))
        context_value = await _prompt_text(
            self.app._session,
            "context window> ",
            default=str(profile.get("context_window_tokens", 4096)),
        )
        gpu_value = str(profile.get("gpu_layers", 0))
        if runtime == "llama_cpp":
            gpu_value = await _prompt_text(self.app._session, "gpu layers> ", default=gpu_value)

        try:
            context_tokens = int(context_value.strip())
            gpu_layers = int(gpu_value.strip())
        except ValueError:
            log.write("[yellow]Context window and GPU layers must be integers.[/]")
            log.write("")
            return

        updated = dict(profile)
        updated["name"] = name_value.strip() or profile["name"]
        updated["context_window_tokens"] = context_tokens
        updated["gpu_layers"] = gpu_layers if runtime == "llama_cpp" else 0
        updated[model_key] = model_value.strip()
        if model_source != "ollama":
            updated["model"] = model_value.strip()
        else:
            updated["recommended_llm"] = model_value.strip()
            updated["model"] = str(profile.get("model") or "").strip()

        try:
            stored = replace_model_profile(self.app.cfg, updated, previous_name=profile["name"])
        except ValueError as exc:
            log.write(f"[yellow]{exc}[/]")
            log.write("")
            return

        active = str(self.app.cfg.get("active_model_profile") or "").strip()
        if active.lower() == str(profile["name"]).strip().lower():
            save_config(self.app.cfg)
            await self.app.activate_model_profile(stored["name"], log)
            return

        save_config(self.app.cfg)
        log.write(f"[bold bright_white]Model preset '{stored['name']}' updated.[/]")
        log.write("")

    async def _resume(self, log: Any) -> None:
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
        self.app._pending_image_paths.clear()
        log.clear()
        log.write(self.banner)
        if not self.app._restore_session_state(log):
            log.write("  [bold bright_white]No previous session state found.[/]")
        log.write("")
        self.app.refresh_token_counter()

    def _reasoning(self, log: Any, raw_arg: str) -> None:
        client = self.app.client
        setter = getattr(client, "set_reasoning_mode", None) if client else None
        getter = getattr(client, "reasoning_status", None) if client else None
        if not callable(setter) or not callable(getter):
            log.write("[yellow]Reasoning mode is only available for the llama.cpp runtime in this app.[/]")
            log.write("")
            return

        arg = raw_arg.strip().lower() or "status"
        if arg == "status":
            log.write(f"[bold bright_white]Reasoning mode: {getter()}[/]")
            log.write("")
            return
        if arg not in {"on", "off", "default"}:
            log.write("[yellow]Usage:[/] /reasoning [status|on|off|default]")
            log.write("")
            return
        setter(arg)
        log.write(f"[bold bright_white]Reasoning mode set to {arg}. Applies to future turns with the current model.[/]")
        log.write("")

    def _air_gapped(self, log: Any, raw_arg: str) -> None:
        if self.app._awaiting_approval:
            log.write("[yellow]Cannot change air-gapped mode while a tool approval prompt is active.[/]")
            log.write("")
            return
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return

        arg = raw_arg.strip().lower() or "status"
        if arg == "status":
            log.write(
                f"[bold bright_white]Air-gapped mode: {'true' if self.app.is_airgapped() else 'false'}[/]"
            )
            log.write("")
            return

        if arg not in {"true", "false"}:
            log.write("[yellow]Usage:[/] /air-gapped [status|true|false]")
            log.write("")
            return

        enabled = arg == "true"
        self.app.set_airgapped(enabled)
        if enabled:
            log.write(
                "[bold bright_white]Air-gapped mode set to true. External network access is now blocked.[/]"
            )
        else:
            log.write(
                "[bold bright_white]Air-gapped mode set to false. External network access is allowed again.[/]"
            )
        log.write("")

    def _mode(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip().lower()
        if not arg or arg == "status":
            log.write(f"[bold bright_white]Harness mode: {self.app.harness_state.mode}[/]")
            log.write("")
            return
        if arg not in {"chat", "code", "review", "debug"}:
            log.write("[yellow]Usage:[/] /mode [chat|code|review|debug|status]")
            log.write("")
            return
        self.app.set_harness_mode(arg)
        log.write(f"[bold bright_white]Harness mode set to {arg}.[/]")
        log.write("")

    def _skills(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip().lower()
        if not arg or arg == "status":
            selected = self.app.harness_state.preferred_skills
            available = ", ".join(self.app.available_harness_skills()) or "none"
            log.write(
                "[bold bright_white]"
                f"Selected skills: {', '.join(selected) if selected else 'none'}"
                "[/]"
            )
            log.write(f"[bold bright_white]Available skills: {available}[/]")
            log.write("")
            return
        if arg == "list":
            available = self.app.available_harness_skills()
            log.write(f"[bold bright_white]Available skills: {', '.join(available) if available else 'none'}[/]")
            log.write("")
            return
        if arg == "clear":
            self.app.clear_harness_skills()
            log.write("[bold bright_white]Selected harness skills cleared.[/]")
            log.write("")
            return
        log.write("[yellow]Usage:[/] /skills [status|list|clear]")
        log.write("")

    def _skill(self, log: Any, raw_arg: str) -> None:
        names = [part.strip() for part in raw_arg.split(",") if part.strip()]
        if not names:
            log.write("[yellow]Usage:[/] /skill <name[,name...]>")
            log.write("")
            return
        applied, missing = self.app.set_harness_skills(names)
        if applied:
            log.write(f"[bold bright_white]Selected skills: {', '.join(applied)}[/]")
        if missing:
            log.write(f"[yellow]Unknown skills:[/] {', '.join(missing)}")
        log.write("")

    def _step(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip().lower()
        if not arg or arg == "status":
            active = self.app.harness_active_step()
            log.write(f"[bold bright_white]Active step: {active or 'n/a'}[/]")
            log.write(f"[bold bright_white]Next action: {self.app.harness_state.next_action or 'n/a'}[/]")
            log.write("")
            return
        if arg == "next":
            self.app.advance_harness_step()
            log.write("[bold bright_white]Advanced to the next step.[/]")
            log.write("")
            return
        if arg == "split":
            self.app.split_harness_step()
            log.write("[bold bright_white]Split the active step into smaller turns.[/]")
            log.write("")
            return
        log.write("[yellow]Usage:[/] /step [status|next|split]")
        log.write("")

    def _util(self, log: Any, raw_arg: str) -> None:
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
