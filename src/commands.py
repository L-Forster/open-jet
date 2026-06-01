"""Slash command handling for the open-jet chat UI."""

from __future__ import annotations

import time
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit.shortcuts import radiolist_dialog

from .api_auth import ApiKeyStore, default_api_key_env, normalize_provider_id
from .codex_auth import CodexAuthError, CodexOAuthProvider
from .config import save_config
from .model_profiles import get_model_profile, list_model_profiles, replace_model_profile
from .runtime_registry import CODEX_RUNTIME, DEFAULT_RUNTIME, LITELLM_RUNTIME, active_runtime
from .memory_reflection import refresh_agent_system_prompt
from .peripherals.system import device_discovery_hint
from .persistent_memory import build_system_prompt, load_persistent_memory, update_persistent_memory
from .setup import _prompt_text
from .skills_registry import skills_manifest_path, sync_skills_manifest
from .surfaces.command_specs import COMMANDS, CommandSpec
from .theme import rich_text

if TYPE_CHECKING:
    from .app import OpenJetApp
    from .session_state import SavedChatEntry


_API_KEY_CONNECT_PROVIDERS = {
    "openai",
    "anthropic",
    "openrouter",
    "google",
    "xai",
    "mistral",
    "deepseek",
    "openai-compatible",
}

_DEFAULT_CODEX_PROFILE: dict[str, object] = {
    "name": "codex",
    "runtime": CODEX_RUNTIME,
    "provider": "openai-codex",
    "model": "gpt-5.5",
    "context_window_tokens": 272000,
    "reasoning_effort": "medium",
    "reasoning_summary": "auto",
    "text_verbosity": "medium",
}


def _ensure_codex_model_profile(cfg: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    existing = get_model_profile(cfg, "codex")
    if existing and active_runtime(existing) == CODEX_RUNTIME:
        return existing, False
    fallback = get_model_profile(cfg, "openai-codex")
    if fallback and active_runtime(fallback) == CODEX_RUNTIME:
        return fallback, False

    profile = dict(_DEFAULT_CODEX_PROFILE)
    if existing:
        profile["name"] = _unique_profile_name(cfg, "openai-codex")
    stored = replace_model_profile(cfg, profile)
    return stored, True


def _unique_profile_name(cfg: dict[str, Any], preferred: str) -> str:
    existing = {str(profile.get("name") or "").strip().lower() for profile in list_model_profiles(cfg)}
    base = preferred.strip() or "profile"
    if base.lower() not in existing:
        return base
    suffix = 2
    while f"{base}-{suffix}".lower() in existing:
        suffix += 1
    return f"{base}-{suffix}"


def _runtime_kind(profile: dict[str, Any]) -> str:
    return "local" if active_runtime(profile) == DEFAULT_RUNTIME else "cloud"


def _preferred_runtime_profile(
    profiles: list[dict[str, Any]],
    *,
    kind: str,
    active: str,
) -> dict[str, Any] | None:
    candidates = [profile for profile in profiles if _runtime_kind(profile) == kind]
    if not candidates:
        return None
    active_key = active.strip().lower()
    for profile in candidates:
        if str(profile.get("name") or "").strip().lower() == active_key:
            return profile
    if kind == "cloud":
        for profile in candidates:
            if str(profile.get("name") or "").strip().lower() in {"codex", "openai-codex"}:
                return profile
    return candidates[0]


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
        if cmd == "usage":
            self._usage(log)
            return True
        if cmd == "device":
            self._device(log, arg)
            return True
        if cmd == "device-add":
            self._device(log, f"add {arg}".strip())
            return True
        if cmd == "device-on":
            self._device(log, f"on {arg}".strip())
            return True
        if cmd == "device-off":
            self._device(log, f"off {arg}".strip())
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
        if cmd == "telemetry":
            self._telemetry(log, arg)
            return True
        if cmd == "connect":
            await self._connect(log, arg)
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
        if cmd == "runtime":
            await self._runtime(log, arg)
            return True
        if cmd == "local":
            await self._runtime(log, "local")
            return True
        if cmd == "cloud":
            await self._cloud(log, arg)
            return True
        if cmd == "edit-model":
            await self._edit_model(log, arg)
            return True
        if cmd == "mode":
            self._mode(log, arg)
            return True
        if cmd == "plan":
            self._plan(log, arg)
            return True
        if cmd == "mcp":
            self._mcp(log, arg)
            return True
        if cmd == "skills":
            await self._skills(log, arg)
            return True
        if cmd == "skill":
            await self._skill(log, arg)
            return True
        if cmd == "todo":
            self._todo(log, arg)
            return True
        if cmd == "util":
            self._util(log, arg)
            return True

        self._render_unknown(log, text)
        return True

    def _render_help(self, log: Any) -> None:
        lines = [rich_text("Slash commands", "assistant")]
        for spec in self.COMMANDS:
            if spec.hidden:
                continue
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
            "",
            Path.cwd(),
            cfg=self.app.cfg,
        )
        self.app.agent.reset_conversation()
        self.app.agent.clear_turn_context()
        self.app.loaded_files.clear()
        self.app._pending_image_paths.clear()
        self.app.harness_state = type(self.app.harness_state)()
        self.app._turn_context_docs = []
        self.app._turn_context_tokens = 0
        self.app._start_new_chat_session()
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

    def _device(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip()
        if not arg or arg.lower() == "list":
            self._devices(log)
            return

        parts = arg.split(maxsplit=1)
        action = parts[0].strip().lower()
        rest = parts[1].strip() if len(parts) > 1 else ""
        if action == "help":
            self._device_help(log)
            return
        if action == "add":
            self._device_add(log, rest)
            return
        if action == "on":
            self._device_toggle(log, rest, enabled=True)
            return
        if action == "off":
            self._device_toggle(log, rest, enabled=False)
            return
        self._device_help(log)

    def _device_help(self, log: Any) -> None:
        log.write("[bold bright_white]Device commands:[/]")
        log.write("[bold bright_white]- /device[/] list currently discovered devices and current ids")
        log.write("[bold bright_white]- /device add <existing_id> <new_id>[/] assign a persistent chat id")
        log.write("[bold bright_white]- /device on <id>[/] enable a device")
        log.write("[bold bright_white]- /device off <id>[/] disable a device")
        log.write("[bold bright_white]- /devices[/] is still an alias for `/device`")
        log.write("")

    def _devices(self, log: Any) -> None:
        registry_path = self.app.write_devices_registry()
        sources = self.app.list_device_sources()
        log.write(f"[bold bright_white]Device registry:[/] {registry_path}")
        if not sources:
            log.write("[yellow]No devices detected.[/]")
            hint = device_discovery_hint()
            if hint:
                log.write(f"[yellow]{hint}[/]")
            log.write("[bold bright_white]Persistent setup uses `open-jet device ...` outside chat.[/]")
            log.write("")
            return
        log.write("[bold bright_white]Discovered devices:[/]")
        for source in sources:
            aliases = ", ".join(f"@{ref}" for ref in source.refs)
            label = source.device.label
            kind = source.device.kind.value
            transport = source.device.transport.value
            state = "enabled" if source.enabled else "disabled"
            log.write(
                "[bold bright_white]"
                f"- {source.primary_ref}: {label} | tag=@{source.primary_ref} | kind={kind} | transport={transport} | state={state} | refs={aliases}"
                "[/]"
            )
        log.write(
            "[bold bright_white]"
            "Use the current id on the left with `/device add <existing_id> <new_id>` or "
            "`open-jet device add <existing_id> <new_id>` if you want a different stable chat id."
            "[/]"
        )
        log.write("")

    def _device_add(self, log: Any, raw_arg: str) -> None:
        parts = raw_arg.split()
        if len(parts) != 2:
            log.write("[yellow]Usage:[/] /device add <existing_id> <new_id>")
            log.write("")
            return
        source_ref, device_id = parts
        try:
            source = self.app.assign_device_alias(source_ref, device_id)
        except ValueError as exc:
            log.write(f"[yellow]{_format_device_error(exc)}[/]")
            log.write("")
            return
        save_config(self.app.cfg)
        registry_path = self.app.write_devices_registry()
        log.write(
            "[bold bright_white]"
            f"Saved device id {source.primary_ref} for {source.device.label}."
            "[/]"
        )
        log.write(f"[bold bright_white]Device registry:[/] {registry_path}")
        log.write(f"[bold bright_white]Use `@{source.primary_ref}` in chat to reference this device.[/]")
        log.write("")

    def _device_toggle(self, log: Any, raw_arg: str, *, enabled: bool) -> None:
        source_ref = raw_arg.strip()
        if not source_ref:
            usage = "/device on <id>" if enabled else "/device off <id>"
            log.write(f"[yellow]Usage:[/] {usage}")
            log.write("")
            return
        try:
            source = self.app.set_device_enabled(source_ref, enabled)
        except ValueError as exc:
            log.write(f"[yellow]{_format_device_error(exc)}[/]")
            log.write("")
            return
        save_config(self.app.cfg)
        registry_path = self.app.write_devices_registry()
        state = "enabled" if source.enabled else "disabled"
        log.write(
            "[bold bright_white]"
            f"Device {source.primary_ref} is now {state}."
            "[/]"
        )
        log.write(f"[bold bright_white]Device registry:[/] {registry_path}")
        log.write("")

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
            log.write("[bold bright_white]Global user preferences:[/]")
            log.write(snapshot.global_user or "(empty)")
            log.write("")
            log.write("[bold bright_white]Global agent memory:[/]")
            log.write(snapshot.global_agent or "(empty)")
            log.write("")
            log.write("[bold bright_white]Local user memory:[/]")
            log.write(snapshot.project_user or "(empty)")
            log.write("")
            log.write("[bold bright_white]Local agent memory:[/]")
            log.write(snapshot.project_agent or "(empty)")
            log.write("")
            return

        parts = arg.split()
        action = parts[0].lower()
        if action == "show":
            location = "project"
            scope = ""
            if len(parts) == 2:
                if parts[1].lower() in {"global", "project"}:
                    location = parts[1].lower()
                else:
                    scope = parts[1]
            elif len(parts) == 3:
                location = parts[1].lower()
                scope = parts[2]
            elif len(parts) > 3:
                log.write("[yellow]Usage:[/] /memory [show [global|project] [user|agent]|clear [global] <user|agent>]")
                log.write("")
                return
            try:
                snapshot = await load_persistent_memory(Path.cwd())
                if scope:
                    text = snapshot.read(location=location, scope=scope)
                    log.write(
                        f"[bold bright_white]{location.title()} {scope.strip().lower()} memory:[/]"
                    )
                    log.write(text or "(empty)")
                    log.write("")
                    return
            except ValueError as exc:
                log.write(f"[yellow]{exc}[/]")
                log.write("")
                return
            selected = (
                (f"{location.title()} user preferences", snapshot.read(location=location, scope="user")),
                (f"{location.title()} agent memory", snapshot.read(location=location, scope="agent")),
            )
            for title, content in selected:
                log.write(f"[bold bright_white]{title}:[/]")
                log.write(content or "(empty)")
                log.write("")
            return

        if action != "clear" or len(parts) not in {2, 3}:
            log.write("[yellow]Usage:[/] /memory [show [global|project] [user|agent]|clear [global] <user|agent>]")
            log.write("")
            return
        location = "project"
        scope = parts[1]
        if len(parts) == 3:
            location = parts[1]
            scope = parts[2]
        try:
            result = await update_persistent_memory(
                Path.cwd(),
                scope=scope,
                action="clear",
                location=location,
            )
        except ValueError as exc:
            log.write(f"[yellow]{exc}[/]")
            log.write("")
            return
        if self.app.agent:
            await refresh_agent_system_prompt(self.app.agent)
        log.write(f"[bold bright_white]{result}[/]")
        log.write("")

    async def _model(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip()
        profiles = self.app.model_profiles()
        active = str(self.app.cfg.get("active_model_profile") or "").strip()

        if not arg or arg.lower() in {"status", "list"}:
            if not profiles:
                log.write("[yellow]No saved model presets yet. Run /setup to add one.[/]")
                log.write("")
                return
            log.write(f"[bold bright_white]Active model preset: {active or 'none'}[/]")
            self._write_model_profile_list(log, profiles, active=active)
            if not arg:
                log.write("[dim]Use /model <preset> to switch model preset, or /runtime local|cloud to switch runtime.[/]")
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
            if arg.strip().lower() in {"codex", "openai-codex"} and self._ensure_connected_codex_profile():
                if await self.app.activate_model_profile("codex", log):
                    return
            log.write(f"[yellow]Unknown model preset:[/] {arg}")
            log.write("")

    async def _runtime(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip().lower() or "status"
        if arg in {"cloud", "codex", "openai-codex"}:
            await self._switch_runtime_kind(log, "cloud")
            return
        if arg in {"local", "llama", "llama_cpp", "llama-cpp"}:
            await self._switch_runtime_kind(log, "local")
            return
        if arg not in {"status", "list"}:
            log.write("[yellow]Usage:[/] /runtime [status|local|cloud]")
            log.write("")
            return

        profiles = self.app.model_profiles()
        active = str(self.app.cfg.get("active_model_profile") or "").strip()
        active_runtime_name = active_runtime(self.app.cfg)
        log.write(f"[bold bright_white]Active runtime:[/] {active_runtime_name} ({active or 'no preset'})")
        local_profiles = [profile for profile in profiles if _runtime_kind(profile) == "local"]
        cloud_profiles = [profile for profile in profiles if _runtime_kind(profile) == "cloud"]
        log.write("[bold bright_white]Local profiles:[/]")
        self._write_model_profile_list(log, local_profiles, active=active, empty="none")
        log.write("[bold bright_white]Cloud profiles:[/]")
        self._write_model_profile_list(log, cloud_profiles, active=active, empty="none")
        log.write("[dim]Use /runtime local or /runtime cloud. Shortcuts: /local, /cloud.[/]")
        log.write("")

    def _usage(self, log: Any) -> None:
        snapshot = self.app.lifetime_token_usage_snapshot()
        total = snapshot["total"]
        if not total["total_tokens"]:
            log.write("[bold bright_white]No model token usage recorded yet.[/]")
            log.write("")
            return

        log.write("[bold bright_white]Token usage:[/]")
        log.write(
            "[bold bright_white]"
            f"Total: {total['prompt_tokens']:,} input | "
            f"{total['completion_tokens']:,} output | "
            f"{total['total_tokens']:,} tokens | "
            f"{total['sessions']:,} saved sessions | "
            f"{snapshot['api_cost']} API Cost"
            "[/]"
        )
        if snapshot.get("has_unsplit_history"):
            log.write("[dim]Some older saved usage has no per-model breakdown.[/]")
        for row in snapshot["models"]:
            log.write(
                "[bold bright_white]"
                f"- {row['label']}: "
                f"{row['prompt_tokens']:,} input | "
                f"{row['completion_tokens']:,} output | "
                f"{row['total_tokens']:,} tokens | "
                f"{row['runtime_requests']:,} requests"
                "[/]"
            )
        if not snapshot["models"]:
            log.write("[dim]Per-model breakdown starts after this version records model-level usage.[/]")
        log.write("")

    async def _switch_runtime_kind(self, log: Any, kind: str) -> None:
        if self.app._awaiting_approval:
            log.write("[yellow]Cannot switch runtime while a tool approval prompt is active.[/]")
            log.write("")
            return
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return

        profiles = self.app.model_profiles()
        if kind == "cloud" and not any(_runtime_kind(profile) == "cloud" for profile in profiles):
            self._ensure_connected_codex_profile()
            profiles = self.app.model_profiles()
        target = _preferred_runtime_profile(profiles, kind=kind, active=str(self.app.cfg.get("active_model_profile") or ""))
        if target is None:
            if kind == "cloud":
                log.write("[yellow]No cloud runtime profile found. Run /connect openai-codex first.[/]")
            else:
                log.write("[yellow]No local runtime profile found. Run /setup to add one.[/]")
            log.write("")
            return

        active = str(self.app.cfg.get("active_model_profile") or "").strip()
        if str(target["name"]).strip().lower() == active.lower():
            log.write(f"[bold bright_white]Already using {kind} runtime: {target['name']}.[/]")
            log.write("")
            return
        if not await self.app.activate_model_profile(str(target["name"]), log):
            log.write(f"[yellow]Runtime switch failed:[/] unknown preset {target['name']}")
            log.write("")
            return

    async def _cloud(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip()
        if not arg:
            await self._switch_runtime_kind(log, "cloud")
            return

        parts = arg.split(maxsplit=1)
        action = parts[0].strip().lower()
        rest = parts[1].strip() if len(parts) > 1 else ""
        if action in {"status", "list", "profiles"}:
            self._cloud_status(log)
            return
        if action == "model":
            await self._set_cloud_model(log, rest)
            return
        if action == "add":
            await self._add_cloud_profile(log)
            return

        profile = get_model_profile(self.app.cfg, arg)
        if profile and _runtime_kind(profile) == "cloud":
            if not await self.app.activate_model_profile(str(profile["name"]), log):
                log.write(f"[yellow]Cloud switch failed:[/] {profile['name']}")
                log.write("")
            return

        log.write("[yellow]Usage:[/] /cloud [status|model <model>|add|<profile>]")
        log.write("")

    def _cloud_status(self, log: Any) -> None:
        profiles = [profile for profile in self.app.model_profiles() if _runtime_kind(profile) == "cloud"]
        active = str(self.app.cfg.get("active_model_profile") or "").strip()
        if not profiles:
            log.write("[yellow]No cloud runtime profiles found. Run /connect openai-codex or /cloud add.[/]")
            log.write("")
            return
        log.write(f"[bold bright_white]Active cloud profile: {active if active_runtime(self.app.cfg) != DEFAULT_RUNTIME else 'none'}[/]")
        self._write_model_profile_list(log, profiles, active=active)
        log.write("[dim]Use /cloud to switch to cloud, /cloud <profile> to pick one, or /cloud model <model> to edit the current cloud profile.[/]")
        log.write("")

    async def _set_cloud_model(self, log: Any, model: str) -> None:
        if self.app._awaiting_approval:
            log.write("[yellow]Cannot change cloud model while a tool approval prompt is active.[/]")
            log.write("")
            return
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return
        model = model.strip()
        if not model:
            log.write("[yellow]Usage:[/] /cloud model <model>")
            log.write("")
            return

        profiles = self.app.model_profiles()
        if not any(_runtime_kind(profile) == "cloud" for profile in profiles):
            self._ensure_connected_codex_profile()
            profiles = self.app.model_profiles()
        profile = _preferred_runtime_profile(
            profiles,
            kind="cloud",
            active=str(self.app.cfg.get("active_model_profile") or ""),
        )
        if profile is None:
            log.write("[yellow]No cloud runtime profile found. Run /connect openai-codex or /cloud add first.[/]")
            log.write("")
            return

        previous_cfg = dict(self.app.cfg)
        updated = dict(profile)
        updated["model"] = model
        updated.pop("llama_model", None)
        try:
            stored = replace_model_profile(self.app.cfg, updated, previous_name=str(profile["name"]))
        except ValueError as exc:
            log.write(f"[yellow]{exc}[/]")
            log.write("")
            return

        save_config(self.app.cfg)
        active = str(self.app.cfg.get("active_model_profile") or "").strip().lower()
        if active == str(stored["name"]).strip().lower():
            if await self.app.activate_model_profile(str(stored["name"]), log):
                return
            self.app.cfg = previous_cfg
            save_config(self.app.cfg)
            try:
                await self.app._init_client()
            except Exception:
                pass
            return
        log.write(f"[bold bright_white]Cloud model for '{stored['name']}' set to {model}.[/]")
        log.write("")

    async def _add_cloud_profile(self, log: Any) -> None:
        if self.app._awaiting_approval:
            log.write("[yellow]Cannot add cloud profile while a tool approval prompt is active.[/]")
            log.write("")
            return
        if self.app._thinking_timer:
            log.write("[yellow]Wait for the current generation to finish, then retry.[/]")
            log.write("")
            return

        existing_names = {str(profile.get("name") or "").strip().lower() for profile in self.app.model_profiles()}
        default_name = "codex" if "codex" not in existing_names else "cloud"
        name_value = await _prompt_text(self.app._session, "cloud profile name> ", default=default_name)
        provider_value = await _prompt_text(self.app._session, "provider> ", default="openai-codex")
        provider_id = normalize_provider_id(provider_value) if provider_value else "openai-codex"
        if provider_id in {"codex", "openai_codex"}:
            provider_id = "openai-codex"

        is_codex = provider_id == "openai-codex"
        default_model = "gpt-5.5" if is_codex else f"{provider_id}/"
        model_value = await _prompt_text(self.app._session, "model> ", default=default_model)
        context_default = "272000" if is_codex else "128000"
        context_value = await _prompt_text(self.app._session, "context window> ", default=context_default)
        try:
            context_tokens = int(context_value.strip())
        except ValueError:
            log.write("[yellow]Context window must be an integer.[/]")
            log.write("")
            return

        profile: dict[str, Any] = {
            "name": name_value.strip() or default_name,
            "runtime": CODEX_RUNTIME if is_codex else LITELLM_RUNTIME,
            "provider": "openai-codex" if is_codex else provider_id,
            "model": model_value.strip(),
            "context_window_tokens": context_tokens,
        }
        if is_codex:
            profile.update(
                {
                    "reasoning_effort": "medium",
                    "reasoning_summary": "auto",
                    "text_verbosity": "medium",
                }
            )
        else:
            base_url_value = await _prompt_text(self.app._session, "base url> ", default="")
            env_default = default_api_key_env(provider_id)
            env_value = await _prompt_text(self.app._session, "api key env> ", default=env_default)
            if base_url_value.strip():
                profile["base_url"] = base_url_value.strip()
            if env_value.strip():
                profile["api_key_env"] = env_value.strip()

        try:
            stored = replace_model_profile(self.app.cfg, profile)
        except ValueError as exc:
            log.write(f"[yellow]{exc}[/]")
            log.write("")
            return
        save_config(self.app.cfg)
        log.write(
            "[bold bright_white]"
            f"Cloud profile '{stored['name']}' added. Use /cloud {stored['name']} to switch."
            "[/]"
        )
        log.write("")

    @staticmethod
    def _write_model_profile_list(
        log: Any,
        profiles: list[dict[str, Any]],
        *,
        active: str,
        empty: str = "",
    ) -> None:
        if not profiles:
            if empty:
                log.write(f"[dim]- {empty}[/]")
            return
        for profile in profiles:
            marker = " (active)" if profile["name"] == active else ""
            runtime = str(profile.get("runtime") or DEFAULT_RUNTIME)
            model_ref = str(profile.get("model") or profile.get("llama_model") or "")
            log.write(
                "[bold bright_white]"
                f"- {profile['name']}{marker}: "
                f"runtime={runtime} context={profile.get('context_window_tokens', 'n/a')} "
                f"gpu={profile.get('gpu_layers', 'n/a')} "
                f"model={model_ref or 'n/a'}"
                "[/]"
            )

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

        name_value = await _prompt_text(self.app._session, "model name> ", default=str(profile["name"]))
        model_value = await _prompt_text(
            self.app._session,
            "model ref> ",
            default=str(profile.get("model") or profile.get("llama_model") or ""),
        )
        context_value = await _prompt_text(
            self.app._session,
            "context window> ",
            default=str(profile.get("context_window_tokens", 4096)),
        )
        gpu_value = await _prompt_text(
            self.app._session,
            "gpu layers> ",
            default=str(profile.get("gpu_layers", 0)),
        )

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
        updated["gpu_layers"] = gpu_layers
        if active_runtime(updated) in {CODEX_RUNTIME, LITELLM_RUNTIME}:
            updated["model"] = model_value.strip()
            updated.pop("llama_model", None)
        else:
            updated["llama_model"] = model_value.strip()

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
        entries = self.app.list_resume_candidates()
        if not entries:
            log.write("  [bold bright_white]No saved chats found under .openjet/state.[/]")
            log.write("")
            self.app.refresh_token_counter()
            return

        selected = await radiolist_dialog(
            title="Resume saved chat",
            text="Choose a saved chat checkpoint to load back into the transcript and runtime.",
            values=[(str(entry.state_path), self._format_resume_entry(entry)) for entry in entries],
        ).run_async()
        if selected is None:
            log.write("[bold bright_white]Resume cancelled.[/]")
            log.write("")
            return

        self.app.loaded_files.clear()
        self.app._pending_image_paths.clear()
        log.clear()
        log.write(self.banner)
        if not await self.app.restore_saved_chat(str(selected), log):
            log.write("  [bold bright_white]Saved chat could not be loaded.[/]")
        log.write("")
        self.app.refresh_token_counter()

    @staticmethod
    def _format_resume_entry(entry: "SavedChatEntry") -> str:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.saved_at))
        model_name = Path(entry.model_ref).name or entry.model_ref or "unknown"
        checkpoint = "KV checkpoint" if entry.kv_cache_available else "transcript only"
        preview = entry.preview or "(no preview)"
        return (
            f"{stamp} | {entry.message_count} messages | {checkpoint} | "
            f"model={model_name} | {preview}"
        )

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

    def _telemetry(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip().lower() or "status"
        telemetry_cfg = self.app.cfg.get("telemetry", {}) or {}
        broadcast_cfg = telemetry_cfg.get("broadcast", {}) or {}
        if arg == "status":
            current = self.app._telemetry_consent_status() or "undecided"
            endpoint = str(broadcast_cfg.get("endpoint", "")).strip() or "(unset)"
            broadcast_active = bool(
                self.app.session_logger
                and self.app.session_logger.broadcast.enabled
                and self.app.session_logger.broadcast.endpoint
            )
            log.write(f"[bold bright_white]Telemetry consent: {current}[/]")
            log.write(f"[dim]Broadcast active: {broadcast_active} — endpoint: {endpoint}[/]")
            if self.app.is_airgapped():
                log.write("[dim]Air-gapped mode is on; telemetry is disabled regardless of consent.[/]")
            log.write("")
            return

        if arg not in {"on", "off", "grant", "granted", "deny", "denied", "true", "false"}:
            log.write("[yellow]Usage:[/] /telemetry [status|on|off]")
            log.write("")
            return

        decision = "granted" if arg in {"on", "grant", "granted", "true"} else "denied"
        self.app._persist_telemetry_consent(decision)
        if decision == "granted":
            log.write("[bold bright_white]Telemetry on — anonymous usage data only.[/]")
            if self.app.is_airgapped():
                log.write("[dim]Air-gapped mode is on; nothing will be sent until you disable it.[/]")
        else:
            log.write("[bold bright_white]Telemetry off.[/]")
        log.write("")

    async def _connect(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip().lower() or "status"
        parts = arg.split()
        api_key_store = ApiKeyStore()
        codex_provider = CodexOAuthProvider()
        if arg == "status":
            status = codex_provider.status()
            if status.get("logged_in"):
                expiry = float(status.get("expires_at") or 0)
                remaining = max(0, int(expiry - time.time()))
                account = str(status.get("account_id") or "").strip()
                account_suffix = f" account={account}" if account else ""
                log.write(
                    "[bold bright_white]"
                    f"OpenAI Codex: connected{account_suffix} expires_in={remaining}s storage={status.get('storage')}"
                    "[/]"
                )
            else:
                log.write(f"[bold bright_white]OpenAI Codex: not connected storage={status.get('storage')}[/]")
            api_status = api_key_store.status(
                ["openai", "anthropic", "openrouter", "google", "xai", "mistral", "deepseek", "openai-compatible"]
            )
            for provider_id, provider_status in api_status.items():
                source = "env" if provider_status.get("env_present") else ("stored" if provider_status.get("stored") else "missing")
                env_name = provider_status.get("env") or "n/a"
                log.write(
                    "[bold bright_white]"
                    f"{provider_id}: {source} env={env_name} storage={provider_status.get('storage')}"
                    "[/]"
                )
            log.write("")
            return
        if arg in {"logout openai-codex", "logout codex", "openai-codex logout"}:
            codex_provider.logout()
            log.write("[bold bright_white]OpenAI Codex disconnected.[/]")
            log.write("")
            return
        if len(parts) == 2 and parts[0] == "logout":
            provider_id = normalize_provider_id(parts[1])
            if provider_id in _API_KEY_CONNECT_PROVIDERS:
                if not api_key_store.clear_key(provider_id):
                    log.write(f"[bold red]{provider_id} API key removal failed:[/] OS keyring is unavailable.")
                    log.write("")
                    return
                log.write(f"[bold bright_white]{provider_id} API key removed from OpenJet storage.[/]")
                log.write("")
                return
        codex_login_requested = bool(parts and parts[0] in {"openai-codex", "codex"})
        codex_login_args = set(parts[1:])
        codex_args_supported = codex_login_requested and codex_login_args.issubset({"--device-auth", "device-auth"})
        if codex_login_requested and codex_args_supported:
            if self.app.is_airgapped():
                log.write("[yellow]Air-gapped mode blocks OpenAI Codex login.[/]")
                log.write("")
                return
            try:
                credentials = await codex_provider.login_browser(device_auth=bool(codex_login_args))
            except CodexAuthError as exc:
                log.write(f"[bold red]OpenAI Codex login failed:[/] {exc}")
                log.write("")
                return
            remaining = max(0, int(credentials.expires_at - time.time()))
            _, added_profile = _ensure_codex_model_profile(self.app.cfg)
            if added_profile:
                save_config(self.app.cfg)
            profile_suffix = " Model preset 'codex' added." if added_profile else ""
            log.write(f"[bold bright_white]OpenAI Codex connected. Token expires in {remaining}s.{profile_suffix}[/]")
            log.write("")
            return
        provider_id = normalize_provider_id(arg)
        if provider_id not in _API_KEY_CONNECT_PROVIDERS:
            log.write(
                "[yellow]Usage:[/] "
                "/connect [status|openai-codex [--device-auth]|openai|anthropic|openrouter|logout <provider>]"
            )
            log.write("")
            return
        if self.app.is_airgapped():
            log.write("[yellow]Air-gapped mode blocks API provider login.[/]")
            log.write("")
            return
        env_name = default_api_key_env(provider_id)
        if env_name and os.environ.get(env_name):
            log.write(f"[bold bright_white]{provider_id} is already available from {env_name}.[/]")
            log.write("")
            return
        api_key = await _prompt_text(self.app._session, f"{provider_id} api key> ", is_password=True)
        if not api_key.strip():
            log.write("[yellow]API key not saved.[/]")
            log.write("")
            return
        try:
            api_key_store.save_key(provider_id, api_key.strip())
        except ValueError as exc:
            log.write(f"[bold red]API key save failed:[/] {exc}")
            log.write("")
            return
        log.write(f"[bold bright_white]{provider_id} API key saved.[/]")
        log.write("")

    def _ensure_connected_codex_profile(self) -> bool:
        status = CodexOAuthProvider().status()
        if not status.get("logged_in"):
            return False
        _, added_profile = _ensure_codex_model_profile(self.app.cfg)
        if added_profile:
            save_config(self.app.cfg)
        return True

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

    def _plan(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip().lower()
        state = self.app.harness_state
        if not arg or arg == "status":
            log.write(
                "[bold bright_white]"
                f"Plan mode: {'on' if state.plan_mode else 'off'} | approved={'yes' if state.plan_approved else 'no'}"
                "[/]"
            )
            if state.plan_summary:
                log.write(f"[bold bright_white]Plan summary:[/] {state.plan_summary}")
            log.write("")
            return
        if arg == "on":
            self.app.enter_harness_plan_mode()
            log.write("[bold bright_white]Plan mode enabled. Edits are blocked until the plan is approved.[/]")
            log.write("")
            return
        if arg == "approve":
            if not state.plan_summary.strip():
                log.write("[yellow]Plan approval requires a recorded plan summary first.[/]")
                log.write("")
                return
            self.app.approve_harness_plan()
            log.write("[bold bright_white]Plan approved. Edit tools are available again.[/]")
            log.write("")
            return
        if arg == "reject":
            self.app.reject_harness_plan()
            log.write("[bold bright_white]Plan approval cleared. Plan mode remains read-only.[/]")
            log.write("")
            return
        log.write("[yellow]Usage:[/] /plan [status|on|approve|reject]")
        log.write("")

    def _mcp(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip().lower() or "status"
        if arg != "status":
            log.write("[yellow]Usage:[/] /mcp status")
            log.write("")
            return
        manager = getattr(self.app, "mcp_manager", None)
        if manager is None:
            from .mcp_support.manager import MCPManager

            manager = MCPManager.from_sources(runtime_cfg=self.app.cfg)
        for line in manager.format_status().splitlines():
            log.write(f"[bold bright_white]{line}[/]")
        log.write("")

    async def _skills(self, log: Any, raw_arg: str) -> None:
        await self._skill(log, raw_arg)

    async def _skill(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip()
        lowered = arg.lower()
        load_only = lowered.startswith("load ")
        if not arg or lowered == "status":
            manifest = sync_skills_manifest(Path.cwd())
            selected = self.app.harness_state.preferred_skills
            available = ", ".join(self.app.available_harness_skills()) or "none"
            log.write(f"[bold bright_white]Selected skills: {', '.join(selected) if selected else 'none'}[/]")
            log.write(f"[bold bright_white]Available skills: {available}[/]")
            log.write(f"[bold bright_white]Skills manifest:[/] {manifest}")
            log.write("")
            return
        if lowered == "list":
            manifest = sync_skills_manifest(Path.cwd())
            available = self.app.available_harness_skills()
            log.write(f"[bold bright_white]Available skills: {', '.join(available) if available else 'none'}[/]")
            log.write(f"[bold bright_white]Skills manifest:[/] {manifest}")
            log.write("")
            return
        if lowered == "clear":
            self.app.clear_harness_skills()
            log.write("[bold bright_white]Selected harness skills cleared.[/]")
            log.write("")
            return
        names_arg = arg[5:].strip() if load_only else arg
        names = [part.strip() for part in names_arg.split(",") if part.strip()]
        if not names:
            log.write("[yellow]Usage:[/] /skill [status|list|clear|load <name[,name...]>|<name[,name...]>]")
            log.write("")
            return
        loaded: list[str] = []
        if load_only:
            available = set(self.app.available_harness_skills())
            requested = [Path(name).stem for name in names if Path(name).stem]
            missing = [name for name in requested if name not in available]
            for name in requested:
                if name in available and await self.app.load_skill_into_context(name, log):
                    loaded.append(name)
            if loaded:
                log.write(f"[bold bright_white]Loaded into current chat: {', '.join(loaded)}[/]")
            if missing:
                log.write(f"[yellow]Unknown skills:[/] {', '.join(missing)}")
            log.write(f"[bold bright_white]Skills manifest:[/] {skills_manifest_path(Path.cwd())}")
            log.write("")
            return

        applied, missing = self.app.set_harness_skills(names)
        for name in applied:
            if await self.app.load_skill_into_context(name, log):
                loaded.append(name)
        if applied:
            log.write(f"[bold bright_white]Selected skills: {', '.join(applied)}[/]")
        if loaded:
            log.write(f"[bold bright_white]Loaded into current chat: {', '.join(loaded)}[/]")
        if missing:
            log.write(f"[yellow]Unknown skills:[/] {', '.join(missing)}")
        log.write(f"[bold bright_white]Skills manifest:[/] {skills_manifest_path(Path.cwd())}")
        log.write("")

    def _todo(self, log: Any, raw_arg: str) -> None:
        arg = raw_arg.strip().lower()
        state = self.app.harness_state
        if not arg or arg == "status":
            if not state.todos:
                log.write("[bold bright_white]Todo ledger: empty[/]")
                log.write("")
                return
            log.write("[bold bright_white]Todo ledger:[/]")
            for todo in state.todos:
                files = f" | files={', '.join(todo.files)}" if todo.files else ""
                log.write(
                    "[bold bright_white]"
                    f"- {todo.id}: {todo.content} | status={todo.status} | kind={todo.kind}{files}"
                    "[/]"
                )
            log.write("")
            return
        if arg == "clear":
            self.app.clear_harness_todos()
            log.write("[bold bright_white]Todo ledger cleared.[/]")
            log.write("")
            return
        log.write("[yellow]Usage:[/] /todo [status|clear]")
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
        return sorted(spec.name for spec in self.COMMANDS if not spec.hidden and spec.name.startswith(needle))

    def command_description(self, canonical_name: str) -> str:
        for spec in self.COMMANDS:
            if spec.name == canonical_name:
                return spec.description
        return ""


def _format_device_error(exc: ValueError) -> str:
    text = str(exc)
    if text.startswith("unknown device reference: "):
        return text
    if text.startswith("unknown source: "):
        return "unknown device reference: " + text.split(": ", 1)[1]
    return text
