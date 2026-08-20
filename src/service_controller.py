"""Narrow OpenJet service boundary for the Pi-owned TypeScript agent loop."""

from __future__ import annotations

import asyncio
import copy
import shlex
import time
from pathlib import Path
from typing import Any

from .airgap import airgapped_from_cfg, set_airgapped
from .config import load_config, save_config
from .codex_auth import CodexAuthError, CodexOAuthProvider
from .device_sources import assign_device_alias, list_device_sources, set_device_enabled
from .hardware import detect_hardware_info, recommended_context_window_tokens
from .model_profiles import apply_model_profile, get_model_profile, list_model_profiles, replace_model_profile, sync_active_model_profile
from .provisioning import provision_setup_artifacts
from .runtime_registry import CODEX_RUNTIME, DEFAULT_CODEX_MODEL, DEFAULT_RUNTIME, LITELLM_RUNTIME, active_model_ref, active_runtime, create_runtime_client
from .runtime_protocol import ToolCall
from .runtime_limits import estimate_tokens, read_memory_snapshot
from .setup import build_recommended_payload
from .session_logging import SessionLogger
from .surfaces.command_specs import COMMANDS
from .system_metrics import SystemMetricsReader
from .tool_executor import execute_tool
from .tools.registry import get_tool_spec


class ServiceError(RuntimeError):
    """User-visible service failure."""


OPENJET_PI_TOOL_NAMES = (
    "device_list",
    "camera_snapshot",
    "microphone_record",
    "microphone_set_enabled",
    "gpio_read",
    "sensor_read",
    "system_info",
    "memory",
    "skills_list",
    "skill_view",
)

QWEN38_NATIVE_CONTEXT_TOKENS = 262_144
QWEN38_FINAL_RESPONSE_TOKENS = 131_072
QWEN38_THINKING_SAMPLING: dict[str, float | int] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
QWEN38_INSTRUCT_SAMPLING: dict[str, float | int] = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}


def _is_qwen38_model(model_ref: str) -> bool:
    normalized = model_ref.lower().replace("_", ".").replace("-", ".")
    return "qwen3.8" in normalized


class _SetupLog:
    """Small Rich-compatible sink used by headless provisioning."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: object) -> None:
        self.lines.append(str(text))


class OpenJetServiceController:
    """Own hardware/runtime services while Pi owns turns, tools, and sessions."""

    def __init__(self, emit, *, force_setup: bool = False, root: Path | None = None) -> None:
        self.emit = emit
        self.force_setup = force_setup
        self.root = Path(root or Path.cwd()).resolve()
        self.cfg = load_config()
        self.cfg["airgapped"] = airgapped_from_cfg(self.cfg)
        set_airgapped(bool(self.cfg["airgapped"]))
        self.runtime_client = None
        self._codex_credentials = None
        self._seen_request_ids: set[str] = set()
        self._terminal_size = (80, 24)
        self._started_at = time.monotonic()
        self._metrics = SystemMetricsReader()
        self._decode_started_at: float | None = None
        self._decode_last_token_at: float | None = None
        self._decode_tokens = 0
        self._last_generation_tps: float | None = None
        self._session_logger: SessionLogger | None = None

    @property
    def configured(self) -> bool:
        return bool(active_model_ref(self.cfg))

    def claim_request_id(self, request_id: str) -> None:
        if request_id in self._seen_request_ids:
            raise ServiceError(f"Duplicate request id: {request_id}")
        self._seen_request_ids.add(request_id)

    async def initialize(self) -> dict[str, Any]:
        await self._start_session_logger()
        setup_required = self.force_setup or not self.configured
        if not setup_required:
            await self._prepare_agent_mode()
        return self._snapshot(setup_required=setup_required)

    async def _start_session_logger(self) -> None:
        if self._session_logger is not None:
            return
        log_cfg = self.cfg.get("logging", {})
        if not isinstance(log_cfg, dict) or not log_cfg.get("enabled", True):
            return
        base_dir = Path(str(log_cfg.get("directory", ".openjet/state/sessions")))
        telemetry_cfg = self.cfg.get("telemetry", {})
        install_id = Path(str(
            telemetry_cfg.get("install_id_path", ".openjet/state/telemetry_identity.json")
            if isinstance(telemetry_cfg, dict) else ".openjet/state/telemetry_identity.json"
        ))
        self._session_logger = SessionLogger(
            base_dir=base_dir if base_dir.is_absolute() else self.root / base_dir,
            label=str(log_cfg.get("label", "open-jet-pi")),
            metrics_interval_seconds=float(log_cfg.get("metrics_interval_seconds", 5)),
            install_id_path=install_id if install_id.is_absolute() else self.root / install_id,
            retention_days=int(log_cfg.get("retention_days", 30)) if log_cfg.get("retention_days") is not None else None,
            max_sessions=int(log_cfg.get("max_sessions", 100)) if log_cfg.get("max_sessions") is not None else None,
            entrypoint="pi-tui",
        )
        await self._session_logger.start()

    def record_agent_trace(self, payload: dict[str, Any]) -> None:
        """Persist model-attributed Pi events in the standard session trace."""
        if self._session_logger is None:
            return
        event = str(payload.get("event") or "activity").strip()
        turn_id = str(payload.get("turnId") or "").strip() or None
        data = payload.get("data")
        self._session_logger.record_agent_trace(
            event,
            data if isinstance(data, dict) else {},
            turn_id=turn_id,
        )

    def _agent_mode(self) -> str:
        mode = str(self.cfg.get("agent_mode") or "local").strip().lower()
        return mode if mode in {"local", "codex", "hybrid"} else "local"

    def _profile_for_runtime(self, runtime: str, preferred_key: str) -> dict[str, Any] | None:
        preferred = str(self.cfg.get(preferred_key) or "").strip().lower()
        profiles = [profile for profile in list_model_profiles(self.cfg) if active_runtime(profile) == runtime]
        for profile in profiles:
            if str(profile.get("name") or "").strip().lower() == preferred:
                return profile
        return profiles[0] if profiles else None

    async def _prepare_agent_mode(self) -> None:
        mode = self._agent_mode()
        if mode in {"local", "hybrid"}:
            local = self._profile_for_runtime(DEFAULT_RUNTIME, "agent_local_profile")
            if local is None:
                raise ServiceError("Local and Slipstream agent modes require a saved local model profile.")
            # Re-apply the selected profile even when its model path is unchanged;
            # reasoning and sampling settings are part of the model configuration.
            apply_model_profile(self.cfg, local)
            self.cfg["agent_local_profile"] = str(local["name"])
            await self._prepare_runtime()
        elif self.runtime_client is not None:
            await self.runtime_client.close()
            self.runtime_client = None
        if mode in {"codex", "hybrid"}:
            codex = self._profile_for_runtime(CODEX_RUNTIME, "agent_codex_profile")
            if codex is None:
                raise ServiceError("Codex and Slipstream agent modes require a Codex profile. Run /connect openai-codex.")
            self.cfg["agent_codex_profile"] = str(codex["name"])
            try:
                self._codex_credentials = await CodexOAuthProvider().credentials()
            except CodexAuthError as exc:
                raise ServiceError(str(exc)) from exc

    async def _prepare_runtime(self) -> None:
        if self.runtime_client is not None:
            return
        runtime = active_runtime(self.cfg)
        if runtime not in {DEFAULT_RUNTIME, LITELLM_RUNTIME}:
            raise ServiceError(
                f"The Pi loop adapter does not yet support the {runtime!r} profile. "
                "Switch to a local llama.cpp or OpenAI-compatible profile."
            )
        if runtime == DEFAULT_RUNTIME:
            await self.emit(
                "status_update",
                {"text": "checking local llama.cpp runtime…", "payload": {"runtimeStarting": True}},
            )
            hardware_info = detect_hardware_info()
            log = _SetupLog()
            resolved = await provision_setup_artifacts(
                dict(self.cfg),
                hardware_info=hardware_info,
                log=log,
                set_status=lambda _text: None,
                clear_status=lambda: None,
            )
            for key in (
                "device",
                "llama_server_path",
                "llama_cpp_ref",
                "llama_mtp",
                "setup_missing_runtime",
                "context_window_tokens",
            ):
                if key in resolved:
                    self.cfg[key] = resolved[key]
            sync_active_model_profile(self.cfg)
            save_config(self.cfg)
        client = create_runtime_client(self.cfg)
        try:
            await self.emit("status_update", {"text": "starting OpenJet runtime…", "payload": {"runtimeStarting": True}})
            await client.start()
            self.runtime_client = client
        except Exception:
            await client.close()
            raise

    def _pi_model(self) -> dict[str, Any] | None:
        if self.runtime_client is None:
            return None
        runtime = active_runtime(self.cfg)
        base_url = str(getattr(self.runtime_client, "base_url", "") or self.cfg.get("base_url") or "").rstrip("/")
        if runtime == DEFAULT_RUNTIME:
            base_url = f"{base_url}/v1"
        if not base_url:
            return None
        model_ref = active_model_ref(self.cfg)
        model_id = Path(model_ref).name or model_ref or "openjet-local"
        reasoning = bool(self.cfg.get("reasoning", True))
        qwen38 = _is_qwen38_model(model_ref)
        context_window = max(1, int(self.cfg.get("context_window_tokens", 32768)))
        configured_max_tokens = self.cfg.get("max_tokens")
        requested_max_tokens = (
            int(configured_max_tokens)
            if configured_max_tokens is not None
            else QWEN38_FINAL_RESPONSE_TOKENS if qwen38 else 8192
        )
        # Pi leaves 4096 context tokens as a safety margin. Do not advertise
        # an output ceiling that cannot fit even with an empty conversation.
        max_tokens = min(max(1, requested_max_tokens), max(1, context_window - 4096))
        model: dict[str, Any] = {
            "provider": "openjet-local" if runtime == DEFAULT_RUNTIME else str(self.cfg.get("provider") or "openjet-compatible"),
            "id": model_id,
            "name": model_id,
            "api": "openai-completions",
            "apiKey": "openjet-local" if runtime == DEFAULT_RUNTIME else str(self.cfg.get("api_key") or "openjet"),
            "baseUrl": base_url,
            "reasoning": reasoning,
            "input": ["text", "image"],
            "contextWindow": context_window,
            "maxTokens": max_tokens,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "compat": {
                "supportsDeveloperRole": False,
                "supportsReasoningEffort": False,
                "supportsUsageInStreaming": True,
                "maxTokensField": "max_tokens",
                "thinkingFormat": "qwen-chat-template",
            },
        }
        if qwen38:
            model["samplingParams"] = dict(
                QWEN38_THINKING_SAMPLING if reasoning else QWEN38_INSTRUCT_SAMPLING
            )
            model["nativeContextWindow"] = QWEN38_NATIVE_CONTEXT_TOKENS
        return model

    def _pi_codex_model(self) -> dict[str, Any] | None:
        profile = self._profile_for_runtime(CODEX_RUNTIME, "agent_codex_profile")
        credentials = self._codex_credentials
        if profile is None or credentials is None:
            return None
        model_id = str(profile.get("model") or DEFAULT_CODEX_MODEL)
        headers = {"originator": "openjet"}
        if credentials.account_id:
            headers["ChatGPT-Account-Id"] = credentials.account_id
        return {
            "provider": "openai-codex",
            "id": model_id,
            "name": model_id,
            "api": "openai-codex-responses",
            "apiKey": credentials.access_token,
            "baseUrl": str(profile.get("codex_base_url") or "https://chatgpt.com/backend-api"),
            "headers": headers,
            "reasoning": True,
            "thinkingLevel": str(profile.get("reasoning_effort") or "medium"),
            "input": ["text", "image"],
            "contextWindow": int(profile.get("context_window_tokens") or 272000),
            "maxTokens": int(profile.get("max_tokens") or 128000),
            "thinkingLevelMap": {"minimal": None, "xhigh": "xhigh", "max": "max"},
            "cost": {"input": 5, "output": 30, "cacheRead": 0.5, "cacheWrite": 6.25},
            "compat": {"supportsOpenAIGrammarTools": True, "supportsAdditionalTools": True, "supportsToolSearch": True},
        }

    def _snapshot(self, *, setup_required: bool | None = None) -> dict[str, Any]:
        power_watts, power_percent = self._metrics.read_power_metrics()
        memory = read_memory_snapshot()
        battery = self._metrics.read_battery_metrics()
        thermal = self._metrics.read_thermal_metrics()
        mode = self._agent_mode()
        local_model = self._pi_model()
        codex_model = self._pi_codex_model()
        primary_model = local_model if mode == "local" else codex_model
        return {
            "agentEngine": "pi",
            "agentMode": mode,
            "workspace": str(self.root),
            "runtime": active_runtime(self.cfg),
            "model": active_model_ref(self.cfg),
            "airgapped": bool(self.cfg.get("airgapped", False)),
            "setupRequired": (not self.configured) if setup_required is None else setup_required,
            "piModel": primary_model,
            "localModel": local_model,
            "codexModel": codex_model,
            "openjetTools": self._pi_tools(),
            "modelProfiles": [
                {
                    "name": str(profile.get("name") or ""),
                    "kind": "local" if active_runtime(profile) == DEFAULT_RUNTIME else "codex" if active_runtime(profile) == CODEX_RUNTIME else "cloud",
                    "model": active_model_ref(profile),
                }
                for profile in list_model_profiles(self.cfg)
            ],
            "agentLocalProfile": str(self.cfg.get("agent_local_profile") or ""),
            "agentCodexProfile": str(self.cfg.get("agent_codex_profile") or ""),
            "localReasoning": bool((self._profile_for_runtime(DEFAULT_RUNTIME, "agent_local_profile") or {}).get("reasoning", self.cfg.get("reasoning", True))),
            "codexModelOptions": ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
            "commands": [
                {"name": spec.name, "description": spec.description, "aliases": list(spec.aliases)}
                for spec in COMMANDS
                if not spec.hidden
            ],
            "status": {
                "uptimeSeconds": round(time.monotonic() - self._started_at, 1),
                "terminalWidth": self._terminal_size[0],
                "terminalHeight": self._terminal_size[1],
                "powerWatts": power_watts,
                "powerPercent": power_percent,
                "cpuPercent": self._metrics.read_cpu_percent(),
                "memoryPercent": memory.used_percent if memory else None,
                "batteryPercent": battery.get("capacity_pct") if battery else None,
                "batteryStatus": battery.get("status") if battery else None,
                "temperatureC": thermal.get("hottest_temp_c") if thermal else None,
                "device": str(self.cfg.get("device", "cpu")).upper(),
                "tps": self._current_tps(),
            },
        }

    def update_generation_metrics(self, payload: dict[str, Any]) -> None:
        """Feed Pi output through OpenJet's original decode-window TPS algorithm."""
        phase = str(payload.get("phase", ""))
        if phase == "start":
            self._decode_started_at = None
            self._decode_last_token_at = None
            self._decode_tokens = 0
            return
        if phase == "chunks":
            chunks = payload.get("chunks")
            if not isinstance(chunks, list):
                return
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                text = chunk.get("text")
                timestamp_ms = chunk.get("timestampMs")
                if not isinstance(text, str) or not isinstance(timestamp_ms, (int, float)):
                    continue
                tokens = estimate_tokens(text)
                if tokens <= 0:
                    continue
                timestamp = float(timestamp_ms) / 1000.0
                if self._decode_started_at is None:
                    self._decode_started_at = timestamp
                self._decode_tokens += tokens
                self._decode_last_token_at = timestamp
            return
        if phase == "end":
            current = self._current_tps()
            if current is not None:
                self._last_generation_tps = current
            self._decode_started_at = None
            self._decode_last_token_at = None
            self._decode_tokens = 0

    def _current_tps(self) -> float | None:
        if (
            self._decode_started_at is not None
            and self._decode_last_token_at is not None
            and self._decode_tokens > 0
        ):
            elapsed = self._decode_last_token_at - self._decode_started_at
            if elapsed > 0:
                return self._decode_tokens / elapsed
        return self._last_generation_tps

    @staticmethod
    def _pi_tools() -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for name in OPENJET_PI_TOOL_NAMES:
            spec = get_tool_spec(name)
            if spec is None:
                continue
            tools.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": dict(spec.parameters),
                        "required": list(spec.required),
                        "additionalProperties": False,
                    },
                }
            )
        return tools

    async def execute_openjet_tool(self, name: str, arguments: Any, *, call_id: str) -> dict[str, Any]:
        if name not in OPENJET_PI_TOOL_NAMES:
            raise ServiceError(f"Pi requested unsupported OpenJet service tool: {name or '<missing>'}")
        if not isinstance(arguments, dict):
            raise ServiceError(f"OpenJet tool {name} requires an object argument payload.")
        result = await execute_tool(ToolCall(name=name, arguments=arguments, id=call_id))
        return {
            "name": name,
            "callId": call_id,
            "ok": result.ok,
            "output": result.output,
            "meta": result.meta,
            "contextContent": result.context_content,
        }

    async def _recommended_setup(self) -> dict[str, Any]:
        await self.emit("status_update", {"text": "provisioning recommended OpenJet setup…"})
        hardware_info = detect_hardware_info()
        setup_result = build_recommended_payload(
            hardware_info=hardware_info,
            recommended_ctx=recommended_context_window_tokens(),
            current_cfg=self.cfg,
        )
        log = _SetupLog()
        setup_result = await provision_setup_artifacts(
            dict(setup_result),
            hardware_info=hardware_info,
            log=log,
            set_status=lambda text: None,
            clear_status=lambda: None,
        )
        profile_name = str(setup_result.pop("model_profile_name", "")).strip() or None
        setup_result.pop("system_prompt", None)
        self.cfg.pop("system_prompt", None)
        self.cfg.update(setup_result)
        sync_active_model_profile(self.cfg, preferred_name=profile_name)
        save_config(self.cfg)
        self.force_setup = False
        await self._prepare_runtime()
        return self._snapshot(setup_required=False)

    async def command(self, text: str) -> dict[str, Any]:
        raw = text.strip().removeprefix("/")
        command, _, argument = raw.partition(" ")
        command = command.lower()
        if command == "status":
            return {"text": "OpenJet hardware/runtime status updated.", "payload": self._snapshot()}
        if command in {"exit", "quit"}:
            return {"text": "Closing OpenJet.", "payload": {"exit": True}}
        if command == "setup":
            mode = argument.strip().lower() or "recommended"
            if mode != "recommended":
                raise ServiceError("The Pi migration currently supports /setup recommended; use the Python setup command for guided/manual setup.")
            snapshot = await self._recommended_setup()
            return {"text": "OpenJet setup complete. Pi is ready.", "payload": snapshot}
        if command in {"mode", "agent", "strategy"}:
            mode = argument.strip().lower() or "status"
            if mode in {"status", "list"}:
                return {
                    "text": f"Mode: {self._agent_mode()}\nAvailable modes: local, codex, hybrid",
                    "payload": self._snapshot(),
                }
            if mode not in {"local", "codex", "hybrid"}:
                raise ServiceError("Usage: /mode [local|codex|hybrid|status]")
            previous_mode = self._agent_mode()
            self.cfg["agent_mode"] = mode
            try:
                await self._prepare_agent_mode()
            except Exception:
                self.cfg["agent_mode"] = previous_mode
                await self._prepare_agent_mode()
                raise
            save_config(self.cfg)
            snapshot = self._snapshot(setup_required=False)
            snapshot["agentChanged"] = True
            return {"text": f"Mode switched to {mode}.", "payload": snapshot}
        if command == "reasoning":
            mode = argument.strip().lower() or "status"
            if mode == "status":
                enabled = bool(self.cfg.get("reasoning", True))
                return {"text": f"Reasoning mode is {'on' if enabled else 'off'}."}
            if mode not in {"on", "off", "default"}:
                raise ServiceError("Usage: /reasoning [status|on|off|default]")
            if mode == "default":
                self.cfg.pop("reasoning", None)
            else:
                self.cfg["reasoning"] = mode == "on"
            save_config(self.cfg)
            snapshot = self._snapshot()
            snapshot["modelChanged"] = True
            enabled = bool(self.cfg.get("reasoning", True))
            return {
                "text": (
                    f"Reasoning mode is {'on' if enabled else 'off'}; "
                    "the matching model sampling preset is active."
                ),
                "payload": snapshot,
            }
        if command == "effort":
            effort = argument.strip().lower() or "status"
            profile = self._profile_for_runtime(CODEX_RUNTIME, "agent_codex_profile")
            if profile is None:
                raise ServiceError("Configure a Codex model before setting effort.")
            if effort == "status":
                return {"text": f"Codex effort: {profile.get('reasoning_effort') or 'medium'}", "payload": self._snapshot()}
            if effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
                raise ServiceError("Usage: /effort [none|low|medium|high|xhigh|max]")
            updated = dict(profile)
            updated["reasoning_effort"] = effort
            replace_model_profile(self.cfg, updated, previous_name=str(profile["name"]))
            self.cfg["agent_codex_profile"] = str(updated["name"])
            save_config(self.cfg)
            snapshot = self._snapshot(setup_required=False)
            if self._agent_mode() in {"codex", "hybrid"}:
                snapshot["agentChanged"] = True
            return {"text": f"Codex effort changed to {effort}.", "payload": snapshot}
        if command in {"device", "devices", "sources"}:
            parts = argument.split()
            action = parts[0].lower() if parts else "list"
            if action in {"list", "status"}:
                sources = list_device_sources(self.cfg)
                return {
                    "text": "No devices detected." if not sources else "Discovered devices:\n" + "\n".join(
                        f"- {source.primary_ref}: {source.device.label} | {source.device.kind.value} | "
                        f"{'enabled' if source.enabled else 'disabled'}" for source in sources
                    )
                }
            try:
                if action == "add" and len(parts) == 3:
                    source = assign_device_alias(self.cfg, reference=parts[1], alias=parts[2])
                    message = f"Saved device id {source.primary_ref} for {source.device.label}."
                elif action in {"on", "off"} and len(parts) == 2:
                    source = set_device_enabled(self.cfg, reference=parts[1], enabled=action == "on")
                    message = f"Device {source.primary_ref} is now {'enabled' if source.enabled else 'disabled'}."
                else:
                    raise ServiceError("Usage: /device [list|add <existing_id> <new_id>|on <id>|off <id>]")
            except ValueError as exc:
                raise ServiceError(str(exc)) from exc
            save_config(self.cfg)
            return {"text": message}
        if command == "model" and argument.strip().lower() in {"", "status", "list"}:
            local = self._profile_for_runtime(DEFAULT_RUNTIME, "agent_local_profile")
            codex = self._profile_for_runtime(CODEX_RUNTIME, "agent_codex_profile")
            return {
                "text": (
                    f"Mode: {self._agent_mode()}\n"
                    f"Codex: {active_model_ref(codex or {}) or 'not configured'} · effort {str((codex or {}).get('reasoning_effort') or 'medium')}\n"
                    f"Local: {active_model_ref(local or {}) or 'not configured'} · reasoning {'on' if bool((local or {}).get('reasoning', True)) else 'off'}"
                ),
                "payload": self._snapshot(),
            }
        if command == "runtime" and argument.strip().lower() in {"", "status", "list"}:
            profiles = list_model_profiles(self.cfg)
            return {
                "text": f"Active runtime: {active_runtime(self.cfg)}\nActive model: {active_model_ref(self.cfg)}\n"
                + "Saved profiles: " + (", ".join(str(item["name"]) for item in profiles) or "none")
            }
        if command == "runtime" and argument.strip():
            raise ServiceError("/runtime controls inference engines such as llama.cpp; use /mode for Local, Codex, or Slipstream.")
        if command == "model" and "=" in argument:
            try:
                options = dict(token.split("=", 1) for token in shlex.split(argument))
            except (ValueError, TypeError) as exc:
                raise ServiceError("Usage: /model codex=<model> effort=<level> local=<profile> reasoning=<on|off>") from exc
            allowed = {"codex", "effort", "local", "reasoning"}
            if not options or set(options) - allowed or any(not value.strip() for value in options.values()):
                raise ServiceError("Usage: /model codex=<model> effort=<level> local=<profile> reasoning=<on|off>")
            previous_cfg = copy.deepcopy(self.cfg)
            restart_local = False
            try:
                codex = self._profile_for_runtime(CODEX_RUNTIME, "agent_codex_profile")
                if "codex" in options or "effort" in options:
                    if codex is None:
                        raise ServiceError("Configure a Codex profile first.")
                    codex = dict(codex)
                    if "codex" in options:
                        codex["model"] = options["codex"]
                        if options["codex"].startswith("gpt-5.6"):
                            codex["context_window_tokens"] = 1_050_000
                    if "effort" in options:
                        effort = options["effort"].lower()
                        if effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
                            raise ServiceError("Codex effort must be none, low, medium, high, xhigh, or max.")
                        codex["reasoning_effort"] = effort
                    replace_model_profile(self.cfg, codex, previous_name=str(codex["name"]))
                    self.cfg["agent_codex_profile"] = str(codex["name"])

                local = self._profile_for_runtime(DEFAULT_RUNTIME, "agent_local_profile")
                if "local" in options:
                    selected = get_model_profile(self.cfg, options["local"])
                    if selected is None or active_runtime(selected) != DEFAULT_RUNTIME:
                        raise ServiceError(f"Unknown local model profile: {options['local']}")
                    restart_local = local is None or active_model_ref(local) != active_model_ref(selected)
                    local = selected
                    self.cfg["agent_local_profile"] = str(selected["name"])
                if "reasoning" in options:
                    if local is None:
                        raise ServiceError("Configure a local model profile first.")
                    value = options["reasoning"].lower()
                    if value not in {"on", "off"}:
                        raise ServiceError("Local reasoning must be on or off.")
                    local = dict(local)
                    local["reasoning"] = value == "on"
                    replace_model_profile(self.cfg, local, previous_name=str(local["name"]))
                if restart_local and self.runtime_client is not None:
                    await self.runtime_client.close()
                    self.runtime_client = None
                await self._prepare_agent_mode()
            except Exception:
                self.cfg = previous_cfg
                await self._prepare_agent_mode()
                raise
            save_config(self.cfg)
            snapshot = self._snapshot(setup_required=False)
            snapshot["agentChanged"] = True
            return {"text": "Model configuration updated.", "payload": snapshot}
        if command == "model" and argument.strip():
            profile_name = argument.strip()
            selected = get_model_profile(self.cfg, profile_name)
            if selected is None:
                raise ServiceError(f"Unknown model profile: {profile_name}")
            previous_cfg = dict(self.cfg)
            selected_runtime = active_runtime(selected)
            key = "agent_local_profile" if selected_runtime == DEFAULT_RUNTIME else "agent_codex_profile"
            self.cfg[key] = str(selected["name"])
            if selected_runtime not in {DEFAULT_RUNTIME, CODEX_RUNTIME}:
                raise ServiceError("Pi agent modes support local and Codex model profiles.")
            if selected_runtime == DEFAULT_RUNTIME:
                if self.runtime_client is not None:
                    await self.runtime_client.close()
                    self.runtime_client = None
                apply_model_profile(self.cfg, selected)
            try:
                await self._prepare_agent_mode()
            except Exception:
                self.cfg = previous_cfg
                await self._prepare_agent_mode()
                raise
            save_config(self.cfg)
            snapshot = self._snapshot(setup_required=False)
            snapshot["agentChanged"] = True
            return {"text": f"Switched to model profile {selected['name']}.", "payload": snapshot}
        raise ServiceError(f"/{command} is handled by the Pi agent UI, not the OpenJet hardware service.")

    def resize(self, width: int, height: int) -> None:
        self._terminal_size = (max(1, width), max(1, height))

    async def close(self) -> None:
        if self.runtime_client is not None:
            await self.runtime_client.close()
            self.runtime_client = None
        if self._session_logger is not None:
            await self._session_logger.stop()
            self._session_logger = None
