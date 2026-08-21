from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from src.model_profiles import get_model_profile
from src.service_controller import OpenJetServiceController, ServiceError
from src.runtime_limits import MemorySnapshot
from src.tui_server import PROTOCOL_VERSION, REQUEST_TYPES, ProtocolError, ProtocolServer


class _FakeController:
    def __init__(self) -> None:
        self.ids: set[str] = set()
        self.commands: list[tuple[str, str | None]] = []
        self.tool_calls: list[tuple[str, object, str]] = []
        self.generation_metrics: list[dict] = []
        self.agent_traces: list[dict] = []
        self.closed = False

    def claim_request_id(self, value: str) -> None:
        if value in self.ids:
            raise ServiceError(f"Duplicate request id: {value}")
        self.ids.add(value)

    async def initialize(self):
        return {"workspace": "/tmp/project", "commands": [], "tools": []}

    async def command(self, text: str, *, api_key: str | None = None):
        self.commands.append((text, api_key))
        return {"text": f"handled {text}"}

    async def execute_openjet_tool(self, name: str, arguments: object, *, call_id: str):
        self.tool_calls.append((name, arguments, call_id))
        return {"name": name, "callId": call_id, "ok": True, "output": "done", "meta": {}}

    def update_generation_metrics(self, payload: dict) -> None:
        self.generation_metrics.append(payload)

    def record_agent_trace(self, payload: dict) -> None:
        self.agent_traces.append(payload)

    def _snapshot(self):
        return {"workspace": "/tmp/project"}

    def resize(self, width: int, height: int):
        self.size = (width, height)

    async def close(self):
        self.closed = True


class ProtocolServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        with patch("src.tui_server.OpenJetServiceController"):
            self.server = ProtocolServer()
        self.controller = _FakeController()
        self.server.controller = self.controller
        self.events: list[tuple[str, dict]] = []

        async def capture(event_type: str, fields: dict | None = None) -> None:
            self.events.append((event_type, fields or {}))

        self.server.emit = capture

    async def test_initialize_emits_ready_snapshot(self) -> None:
        await self.server.handle({"protocolVersion": PROTOCOL_VERSION, "type": "initialize", "id": "a"})
        self.assertEqual(self.events[0][0], "ready")
        self.assertEqual(self.events[0][1]["requestId"], "a")
        self.assertEqual(self.events[0][1]["payload"]["workspace"], "/tmp/project")

    async def test_duplicate_request_is_rejected(self) -> None:
        message = {"protocolVersion": PROTOCOL_VERSION, "type": "status", "id": "same"}
        await self.server.handle(message)
        with self.assertRaisesRegex(ProtocolError, "Duplicate request id"):
            await self.server.handle(message)

    async def test_protocol_mismatch_is_rejected_before_dispatch(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "Protocol mismatch"):
            await self.server.handle({"protocolVersion": 2, "type": "initialize", "id": "x"})

    async def test_stale_or_unknown_types_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "Unknown request type"):
            await self.server.handle({"protocolVersion": PROTOCOL_VERSION, "type": "wat", "id": "x"})

    async def test_tool_execute_returns_correlated_result(self) -> None:
        await self.server.handle(
            {
                "protocolVersion": PROTOCOL_VERSION,
                "type": "tool_execute",
                "id": "tool-request",
                "callId": "pi-call",
                "payload": {"name": "device_list", "arguments": {"kind": "camera"}},
            }
        )
        self.assertEqual(self.controller.tool_calls, [("device_list", {"kind": "camera"}, "pi-call")])
        self.assertEqual(self.events[0][0], "tool_result")
        self.assertEqual(self.events[0][1]["requestId"], "tool-request")
        self.assertEqual(self.events[0][1]["callId"], "pi-call")

    async def test_generation_metrics_is_accepted_and_dispatched(self) -> None:
        await self.server.handle(
            {
                "protocolVersion": PROTOCOL_VERSION,
                "type": "generation_metrics",
                "id": "metrics-start",
                "payload": {"phase": "start"},
            }
        )
        self.assertEqual(self.controller.generation_metrics, [{"phase": "start"}])

    async def test_command_passes_api_key_outside_text(self) -> None:
        await self.server.handle(
            {
                "protocolVersion": PROTOCOL_VERSION,
                "type": "command",
                "id": "connect-1",
                "text": "/connect openrouter",
                "apiKey": "sk-or-secret",
            }
        )
        self.assertEqual(self.controller.commands, [("/connect openrouter", "sk-or-secret")])
        encoded = json.dumps(self.events)
        self.assertNotIn("sk-or-secret", encoded)
        self.assertEqual(self.events[0][0], "notification")

    async def test_agent_trace_is_accepted_and_dispatched(self) -> None:
        payload = {"event": "model_tool_start", "turnId": "turn-1", "data": {"lane": "local"}}
        await self.server.handle(
            {"protocolVersion": PROTOCOL_VERSION, "type": "agent_trace", "id": "trace-1", "payload": payload}
        )
        self.assertEqual(self.controller.agent_traces, [payload])


class ProtocolEncodingTests(unittest.TestCase):
    def test_schema_is_valid_json_and_pins_v1(self) -> None:
        with open("protocol/openjet-tui-v1.schema.json", encoding="utf-8") as handle:
            schema = json.load(handle)
        self.assertEqual(schema["properties"]["protocolVersion"]["const"], PROTOCOL_VERSION)
        self.assertNotIn("approval_decision", schema["properties"]["type"]["enum"])
        self.assertEqual(schema["x-requestTypes"], ["initialize", "command", "tool_execute", "generation_metrics", "agent_trace", "status", "resize", "shutdown"])
        self.assertEqual(set(schema["x-requestTypes"]), REQUEST_TYPES)


class ServiceControllerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        cfg = {"airgapped": False, "telemetry": {}, "state": {"enabled": False}}
        self.load_patch = patch("src.service_controller.load_config", return_value=cfg)
        self.save_patch = patch("src.service_controller.save_config")
        self.load_patch.start()
        self.save_patch.start()
        self.controller = OpenJetServiceController(
            AsyncMock(), root=Path(self.temp.name)
        )

    def tearDown(self) -> None:
        self.load_patch.stop()
        self.save_patch.stop()
        self.temp.cleanup()

    async def test_snapshot_identifies_pi_as_agent_engine(self) -> None:
        snapshot = self.controller._snapshot(setup_required=True)
        self.assertEqual(snapshot["agentEngine"], "pi")
        self.assertIsNone(snapshot["piModel"])
        self.assertIn("device_list", {tool["name"] for tool in snapshot["openjetTools"]})
        self.assertNotIn("shell", {tool["name"] for tool in snapshot["openjetTools"]})

    async def test_prepare_runtime_reprovisions_llama_server_before_start(self) -> None:
        self.controller.cfg.update(
            {
                "runtime": "llama_cpp",
                "llama_model": "/models/Qwen3.8-27B-Q4_K_M.gguf",
                "device": "vulkan",
                "llama_server_path": "/opt/openjet/bin/llama-server",
                "llama_mtp": True,
            }
        )
        client = SimpleNamespace(start=AsyncMock(), close=AsyncMock(), base_url="http://127.0.0.1:18080")
        resolved = {
            **self.controller.cfg,
            "device": "cuda",
            "llama_server_path": "/opt/openjet/llama.cpp/build/bin/llama-server",
        }
        with patch("src.service_controller.detect_hardware_info"), patch(
            "src.service_controller.provision_setup_artifacts",
            AsyncMock(return_value=resolved),
        ) as provision, patch(
            "src.service_controller.create_runtime_client",
            return_value=client,
        ):
            await self.controller._prepare_runtime()

        provision.assert_awaited_once()
        self.assertEqual(self.controller.cfg["device"], "cuda")
        self.assertEqual(
            self.controller.cfg["llama_server_path"],
            "/opt/openjet/llama.cpp/build/bin/llama-server",
        )
        client.start.assert_awaited_once()
        self.assertIs(self.controller.runtime_client, client)

    async def test_model_attribution_is_forwarded_to_session_trace(self) -> None:
        logger = SimpleNamespace(record_agent_trace=Mock())
        self.controller._session_logger = logger
        self.controller.record_agent_trace({
            "event": "delegation_end",
            "turnId": "turn-1",
            "data": {"lane": "local", "model": "qwen", "outputTokens": 120},
        })
        logger.record_agent_trace.assert_called_once_with(
            "delegation_end",
            {"lane": "local", "model": "qwen", "outputTokens": 120},
            turn_id="turn-1",
        )

    async def test_agent_hybrid_prepares_codex_primary_and_warm_local_model(self) -> None:
        self.controller.cfg.update(
            {
                "runtime": "llama_cpp",
                "llama_model": "/models/qwen.gguf",
                "context_window_tokens": 32768,
                "active_model_profile": "local",
                "model_profiles": [
                    {
                        "name": "local",
                        "runtime": "llama_cpp",
                        "llama_model": "/models/qwen.gguf",
                        "context_window_tokens": 32768,
                    },
                    {
                        "name": "codex",
                        "runtime": "openai_codex",
                        "model": "gpt-5.6-sol",
                        "reasoning_effort": "medium",
                    },
                ],
            }
        )

        async def prepare_local() -> None:
            self.controller.runtime_client = SimpleNamespace(base_url="http://127.0.0.1:18080")

        self.controller._prepare_runtime = AsyncMock(side_effect=prepare_local)
        credentials = SimpleNamespace(access_token="token", account_id="acct")
        with patch("src.service_controller.CodexOAuthProvider") as provider:
            provider.return_value.credentials = AsyncMock(return_value=credentials)
            result = await self.controller.command("/mode hybrid")

        payload = result["payload"]
        self.assertEqual(payload["agentMode"], "hybrid")
        self.assertEqual(payload["piModel"]["provider"], "openai-codex")
        self.assertEqual(payload["piModel"]["id"], "gpt-5.6-sol")
        self.assertEqual(payload["piModel"]["thinkingLevel"], "medium")
        self.assertIsNone(payload["piModel"]["thinkingLevelMap"]["minimal"])
        self.assertEqual(payload["localModel"]["provider"], "openjet-local")
        self.assertEqual(payload["piModel"]["headers"]["ChatGPT-Account-Id"], "acct")
        self.assertTrue(payload["agentChanged"])

    async def test_runtime_does_not_accept_agent_modes(self) -> None:
        with self.assertRaisesRegex(ServiceError, "use /mode"):
            await self.controller.command("/runtime hybrid")

    async def test_effort_updates_the_codex_side_without_changing_mode(self) -> None:
        self.controller.cfg.update({
            "agent_mode": "codex",
            "agent_codex_profile": "codex",
            "model_profiles": [{
                "name": "codex",
                "runtime": "openai_codex",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "medium",
            }],
        })
        self.controller._codex_credentials = SimpleNamespace(access_token="token", account_id="acct")

        result = await self.controller.command("/effort xhigh")

        self.assertEqual(result["payload"]["agentMode"], "codex")
        self.assertEqual(result["payload"]["piModel"]["id"], "gpt-5.6-sol")
        self.assertEqual(result["payload"]["piModel"]["thinkingLevel"], "xhigh")
        self.assertTrue(result["payload"]["agentChanged"])

    async def test_effort_rejects_levels_not_supported_by_gpt_5_6(self) -> None:
        self.controller.cfg.update({
            "agent_codex_profile": "codex",
            "model_profiles": [{"name": "codex", "runtime": "openai_codex", "model": "gpt-5.6-sol"}],
        })
        with self.assertRaisesRegex(ServiceError, "Usage: /effort"):
            await self.controller.command("/effort minimal")

    async def test_model_configuration_updates_hybrid_pair_atomically(self) -> None:
        self.controller.cfg.update({
            "agent_mode": "hybrid",
            "runtime": "llama_cpp",
            "llama_model": "/models/qwen.gguf",
            "agent_local_profile": "local",
            "agent_codex_profile": "codex",
            "model_profiles": [
                {"name": "local", "runtime": "llama_cpp", "llama_model": "/models/qwen.gguf", "reasoning": True},
                {"name": "codex", "runtime": "openai_codex", "model": "gpt-5.6-sol", "reasoning_effort": "medium"},
            ],
        })
        self.controller.runtime_client = SimpleNamespace(base_url="http://127.0.0.1:18080")
        credentials = SimpleNamespace(access_token="token", account_id="acct")
        with patch("src.service_controller.CodexOAuthProvider") as provider:
            provider.return_value.credentials = AsyncMock(return_value=credentials)
            result = await self.controller.command(
                "/model codex=gpt-5.6-terra effort=high local=local reasoning=off"
            )

        payload = result["payload"]
        self.assertEqual(payload["agentMode"], "hybrid")
        self.assertEqual(payload["piModel"]["id"], "gpt-5.6-terra")
        self.assertEqual(payload["piModel"]["thinkingLevel"], "high")
        self.assertFalse(payload["localReasoning"])
        self.assertFalse(payload["localModel"]["reasoning"])
        self.assertTrue(payload["agentChanged"])

    def test_snapshot_exposes_live_local_runtime_metrics(self) -> None:
        self.controller.cfg["device"] = "cuda"
        self.controller._metrics = SimpleNamespace(
            read_power_metrics=lambda: (184.2, 72.0),
            read_cpu_percent=lambda: 37.0,
            read_battery_metrics=lambda: {"capacity_pct": 81.0, "status": "Discharging"},
            read_thermal_metrics=lambda: {"hottest_temp_c": 68.0},
        )
        with patch(
            "src.service_controller.read_memory_snapshot",
            return_value=MemorySnapshot(total_mb=32768, available_mb=12780, used_percent=61.0),
        ):
            status = self.controller._snapshot(setup_required=True)["status"]

        self.assertEqual(status["device"], "CUDA")
        self.assertEqual(status["cpuPercent"], 37.0)
        self.assertEqual(status["memoryPercent"], 61.0)
        self.assertEqual(status["powerWatts"], 184.2)
        self.assertEqual(status["temperatureC"], 68.0)
        self.assertEqual(status["batteryPercent"], 81.0)

    def test_legacy_decode_window_tps_is_preserved_for_pi_output(self) -> None:
        with patch("src.service_controller.estimate_tokens", side_effect=lambda text: len(text.split())):
            self.controller.update_generation_metrics({"phase": "start"})
            self.controller.update_generation_metrics({
                "phase": "chunks",
                "chunks": [
                    {"text": "one two", "timestampMs": 100_000},
                    {"text": "three four five six", "timestampMs": 102_000},
                ],
            })
        self.assertEqual(self.controller._current_tps(), 3.0)
        self.controller.update_generation_metrics({"phase": "end"})
        self.assertEqual(self.controller._current_tps(), 3.0)

    def test_qwen38_thinking_defaults_are_sent_to_pi(self) -> None:
        self.controller.cfg.update(
            {
                "runtime": "llama_cpp",
                "llama_model": "/models/Qwen3.8-27B-Q4_K_M.gguf",
                "context_window_tokens": 262144,
            }
        )
        self.controller.runtime_client = SimpleNamespace(base_url="http://127.0.0.1:18080")

        model = self.controller._pi_model()

        self.assertIsNotNone(model)
        self.assertEqual(model["nativeContextWindow"], 262144)
        self.assertEqual(model["maxTokens"], 131072)
        self.assertEqual(
            model["samplingParams"],
            {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": 1.0,
            },
        )

    def test_pi_completion_budget_cannot_exceed_available_context(self) -> None:
        self.controller.cfg.update(
            {
                "runtime": "llama_cpp",
                "llama_model": "/models/Qwen3.8-27B-Q4_K_M.gguf",
                "context_window_tokens": 80000,
            }
        )
        self.controller.runtime_client = SimpleNamespace(base_url="http://127.0.0.1:18080")

        model = self.controller._pi_model()

        self.assertIsNotNone(model)
        self.assertEqual(model["contextWindow"], 80000)
        self.assertEqual(model["maxTokens"], 75904)

    async def test_qwen38_reasoning_off_uses_instruct_defaults(self) -> None:
        self.controller.cfg.update(
            {
                "runtime": "llama_cpp",
                "llama_model": "/models/Qwen3.8-27B-Q4_K_M.gguf",
                "context_window_tokens": 80000,
            }
        )
        self.controller.runtime_client = SimpleNamespace(base_url="http://127.0.0.1:18080")

        result = await self.controller.command("/reasoning off")
        model = result["payload"]["piModel"]

        self.assertTrue(result["payload"]["modelChanged"])
        self.assertFalse(model["reasoning"])
        self.assertEqual(
            model["samplingParams"],
            {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
            },
        )


    async def test_help_lists_available_commands(self) -> None:
        result = await self.controller.command("/help")
        self.assertIn("/mode", result["text"])
        self.assertIn("/cloud", result["text"])

    async def test_advertised_but_unsupported_command_degrades_to_notice(self) -> None:
        result = await self.controller.command("/resume")
        self.assertIn("not available in this interface yet", result["text"])

    async def test_unknown_command_reports_unknown(self) -> None:
        with self.assertRaises(ServiceError):
            await self.controller.command("/definitely-not-a-command")

    async def test_bare_cloud_opens_the_model_picker(self) -> None:
        result = await self.controller.command("/cloud")
        self.assertTrue(result["payload"]["openCloudPicker"])
        self.assertEqual(result["text"], "")

    async def test_cloud_without_key_asks_for_openrouter_api_key(self) -> None:
        with patch("src.service_controller.ApiKeyStore") as store:
            store.return_value.resolve_key.return_value = None
            result = await self.controller.command("/cloud stealth/ox-alpha")

        self.assertEqual(result["payload"]["needsApiKey"], "openrouter")
        self.assertIn("/login", result["text"])

    async def test_cloud_switches_to_openrouter_when_key_is_present(self) -> None:
        with patch("src.service_controller.ApiKeyStore") as store:
            store.return_value.resolve_key.return_value = "or-key"
            store.return_value.status.return_value = {}
            result = await self.controller.command("/cloud stealth/ox-alpha")

        payload = result["payload"]
        self.assertTrue(payload["agentChanged"])
        self.assertEqual(payload["orchestratorKind"], "openrouter")
        self.assertEqual(payload["piModel"]["provider"], "openrouter")
        self.assertEqual(payload["piModel"]["id"], "stealth/ox-alpha")
        self.assertEqual(payload["piModel"]["apiKey"], "or-key")
        self.assertIn("not loaded", result["text"])

    async def test_model_effort_does_not_flip_openrouter_orchestrator_to_codex(self) -> None:
        self.controller.cfg.update({
            "agent_mode": "codex",
            "model_profiles": [
                {
                    "name": "or",
                    "runtime": "litellm",
                    "provider": "openrouter",
                    "model": "openrouter/stealth/ox-alpha",
                    "api_key_env": "OPENROUTER_API_KEY",
                },
                {"name": "cx", "runtime": "openai_codex", "model": "gpt-5.6-sol"},
            ],
            "agent_orchestrator_profile": "or",
            "agent_codex_profile": "cx",
        })
        with patch.object(self.controller, "_prepare_agent_mode", AsyncMock()):
            await self.controller.command("/model effort=high")

        self.assertEqual(self.controller.cfg["agent_orchestrator_profile"], "or")
        codex = get_model_profile(self.controller.cfg, "cx")
        self.assertEqual(codex["reasoning_effort"], "high")

    async def test_model_local_edit_survives_missing_openrouter_key(self) -> None:
        self.controller.cfg.update({
            "agent_mode": "codex",
            "model_profiles": [
                {"name": "loc", "runtime": "llama_cpp", "llama_model": "/models/a.gguf"},
                {"name": "or", "runtime": "litellm", "provider": "openrouter",
                 "model": "openrouter/stealth/ox-alpha", "api_key_env": "OPENROUTER_API_KEY"},
            ],
            "agent_orchestrator_profile": "or",
        })
        with patch("src.service_controller.ApiKeyStore") as store, patch.object(
            self.controller, "_prepare_agent_mode", AsyncMock()
        ):
            store.return_value.resolve_key.return_value = None
            result = await self.controller.command("/model openrouter=stealth/ox-alpha local=loc")

        self.assertEqual(result["payload"]["needsApiKey"], "openrouter")
        self.assertEqual(self.controller.cfg["agent_local_profile"], "loc")

    async def test_bare_connect_opens_the_credential_picker(self) -> None:
        result = await self.controller.command("/connect")
        self.assertTrue(result["payload"]["openConnectPicker"])
        self.assertEqual(result["text"], "")

    async def test_connect_openrouter_saves_key_without_echoing_it(self) -> None:
        with patch("src.service_controller.ApiKeyStore") as store:
            store.return_value.save_key.return_value = None
            store.return_value.resolve_key.return_value = "or-key"
            result = await self.controller.command("/connect openrouter sk-or-secret")

        store.return_value.save_key.assert_called_once_with("openrouter", "sk-or-secret")
        self.assertNotIn("sk-or-secret", result["text"])
        self.assertNotIn("sk-or-secret", json.dumps(result["payload"], default=str))
        self.assertIn("API key saved", result["text"])

    async def test_connect_openrouter_accepts_api_key_field(self) -> None:
        with patch("src.service_controller.ApiKeyStore") as store:
            store.return_value.save_key.return_value = None
            store.return_value.resolve_key.return_value = "or-key"
            result = await self.controller.command("/connect openrouter", api_key="sk-or-secret")

        store.return_value.save_key.assert_called_once_with("openrouter", "sk-or-secret")
        self.assertNotIn("sk-or-secret", result["text"])
        self.assertNotIn("sk-or-secret", json.dumps(result["payload"], default=str))
        self.assertIn("API key saved", result["text"])

    async def test_connect_openrouter_reports_save_failure_without_leaking_key(self) -> None:
        with patch("src.service_controller.ApiKeyStore") as store:
            store.return_value.save_key.side_effect = ValueError("OS keyring is unavailable.")
            result = await self.controller.command("/connect openrouter sk-or-secret")

        self.assertIn("Could not save", result["text"])
        self.assertIn("keyring", result["text"])
        self.assertNotIn("sk-or-secret", result["text"])
        self.assertNotIn("sk-or-secret", json.dumps(result["payload"], default=str))


if __name__ == "__main__":
    unittest.main()
