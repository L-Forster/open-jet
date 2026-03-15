from __future__ import annotations

import asyncio
import socket
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src import create_agent
from src.airgap import AirgapViolationError, assert_endpoint_allowed, set_airgapped
from src.app import OpenJetApp
from src.runtime_registry import create_runtime_client


class _CreateSessionClient:
    model = "fake"
    context_window_tokens = 4096
    gpu_layers = 0

    def __init__(self) -> None:
        self.start = AsyncMock()
        self.close = AsyncMock()
        self.reset_kv_cache = AsyncMock()

    async def chat_stream(self, messages, *, use_tools=True):
        if False:
            yield messages, use_tools


class AirgapBaseTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        set_airgapped(False)


class AirgapGuardTests(AirgapBaseTestCase):
    def test_assert_endpoint_allowed_blocks_remote_hosts(self) -> None:
        set_airgapped(True)

        with self.assertRaises(AirgapViolationError):
            assert_endpoint_allowed("https://api.openai.com/v1", label="remote API")

    def test_socket_connect_is_hard_blocked_for_remote_addresses(self) -> None:
        set_airgapped(True)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with self.assertRaises(AirgapViolationError):
                sock.connect(("8.8.8.8", 53))


class RuntimeRegistryAirgapTests(AirgapBaseTestCase):
    def test_create_runtime_client_blocks_openrouter_when_airgapped(self) -> None:
        with self.assertRaises(AirgapViolationError):
            create_runtime_client(
                {
                    "runtime": "openrouter",
                    "openrouter_model": "openai/gpt-4o-mini",
                    "airgapped": True,
                }
            )

    def test_create_runtime_client_allows_loopback_openai_compatible_when_airgapped(self) -> None:
        client = create_runtime_client(
            {
                "runtime": "openai_compatible",
                "openai_compatible_model": "local",
                "openai_compatible_base_url": "http://127.0.0.1:9000",
                "airgapped": True,
            }
        )

        self.assertEqual(client.base_url, "http://127.0.0.1:9000")
        asyncio.run(client.close())


class AppAirgapTests(AirgapBaseTestCase):
    def test_runtime_status_snapshot_reports_airgapped_mode(self) -> None:
        app = OpenJetApp()
        app.agent = Mock()
        app.agent.conversation_message_count.return_value = 0
        app.agent.context_budget.return_value = None
        app.agent.estimated_context_tokens.return_value = 0

        with patch("src.app.read_memory_snapshot", return_value=None):
            app.set_airgapped(True, persist=False)
            snapshot = app.runtime_status_snapshot()

        self.assertTrue(snapshot["airgapped"])

    def test_air_gapped_slash_command_updates_app_state(self) -> None:
        app = OpenJetApp()

        with patch("src.app.save_config") as save_config:
            asyncio.run(app.commands.maybe_handle("/air-gapped true"))

        self.assertTrue(app.is_airgapped())
        self.assertTrue(any("Air-gapped mode set to true" in str(entry) for entry in app.query_one("#chat-log")._entries))
        save_config.assert_called_once()

    def test_persist_session_state_includes_airgapped_flag(self) -> None:
        app = OpenJetApp()
        app.agent = Mock()
        app.agent.messages = [{"role": "system", "content": "system"}]
        app.agent.estimated_context_tokens.return_value = 0
        app.agent.context_budget.return_value = None
        app.set_airgapped(True, persist=False)

        with patch.object(app.state_store, "save") as save:
            app.persist_session_state(reason="test")

        payload = save.call_args.args[0]
        self.assertTrue(payload["airgapped"])


class SDKAirgapTests(AirgapBaseTestCase):
    def test_create_agent_supports_airgapped_sessions(self) -> None:
        fake_client = _CreateSessionClient()

        async def _run() -> None:
            with patch("src.sdk.create_runtime_client", return_value=fake_client) as create_client, patch(
                "src.sdk.build_system_prompt",
                new=AsyncMock(return_value="system prompt"),
            ):
                session = await create_agent(
                    cfg={"runtime": "llama_cpp", "llama_model": "model.gguf"},
                    airgapped=True,
                )

            self.assertTrue(session.airgapped)
            self.assertTrue(create_client.call_args.args[0]["airgapped"])
            fake_client.start.assert_awaited_once()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
