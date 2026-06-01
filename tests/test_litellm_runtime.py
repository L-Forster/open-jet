from __future__ import annotations

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.airgap import AirgapViolationError, set_airgapped
from src.api_auth import ApiKeyStore
from src.litellm_client import LiteLLMClient, LiteLLMUnavailableError
from src.runtime_registry import LITELLM_RUNTIME, create_runtime_client


async def _stream_chunks(chunks):
    for chunk in chunks:
        yield chunk


class _FakeLiteLLM:
    def __init__(self) -> None:
        self.last_kwargs = None

    async def acompletion(self, **kwargs):
        self.last_kwargs = kwargs
        return {}


class LiteLLMRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        set_airgapped(False)

    async def test_create_runtime_client_returns_litellm_client(self) -> None:
        client = create_runtime_client(
            {
                "runtime": LITELLM_RUNTIME,
                "provider": "openai",
                "model": "openai/gpt-5.5",
                "api_key_env": "OPENAI_API_KEY",
            }
        )

        self.assertIsInstance(client, LiteLLMClient)
        self.assertEqual(client.model, "openai/gpt-5.5")
        self.assertEqual(client.provider, "openai")

    async def test_missing_litellm_extra_has_actionable_error(self) -> None:
        client = LiteLLMClient(model="openai/gpt-5.5", provider="openai", auth_store=ApiKeyStore())
        error = LiteLLMUnavailableError(
            "LiteLLM support is not installed. Install it with `pip install open-jet[cloud]`."
        )

        with patch.object(client, "_import_litellm", side_effect=error), self.assertRaises(LiteLLMUnavailableError) as raised:
            await client.start()

        self.assertIn("pip install open-jet[cloud]", str(raised.exception))

    async def test_start_preflights_litellm_model(self) -> None:
        fake_litellm = _FakeLiteLLM()
        store = ApiKeyStore()
        client = LiteLLMClient(model="openai/gpt-5.5", provider="openai", api_key_env="OPENAI_API_KEY", auth_store=store)

        with patch.object(client, "_import_litellm", return_value=fake_litellm), patch.object(
            store, "resolve_key", return_value="test-key"
        ):
            await client.start()

        self.assertEqual(fake_litellm.last_kwargs["model"], "openai/gpt-5.5")
        self.assertFalse(fake_litellm.last_kwargs["stream"])
        self.assertEqual(fake_litellm.last_kwargs["max_tokens"], 1)
        self.assertEqual(fake_litellm.last_kwargs["api_key"], "test-key")

    async def test_provider_matrix_preflights_through_litellm(self) -> None:
        cases = [
            ("openai", "openai/gpt-5.5", "OPENAI_API_KEY"),
            ("anthropic", "anthropic/claude-opus-4-8", "ANTHROPIC_API_KEY"),
            ("openrouter", "openrouter/anthropic/claude-opus-4.8", ""),
            ("openrouter", "openrouter/google/gemini-3.1-pro-preview", ""),
            ("openrouter", "openrouter/x-ai/grok-4.20", ""),
            ("openrouter", "openrouter/deepseek/deepseek-v4-pro", ""),
            ("openrouter", "openrouter/z-ai/glm-5.1", ""),
            ("openrouter", "openrouter/moonshotai/kimi-k2.5", ""),
        ]

        for provider, model, env_name in cases:
            with self.subTest(provider=provider):
                fake_litellm = _FakeLiteLLM()
                store = ApiKeyStore()
                client = LiteLLMClient(
                    model=model,
                    provider=provider,
                    api_key_env=env_name,
                    auth_store=store,
                )
                with patch.object(client, "_import_litellm", return_value=fake_litellm), patch.object(
                    store, "resolve_key", return_value="test-key"
                ):
                    await client.start()

                kwargs = fake_litellm.last_kwargs
                self.assertEqual(kwargs["model"], model)
                self.assertFalse(kwargs["stream"])
                self.assertEqual(kwargs["max_tokens"], 1)
                self.assertEqual(kwargs["api_key"], "test-key")

    async def test_provider_stream_shapes_map_to_openjet_chunks(self) -> None:
        client = LiteLLMClient(model="anthropic/claude-opus-4-8", provider="anthropic", auth_store=ApiKeyStore())
        stream = _stream_chunks(
            [
                {"choices": [{"delta": {"reasoning": "thinking"}}]},
                {"choices": [{"delta": {"text": "hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
            ]
        )

        chunks = [chunk async for chunk in client._chunks_to_stream(stream, use_tools=False)]

        self.assertEqual("".join(chunk.reasoning for chunk in chunks), "thinking")
        self.assertEqual("".join(chunk.text for chunk in chunks), "hello world")
        self.assertTrue(chunks[-1].done)

    async def test_cfg_airgapped_blocks_remote_litellm_provider_without_global_state(self) -> None:
        set_airgapped(False)

        with self.assertRaises(AirgapViolationError):
            create_runtime_client(
                {
                    "runtime": LITELLM_RUNTIME,
                    "provider": "openai",
                    "model": "openai/gpt-5.5",
                    "airgapped": True,
                }
            )

    async def test_cfg_airgapped_allows_loopback_litellm_provider_without_global_state(self) -> None:
        set_airgapped(False)

        client = create_runtime_client(
            {
                "runtime": LITELLM_RUNTIME,
                "provider": "openai-compatible",
                "model": "openai/local",
                "base_url": "http://127.0.0.1:1234/v1",
                "airgapped": True,
            }
        )

        self.assertIsInstance(client, LiteLLMClient)

    async def test_litellm_stream_maps_text_and_tool_calls(self) -> None:
        client = LiteLLMClient(model="openai/gpt-5.5", provider="openai", auth_store=ApiKeyStore())
        stream = _stream_chunks(
            [
                {"choices": [{"delta": {"content": "hello "}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "content": "<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>"
                            }
                        }
                    ]
                },
            ]
        )

        chunks = [chunk async for chunk in client._chunks_to_stream(stream, use_tools=True)]

        self.assertEqual("".join(chunk.text for chunk in chunks), "hello ")
        self.assertTrue(chunks[-1].done)
        self.assertEqual(len(chunks[-1].tool_calls), 1)
        self.assertEqual(chunks[-1].tool_calls[0].arguments, {"path": "README.md"})

    async def test_local_dummy_key_requires_actual_loopback_host(self) -> None:
        local = LiteLLMClient(
            model="openai/local",
            provider="openai-compatible",
            base_url="http://localhost:1234/v1",
            auth_store=ApiKeyStore(),
        )
        remote = LiteLLMClient(
            model="openai/local",
            provider="openai-compatible",
            base_url="https://localhost.evil.com/v1",
            auth_store=ApiKeyStore(),
        )

        self.assertEqual(local._api_key(), "openjet-local")
        with self.assertRaisesRegex(RuntimeError, "Missing API key"):
            remote._api_key()


class ApiKeyStoreTests(unittest.TestCase):
    def test_openai_compatible_does_not_default_to_openai_api_key(self) -> None:
        store = ApiKeyStore()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-secret"}):
            self.assertIsNone(store.resolve_key("openai-compatible"))

    def test_env_overrides_stored_api_key(self) -> None:
        store = ApiKeyStore()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}), patch.object(
            store, "_load_keyring", return_value="stored-key"
        ):

            self.assertEqual(store.resolve_key("openai", env_name="OPENAI_API_KEY"), "env-key")

    def test_status_does_not_leak_key_material(self) -> None:
        store = ApiKeyStore()
        with patch.object(store, "_load_keyring", return_value="secret-key"):

            status = store.status(["anthropic"])

            self.assertTrue(status["anthropic"]["stored"])
            self.assertNotIn("secret-key", str(status))

    def test_save_key_requires_keyring(self) -> None:
        store = ApiKeyStore()

        with patch.object(store, "_save_keyring", return_value=False), self.assertRaisesRegex(ValueError, "keyring"):
            store.save_key("openai", "sk-secret")

    def test_clear_key_reports_keyring_failure(self) -> None:
        store = ApiKeyStore()

        with patch.object(store, "_clear_keyring", return_value=False):
            self.assertFalse(store.clear_key("openai"))

    def test_clear_key_returns_false_when_delete_raises_for_existing_key(self) -> None:
        store = ApiKeyStore()
        fake_keyring = SimpleNamespace(
            get_password=lambda *_args: "stored-key",
            delete_password=lambda *_args: (_ for _ in ()).throw(RuntimeError("locked")),
        )

        with patch.dict(sys.modules, {"keyring": fake_keyring}):
            self.assertFalse(store.clear_key("openai"))

    def test_clear_key_returns_true_when_key_is_already_absent(self) -> None:
        store = ApiKeyStore()
        fake_keyring = SimpleNamespace(
            get_password=lambda *_args: None,
            delete_password=lambda *_args: (_ for _ in ()).throw(RuntimeError("should not delete")),
        )

        with patch.dict(sys.modules, {"keyring": fake_keyring}):
            self.assertTrue(store.clear_key("openai"))


if __name__ == "__main__":
    unittest.main()
