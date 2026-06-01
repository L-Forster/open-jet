from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.airgap import set_airgapped
from src.codex_auth import CodexAuthStore, CodexCredentials, CodexOAuthProvider, _run_codex_login
from src.model_profiles import apply_model_profile, get_model_profile, list_model_profiles
from src.openai_codex_client import OpenAICodexClient
from src.runtime_protocol import stream_openai_responses
from src.runtime_registry import CODEX_RUNTIME, create_runtime_client


class _FakeResponse:
    def __init__(self, lines: list[str], *, status_code: int = 200, body: str = "") -> None:
        self._lines = lines
        self.status_code = status_code
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            request = httpx.Request("POST", "https://api.openai.com/v1/responses")
            response = httpx.Response(self.status_code, request=request, content=self._body.encode("utf-8"))
            raise httpx.HTTPStatusError("error", request=request, response=response)

    async def aread(self) -> bytes:
        return self._body.encode("utf-8")

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeHTTPClient:
    def __init__(self, lines: list[str], *, status_code: int = 200, body: str = "") -> None:
        self.lines = lines
        self.status_code = status_code
        self.body = body
        self.last_url: str | None = None
        self.last_payload: dict | None = None
        self.last_headers: dict | None = None

    def stream(self, method: str, url: str, *, json: dict, headers=None) -> _FakeResponse:
        self.last_url = url
        self.last_payload = json
        self.last_headers = headers
        return _FakeResponse(self.lines, status_code=self.status_code, body=self.body)


def _jwt(claims: dict) -> str:
    def enc(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{enc({'alg': 'none'})}.{enc(claims)}.sig"


class CodexRuntimeRegistryTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_airgapped(False)

    def test_existing_local_profiles_default_to_llama_runtime(self) -> None:
        profiles = list_model_profiles(
            {
                "model_profiles": [
                    {"name": "local", "llama_model": "/models/local.gguf", "context_window_tokens": 4096}
                ]
            }
        )

        self.assertEqual(profiles[0]["runtime"], "llama_cpp")
        self.assertEqual(profiles[0]["llama_model"], "/models/local.gguf")

    def test_codex_profile_is_listed_and_applied(self) -> None:
        cfg = {
            "model_profiles": [
                {
                    "name": "codex",
                    "runtime": CODEX_RUNTIME,
                    "model": "gpt-5.5",
                    "context_window_tokens": 272000,
                }
            ]
        }

        profile = get_model_profile(cfg, "codex")
        assert profile is not None
        apply_model_profile(cfg, profile)

        self.assertEqual(cfg["runtime"], CODEX_RUNTIME)
        self.assertEqual(cfg["model"], "gpt-5.5")
        self.assertNotIn("llama_model", cfg)

    def test_create_runtime_client_returns_codex_client(self) -> None:
        with patch(
            "src.openai_codex_client.CodexOAuthProvider.credentials",
            new=AsyncMock(
                return_value=CodexCredentials(
                    access_token="token",
                    refresh_token="refresh",
                    expires_at=time.time() + 3600,
                )
            ),
        ):
            client = create_runtime_client(
                {
                    "runtime": CODEX_RUNTIME,
                    "model": "gpt-5.5",
                    "context_window_tokens": 272000,
                }
            )

        self.assertIsInstance(client, OpenAICodexClient)
        self.assertEqual(client.model, "gpt-5.5")
        asyncio.run(client.close())

    def test_airgapped_blocks_codex_runtime_creation(self) -> None:
        set_airgapped(True)

        with self.assertRaises(ValueError):
            create_runtime_client({"runtime": CODEX_RUNTIME, "model": "gpt-5.5", "airgapped": True})


class CodexAuthStoreTests(unittest.TestCase):
    def test_store_reads_official_codex_auth_json_without_token_leak_in_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth_path = Path(tmp) / "auth.json"
            access = _jwt({"exp": int(time.time()) + 3600})
            auth_path.write_text(
                json.dumps(
                    {
                        "auth_mode": "chatgpt",
                        "tokens": {
                            "id_token": _jwt(
                                {
                                    "email": "user@example.com",
                                    "https://api.openai.com/auth": {
                                        "chatgpt_account_id": "acct",
                                        "chatgpt_plan_type": "pro",
                                    },
                                }
                            ),
                            "access_token": access,
                            "refresh_token": "refresh-secret",
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = CodexAuthStore(auth_path)
            loaded = store.load()
            status = store.status()

            assert loaded is not None
            self.assertEqual(loaded.access_token, access)
            self.assertEqual(loaded.account_id, "acct")
            self.assertEqual(loaded.plan_type, "pro")
            self.assertTrue(status["logged_in"])
            self.assertNotIn(access, json.dumps(status))
            self.assertNotIn("refresh-secret", json.dumps(status))

    def test_store_writes_official_codex_auth_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth_path = Path(tmp) / "auth.json"
            store = CodexAuthStore(auth_path)
            credentials = CodexCredentials(
                access_token="access-secret",
                refresh_token="refresh-secret",
                expires_at=time.time() + 3600,
                account_id="acct",
            )

            store.save(credentials)
            loaded = store.load()
            raw = json.loads(auth_path.read_text(encoding="utf-8"))

            assert loaded is not None
            self.assertEqual(loaded.access_token, "access-secret")
            self.assertEqual(raw["auth_mode"], "chatgpt")
            self.assertEqual(raw["tokens"]["refresh_token"], "refresh-secret")

    def test_clear_unlinks_auth_file_when_codex_logout_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth_path = Path(tmp) / "auth.json"
            auth_path.write_text("{}", encoding="utf-8")
            store = CodexAuthStore(auth_path)

            with patch("src.codex_auth._codex_binary", return_value="codex"), patch(
                "src.codex_auth.subprocess.run",
                return_value=type("_Result", (), {"returncode": 1})(),
            ):
                store.clear()

            self.assertFalse(auth_path.exists())

    def test_provider_refresh_persists_replacement_refresh_token(self) -> None:
        class _HTTP:
            async def post(self, *_args, **_kwargs):
                return _TokenResponse()

        class _TokenResponse:
            status_code = 200

            def json(self):
                return {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                }

        async def _run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                store = CodexAuthStore(Path(tmp) / "auth.json")
                provider = CodexOAuthProvider(store=store, http=_HTTP())
                old = CodexCredentials(access_token="old-access", refresh_token="old-refresh", expires_at=0)
                refreshed = await provider.refresh(old)
                loaded = store.load()

                self.assertEqual(refreshed.access_token, "new-access")
                assert loaded is not None
                self.assertEqual(loaded.refresh_token, "new-refresh")

        asyncio.run(_run())

    def test_refresh_response_uses_expires_in_when_access_token_has_no_exp(self) -> None:
        previous = CodexCredentials(access_token="old", refresh_token="old-refresh", expires_at=0)

        refreshed = CodexCredentials.from_refresh_response(
            {
                "access_token": "opaque-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
            previous=previous,
        )

        self.assertGreater(refreshed.expires_at, time.time() + 3500)
        self.assertEqual(refreshed.refresh_token, "new-refresh")

    def test_store_preserves_expiry_for_opaque_refreshed_access_token(self) -> None:
        previous = CodexCredentials(access_token="old", refresh_token="old-refresh", expires_at=0)
        refreshed = CodexCredentials.from_refresh_response(
            {
                "access_token": "opaque-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
            previous=previous,
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = CodexAuthStore(Path(tmp) / "auth.json")
            store.save(refreshed)
            loaded = store.load()

        assert loaded is not None
        self.assertEqual(loaded.access_token, "opaque-access")
        self.assertGreater(loaded.expires_at, time.time() + 3500)

    def test_run_codex_login_uses_device_auth_flag_when_requested(self) -> None:
        with patch("src.codex_auth._codex_binary", return_value="codex"), patch(
            "src.codex_auth.subprocess.run",
            return_value=type("_Result", (), {"returncode": 0})(),
        ) as run:
            _run_codex_login(device_auth=True)

        run.assert_called_once_with(["codex", "login", "--device-auth"], check=False)


class CodexResponsesStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_start_preflights_configured_model(self) -> None:
        class _Provider:
            async def credentials(self):
                return CodexCredentials(access_token="access", refresh_token="refresh", expires_at=time.time() + 3600)

        http = _FakeHTTPClient(['data: {"type":"response.completed"}'])
        client = OpenAICodexClient(model="gpt-5.5", auth_provider=_Provider())
        client._http = http

        await client.start()

        self.assertEqual(http.last_url, "https://chatgpt.com/backend-api/codex/responses")
        self.assertEqual(http.last_payload["model"], "gpt-5.5")
        self.assertEqual(http.last_payload["max_output_tokens"], 1)

    async def test_codex_start_uses_supplied_preflight_system_prompt(self) -> None:
        class _Provider:
            async def credentials(self):
                return CodexCredentials(access_token="access", refresh_token="refresh", expires_at=time.time() + 3600)

        http = _FakeHTTPClient(['data: {"type":"response.completed"}'])
        client = OpenAICodexClient(model="gpt-5.5", auth_provider=_Provider())
        client._http = http

        await client.start(
            [
                {"role": "system", "content": "real system prompt"},
                {"role": "user", "content": "Reply OK."},
            ]
        )

        self.assertEqual(http.last_payload["instructions"], "real system prompt")
        self.assertEqual(http.last_payload["input"], [{"role": "user", "content": "Reply OK."}])

    async def test_codex_start_maps_unsupported_model_to_cloud_model_hint(self) -> None:
        class _Provider:
            async def credentials(self):
                return CodexCredentials(access_token="access", refresh_token="refresh", expires_at=time.time() + 3600)

        http = _FakeHTTPClient([], status_code=400, body='{"detail":"model not supported"}')
        client = OpenAICodexClient(model="bad-codex-model", auth_provider=_Provider())
        client._http = http

        with self.assertRaisesRegex(RuntimeError, "/cloud model <model>"):
            await client.start()

    async def test_codex_client_uses_codex_oauth_backend_and_account_header(self) -> None:
        class _Provider:
            async def credentials(self):
                return CodexCredentials(
                    access_token="access",
                    refresh_token="refresh",
                    expires_at=time.time() + 3600,
                    account_id="acct",
                )

        http = _FakeHTTPClient(['data: {"type":"response.completed"}'])
        client = OpenAICodexClient(model="gpt-5.5", auth_provider=_Provider())
        client._http = http

        chunks = [chunk async for chunk in client.chat_stream([{"role": "user", "content": "hi"}], use_tools=False)]

        self.assertTrue(chunks[-1].done)
        self.assertEqual(http.last_url, "https://chatgpt.com/backend-api/codex/responses")
        self.assertEqual(http.last_headers["Authorization"], "Bearer access")
        self.assertEqual(http.last_headers["ChatGPT-Account-Id"], "acct")

    async def test_stream_openai_responses_maps_text_and_done(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"type":"response.output_text.delta","delta":"hello"}',
                'data: {"type":"response.output_text.delta","delta":" world"}',
                'data: {"type":"response.completed"}',
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_responses(
                http,
                base_url="https://api.openai.com/v1",
                model="gpt-5.5",
                messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}],
                use_tools=False,
                headers={"Authorization": "Bearer token"},
            )
        ]

        self.assertEqual("".join(chunk.text for chunk in chunks), "hello world")
        self.assertTrue(chunks[-1].done)
        self.assertEqual(http.last_payload["store"], False)
        self.assertEqual(http.last_payload["include"], ["reasoning.encrypted_content"])

    async def test_stream_openai_responses_extracts_xml_tool_call(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"type":"response.output_text.delta","delta":"<tool_call><function=read_file><parameter=path>README.md</parameter></function></tool_call>"}',
                'data: {"type":"response.completed"}',
            ]
        )

        chunks = [
            chunk
            async for chunk in stream_openai_responses(
                http,
                base_url="https://api.openai.com/v1",
                model="gpt-5.5",
                messages=[{"role": "user", "content": "read"}],
                use_tools=True,
            )
        ]

        self.assertEqual(len(chunks[-1].tool_calls), 1)
        self.assertEqual(chunks[-1].tool_calls[0].name, "read_file")
        self.assertEqual(chunks[-1].tool_calls[0].arguments, {"path": "README.md"})

    async def test_stream_openai_responses_raises_on_incomplete_response(self) -> None:
        http = _FakeHTTPClient(
            [
                'data: {"type":"response.output_text.delta","delta":"partial"}',
                'data: {"type":"response.incomplete","incomplete_details":{"reason":"max_output_tokens"}}',
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "incomplete: max_output_tokens"):
            [
                chunk
                async for chunk in stream_openai_responses(
                    http,
                    base_url="https://api.openai.com/v1",
                    model="gpt-5.5",
                    messages=[{"role": "user", "content": "hi"}],
                    use_tools=False,
                )
            ]


if __name__ == "__main__":
    unittest.main()
