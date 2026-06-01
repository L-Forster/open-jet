from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from .airgap import assert_endpoint_allowed
from .codex_auth import CodexAuthError, CodexOAuthProvider
from .runtime_protocol import StreamChunk, stream_openai_responses


class OpenAICodexClient:
    """RuntimeClient implementation backed by OpenAI Codex OAuth."""

    def __init__(
        self,
        *,
        model: str,
        context_window_tokens: int = 272000,
        base_url: str = "https://chatgpt.com/backend-api/codex",
        auth_provider: CodexOAuthProvider | None = None,
        reasoning_effort: str | None = "medium",
        reasoning_summary: str | None = "auto",
        text_verbosity: str | None = "medium",
    ) -> None:
        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("Missing model for OpenAI Codex runtime (`model`).")
        self.context_window_tokens = max(512, int(context_window_tokens))
        self.gpu_layers = 0
        self.base_url = str(base_url or "https://chatgpt.com/backend-api/codex").rstrip("/")
        self.auth_provider = auth_provider or CodexOAuthProvider()
        self.reasoning_effort = _optional_enum(reasoning_effort, {"none", "minimal", "low", "medium", "high", "xhigh"})
        self.reasoning_summary = _optional_enum(reasoning_summary, {"auto", "detailed"})
        self.text_verbosity = _optional_enum(text_verbosity, {"low", "medium", "high"})
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0))

    async def start(self, messages: list[dict] | None = None) -> None:
        assert_endpoint_allowed(self.base_url, label="OpenAI Codex runtime")
        try:
            async for chunk in self._stream_once(
                messages or [{"role": "user", "content": "Reply OK."}],
                use_tools=False,
                extra_body={"max_output_tokens": 1},
            ):
                if chunk.done:
                    break
        except CodexAuthError:
            raise
        except RuntimeError as exc:
            raise RuntimeError(_format_codex_error(exc, model=self.model)) from exc

    async def close(self) -> None:
        await self._http.aclose()

    async def reset_kv_cache(self) -> None:
        return None

    async def save_kv_cache(self, path: Path) -> bool:
        return False

    async def restore_kv_cache(self, path: Path) -> bool:
        return False

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        use_tools: bool = True,
    ) -> AsyncIterator[StreamChunk]:
        assert_endpoint_allowed(self.base_url, label="OpenAI Codex runtime")
        try:
            async for chunk in self._stream_once(messages, use_tools=use_tools):
                yield chunk
        except RuntimeError as exc:
            if not _looks_like_auth_expired(str(exc)):
                raise
            credentials = self.auth_provider.store.load()
            if credentials is None:
                raise CodexAuthError("Not logged in to OpenAI Codex. Run /connect openai-codex first.") from exc
            await self.auth_provider.refresh(credentials)
            async for chunk in self._stream_once(messages, use_tools=use_tools):
                yield chunk

    async def _stream_once(
        self,
        messages: list[dict],
        *,
        use_tools: bool,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        credentials = await self.auth_provider.credentials()
        request_body = self._extra_body()
        if extra_body:
            request_body.update(extra_body)
        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "Accept": "text/event-stream",
            "originator": "openjet",
        }
        if credentials.account_id:
            headers["ChatGPT-Account-Id"] = credentials.account_id
        async for chunk in stream_openai_responses(
            self._http,
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            use_tools=use_tools,
            headers=headers,
            extra_body=request_body,
        ):
            yield chunk

    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
            if self.reasoning_summary:
                body["reasoning"]["summary"] = self.reasoning_summary
        if self.text_verbosity:
            body["text"] = {"verbosity": self.text_verbosity}
        return body


def _optional_enum(value: str | None, allowed: set[str]) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized or normalized == "default":
        return None
    if normalized not in allowed:
        return None
    return normalized


def _looks_like_auth_expired(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in (" 401", "unauthorized", "expired", "invalid_token"))


def _format_codex_error(exc: Exception, *, model: str) -> str:
    text = str(exc).strip() or type(exc).__name__
    lowered = text.lower()
    if "not supported" in lowered or "unsupported" in lowered or ("model" in lowered and "400" in lowered):
        return f"OpenAI Codex rejected model `{model}`. Change it with `/cloud model <model>`. Details: {text}"
    if "401" in lowered or "unauthorized" in lowered or "invalid_token" in lowered:
        return "OpenAI Codex authentication failed. Run /connect openai-codex again."
    if "quota" in lowered or "rate limit" in lowered or "429" in lowered or "plan" in lowered:
        return f"OpenAI Codex is unavailable for this account or quota: {text}"
    return f"OpenAI Codex preflight failed: {text}"
