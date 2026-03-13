"""HTTP client for remote OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import httpx

from .runtime_protocol import StreamChunk, stream_openai_chat


def _normalize_base_url(base_url: str | None) -> str:
    src = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").strip()
    if not src:
        return "https://api.openai.com"
    return src[:-1] if src.endswith("/") else src


class OpenAICompatibleClient:
    """Connects to a remote OpenAI-compatible API without managing a local runtime."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        context_window_tokens: int = 8192,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        verify_connection: bool = False,
    ) -> None:
        resolved_model = (model or "").strip()
        if not resolved_model:
            raise ValueError("Missing model for openai-compatible runtime (`openai_compatible_model` or `model`).")
        self.model = resolved_model
        self.base_url = _normalize_base_url(base_url)
        self.context_window_tokens = max(512, int(context_window_tokens))
        self.gpu_layers = 0
        self.verify_connection = bool(verify_connection)
        self.extra_body = dict(extra_body or {})
        self._http = httpx.AsyncClient(timeout=120.0)

        resolved_key = (api_key or "").strip()
        if not resolved_key:
            env_name = (api_key_env or "OPENAI_API_KEY").strip() or "OPENAI_API_KEY"
            resolved_key = os.getenv(env_name, "").strip()

        headers: dict[str, str] = {}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"
        for key, value in (extra_headers or {}).items():
            header_key = str(key).strip()
            header_value = str(value).strip()
            if header_key and header_value:
                headers[header_key] = header_value
        self._headers = headers

    async def start(self) -> None:
        if not self.verify_connection:
            return
        resp = await self._http.get(f"{self.base_url}/v1/models", headers=self._headers or None)
        resp.raise_for_status()

    async def close(self) -> None:
        await self._http.aclose()

    async def reset_kv_cache(self) -> None:
        return None

    async def chat_stream(
        self, messages: list[dict], *, use_tools: bool = True
    ) -> AsyncIterator[StreamChunk]:
        async for chunk in stream_openai_chat(
            self._http,
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            use_tools=use_tools,
            headers=self._headers or None,
            extra_body=self.extra_body or None,
        ):
            yield chunk
