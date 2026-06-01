from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator

from .airgap import assert_endpoint_allowed, endpoint_is_loopback
from .api_auth import ApiKeyStore, normalize_provider_id
from .runtime_protocol import (
    StreamChunk,
    _TOOL_CALL_MARKER,
    _TOOL_CALL_MARKER_OVERLAP,
    _messages_with_tool_guidelines,
    parse_tool_calls,
)


class LiteLLMUnavailableError(RuntimeError):
    pass


class LiteLLMClient:
    """RuntimeClient implementation backed by LiteLLM."""

    def __init__(
        self,
        *,
        model: str,
        provider: str = "",
        base_url: str = "",
        api_key_env: str = "",
        context_window_tokens: int = 128000,
        auth_store: ApiKeyStore | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("Missing model for LiteLLM runtime (`model`).")
        self.provider = normalize_provider_id(provider) or _provider_from_model(self.model)
        self.base_url = str(base_url or "").strip()
        self.api_key_env = str(api_key_env or "").strip()
        self.context_window_tokens = max(512, int(context_window_tokens))
        self.gpu_layers = 0
        self.auth_store = auth_store or ApiKeyStore()
        self.extra_body = dict(extra_body or {})

    async def start(self, messages: list[dict] | None = None) -> None:
        self._assert_network_allowed()
        litellm = self._import_litellm()
        kwargs = self._completion_kwargs(messages or [{"role": "user", "content": "Reply OK."}], stream=False)
        kwargs["max_tokens"] = 1
        try:
            await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise RuntimeError(_format_litellm_error(exc, provider=self.provider)) from exc

    async def close(self) -> None:
        return None

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
        self._assert_network_allowed()
        litellm = self._import_litellm()
        payload_messages = _messages_with_tool_guidelines(messages) if use_tools else messages
        kwargs = self._completion_kwargs(payload_messages, stream=True)
        try:
            stream = await litellm.acompletion(**kwargs)
            async for chunk in self._chunks_to_stream(stream, use_tools=use_tools):
                yield chunk
        except Exception as exc:
            raise RuntimeError(_format_litellm_error(exc, provider=self.provider)) from exc

    def _completion_kwargs(self, messages: list[dict], *, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        api_key = self._api_key()
        if api_key:
            kwargs["api_key"] = api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
            kwargs["base_url"] = self.base_url
        kwargs.update(self.extra_body)
        return kwargs

    async def _chunks_to_stream(self, stream, *, use_tools: bool) -> AsyncIterator[StreamChunk]:
        content_buf = ""
        reasoning_buf = ""
        text_emit_buffer = ""
        saw_content_tool_markup = False
        async for chunk in stream:
            delta = _first_delta(chunk)
            text = _field(delta, "content") or _field(delta, "text")
            if text:
                text = str(text)
                content_buf += text
                if not use_tools:
                    yield StreamChunk(text=text)
                elif saw_content_tool_markup:
                    yield StreamChunk(tool_args_delta=text)
                else:
                    text_emit_buffer += text
                    marker_index = text_emit_buffer.lower().find(_TOOL_CALL_MARKER)
                    if marker_index >= 0:
                        visible = text_emit_buffer[:marker_index]
                        if visible:
                            yield StreamChunk(text=visible)
                        hidden = text_emit_buffer[marker_index:]
                        if hidden:
                            yield StreamChunk(tool_args_delta=hidden)
                        text_emit_buffer = ""
                        saw_content_tool_markup = True
                    elif len(text_emit_buffer) > _TOOL_CALL_MARKER_OVERLAP:
                        visible = text_emit_buffer[:-_TOOL_CALL_MARKER_OVERLAP]
                        text_emit_buffer = text_emit_buffer[-_TOOL_CALL_MARKER_OVERLAP:]
                        if visible:
                            yield StreamChunk(text=visible)
            reasoning = _field(delta, "reasoning_content") or _field(delta, "reasoning")
            if reasoning:
                reasoning = str(reasoning)
                reasoning_buf += reasoning
                yield StreamChunk(reasoning=reasoning)
        if use_tools and not saw_content_tool_markup and text_emit_buffer:
            yield StreamChunk(text=text_emit_buffer)
        tool_calls = parse_tool_calls(f"{content_buf}\n{reasoning_buf}") if use_tools else []
        yield StreamChunk(tool_calls=tool_calls, done=True)

    def _api_key(self) -> str | None:
        api_key = self.auth_store.resolve_key(self.provider, env_name=self.api_key_env)
        if api_key:
            return api_key
        if self.base_url and _is_probably_local_api(self.base_url):
            return "openjet-local"
        raise RuntimeError(
            f"Missing API key for provider `{self.provider}`. "
            f"Run /connect {self.provider} or set {self.api_key_env or 'the provider API key environment variable'}."
        )

    def _assert_network_allowed(self) -> None:
        target = self.base_url or _provider_default_endpoint(self.provider)
        assert_endpoint_allowed(target, label=f"LiteLLM provider `{self.provider}`")

    @staticmethod
    def _import_litellm():
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise LiteLLMUnavailableError(
                "LiteLLM support is not installed. Install it with `pip install open-jet[cloud]`."
            ) from exc
        return litellm


def _provider_from_model(model: str) -> str:
    prefix = str(model or "").split("/", 1)[0].strip().lower()
    return prefix or "openai-compatible"


def _provider_default_endpoint(provider: str) -> str:
    provider_id = normalize_provider_id(provider)
    if provider_id == "anthropic":
        return "https://api.anthropic.com"
    if provider_id == "openrouter":
        return "https://openrouter.ai"
    if provider_id == "google":
        return "https://generativelanguage.googleapis.com"
    if provider_id == "xai":
        return "https://api.x.ai"
    if provider_id == "mistral":
        return "https://api.mistral.ai"
    if provider_id == "deepseek":
        return "https://api.deepseek.com"
    return "https://api.openai.com"


def _is_probably_local_api(base_url: str) -> bool:
    return endpoint_is_loopback(base_url)


def _first_delta(chunk: Any) -> Any:
    choices = _field(chunk, "choices") or []
    if not choices:
        return {}
    choice = choices[0]
    return _field(choice, "delta") or {}


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _format_litellm_error(exc: Exception, *, provider: str) -> str:
    text = str(exc).strip() or type(exc).__name__
    lowered = text.lower()
    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered or "401" in lowered:
        return f"LiteLLM authentication failed for `{provider}`. Check /connect {provider} or the API key env var."
    if "rate limit" in lowered or "429" in lowered or "quota" in lowered:
        return f"LiteLLM provider `{provider}` is rate limited or out of quota: {text}"
    if "not found" in lowered or "404" in lowered or "model" in lowered and "invalid" in lowered:
        return f"LiteLLM provider `{provider}` rejected the configured model: {text}"
    if "connection" in lowered or "connect" in lowered or "base" in lowered:
        return f"LiteLLM provider `{provider}` connection failed. Check base_url and network access: {text}"
    return f"LiteLLM provider `{provider}` failed: {text}"
