"""`openjet serve` and `openjet connect`: run the local model as an HTTP server and point other coding tools at it.

The server is whichever engine setup chose: OpenJet's Qwen3.8-27B engine or llama-server. Both answer
OpenAI Chat Completions (/v1/chat/completions), OpenAI Responses (/v1/responses) and Anthropic Messages
(/v1/messages), so Claude Code, Codex, OpenCode and OpenClaw talk to either without a proxy.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .config import load_config
from .runtime_registry import DEFAULT_RUNTIME, active_model_ref, active_runtime, create_runtime_client

DEFAULT_SERVE_PORT = 8080
CLIENTS = ("claude-code", "codex", "opencode", "openclaw")


def _model_id(cfg: dict) -> str:
    name = Path(active_model_ref(cfg) or "local").name
    return name.removesuffix(".gguf")


def connect_recipe(client: str, cfg: dict, *, host: str = "127.0.0.1", port: int = DEFAULT_SERVE_PORT) -> str:
    """Configuration that points *client* at `openjet serve` on host:port."""
    model = _model_id(cfg)
    context = int(cfg.get("context_window_tokens") or 32768)
    root = f"http://{host}:{port}"
    start = f"Start the server in another terminal with `openjet serve --port {port}`, then:\n\n"
    if client == "claude-code":
        return start + (
            "Run Claude Code with these variables (Anthropic Messages API):\n\n"
            f"  export ANTHROPIC_BASE_URL={root}\n"
            "  export ANTHROPIC_AUTH_TOKEN=openjet\n"
            f"  export ANTHROPIC_MODEL={model}\n"
            f"  export ANTHROPIC_DEFAULT_HAIKU_MODEL={model}\n"
            "  claude\n\n"
            "To keep them, put the same keys in the \"env\" block of ~/.claude/settings.json.\n"
            "Claude Code's system prompt and tools are large: give the server the largest context that fits\n"
            "(`openjet context <tokens>`). When a prompt overflows, Claude Code compacts and retries."
        )
    if client == "codex":
        return start + (
            "Add the provider to ~/.codex/config.toml (OpenAI Responses API):\n\n"
            "  [model_providers.openjet]\n"
            "  name = \"OpenJet\"\n"
            f"  base_url = \"{root}/v1\"\n"
            "  wire_api = \"responses\"\n\n"
            "Save this as ~/.codex/openjet.config.toml:\n\n"
            f"  model = \"{model}\"\n"
            "  model_provider = \"openjet\"\n"
            f"  model_context_window = {context}\n\n"
            "Then run `codex --profile openjet`."
        )
    if client == "opencode":
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {"openjet": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "OpenJet (local)",
                "options": {"baseURL": f"{root}/v1"},
                "models": {model: {"name": f"{model} (OpenJet)", "limit": {"context": context, "output": min(32768, context)}}},
            }},
        }
        return start + (
            "Merge this into opencode.json in your project, or ~/.config/opencode/opencode.json (OpenAI Chat API):\n\n"
            + _indent(json.dumps(config, indent=2))
            + f"\n\nThen run `opencode` and pick openjet/{model} with /models."
        )
    if client == "openclaw":
        config = {
            "models": {"providers": {"openjet": {
                "baseUrl": f"{root}/v1",
                "apiKey": "openjet",
                "api": "openai-completions",
                "models": [{"id": model, "name": f"{model} (OpenJet)", "contextWindow": context,
                            "maxTokens": min(32768, context)}],
            }}},
            "agents": {"defaults": {"model": {"primary": f"openjet/{model}"}}},
        }
        return start + (
            "Merge this into ~/.openclaw/openclaw.json (OpenAI Chat API):\n\n" + _indent(json.dumps(config, indent=2))
        )
    raise ValueError(f"Unknown client `{client}`. Choose one of: {', '.join(CLIENTS)}.")


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in text.splitlines())


async def serve(*, host: str, port: int) -> None:
    """Run the configured local model in the foreground until interrupted."""
    cfg = load_config()
    if active_runtime(cfg) != DEFAULT_RUNTIME:
        raise SystemExit("`openjet serve` runs a local model; the active profile uses a remote runtime. "
                         "Pick a local model with `/model` or `openjet setup`.")
    client = create_runtime_client(cfg, host=host, port=port)
    print(f"Loading {_model_id(cfg)}...", flush=True)
    await client.start()
    engine = "OpenJet engine" if client.active_backend == "megakernel" else "llama.cpp"
    root = f"http://{host}:{port}"
    print(
        f"Serving {_model_id(cfg)} with {engine} at {root}\n"
        f"  OpenAI Chat Completions  {root}/v1/chat/completions\n"
        f"  OpenAI Responses         {root}/v1/responses\n"
        f"  Anthropic Messages       {root}/v1/messages\n"
        f"Connect a tool with `openjet connect {{{','.join(CLIENTS)}}}`. Ctrl-C stops the server.",
        flush=True,
    )
    try:
        await client._proc.wait()
        print("The model server exited.", flush=True)
    finally:
        await client.close()


def run_serve(*, host: str, port: int) -> None:
    try:
        asyncio.run(serve(host=host, port=port))
    except KeyboardInterrupt:
        pass
