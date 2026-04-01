# Python SDK

Use the Python SDK when you want to embed OpenJet inside another app, agent, worker, or script without using the TUI.

Primary imports:

```python
from openjet.sdk import OpenJetSession, create_agent, recommend_hardware_config
```

The SDK surface has two main jobs:

- embed OpenJet sessions and tool execution in another Python application
- profile hardware and recommend local `llama.cpp` settings

## Basic local session

```python
import asyncio

from openjet.sdk import OpenJetSession


async def main() -> None:
    session = await OpenJetSession.create()
    try:
        result = await session.run("Summarize the current README")
        print(result.text)
    finally:
        await session.close()


asyncio.run(main())
```

`OpenJetSession.create()` reads `config.yaml` by default, so the SDK and TUI can share the same runtime setup.

## Explicit local config

```python
import asyncio

from openjet.sdk import OpenJetSession


async def main() -> None:
    session = await OpenJetSession.create(
        cfg={
            "llama_model": "/home/you/models/Qwen3.5-4B-Q4_K_M.gguf",
            "llama_server_path": "/home/you/llama.cpp/build/bin/llama-server",
            "device": "cuda",
            "gpu_layers": 99,
            "context_window_tokens": 4096,
        }
    )
    try:
        result = await session.run("Explain the repo layout briefly.")
        print(result.text)
    finally:
        await session.close()


asyncio.run(main())
```

## Streaming integration

```python
async for event in session.stream("Inspect README.md with tools if needed"):
    if event.text:
        print(event.text, end="")
    if event.tool_result:
        print(f"\n[{event.tool_result.tool_call.name}] {event.tool_result.output}")
```

Event kinds:

- `TEXT`
- `TOOL_REQUEST`
- `TOOL_RESULT`
- `CONDENSE`
- `DONE`
- `ERROR`

## Hardware profiling and auto-configuration

Use `recommend_hardware_config()` when you want OpenJet to turn machine details
into a recommended model/runtime shape for local `llama.cpp`.

```python
from openjet.sdk import recommend_hardware_config

result = recommend_hardware_config(
    {
        "total_ram_gb": 16,
        "gpu": "cuda",
        "vram_mb": 12288,
        "label": "RTX 4070 workstation",
    }
)

print(result.model.label)
print(result.llama.device)
print(result.llama.gpu_layers)
print(result.llama.context_window_tokens)
```

This is the same recommendation path OpenJet uses to help set up the local CLI
and TUI experience.

## Session creation options

`OpenJetSession.create()` and `create_agent()` accept:

- `cfg`: explicit config override dict
- `system_prompt`: replacement base system prompt
- `approval_handler`: sync or async callback for gated tools
- `allowed_tools`: explicit allowed tool-name set
- `airgapped`: override air-gapped mode for the session

## Approval and tool limits

```python
session = await OpenJetSession.create(
    approval_handler=lambda tool_call: tool_call.name == "shell",
    allowed_tools={"shell", "read_file", "load_file", "grep"},
)
```

If no `approval_handler` is provided, approval-gated tools are denied by default.

## Runtime controls

```python
session.set_airgapped(True)
session.add_turn_context(
    [{"role": "system", "content": "Focus on files under src/ only."}]
)
session.clear_turn_context()
```

Use these when another orchestrator needs to clamp network access or inject temporary per-turn guidance.

## Responses

`run()` returns an `SDKResponse` with:

- `text`: final assistant text
- `tool_results`: executed tool outputs and metadata
- `condense_messages`: context-condense notices emitted during the turn

## Integration guidance

OpenJet works best as the session layer beneath another agent when you want:

- a bounded-memory chat/runtime loop
- explicit tool approvals
- local `llama.cpp` support

If you already have your own orchestrator, prefer:

1. create one `OpenJetSession` per task or worker
2. keep the local model path in `cfg`
3. use `stream()` if your outer agent needs incremental tokens or tool events
4. use `allowed_tools` and `approval_handler` to enforce your own policy

## Related surfaces

- If you want the interactive terminal app, see [CLI usage](../usage/cli.md).
- If you want throughput measurements for the active model profile, see [Benchmarking](../benchmarking.md).
