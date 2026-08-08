# Python SDK

Use the Python SDK when you want to embed OpenJet inside another app, agent, worker, or script without using the TUI.

Primary imports:

```python
from openjet.sdk import (
    OpenJetSession,
    create_agent,
    create_inference_session,
    recommend_hardware_config,
)
```

The SDK surface has three main jobs:

- run a local model as plain text in, text out, inside an application you ship
- embed OpenJet sessions and tool execution in another Python application
- profile hardware and recommend local `llama.cpp` settings

If you are shipping a model inside your own application, start with the
[SDK quickstart](quickstart.md) — the model has to be provisioned into the project with
`openjet project` before any session will start.

## Three ways to create a session

| Constructor | Tools | Use it for |
|---|---|---|
| `create_inference_session()` | none — every tool refused | a model embedded in something you ship |
| `OpenJetSession.create()` | whatever `allowed_tools` permits | your own policy, explicitly stated |
| `create_agent()` | the full agent surface | the same loop the terminal agent runs |

### Inference-only sessions

```python
import asyncio

from openjet.sdk import create_inference_session


async def main() -> None:
    session = await create_inference_session(
        system_prompt="You are a shopkeeper in a fantasy village. Two sentences maximum.",
    )
    try:
        result = await session.run("The player asks what you have for sale.")
        print(result.text)
    finally:
        await session.close()


asyncio.run(main())
```

`create_inference_session()` passes an empty `allowed_tools` set, so the model cannot
reach the shell or the filesystem no matter what it generates. That is the shape you want
for anything facing your users. It accepts `cfg`, `system_prompt`, `root`, and
`airgapped`; there is no `approval_handler`, because nothing can be approved.

Sessions pick up `.openjet/config.yaml` from anywhere inside the project, so behaviour is
the same from `src/`, from a test, or from your app's entrypoint. See
[Configuration](../configuration.md#project-configuration-openjetconfigyaml).

### Models are never fetched at runtime

`OpenJetSession.create()` verifies the configured local model exists and raises if it
does not. It does not download it:

```
RuntimeError: Configured model is missing: /app/.openjet/models/Qwen3.5-4B-Q4_K_M.gguf
Run `openjet project` to provision it. OpenJet does not download models at runtime.
```

Acquisition is a build-time step. An application you ship must not reach the network on a
user's machine, and a user must never be asked to fetch a file and put it somewhere. The
check applies to the local `llama.cpp` runtime; remote runtimes have no local model to
verify.

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

`OpenJetSession.create()`, `create_agent()`, and `create_inference_session()` accept:

- `cfg`: explicit config override dict
- `system_prompt`: replacement base system prompt
- `approval_handler`: sync or async callback for gated tools
- `allowed_tools`: explicit allowed tool-name set
- `airgapped`: override air-gapped mode for the session
- `root`: project root to resolve file work and the `.openjet/` overlay against

`create_inference_session()` takes `cfg`, `system_prompt`, `root`, and `airgapped` only —
it fixes `allowed_tools` to the empty set, which leaves nothing for an
`approval_handler` to decide.

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

- If you are embedding a model in an application you ship, see the [SDK quickstart](quickstart.md).
- For the embedded model catalog, use cases, and target devices, see [Choosing a model](../models.md).
- For the project overlay and config precedence, see [Configuration](../configuration.md#project-configuration-openjetconfigyaml).
- If you want the interactive terminal app, see [CLI usage](../usage/cli.md).
- If you want throughput measurements for the active model profile, see [Benchmarking](../benchmarking.md).
