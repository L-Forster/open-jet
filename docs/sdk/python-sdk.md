# Python SDK

`open-jet` exposes a programmatic session API so you can drive the same bounded-memory agent backend from scripts.

Public exports:

- `OpenJetSession`
- `SDKEvent`
- `SDKEventKind`
- `SDKResponse`
- `ToolResult`
- `create_agent()`

## Basic example

```python
import asyncio

from src import OpenJetSession


async def main() -> None:
    session = await OpenJetSession.create()
    try:
        response = await session.run("Summarize the current README")
        print(response.text)

        vision = await session.run("Describe this image", image_paths=["./example.png"])
        print(vision.text)

        async for event in session.stream("Inspect README.md with tools if needed"):
            if event.text:
                print(event.text, end="")
            if event.tool_result:
                print(f"\n[{event.tool_result.tool_call.name}] {event.tool_result.output}")
    finally:
        await session.close()


asyncio.run(main())
```

`run()` returns an `SDKResponse` with:

- `text`: final assistant text
- `tool_results`: executed tool outputs and metadata
- `condense_messages`: any context-condense notices emitted during the turn

`stream()` yields `SDKEvent` values with kinds:

- `TEXT`
- `TOOL_REQUEST`
- `TOOL_RESULT`
- `CONDENSE`
- `DONE`
- `ERROR`

## Session creation options

```python
session = await OpenJetSession.create(
    cfg={"runtime": {"active": "llama_cpp"}},
    system_prompt="You are concise and tool-aware.",
    airgapped=True,
)
```

`OpenJetSession.create()` and `create_agent()` accept:

- `cfg`: config override dict
- `system_prompt`: replacement base system prompt
- `approval_handler`: sync or async callback for approval-gated tools
- `allowed_tools`: explicit allowed tool-name set
- `airgapped`: override air-gapped mode for the session

## Approval handler for gated tools

```python
session = await OpenJetSession.create(
    approval_handler=lambda tool_call: tool_call.name == "shell"
)
```

If no `approval_handler` is provided, approval-gated tools are denied by default.

## Restricting tools

```python
session = await OpenJetSession.create(allowed_tools={"read_file", "load_file", "grep"})
```

## Runtime controls

```python
session.set_airgapped(True)
session.add_turn_context(
    [{"role": "system", "content": "Focus on files under src/ only."}]
)
session.clear_turn_context()
```

- `set_airgapped()` updates the session's network-restriction mode
- `add_turn_context()` injects bounded extra messages for the next turn
- `clear_turn_context()` removes that temporary turn context
