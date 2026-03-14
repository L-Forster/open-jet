# Python SDK

`open-jet` exposes a programmatic session API so you can drive the same bounded-memory agent backend from scripts.

## Example

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

## Approval handler for mutating/shell tools

```python
session = await OpenJetSession.create(
    approval_handler=lambda tool_call: tool_call.name == "shell"
)
```

## Restricting tools

```python
session = await OpenJetSession.create(allowed_tools={"read_file", "load_file", "grep"})
```
