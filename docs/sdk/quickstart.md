# SDK quickstart

Embedding a local model in an application you ship. If you want an agent in your own
terminal instead, see [Getting started](../getting-started.md).

## 1. Install

```bash
pip install open-jet
```

## 2. Provision a model for the project

Run this once, in your project, at build time:

```bash
cd your-project
openjet project
```

It asks three questions:

**What is the model for?** Selection is driven by use case, not raw capability. A
real-time NPC has a 350ms first-token budget; a summarizer has 2.5s. Structured output
carries a higher quant floor than free-form chat, because schema adherence degrades
before prose does.

**Which device are you shipping to?** Declared, not detected — the machine you build on
is not the machine your users have. Provisioning on a 64GB workstation for a handheld
target picks for the handheld.

**How much memory may the model use?** Your application owns the rest of the device. A
chatbot inside a game gets a slice, not the whole GPU.

Then it picks the strongest model that satisfies all three, downloads it into
`.openjet/models/`, and writes `.openjet/config.yaml`.

Non-interactive, for a build script or CI:

```bash
openjet project --use-case dialogue --target handheld --budget 4 --context-tokens 4096
```

See [Choosing a model](../models.md#embedded-in-your-application) for the full catalog
and the `--use-case` / `--target` values.

## 3. Use it

```python
import asyncio

from openjet.sdk import create_inference_session


async def main() -> None:
    session = await create_inference_session()
    try:
        result = await session.run("Greet the player and ask where they are headed.")
        print(result.text)
    finally:
        await session.close()


asyncio.run(main())
```

`create_inference_session()` refuses every tool: text in, text out. An embedded model
cannot reach the shell or the filesystem no matter what it generates, which is what you
want for anything facing your users.

Use `create_agent()` instead when you *do* want the full agent — files, shell, tools,
approvals. That is the same surface the terminal agent uses, and it is unchanged.

Either way the session picks up `.openjet/config.yaml` from anywhere inside the project,
so it behaves the same from `src/`, from a test, or from your app's entrypoint.

## Downloading is never a runtime behaviour

The SDK does not fetch models. `OpenJetSession.create()` verifies the provisioned model
is present and raises if it is not:

```
RuntimeError: Configured model is missing: /app/.openjet/models/Qwen3.5-4B-Q4_K_M.gguf
Run `openjet project` to provision it. OpenJet does not download models at runtime.
```

This is deliberate. An application you ship must not reach the network on a user's
machine, and a user must never be asked to fetch a file and put it somewhere. The model
is acquired once, by you, before you ship.

For a build, that means `openjet project` belongs alongside your other install steps:

```dockerfile
COPY . /app
WORKDIR /app
RUN pip install open-jet && \
    openjet project --use-case chat --target server --budget 12
```

## Shipping the model

The model lives in `.openjet/models/` inside the project so your packaging step can
bundle it like any other asset. `.openjet/` ignores itself in git — weights do not
belong in a repository, and the config records machine-local absolute paths.

Model choice is a hardware decision, not a project decision, so it is not shared through
your repo: a pinned 27B means nothing to a teammate on a 16GB laptop. Each developer and
each build target runs `openjet project` for the machine in front of them.

## Checking what a project is using

```bash
openjet --status
```

```
Runtime: Local model: llama.cpp (GGUF)
Model ref: /app/.openjet/models/Qwen3.5-4B-Q4_K_M.gguf
Context window: 4096
Project model: qwen35-4b-q4km (use case: dialogue, target: handheld)
Config: /home/you/.openjet/config.yaml <- /app/.openjet/config.yaml
```

The `Config:` line lists every file feeding the active configuration, lowest precedence
first, so you can tell whether a project overlay is in effect.

## Related

- [Python SDK reference](python-sdk.md) — session API, tools, approvals, streaming
- [Choosing a model](../models.md) — catalogs for both paths
- [Deployment](../deployment/cpu-only.md) — CPU-only, Jetson, and x86 NVIDIA targets
