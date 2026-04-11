# Examples

This folder is part of the docs and collects practical workflows you can run with OpenJet on constrained edge hardware.

### Reset context memory vs unload the actual model

If you want to clear conversation/KV state but keep the runtime process available:

```text
/clear
```

`/clear` resets conversation + KV cache. It is useful for context pressure, but it is **not** the same as fully unloading model weights from RAM/VRAM.

For memory-heavy shell tasks (for example `python`, `make/cmake`, `cargo`, or `node` workflows), OpenJet can auto-unload the runtime model before running the tool command, then auto-reload it after command completion when available memory is low.

This keeps large non-LLM jobs from competing with resident model weights on constrained systems.

Manual fallback is still available if you want explicit control:

```text
/exit
```

Then run your other task while the model is not resident in memory, and relaunch with:

```bash
open-jet
```

## 2) Unload model for a heavy task, then reload and continue analysis

A practical low-memory flow:

1. Run inference and capture outputs to files.
2. Let OpenJet auto-unload/reload runtime around a heavy shell task when memory is tight.
3. If needed, force manual unload with `/exit`, then run the heavy non-LLM task.
4. Start OpenJet again with `open-jet` (manual path only).
5. Recover analysis context with:

```text
/resume
/load ./path/to/results.txt
```

## 3) Analyze results dynamically with bounded context

After reloading, keep the conversation focused without losing important state:

```text
/condense
/status
```

- `/condense` compresses older context so long sessions remain useful.
- `/status` shows context/RAM state so you can monitor pressure.

If you need to bring artifacts back into context:

```text
/load ./path/to/results.txt
```

## 4) Use shell workflows (including SSH) from the agent

OpenJet supports controlled tool use, including shell commands (subject to your approval settings).

Example request to the agent:

```text
SSH into 192.168.1.50 as ubuntu, list GPU/CPU/memory config files, and summarize differences from this host.
```

A typical shell-level command sequence the agent may propose:

```bash
ssh ubuntu@192.168.1.50 'uname -a; lscpu; free -h; nvidia-smi || true; ls /etc'
```

This is useful for inventory, validation, and hardware triage in multi-device deployments.

## 5) Connect OpenJet to larger workflows through CLI + SDK

### CLI helpers

```bash
open-jet status
open-jet models
open-jet commands
open-jet benchmark --sweep
```

### Python SDK orchestration

```python
import asyncio

from openjet.sdk import OpenJetSession


async def main() -> None:
    session = await OpenJetSession.create()
    try:
        result = await session.run("Run a quick quality check on ./reports/latest.json")
        print(result.text)
    finally:
        await session.close()


asyncio.run(main())
```

This lets you compose OpenJet with schedulers, CI jobs, robotics pipelines, or other agent workflows while reusing the same bounded-memory runtime behavior.

## 6) Benchmark the active model profile

When you want to compare the currently configured runtime shape on one machine:

```bash
open-jet benchmark
open-jet benchmark --sweep
```

This is useful for:

- validating whether a new GGUF preset is actually faster
- comparing GPU offload settings on the same host
- collecting quick local throughput checks before embedding OpenJet elsewhere

## 7) Workflow harness for structured task execution

To keep agent work organized over many turns:

```text
/mode code
/skill repo-maintainer
/todo status
```

The workflow harness keeps todo and verification state under `.openjet/`, helping constrained local models stay on-task across longer jobs.
