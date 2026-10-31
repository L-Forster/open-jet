# TypeScript terminal UI

OpenJet's interactive terminal and agent loop are a standalone TypeScript
executable built on the Earendil Works Pi fork. Pi `AgentSession` is authoritative
for turns, built-in coding tools, streaming, cancellation, retries, compaction,
context accounting, and interactive session persistence.

```text
openjet / open-jet
  -> Python CLI dispatch
     -> openjet-tui
        ├── Pi AgentSession + built-in coding tools
        └── python -m src.tui_server (narrow OpenJet hardware service)
```

The Python service starts and stops OpenJet's configured llama.cpp runtime and
reports hardware, device, air-gap, setup, and model-profile metadata. It does not
receive prompts, execute model turns, approve tools, or persist conversations.
It also exposes OpenJet's device capture, GPIO/sensor, resource-inspection,
persistent-memory, and skill tools as Pi custom tools. File access, editing, and
shell execution remain Pi built-ins, so there is only one coding-tool stack.
The narrow service protocol uses newline-delimited JSON over stdin/stdout; its v1
contract lives in `protocol/openjet-tui-v1.schema.json`. Optional `apiKey` on
`command` requests carries OpenRouter credentials outside slash-command text so
editor history and command strings do not store the secret.

`OpenJetServiceController` is the Python boundary. The frontend creates a Pi model
definition from the prepared OpenJet endpoint, then owns the `AgentSession` and
stores its data under `.openjet/pi/` rather than `.pi/` (including sessions under
`.openjet/pi/sessions/`). Workspace `.openjet/` is gitignored; Pi model config that
must hold a live token for the agent is written owner-only (`0o600`).

OpenRouter login is Pi-owned UI (`/login`, `/cloud`) backed by OpenJet's
`ApiKeyStore` (OS keyring; `OPENROUTER_API_KEY` wins when set). Curated picker
rows come from `src/openrouter_catalog.py` and are generated into
`ui/src/openrouter-models.generated.ts` during the TUI build so Python and
TypeScript cannot drift.

The frontend and agent harness live in
[open-jet-harness](https://github.com/L-Forster/open-jet-harness), checked out as
the `ui/` git submodule. It uses Pi's terminal lifecycle, editor, completion,
markdown, layout, AgentSession, model runtime, session manager, coding tools,
queueing, retry, and compaction primitives. Dependencies are pinned to the
Earendil Works Pi fork at version `1.4.0`; its MIT attribution is recorded in
`ui/NOTICE`. The harness keeps a copy of the protocol schema for standalone
builds; inside an OpenJet checkout its build reads `protocol/` here, and a test
checks the two copies match.

For the managed local runtime (llama-server or the megakernel server) the Python
service advertises a text-only model whose thinking is selected by Pi's thinking
level. Pi then sends `chat_template_kwargs.enable_thinking` on every request and
picks the Qwen thinking or instruct sampling preset through
`samplingParamsByThinkingLevel`. The harness sizes Pi's compaction thresholds to
each model's context window.

## Building

Release wheels contain a platform-matched Bun standalone executable, so installed
users do not need Node.js or Bun:

```bash
git submodule update --init ui
python scripts/build_tui.py
python -m build --wheel
```

`scripts/build_tui.py` regenerates the OpenRouter TypeScript catalog from Python
before `npm run build`. The native build supports Linux x86_64/aarch64, macOS
x86_64/arm64, and Windows x86_64. Source installers either build with the available
toolchain or take the TUI out of the matching platform wheel on PyPI, checksum-verified
(`scripts/install_tui_asset.py`).

For frontend development:

```bash
python -c "from src.openrouter_catalog import write_openrouter_catalog_ts; write_openrouter_catalog_ts()"
cd ui
npm ci --legacy-peer-deps
npm run build
npm test
```

Set `OPENJET_TUI_BINARY` to an alternate executable when testing a development
build. The launcher supplies the exact Python interpreter to the frontend through
`OPENJET_PYTHON`.

## Failure boundary

- A missing, corrupt, unsupported, or incompatible executable produces an
  actionable launcher error and a non-zero exit.
- Backend EOF stops the OpenJet-managed runtime; Pi owns active-turn cancellation
  and session durability.
- Pi tools run directly without OpenJet approval popups. A future targeted safety
  extension can gate destructive commands without reintroducing a second loop.
- The frontend always stops the Pi terminal lifecycle before exiting, including
  fatal backend and signal paths.

Non-interactive CLI commands and the public Python SDK do not import or launch the
terminal frontend.
