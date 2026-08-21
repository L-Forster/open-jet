# Slash Commands

Commands below match the Pi TypeScript TUI. Some older Textual-app spellings remain accepted as aliases.

## Modes and models

- `/mode` open the Local / Codex / OpenRouter / Slipstream picker
- `/mode [status|local|codex|openrouter|slipstream]` inspect or switch mode (`hybrid` remains accepted for Slipstream; `/agent` and `/strategy` remain aliases for `/mode`)
- `/model` configure the active mode (Codex model and effort; OpenRouter model; local model and reasoning; both sides in Slipstream)
- `/model openrouter=stealth/ox-alpha local=local reasoning=on` apply a Slipstream OpenRouter + local pair directly
- `/model codex=gpt-5.6-sol effort=high local=local reasoning=on` apply a Slipstream Codex + local pair directly
- `/effort [none|low|medium|high|xhigh|max]` change Codex reasoning effort (Codex orchestrator only)
- `/runtime [status]` inspect the inference engine (for example llama.cpp)
- `/local` switch to the local runtime profile

## OpenRouter and credentials

- `/login` Pi OpenRouter login (paste an API key; saved to the OS keyring, not a config file)
- `/cloud` open the curated OpenRouter model list (pricing and context in the picker). Selecting a model activates OpenRouter-only mode unless Slipstream is already pairing it as orchestrator
- `/cloud <openrouter-id>` activate a specific OpenRouter model (for example `/cloud stealth/ox-alpha`)
- `/connect` open the credential picker
- `/connect openrouter` same as `/login` when no key is supplied
- `/connect openrouter <api-key>` save an OpenRouter key (the TUI stores only `/connect` in editor history and sends the key on a dedicated RPC field)
- `/connect [status|openai-codex [--device-auth]|openai|anthropic|logout <provider>]` manage other external providers from the Textual app path

## Session

- `/clear` (`/new`, `/reset`) start a new conversation (blocked while a turn is streaming)
- `/clear-chat` clear chat only (keep current server/KV state) where supported
- `/resume` pick a saved Pi chat from `.openjet/pi/sessions/` (blocked while a turn is streaming)
- `/exit` (`/quit`) quit the app

## Workspace and status

- `/help` show commands
- `/status` show context/RAM status
- `/usage` show lifetime token usage by model
- `/device [list|add <existing_id> <new_id>|on <id>|off <id>|help]` list or configure devices in chat
- `/devices` alias for `/device`
- `/condense` condense older context
- `/load <path>` load a file into context
- `/setup` reopen setup wizard (Pi path supports `/setup recommended`)
- `/edit-model [name]` edit a saved model preset
- `/memory [show|clear <user|agent>]` inspect or clear persistent memory
- `/reasoning [status|on|off|default]` control llama.cpp reasoning mode
- `/air-gapped [status|true|false]` control air-gapped mode
- `/plan [status|on|approve|reject]` inspect or control read-only plan mode
- `/mcp status` show configured MCP server status
- `/skill [status|list|clear|load <name[,name...]>|<name[,name...]>]` inspect, load into the current chat, and pin harness skills
- `/skills` alias for `/skill`
- `/todo [status|clear]` inspect or clear the current todo ledger
- `/util [show|hide|toggle|status]` show or hide the utilization line

Persistent device setup is usually clearer from the regular CLI:

```bash
openjet device list
openjet device add <existing_id> <new_id>
openjet device on <id>
openjet device off <id>
```
