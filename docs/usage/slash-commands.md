# Slash Commands

- `/help` show commands
- `/exit` quit app
- `/clear` clear chat and restart runtime (flush KV cache)
- `/clear-chat` clear chat only
- `/status` show context/RAM status
- `/device [list|add <existing_id> <new_id>|on <id>|off <id>|help]` list or configure devices in chat
- `/devices` alias for `/device`
- `/condense` condense older context
- `/load <path>` load a file into context
- `/resume` pick a saved chat checkpoint from `.openjet/state/` and load it back into the TUI
- `/setup` reopen setup wizard
- `/model` open an arrow-key picker to switch saved model presets
- `/model [status|list|<name>]` show or switch saved model presets
- `/models` alias for `/model`
- `/edit-model [name]` edit a saved model preset
- `/memory [show|clear <user|agent>]` inspect or clear persistent memory
- `/reasoning [status|on|off|default]` control llama.cpp reasoning mode
- `/air-gapped [status|true|false]` control air-gapped mode
- `/mode [chat|code|review|debug|status]` set harness mode
- `/plan [status|on|approve|reject]` inspect or control read-only plan mode
- `/skill [status|list|clear|load <name[,name...]>|<name[,name...]>]` inspect, load into the current chat, and pin harness skills
- `/skills [status|list|clear|load <name[,name...]>|<name[,name...]>]` alias for `/skill`
- `/todo [status|clear]` inspect or clear the current todo ledger
- `/util [show|hide|toggle|status]` show or hide the utilization line

Persistent device setup is usually clearer from the regular CLI:

```bash
open-jet device list
open-jet device add <existing_id> <new_id>
open-jet device on <id>
open-jet device off <id>
```
