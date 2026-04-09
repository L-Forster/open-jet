# Slash Commands

- `/help` show commands
- `/exit` quit app
- `/clear` clear chat and restart runtime (flush KV cache)
- `/clear-chat` clear chat only
- `/status` show context/RAM status
- `/voice [status|start [mic]|stop]` start microphone dictation mode; spoken utterances build a local draft, `send message` submits it, `clear message` discards it, and `stop listening` exits voice mode. Spoken assistant/tool output is not implemented in this build.
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
- `/skill [status|list|clear|load <name[,name...]>|<name[,name...]>]` inspect, load into the current chat, and pin harness skills
- `/skills [status|list|clear|load <name[,name...]>|<name[,name...]>]` alias for `/skill`
- `/step [status|next|split]` inspect or control the active workflow step
- `/util [show|hide|toggle|status]` show or hide the utilization line

Persistent device setup is usually clearer from the regular CLI:

```bash
open-jet device list
open-jet device add <existing_id> <new_id>
open-jet device on <id>
open-jet device off <id>
```

Voice output currently has no shipped provider implementation. Any configured `voice_output.provider` value will be reported as unavailable until a real TTS backend is added.
