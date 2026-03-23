# Devices

OpenJet keeps a shared device registry in `devices.md` and lets you tag device ids in the TUI.

Persistent device setup is usually clearer from the regular CLI, not from chat. Use:

```bash
open-jet device list
open-jet device add <existing_id> <new_id>
open-jet device on <id>
open-jet device off <id>
```

Run `open-jet device list` first. Do not guess ids. Use the current id shown on the left as `<existing_id>` if you want to rename a device for chat.

## What Exists At Startup

- OpenJet keeps the device registry at `.openjet/state/devices.md`.
- The agent system prompt always includes the concrete path to that file.
- The registry is rewritten from the currently discovered devices plus saved ids and enabled/disabled state.
- The registry lists ids, device kind, transport, hardware path, latest observation file, and latest payload file.
- The registry does not preload any device logs or payload files into model context by itself.

## Commands

- `open-jet device list`
  Lists discovered devices, writes `devices.md`, and shows the current ids you can tag in chat.

- `open-jet device add <existing_id> <new_id>`
  Adds or renames a persistent device id for a discovered device and rewrites `devices.md`.

- `open-jet device on <id>`
  Enables a device that was previously turned off.

- `open-jet device off <id>`
  Disables a device. This is useful for microphones when you want them visible but not capturable.

- `/device [list|add <existing_id> <new_id>|on <id>|off <id>|help]`
  Chat-side mirror of the same persistent device configuration.

- `/devices`
  Backward-compatible alias for `/device`.

Examples:

```text
open-jet device list
open-jet device add camera0 desk_camera
open-jet device off mic0
/device list
/device add camera0 desk_camera
```

## User Flows

### 1. List Devices

```text
open-jet device list
```

Current flow:

1. OpenJet discovers currently available local peripherals.
2. It merges those with any saved ids and disabled state from config.
3. It rewrites `.openjet/state/devices.md`.
4. It prints each usable id.

This command does not capture a frame, record audio, or read GPIO by itself.

### 2. Add Or Rename A Device Id

```text
open-jet device add camera0 desk_camera
```

Current flow:

1. OpenJet resolves the existing device id such as `camera0`.
2. It saves the chosen id such as `desk_camera` into config as a persistent alias.
3. It rewrites `.openjet/state/devices.md`.
4. Future chats can use `@desk_camera`.

This command does not create a new hardware device. It assigns a stable chat id to an already discovered device.

### 3. Enable Or Disable A Device

```text
open-jet device off mic0
open-jet device on mic0
```

Current flow:

1. OpenJet resolves the device id.
2. It updates the persistent enabled/disabled state in config.
3. It rewrites `.openjet/state/devices.md`.

Disabled devices still appear in the registry, but capture tools reject them until they are re-enabled.

### 4. Mention A Device In Chat

```text
@camera0 what is on the desk?
@desk_camera compare this with @gpio0
```

Current flow:

1. The TUI parses `@...` mentions.
2. If a mention matches a device id, the TUI treats it as a device ref instead of a file ref.
3. The TUI rewrites `.openjet/state/devices.md`.
4. The TUI removes the consumed `@device_id` tokens from the user request.
5. The TUI injects a short text note into that turn:
   `IO device registry located in /absolute/path/to/.openjet/state/devices.md. Open if wanting to interact with devices.`
6. The turn also includes the referenced device ids for that request.

Important:

- Tagging a device does not directly capture it.
- Tagging a device does not load all logs, transcripts, or image files.
- Tagging a device only tells the agent where the registry is and which ids were referenced in that turn.

### 5. Ask For A Device Result Without Tagging

The agent can still decide to use devices even if you do not tag one.

Current flow:

1. The permanent system prompt already contains the absolute path to `.openjet/state/devices.md`.
2. The agent can open that file to inspect what devices exist.
3. If needed, the agent can then call a device tool for a specific id.

So tagging helps steer the turn, but it is not the only way the agent can discover devices.

### 6. Agent Gets A Device Result

After the agent decides to interact with a device, it uses the built-in tool layer.

Current flow:

1. The agent opens `devices.md` or relies on the turn hint.
2. It chooses a device id.
3. It calls a device tool such as `camera_snapshot`, `microphone_record`, or `gpio_read`.
4. OpenJet captures or reads the requested device.
5. OpenJet stores the resulting observation under `.openjet/state/observations/...`.
6. OpenJet rewrites `.openjet/state/devices.md` so `latest_observation_file` and `latest_payload_file` stay current.
7. The tool result is returned to the model as tool context.

The exact result depends on the device type:

- camera: one still image plus summary text
- microphone: one short recording, local transcription when available, otherwise speech-detection text
- GPIO: one text snapshot or rolling GPIO buffer

## Direct Tags

Once a device exists, tag its id directly in the prompt. Tagging the id adds a pointer to `devices.md` into context for that turn, but it does not preload every device log or payload file:

```text
@camera0 what is on the desk?
@mic0 detect whether someone is speaking
@gpio0 summarize the current GPIO state
@camera0 @gpio0 what is happening here?
```

## Current Behavior

- `devices.md` is a registry, not a payload dump
- `open-jet device ...` and `/device ...` both rewrite `devices.md` when they change naming or enabled state
- device tools that capture or read devices also rewrite `devices.md`
- tagged device ids do not capture or load logs by themselves
- the model can open `devices.md` and then choose a specific backing file only when relevant
- device tools are available in normal chat/code/review/debug flows

This does not start a background workflow or daemon.

## GPIO Bindings

Raw `gpiochip` devices are controllers, not your actual logical sensors. If you have multiple devices on one GPIO controller, define named bindings in `config.yaml`:

```yaml
gpio_bindings:
  - name: door
    label: Door Sensor
    chip: gpio0
    lines: [17]
  - name: relay-bank
    label: Relay Bank
    chip: /dev/gpiochip0
    lines: [22, 23]
```

Those bindings become separate sources like `@door` and `@relay-bank`, and `gpio_read` will scope the snapshot to those configured lines instead of dumping the whole chip.

## Tool Calls

The same device layer is now exposed through the built-in tool runtime:

- `device_list`
  Lists detected devices and their refs.

- `camera_snapshot`
  Captures a single frame from a camera source and returns it as multimodal tool context.

- `microphone_record`
  Records a short clip, tries bundled local transcription first, and falls back to local speech detection if transcription is unavailable.

- `microphone_set_enabled`
  Turns a microphone source on or off persistently.

- `gpio_read`
  Reads a specific GPIO source. This is the preferred GPIO tool name.

- `sensor_read`
  Legacy alias for `gpio_read`.

## What `devices.md` Is For

`devices.md` is the single user-visible registry that connects the TUI, the agent, and the stored observations.

Use it to answer:

- what devices exist right now
- which ids the chat should use
- whether a device is enabled
- where the latest observation JSON lives
- where the latest payload file lives

Do not treat it as the device data itself. The actual camera images, transcript buffers, and GPIO text buffers remain in the observation store until the agent opens or captures them explicitly.
