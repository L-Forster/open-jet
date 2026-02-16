# open-jet

Minimal agentic TUI for running LLMs on edge devices.

## Setup

1. Install [Ollama](https://ollama.com) on your device.
2. Pull a model: `ollama pull llama3.2`
3. Create a venv and install open-jet (on Arch, use `python -m pip` so the venv is used):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

4. Run:

```bash
open-jet
```

## Configuration

Edit `config.yaml` to change the Ollama endpoint, model, or system prompt.

## Usage

- Type a message and press Enter to chat.
- When the model proposes a shell command, a confirmation dialog appears.
- Press `y` to approve or `n` to deny.
- Press `Ctrl+C` to quit.
