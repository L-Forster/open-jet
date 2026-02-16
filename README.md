# open-jetson

Minimal agentic TUI for running LLMs on NVIDIA Jetson edge devices.

## Setup

1. Install [Ollama](https://ollama.com) on your Jetson device.
2. Pull a model: `ollama pull llama3.2`
3. Install open-jetson:

```bash
pip install -e .
```

4. Run:

```bash
open-jetson
```

## Configuration

Edit `config.yaml` to change the Ollama endpoint, model, or system prompt.

## Usage

- Type a message and press Enter to chat.
- When the model proposes a shell command, a confirmation dialog appears.
- Press `y` to approve or `n` to deny.
- Press `Ctrl+C` to quit.
