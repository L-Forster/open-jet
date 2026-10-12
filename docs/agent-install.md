# Install OpenJet with a coding agent

These are instructions for Codex, Claude Code, or another agent with terminal access. Carry out the installation, inspect the user's hardware, configure a compatible local coding model without the setup wizard, verify inference, and give the user working usage commands.

## Inspect the machine

Check OS, architecture, Python (3.10 or newer), free disk, RAM, existing OpenJet installations, and available package tools. On NVIDIA hardware run:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
```

On Linux inspect RAM with `free -m` and disk with `df -h`. On macOS use `uname -m`, `sysctl -n hw.memsize`, and `df -h`; Apple silicon shares RAM between CPU and GPU. On Windows use PowerShell `Get-CimInstance Win32_ComputerSystem` and `Get-PSDrive -PSProvider FileSystem`, plus `nvidia-smi` when available. A Jetson or AMD machine may have GPU acceleration without `nvidia-smi`; use OpenJet's hardware detection below.

Do not add GPU VRAM together when choosing a model. Check available memory as well as total memory. Do not change drivers, delete models, or install privileged dependencies without the environment's required authorization.

## Install

Reuse an installation if `openjet setup --help` lists `--non-interactive`. Otherwise install the package in isolation:

```bash
pipx install open-jet
```

Use the actual launcher path reported by pipx if the current shell's PATH does not include it. If the published package does not yet contain `--non-interactive`, install from the source checkout instead; do not fall back to the wizard.

For a source installation on Linux or macOS:

```bash
git clone https://github.com/l-forster/open-jet.git "$HOME/.openjet/source"
cd "$HOME/.openjet/source"
./install.sh
```

If the destination already exists, inspect and reuse a valid checkout rather than overwriting it. Use `./install-macos.sh` on macOS if preferred. The installer exits after installing and reports launcher paths. The checkout's `.venv/bin/openjet` works without a PATH change. On Windows, clone a checkout, run `install.bat` from it, and use `.venv\Scripts\openjet.exe`. If pipx is missing, use the source installer rather than installing into system Python.

## Inspect the recommendation without changing setup

Use the installed launcher for all commands below:

```bash
openjet setup --non-interactive --dry-run --json
```

This detects hardware and returns the recommended model path, context, device, download size, configuration target, and available catalog IDs. Check these against the machine inspection. OpenJet chooses from its own catalog; do not invent model names or maintain a separate model/VRAM table.

Run setup from a directory without a project `.openjet/config.yaml` overlay. If OpenJet reports a configuration conflict, move to a neutral directory and inspect again. Preserve a compatible existing local model. If the user requested a particular model, use its catalog filename or a local `.gguf` path with `--model`.

## Configure without the wizard

For the automatic recommendation:

```bash
openjet setup --non-interactive --yes --json
```

For a specific catalog model returned by dry-run:

```bash
openjet setup --non-interactive --yes --model "<catalog filename>" --json
```

For an existing GGUF:

```bash
openjet setup --non-interactive --yes --model "/absolute/path/model.gguf" --json
```

Replace placeholders with actual values. `--context-tokens <number>` may lower the recommended context. `--yes` accepts model download and runtime provisioning. The command never opens the wizard or TUI, keeps other model files, tests inference, and saves configuration only after verification succeeds.

JSON is written to stdout; progress and build/download logs go to stderr. Let builds and downloads finish. On a nonzero exit, read `stage` and `error`, fix that specific problem, and retry the same command. Missing source-build tools typically require git, CMake, and a compiler; CUDA builds can also require the CUDA toolkit. Use the platform deployment docs for details. Do not silently switch to cloud inference or repeatedly download different models.

## Verify and hand off

Require `status: "ready"` and `verification.passed: true`. Configuration or a successful download alone does not prove inference works. Then run:

```bash
openjet status --json
```

Run status from the user's intended project directory too; confirm no project overlay changes the selected model. Do not launch the interactive TUI in the installing agent's terminal.

Finish by telling the user their detected hardware, model and quantization, actual backend from `verification.backend`, context size, and that local inference passed. Include these usage commands, with their project path and the actual executable path where needed:

```bash
cd /path/to/your/project
openjet
```

Explain: enter a coding task and press Enter; use `@file` to include a file, `/help` for commands, and `/exit` to quit. Include a one-shot example:

```bash
openjet chat "Explain this repository"
```

Mention material limitations such as CPU-only performance or a fallback backend. If installation remains blocked, state the failed stage and exact recovery command instead of claiming it is ready.
