# Getting started

OpenJet runs local models two ways, and they need almost nothing in common. Pick the one
that matches what you are building.

## I want an agent in my terminal

A local coding agent that works with files, runs shell commands with approval, and keeps
session state across turns. The model runs on your machine.

```bash
pipx install open-jet
openjet setup
openjet
```

`openjet setup` profiles the machine it is running on, picks the strongest coding model
that machine can hold, downloads it, and configures `llama.cpp`.

Continue with:

- [Quickstart](quickstart.md)
- [CLI usage](usage/cli.md)
- [Choosing a model](models.md#terminal-coding-agent)
- [Slash commands](usage/slash-commands.md)

## I want a model inside my own application

A model embedded in something you ship — an in-app assistant, an NPC that talks, a
classifier, an extraction pipeline. Your users never see a model, a download, or a
config file.

```bash
pip install open-jet
cd your-project
openjet project
```

`openjet project` is a build-time step. It asks what the model is for, what device you
are shipping to, and how much memory your application will concede to the model, then
downloads the model into your project so your build can bundle it.

Nothing in the SDK downloads at runtime. If the model is missing, the session raises
immediately rather than reaching for the network on a user's machine.

Continue with:

- [SDK quickstart](sdk/quickstart.md)
- [Choosing a model](models.md#embedded-in-your-application)
- [Python SDK reference](sdk/python-sdk.md)
- [Project configuration](configuration.md#project-configuration-openjetconfigyaml)
- [Deployment](deployment/cpu-only.md)

## Which one am I?

| | Terminal agent | Embedded in an application |
|---|---|---|
| Provisioning command | `openjet setup` | `openjet project` |
| Hardware | Detected on this machine | Declared: the device you ship to |
| Memory available to the model | Whatever the machine has | The slice your application concedes |
| Selection optimises for | Coding capability | Use case and latency |
| Model lives in | A machine-wide store | Your project, so the build bundles it |
| Config | Machine-wide `config.yaml` | `.openjet/config.yaml` in the project |

Both can coexist on one machine. A project overlay only applies inside that project.
