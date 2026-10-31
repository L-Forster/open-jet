<h1 align="center">OpenJet</h1>

<p align="center"><a href="README.md">English</a> · 简体中文</p>

<p align="center"><b>在你自己的 GPU 上运行 Qwen3.8-27B，速度是 llama.cpp 的 1.5 倍。</b><br>
24 GB 显存的 RTX 30、40、50 系列 · 支持 Claude Code、Codex、OpenCode 和 OpenClaw · 可离线运行 · 开源</p>

<p align="center">
  <a href="https://github.com/L-Forster/open-jet/stargazers"><img src="https://img.shields.io/github/stars/L-Forster/open-jet" alt="Stars"></a>
  <a href="https://pypi.org/project/open-jet/"><img src="https://img.shields.io/pypi/v/open-jet" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/L-Forster/open-jet" alt="License"></a>
  <a href="https://discord.com/invite/pspKHtExSa"><img src="https://img.shields.io/badge/Discord-join-5865F2" alt="Discord"></a>
</p>



https://github.com/user-attachments/assets/c9ac0b21-67b1-4f20-af70-21741accc774


| RTX 3090，Qwen3.8-27B Q4_K_M | llama.cpp | OpenJet |
|---|---|---|
| 思考模式，短提示词（编程智能体的典型用法） | 60 tok/s | **92 tok/s**（1.5 倍） |
| 思考模式，上下文中有 98K token | 37 tok/s | **79 tok/s**（2.2 倍） |
| 读取 98K token 的提示词 | 112 秒 | **82 秒**（1.4 倍） |
| 贪心解码，4 个提示词的平均值 | 63 tok/s | **107 tok/s**（1.7 倍） |
| 贪心解码，写代码 | 73 tok/s | **140 tok/s**（1.9 倍） |
| 读取你的提示词，2-8K token | 1050-1140 tok/s | **1530-1600 tok/s** |

每个数字都是一块 RTX 3090 上的单个对话，不是批处理。llama.cpp 使用与 OpenJet 相同的参数运行，包括 MTP
投机解码。前三行是当前引擎与 llama.cpp b11982 的对比，用
[`megakernel/bench/run_compare.sh`](megakernel/bench/run_compare.sh) 测得：思考模式开启，强度 xhigh，温度
1.0，5 个提示词 x 4 个随机种子，以及一个 98K token 的提示词 x 4 个随机种子。其余几行来自较早的引擎版本。在相同权重下，OpenJet
的输出与 llama.cpp 一致：KL 散度 0.0009，首选 token 一致率 99.08%
（[结果和复现脚本](megakernel/README.md#results)）。

## 安装

```bash
pip install open-jet
openjet setup      # 检测你的 GPU，下载模型和适配该 GPU 的 OpenJet 预编译引擎
openjet            # OpenJet 自带的编程智能体
openjet serve      # 或者把模型提供给 Claude Code、Codex、OpenCode 或 OpenClaw 使用
```

更多：[docs/zh/installation.md](docs/zh/installation.md)

## 硬件要求

Linux，以及 24 GB 显存的 NVIDIA RTX 30、40 或 50 系列 GPU（sm_86、sm_89、sm_120），驱动 570 或更新版本。
`openjet setup` 会下载为你的 GPU 预编译好的 OpenJet 引擎（[具体是哪一个](docs/zh/installation.md#预编译引擎)）。

24 GB 显卡上的上下文：使用默认的 q8_0 KV 缓存时约 120K token，使用 `openjet setup --long-context` 时最多约 234K
（[KV 缓存预设](megakernel/README.md#kv-cache-presets-mk_kv)）。

更多：[docs/zh/installation.md](docs/zh/installation.md#硬件要求)

## 一块 GPU 上跑 4 个编程智能体

https://github.com/user-attachments/assets/a7fc5010-110d-45fd-9774-a42afa1f3eb9

最多 4 个智能体在同一次权重遍历中一起解码：在 RTX 3090 上，短上下文时总速度约 290 tok/s，运行真实编程智能体时为
150-200 tok/s。

更多：[docs/zh/features.md](docs/zh/features.md#一块-gpu-上跑-4-个编程智能体)

## 你的 GPU 上的结果

`openjet bench --github <你的用户名>` 用固定的工作负载测试你的显卡。把生成的文件通过 pull request 提交到
[`benchmarks/community/`](benchmarks/community/)（[测试内容说明](benchmarks/community/README.md#what-a-run-measures)）。
已提交的结果见[英文 README 中的表格](README.md#results-on-your-gpu)。

## 配合你的编程工具使用

`openjet serve` 支持 Anthropic Messages、OpenAI Responses 和 OpenAI Chat Completions 三种 API；
`openjet connect claude-code`（或 `codex`、`opencode`、`openclaw`）会打印对应的配置。

更多：[docs/zh/features.md](docs/zh/features.md#配合你的编程工具使用)

## OpenJet 的终端智能体

为能装进你 GPU 的模型而设计的编程智能体：有界的上下文、自动压缩上下文、命令需你批准才执行、可恢复的会话，以及
[skills](docs/usage/skills.md) 和 [MCP](docs/usage/mcp.md)。

更多：[docs/zh/features.md](docs/zh/features.md#openjet-的终端智能体)

## Slipstream

前沿模型负责规划和审查；你的本地模型负责阅读、编辑和测试。

更多：[docs/zh/features.md](docs/zh/features.md#slipstream前沿模型规划本地模型干活)

## Python SDK

在你的应用中嵌入本地模型。

更多：[docs/zh/features.md](docs/zh/features.md#python-sdk在应用中嵌入本地-llm)

另见：[针对硬件调优](docs/zh/features.md#针对硬件调优) · [Ollama、LM Studio 或 llama-server](docs/zh/features.md#使用-ollamalm-studio-或已有的-llama-server) · [常见问题](docs/zh/features.md#常见问题)

## 文档

中文文档：[安装](docs/zh/installation.md) · [编程工具](docs/zh/clients.md) · [功能](docs/zh/features.md)。
其余文档暂时只有英文：

- 入门：[Getting started](docs/getting-started.md) · [Quickstart](docs/quickstart.md) ·
  [Configuration](docs/configuration.md) · [Choosing a model](docs/models.md) · [Overview](docs/overview.md)
- 运行时：[OpenJet 引擎](megakernel/README.md) · [llama.cpp](docs/runtimes/llama-cpp.md) ·
  [Benchmarking](docs/benchmarking.md)
- 智能体：[CLI](docs/usage/cli.md) · [Slash commands](docs/usage/slash-commands.md) ·
  [Slipstream](docs/usage/slipstream.md) · [Skills](docs/usage/skills.md) ·
  [MCP](docs/usage/mcp.md) · [Device sources](docs/usage/device-sources.md) ·
  [Workflow harness](docs/usage/workflow-harness.md) ·
  [Session state and logging](docs/usage/session-state-and-logging.md)
- SDK：[Quickstart](docs/sdk/quickstart.md) · [Python SDK](docs/sdk/python-sdk.md)
- 面向 AI 工具和作者：[AGENTS.md](AGENTS.md) · [让智能体来安装](docs/agent-install.md) ·
  [AI reference](docs/AI_REFERENCE.md) · [Media kit](docs/MEDIA_KIT.md)
- 部署：[Jetson](docs/deployment/jetson.md) ·
  [Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md) · [CPU-only](docs/deployment/cpu-only.md)

## 社区与许可证

[Discord](https://discord.gg/pspKHtExSa) · [X / Twitter](https://x.com/FlouisLF) · `AGPL-3.0-only`（[许可说明](docs/licensing.md)、[CONTRIBUTING.md](CONTRIBUTING.md)、[CLA.md](CLA.md)）
