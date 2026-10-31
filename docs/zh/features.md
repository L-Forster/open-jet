# 功能

[English](../features.md) · 简体中文

## 配合你的编程工具使用

`openjet serve` 用同一个本地模型提供编程工具使用的三种 API：

| 工具 | API | 配置方式 |
|---|---|---|
| Claude Code | Anthropic Messages（`/v1/messages`） | `openjet connect claude-code` |
| Codex CLI | OpenAI Responses（`/v1/responses`） | `openjet connect codex` |
| OpenCode | OpenAI Chat Completions（`/v1/chat/completions`） | `openjet connect opencode` |
| OpenClaw | OpenAI Chat Completions | `openjet connect openclaw` |
| OpenJet 的终端智能体 | 内置 | `openjet` |
| 你自己的应用 | 三种 API 任选其一，或 [Python SDK](../sdk/quickstart.md) | |

每条 `connect` 命令都会按你的服务器地址和模型打印确切的配置。完整配置和已知限制见[编程工具](clients.md)。

## 一块 GPU 上跑 4 个编程智能体

https://github.com/user-attachments/assets/a7fc5010-110d-45fd-9774-a42afa1f3eb9


硬件要求与 OpenJet 引擎相同。从源码检出构建引擎：

```bash
git clone https://github.com/l-forster/open-jet.git && cd open-jet
git clone --depth 1 --branch b10246 https://github.com/ggml-org/llama.cpp.git llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --config Release -j 2
make -C megakernel libqwen38mk.so
pip install jinja2 rich  # rich：mk_top.py 需要
```

用 Qwen3.8-27B Q4_K_M 的 GGUF 启动服务器（如果是 `openjet setup` 下载的，它的路径就是 OpenJet `config.yaml` 中的
`llama_model`）：

```bash
python3 megakernel/server/mk_server.py -m /path/to/Qwen3.8-27B-Q4_K_M.gguf --agents 4
```

然后把最多 4 个智能体（任何支持 OpenAI chat API 的工具）指向 `http://127.0.0.1:18080/v1`。它们在同一次权重遍历中一起解码，所以是共享
GPU，而不是轮流使用：

| RTX 3090，Qwen3.8-27B Q4_K_M | 每个智能体 | 总计 |
|---|---|---|
| 1 个智能体生成 | 117 tok/s | 117 tok/s |
| 4 个智能体生成，短上下文 | ~75 tok/s | ~290 tok/s |
| 4 个智能体生成，每个 10K 上下文 | ~40 tok/s | ~160 tok/s |
| 4 个真实编程智能体（会暂停去运行工具和测试） | 40-75 tok/s | 150-200 tok/s |

4 个智能体共享一个 132K token 的 KV 缓存（q8_0），所以当其他智能体的上下文较短时，一个智能体可以用掉其中大部分。它们共有的系统提示词和工具列表只处理一次。实时查看它们的状态（每个智能体的
tok/s、上下文和任务）：

```bash
python3 megakernel/tools/mk_top.py
```

## OpenJet 的终端智能体


大多数编程智能体假设使用的是上下文窗口巨大的前沿模型。OpenJet 的智能体假设的是能装进你 GPU 的模型，它的上下文有限，在长任务中容易偏离。所以框架本身承担了更多工作：

- `chat`、`code`、`review` 和 `debug` 模式
- 跨轮次保持的步骤状态
- 项目文档和 skill 文档加载到每轮有界的上下文中
- 自动压缩上下文
- shell 命令需要你批准才会执行
- 可以关闭再恢复的会话
- MCP 工具，以及摄像头、麦克风、GPIO 等设备数据源

![Qwen3.6-27B Terminal-Bench 2.0](https://img.shields.io/badge/Qwen3.6--27B%20Terminal--Bench%202.0-59.3-blue)

终端界面用 TypeScript 编写，基于 Pi 终端框架；Python 负责硬件、环境准备和模型生命周期。发布的 wheel 已内置界面，所以不需要
Node.js 或 Bun。见 [TypeScript terminal UI](../architecture/typescript-tui.md)。

使用本地模型时，代码和提示词都不会离开你的电脑。`airgapped: true` 还会屏蔽所有非本机回环地址的模型和工具端点。

## 针对硬件调优

在托管的 llama.cpp 路径上，OpenJet 掌管整个模型生命周期，并按所在机器进行配置：

- **模型选择。** 根据实测的显存和内存选择模型、量化方式和上下文窗口。
- **GPU 卸载。** 设置 GPU 层数，在统一内存的机器上把 MoE 专家卸载到 CPU。
- **设备特性。** 在 Vulkan 上限制上下文长度以避开单次分配上限，并配置 Jetson 的统一内存和时钟频率。
- **内存压力。** 在资源紧张的机器上，执行重型 shell 命令前后会卸载并重新加载模型。
- **KV 缓存。** 在轮次之间保存和恢复缓存，所以对话不会从头重新处理。
- **基准测试。** `openjet bench` 用固定的工作负载测试已配置的模型，并生成可分享的结果文件（[你的 GPU 上的结果](../../README.zh-CN.md#你的-gpu-上的结果)）。`openjet benchmark --sweep`
  用 `llama-bench` 调整 llama.cpp 的 GPU 层数、批大小和线程数（[Benchmarking](../benchmarking.md)）。

## Slipstream：前沿模型规划，本地模型干活

Slipstream 把一个前沿模型（通过你的 ChatGPT 订阅使用的 OpenAI Codex，或任何 OpenRouter 模型）和你的本地模型搭配使用。前沿模型负责意图、架构、任务拆分和最终审查。本地模型负责阅读仓库、编辑文件、运行测试并逐个解决失败。只有它的结论会返回给前沿模型，探索过程不会。

目标是前沿模型只占 20% 的 token，即同一份计划能完成约 5 倍的工作。每次会话的底栏都会显示实际占比；小任务或高风险任务占比更高。本地这一侧也不是免费的：它消耗
GPU 时间。

用 `/mode` → **Slipstream** 启动。用 `/connect openai-codex` 连接，或用 `/login` / `/connect openrouter`。见
[Usage: Slipstream](../usage/slipstream.md)。

## 使用 Ollama、LM Studio 或已有的 llama-server

如果你已经在运行模型服务器，OpenJet 的智能体可以直接连接它。模型由那个服务器控制，所以 OpenJet 的硬件调优和引擎不会生效。

```bash
pipx install 'open-jet[cloud]'
```

在 `config.yaml` 中添加一个 profile（`openjet --status` 会显示文件位置），然后用 `/model <名称>` 选择它：

```yaml
model_profiles:
  - name: ollama-qwen
    runtime: litellm
    provider: openai-compatible
    model: openai/qwen3.5:9b
    base_url: http://127.0.0.1:11434/v1
    context_window_tokens: 32768
```

| 服务器 | `base_url` | `model` |
|---|---|---|
| Ollama | `http://127.0.0.1:11434/v1` | `openai/<ollama 模型>` |
| LM Studio | `http://127.0.0.1:1234/v1` | `openai/<LM Studio 模型 ID>` |
| llama-server | `http://127.0.0.1:8080/v1` | `openai/local` |

## Python SDK：在应用中嵌入本地 LLM

把本地模型打包进你的应用。在构建时按目标设备确定模型大小：

```bash
pip install open-jet
cd your-project
openjet project --use-case dialogue --target handheld --budget 4
```

```python
import asyncio

from openjet.sdk import create_inference_session


async def main() -> None:
    session = await create_inference_session()
    try:
        result = await session.run("Summarise this support ticket in one line.")
        print(result.text)
    finally:
        await session.close()


asyncio.run(main())
```

权重放在 `.openjet/models/` 中，由你的构建打包；用户的电脑上不会下载任何东西。见 [SDK quickstart](../sdk/quickstart.md)。

## 常见问题

**OpenJet 是什么？** 在你自己的 GPU 上快速运行 Qwen3.8-27B 及其他开源模型的工具。它为你的硬件配置好模型和引擎，并提供给 Claude
Code、Codex、OpenCode、OpenClaw、它自带的终端智能体或你的应用使用。从 PyPI 安装 `open-jet` 包即可。

**它和 Ollama 或 LM Studio 有什么不同？** 它们负责运行和提供模型。OpenJet 也做这些，并且在支持的 NVIDIA GPU 上用自己的引擎运行
Qwen3.8-27B，速度是 llama.cpp 的 1.5 倍。它还支持 Anthropic Messages API，所以 Claude Code 无需代理就能连接。

**能离线使用吗？** 能。模型下载到本地后，一切都在本地运行。`airgapped: true` 还会屏蔽云端运行时和远程 MCP 服务器。

**能用云端模型吗？** 只有你主动选择时才会。在 OpenJet 的智能体中，`/mode` 可以在 Local、Codex、OpenRouter 和 Slipstream
之间切换；默认是 Local。

**引擎能用于其他 GPU 或模型吗？** 它支持在 24 GB 或以上显存的 RTX 30、40、50 系列 GPU（sm_86、sm_89、sm_120）上运行
Qwen3.8-27B 的 Q4_K_M 和 UD-Q4_K_XL。它是在 RTX 3090 上调优的。其他情况都用 llama.cpp 运行。
