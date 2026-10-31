# 安装

[English](../installation.md) · 简体中文

## 硬件要求

| 电脑 | 能运行什么 |
|---|---|
| Linux，NVIDIA RTX 30、40 或 50 系列，24 GB 或以上显存（sm_86、sm_89、sm_120，以及对应的 RTX 工作站显卡） | [OpenJet 引擎](../../megakernel/README.md)上的 Qwen3.8-27B |
| Windows | llama.cpp。OpenJet 引擎在 Windows 上只提供预编译版本，目前仍在构建中 |
| NVIDIA Turing，24 GB 或以上显存（sm_75，例如 48 GB 的 Quadro RTX 8000） | OpenJet 引擎，实验性支持（[Turing](../../megakernel/README.md#turing-rtx-8000)） |
| 其他 GPU、Apple 芯片，或只有 CPU | llama.cpp 上能装下的最好的模型（[按内存选择](../models.md#pick-by-memory)、[纯 CPU](../deployment/cpu-only.md)） |
| Jetson | llama.cpp（[Jetson 设置](../deployment/jetson.md)） |

使用 OpenJet 引擎还需要支持 CUDA 12.8 的 NVIDIA 驱动（570 或更新版本）。`openjet setup`
会下载为你的 GPU 预编译好的引擎（[具体是哪一个](#预编译引擎)），所以不需要安装 CUDA 工具包。如果没有匹配的预编译引擎，setup
会自行编译，这需要 `nvcc`（CUDA 12 以上，RTX 50 系列需要 12.8 以上）。如果引擎无法安装，setup 会说明原因，并改用 llama.cpp
运行模型。

在 24 GB 显卡上，使用默认的 q8_0 KV 缓存时上下文约为 120K token；使用 `openjet setup --long-context`
时最多约 234K，此时缓存以 5 bit 存储，最近的 2K 个位置保持 q8_0
（[KV 缓存预设](../../megakernel/README.md#kv-cache-presets-mk_kv)）。

哪些模型文件能在哪些硬件上运行，以及下载链接：[选择模型](../models.md)。

### 无法访问 Hugging Face 时

`openjet setup` 从 huggingface.co 下载模型。如果你的网络无法访问它，可以先从你能访问的镜像下载同一个 GGUF
文件（例如 `Qwen3.8-27B-Q4_K_M.gguf`），然后在 setup 中选择这个本地 `.gguf` 文件。

## 预编译引擎

每个 [GitHub release](https://github.com/L-Forster/open-jet/releases) 都附带为各个 GPU 架构编译好的 OpenJet
引擎。`openjet setup` 读取你 GPU 的计算能力，并从与你的 `open-jet` 包版本相同的 release 中下载对应的压缩包，所以你不需要自己选择。下面的链接指向最新的
release。

| GPU（24 GB 或以上） | 架构 | Linux |
|---|---|---|
| RTX 5090、RTX PRO Blackwell | sm_120 | [openjet-engine-linux-x64-sm120.tar.gz](https://github.com/L-Forster/open-jet/releases/latest/download/openjet-engine-linux-x64-sm120.tar.gz) |
| RTX 4090、RTX 6000 Ada、L40、L4 | sm_89 | [openjet-engine-linux-x64-sm89.tar.gz](https://github.com/L-Forster/open-jet/releases/latest/download/openjet-engine-linux-x64-sm89.tar.gz) |
| RTX 3090、3090 Ti、RTX A5000、A6000 | sm_86 | [openjet-engine-linux-x64-sm86.tar.gz](https://github.com/L-Forster/open-jet/releases/latest/download/openjet-engine-linux-x64-sm86.tar.gz) |
| Titan RTX、Quadro RTX 6000 和 8000（实验性） | sm_75 | [openjet-engine-linux-x64-sm75.tar.gz](https://github.com/L-Forster/open-jet/releases/latest/download/openjet-engine-linux-x64-sm75.tar.gz) |

每个压缩包包含 Q4_K_M 的引擎（`libqwen38mk.so`）、UD-Q4_K_XL 的引擎（`libqwen38mk_mixed.so`，sm_75 中没有，所以 Turing
用 llama.cpp 运行 UD-Q4_K_XL），以及一个只用于分词器的 CPU 版 `libllama`。CUDA 已静态链接，所以只需要 NVIDIA
驱动。用 `nvidia-smi --query-gpu=name,compute_cap --format=csv` 查看 GPU 的架构（8.6 即 sm_86）。

Windows 压缩包（`openjet-engine-windows-x64-sm*.zip`）还没有发布；在那之前，Windows 上的 OpenJet 使用 llama.cpp
运行模型。

## 一行命令安装

Linux 和 macOS：

```bash
curl -fsSL https://github.com/L-Forster/open-jet/releases/latest/download/install.sh | sh
```

Windows PowerShell：

```powershell
irm https://github.com/L-Forster/open-jet/releases/latest/download/install.ps1 | iex
```

安装脚本使用 [uv](https://docs.astral.sh/uv/) 下载一份独立于系统的 Python 3.12，并用它从 PyPI 安装 `open-jet`
包。`openjet` 放在 `~/.local/bin`（Windows 上是 `%USERPROFILE%\.local\bin`），安装脚本会把它加入
PATH。要安装特定版本，先设置 `OPENJET_VERSION`（例如 `OPENJET_VERSION=0.5.4`）。再次运行同一条命令即可更新。如果已经装有
Python，`pipx install open-jet` 效果相同。

## 让你的智能体来安装

让 Codex 或 Claude Code [按照智能体安装说明操作](../agent-install.md)。它可以检查你的硬件、选择模型、不经过向导直接配置
OpenJet、验证推理，并告诉你常用命令。

安装好之后，非交互方式是：

```bash
openjet setup --non-interactive --dry-run --json
openjet setup --non-interactive --yes --json
```

## 克隆并安装

```bash
git clone https://github.com/l-forster/open-jet.git
cd open-jet
./install.sh
```

在 macOS 上使用：

```bash
./install-macos.sh
```

在 Windows 上运行：

```bat
curl -L https://www.openjet.dev/install.bat -o install.bat && install.bat
```

这会创建本地虚拟环境、安装仓库、创建 `open-jet` / `openjet` 命令，然后退出。运行 `openjet setup --yes`
接受硬件推荐配置，或运行 `openjet setup` 使用向导。这是可编辑的 Python 安装，不是编译后的发布版本。

如果 `src/` 下残留旧的原地编译产物，安装脚本会删除它们，确保导入的仍是源码树。

## 可选：编译后的 wheel

如果需要编译后的包，单独构建：

```bash
./scripts/build-compiled.sh
```

这会在 `dist/` 中生成一个 wheel，并且与普通的可编辑安装互不影响。

## 运行时

OpenJet 在本地运行 GGUF 模型：GPU 支持时，Qwen3.8-27B 运行在 OpenJet 引擎上，其他一切运行在 `llama.cpp` 的
`llama-server` 上。

## 推荐流程

先运行：

```bash
openjet setup
```

OpenJet 会先尝试复用本机已有的配置。缺少的部分，setup 可以帮你准备好。

setup 可以：

- 复用已经安装的 `llama-server`
- 克隆并编译 `llama.cpp` 得到 `llama-server`
- 使用本地的 `.gguf` 文件
- 直接下载推荐的 GGUF

自动准备仍然需要：

- 网络连接
- `git`
- `cmake`
- 可用的编译工具链

如果想自己控制运行时，仍可参考手动构建指南：

- [Jetson 设置](../deployment/jetson.md)
- [Linux x86 + NVIDIA](../deployment/linux-x86-nvidia.md)
- [纯 CPU](../deployment/cpu-only.md)

## 公开的 SDK 导入方式

在脚本和集成中使用：

```python
from openjet.sdk import OpenJetSession
```
