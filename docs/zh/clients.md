# 编程工具：Claude Code、Codex、OpenCode 和 OpenClaw

[English](../clients.md) · 简体中文

`openjet serve` 运行 `openjet setup` 配置好的模型：GPU 和模型支持时用 OpenJet 引擎，否则用 llama.cpp。它提供三种 API：

| API | 端点 | 使用者 |
|---|---|---|
| Anthropic Messages | `POST /v1/messages`、`POST /v1/messages/count_tokens` | Claude Code |
| OpenAI Responses | `POST /v1/responses` | Codex CLI |
| OpenAI Chat Completions | `POST /v1/chat/completions` | OpenCode、OpenClaw 及大多数其他工具 |

```bash
openjet serve                 # http://127.0.0.1:8080；用 --host 和 --port 修改
```

让它在单独的终端里一直运行。它会打印正在使用哪个引擎。先关掉 OpenJet 自带的智能体：两者都会把模型加载到同一块 GPU 上。

`openjet connect <工具>` 会打印下面的配置，并填好你的服务器地址、模型名和上下文窗口。传入与 `openjet serve` 相同的
`--host` / `--port`。

## Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=openjet
export ANTHROPIC_MODEL=Qwen3.8-27B-Q4_K_M
export ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.8-27B-Q4_K_M
claude
```

token 可以是任意值，服务器不会检查。设置 Haiku 模型后，Claude Code 的后台请求（例如会话标题）也会发到同一个服务器。要保存这些设置，把同样的键写进
`~/.claude/settings.json` 的 `env` 块。

Claude Code 的系统提示词和工具定义很大。用 `openjet context <token 数>` 给服务器设置你的 GPU 能容纳的最大上下文，然后重启
`openjet serve`。提示词放不下时，服务器返回 Anthropic 的 `prompt is too long` 错误，Claude Code 会压缩对话并重试。

## Codex CLI

在 `~/.codex/config.toml` 中添加 provider：

```toml
[model_providers.openjet]
name = "OpenJet"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
```

把下面的内容保存为 `~/.codex/openjet.config.toml`：

```toml
model = "Qwen3.8-27B-Q4_K_M"
model_provider = "openjet"
model_context_window = 100000
```

然后运行 `codex --profile openjet`。Codex 的 `model_reasoning_effort` 对应模型的思考强度：`low` 和 `minimal` 对应
low，`medium` 对应 medium，`high` 和 `xhigh` 对应 xhigh，`none` 关闭思考。

## OpenCode

把下面的内容合并到项目中的 `opencode.json`，或 `~/.config/opencode/opencode.json`：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "openjet": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenJet (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "Qwen3.8-27B-Q4_K_M": {
          "name": "Qwen3.8-27B-Q4_K_M (OpenJet)",
          "limit": { "context": 100000, "output": 32768 }
        }
      }
    }
  }
}
```

运行 `opencode`，用 `/models` 选择 `openjet/Qwen3.8-27B-Q4_K_M`。

## OpenClaw

把下面的内容合并到 `~/.openclaw/openclaw.json`：

```json
{
  "models": {
    "providers": {
      "openjet": {
        "baseUrl": "http://127.0.0.1:8080/v1",
        "apiKey": "openjet",
        "api": "openai-completions",
        "models": [
          { "id": "Qwen3.8-27B-Q4_K_M", "name": "Qwen3.8-27B-Q4_K_M (OpenJet)",
            "contextWindow": 100000, "maxTokens": 32768 }
        ]
      }
    }
  },
  "agents": { "defaults": { "model": { "primary": "openjet/Qwen3.8-27B-Q4_K_M" } } }
}
```

## OpenJet 引擎上的限制

使用 llama.cpp 时，这些端点由 llama-server 自己提供。使用 OpenJet 引擎时：

- **一次一个请求。** 同时到达的请求会相互等待。
- **只缓存一个对话。** 接续上一个对话的请求只处理新增的 token。来自另一个对话的请求（例如后台的标题请求）会处理完整的提示词，之后第一个对话的下一个请求也会如此。
- **只支持文本。** 请求中的图片和文档会被替换成简短的占位符。
- **Anthropic Messages。** 思考块的签名为空。`stop_sequences` 和 `tool_choice` 会被忽略。网页搜索等服务端工具会从工具列表中移除。
- **OpenAI Responses。** 不存储响应：`previous_response_id` 会被拒绝，所以客户端必须在 `input` 中发送完整对话。支持函数工具和自定义（自由格式）工具；其他工具类型（例如网页搜索）会被跳过，并在服务器日志中给出警告。
- **采样。** `top_k` 最大为 32；presence、frequency 和 repetition 惩罚会被忽略。
