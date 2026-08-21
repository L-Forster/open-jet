// Generated from src/openrouter_catalog.py. Do not edit.
export const CURATED_OPENROUTER_MODELS = [
  {
    "id": "stealth/ox-alpha",
    "name": "Ox Alpha (free)",
    "contextWindow": 1048576,
    "maxTokens": 131072,
    "reasoning": true,
    "featured": true,
    "cost": {
      "input": 0,
      "output": 0,
      "cacheRead": 0,
      "cacheWrite": 0
    }
  },
  {
    "id": "openrouter/free",
    "name": "Free router",
    "contextWindow": 200000,
    "maxTokens": 4096,
    "reasoning": true,
    "featured": false,
    "cost": {
      "input": 0,
      "output": 0,
      "cacheRead": 0,
      "cacheWrite": 0
    }
  },
  {
    "id": "anthropic/claude-opus-4.8",
    "name": "Claude Opus 4.8",
    "contextWindow": 1000000,
    "maxTokens": 128000,
    "reasoning": true,
    "featured": false,
    "cost": {
      "input": 5,
      "output": 25,
      "cacheRead": 0.5,
      "cacheWrite": 6.25
    }
  },
  {
    "id": "google/gemini-3.1-pro-preview",
    "name": "Gemini 3.1 Pro",
    "contextWindow": 1048576,
    "maxTokens": 65536,
    "reasoning": true,
    "featured": false,
    "cost": {
      "input": 2,
      "output": 12,
      "cacheRead": 0.2,
      "cacheWrite": 0.375
    }
  },
  {
    "id": "x-ai/grok-4.20",
    "name": "Grok 4.20",
    "contextWindow": 2000000,
    "maxTokens": 4096,
    "reasoning": true,
    "featured": false,
    "cost": {
      "input": 1.25,
      "output": 2.5,
      "cacheRead": 0.2,
      "cacheWrite": 0
    }
  },
  {
    "id": "deepseek/deepseek-v4-pro",
    "name": "DeepSeek V4 Pro",
    "contextWindow": 1048576,
    "maxTokens": 131072,
    "reasoning": true,
    "featured": false,
    "cost": {
      "input": 1.168,
      "output": 2.336,
      "cacheRead": 0.09855,
      "cacheWrite": 0
    }
  },
  {
    "id": "z-ai/glm-5.1",
    "name": "GLM 5.1",
    "contextWindow": 202752,
    "maxTokens": 131072,
    "reasoning": true,
    "featured": false,
    "cost": {
      "input": 1.4,
      "output": 4.4,
      "cacheRead": 0.26,
      "cacheWrite": 0
    }
  },
  {
    "id": "moonshotai/kimi-k2.5",
    "name": "Kimi K2.5",
    "contextWindow": 262144,
    "maxTokens": 4096,
    "reasoning": true,
    "featured": false,
    "cost": {
      "input": 0.41,
      "output": 2.06,
      "cacheRead": 0.07,
      "cacheWrite": 0
    }
  }
] as const;
