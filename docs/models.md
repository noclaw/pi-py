# Custom Models

Add local servers, private endpoints, or any OpenAI-compatible API to pi-py by
editing `~/.pi-py/models.json`. No code changes needed.

The file is merged with the bundled catalog at import time:
- New providers are added alongside the built-in ones.
- Existing providers (e.g. `openai`) are replaced entirely when present in your file.

Use the root [`models.json`](../models.json) in this repo as a starting-point
template — copy it to `~/.pi-py/models.json` and edit from there.

---

## Quick examples

### Ollama (local)

```json
{
  "providers": {
    "ollama": {
      "api": "openai-completions",
      "baseUrl": "http://localhost:11434/v1",
      "apiKey": "ollama",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        { "id": "llama3.1:8b" },
        { "id": "qwen2.5-coder:7b", "input": ["text"] },
        {
          "id": "deepseek-r1:14b",
          "reasoning": true,
          "compat": { "thinkingFormat": "deepseek" }
        }
      ]
    }
  }
}
```

`apiKey` is required by the OpenAI SDK but Ollama ignores the value — any
non-empty string works. `supportsDeveloperRole: false` prevents pi-ai from
sending a `developer`-role system message, which Ollama doesn't understand.

### LM Studio

```json
{
  "providers": {
    "lmstudio": {
      "api": "openai-completions",
      "baseUrl": "http://localhost:1234/v1",
      "apiKey": "lm-studio",
      "compat": {
        "maxTokensField": "max_tokens",
        "supportsDeveloperRole": false,
        "supportsUsageInStreaming": false
      },
      "models": [
        { "id": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF" }
      ]
    }
  }
}
```

### vLLM

```json
{
  "providers": {
    "vllm": {
      "api": "openai-completions",
      "baseUrl": "http://localhost:8000/v1",
      "apiKey": "VLLM_API_KEY",
      "compat": {
        "maxTokensField": "max_tokens",
        "supportsDeveloperRole": false
      },
      "models": [
        {
          "id": "Qwen/Qwen2.5-72B-Instruct",
          "name": "Qwen 2.5 72B",
          "contextWindow": 131072,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

### Proxy / LiteLLM

```json
{
  "providers": {
    "litellm": {
      "api": "openai-completions",
      "baseUrl": "http://localhost:4000/v1",
      "apiKey": "LITELLM_API_KEY",
      "models": [
        {
          "id": "gpt-4o",
          "name": "GPT-4o (via LiteLLM)",
          "input": ["text", "image"],
          "contextWindow": 128000,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

---

## Provider fields

| Field | Required | Description |
|---|---|---|
| `api` | Yes | API type — see [Supported APIs](#supported-apis) |
| `baseUrl` | Yes | API endpoint URL |
| `apiKey` | No | API key. Accepts: literal value, env var name, or `"!shell-command"` |
| `authHeader` | No | `true` → include `Authorization: Bearer <apiKey>` header |
| `headers` | No | Extra headers sent with every request |
| `compat` | No | Shared compatibility defaults for all models in this provider |
| `models` | No | Model definitions (see below) |

### `apiKey` resolution

Three formats, resolved in order:

```json
"apiKey": "sk-..."                          // literal
"apiKey": "MY_API_KEY"                      // environment variable name
"apiKey": "!security find-generic-password -ws 'openai'"   // shell command
```

Shell commands are executed at request time. pi-py has no built-in caching —
wrap slow commands in your own script if needed.

---

## Model fields

| Field | Required | Default | Description |
|---|---|---|---|
| `id` | Yes | — | Model identifier sent to the API |
| `name` | No | `id` | Human-readable label |
| `reasoning` | No | `false` | Supports extended thinking / reasoning |
| `input` | No | `["text"]` | Input modalities: `["text"]` or `["text", "image"]` |
| `contextWindow` | No | `128000` | Context window size in tokens |
| `maxTokens` | No | `16384` | Maximum output tokens |
| `cost` | No | all zeros | `{ "input", "output", "cacheRead", "cacheWrite" }` — $/million tokens |
| `hint` | No | — | Short alias (e.g. `"sonnet"`) for model matching |
| `thinkingLevelMap` | No | — | Map pi thinking levels to provider values — see below |
| `compat` | No | provider `compat` | Per-model overrides — merged with provider-level `compat` |

---

## `thinkingLevelMap`

Maps pi's thinking levels (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`)
to what your model actually supports. Three tristate values:

| JSON value | Meaning |
|---|---|
| omitted | Level is supported; use the provider's default mapping |
| `"string"` | Level is supported; send this specific value |
| `null` | Level is unsupported — hidden from the selector |

Example — model supports only off, high, and a custom "max" level:

```json
{
  "id": "my-model",
  "reasoning": true,
  "thinkingLevelMap": {
    "minimal": null,
    "low": null,
    "medium": null,
    "high": "high",
    "xhigh": "max"
  }
}
```

Example — model cannot disable thinking:

```json
{
  "id": "always-thinking",
  "reasoning": true,
  "thinkingLevelMap": { "off": null }
}
```

---

## OpenAI compatibility (`compat`)

For providers using `api: "openai-completions"`. All fields are optional;
set only what differs from the defaults.

| Field | Default | Description |
|---|---|---|
| `supportsStore` | `false` | Provider accepts the `store` field |
| `supportsDeveloperRole` | `false` | Use `developer` role for system prompt; `false` → use `system` |
| `supportsReasoningEffort` | `false` | Provider accepts the `reasoning_effort` parameter |
| `supportsUsageInStreaming` | `true` | Provider supports `stream_options: { include_usage: true }` |
| `maxTokensField` | `"max_completion_tokens"` | Use `"max_tokens"` for most local/third-party servers |
| `requiresToolResultName` | `false` | Include `name` field on tool result messages |
| `requiresAssistantAfterToolResult` | `false` | Insert empty assistant message after tool results |
| `requiresThinkingAsText` | `false` | Convert thinking blocks to plain text |
| `requiresReasoningContentOnAssistantMessages` | `false` | Include empty `reasoning_content` on all replayed assistant messages (DeepSeek) |
| `thinkingFormat` | `null` | Thinking parameter style: `"deepseek"`, `"openrouter"`, `"together"`, `"zai"`, `"qwen"`, `"qwen-chat-template"` |
| `cacheControlFormat` | `null` | `"anthropic"` for providers that accept Anthropic-style `cache_control` markers |
| `supportsStrictMode` | `true` | Include `strict: true` in tool definitions |
| `supportsLongCacheRetention` | `true` | Provider accepts long cache retention (24h for OpenAI, 1h for Anthropic-style) |

**Common presets for local servers:**

```json
{
  "compat": {
    "supportsDeveloperRole": false,
    "supportsReasoningEffort": false,
    "maxTokensField": "max_tokens"
  }
}
```

---

## Anthropic compatibility (`compat`)

For providers using `api: "anthropic-messages"`.

| Field | Default | Description |
|---|---|---|
| `supportsEagerToolInputStreaming` | `true` | Provider accepts per-tool `eager_input_streaming`. Set `false` for Anthropic-compatible proxies that reject this field |
| `supportsLongCacheRetention` | `true` | Provider accepts `cache_control.ttl: "1h"` for long cache retention |

---

## Overriding built-in providers

Route a built-in provider through a proxy without redefining its models:

```json
{
  "providers": {
    "anthropic": {
      "baseUrl": "https://my-proxy.example.com",
      "apiKey": "PROXY_KEY"
    }
  }
}
```

All built-in Anthropic models remain available; only the endpoint and auth change.

---

## Supported APIs

| `api` value | Description |
|---|---|
| `openai-completions` | OpenAI Chat Completions — most compatible with third-party servers |
| `anthropic-messages` | Anthropic Messages API |

Note: pi-py does not currently have native support for the OpenAI Responses API
or the Google Generative AI API. Gemini models work via Google's
OpenAI-compatible endpoint — see [providers.md](providers.md).

---

## Image generation models

Image generation uses a separate model type and API. See [image_models.md](image_models.md).
