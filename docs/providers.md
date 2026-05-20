# Providers

## Default provider and model

Set the provider and model used when `model=None` in `create_agent()` or `pi-py prompt`:

**`~/.pi-py/settings.json`**
```json
{
  "defaultProvider": "anthropic",
  "defaultModel": "claude-sonnet-4-6"
}
```

A project-level `.pi-py/settings.json` anywhere on the path from the working
directory up to the filesystem root overrides the global file — useful for
per-project model selection.

```python
import pi_agent

model = pi_agent.get_default_model()          # resolves from settings
harness = await pi_agent.create_agent()        # model=None uses the default
```

---

## Credential resolution order

For each provider, pi-py checks in this order:

1. `get_api_key_and_headers` callable passed to `AgentHarness` or `create_agent()`
2. `~/.pi-py/auth.json` (see below)
3. Environment variable

---

## Built-in providers

These providers have models in the bundled catalog and auto-resolve their API key
from environment variables.

| Provider | Environment variable | `auth.json` key | Notes |
|---|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` or `ANTHROPIC_OAUTH_TOKEN` | `anthropic` | OAuth token uses `Authorization: Bearer` |
| OpenAI | `OPENAI_API_KEY` | `openai` | |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek` | |
| Google (Gemini) | `GEMINI_API_KEY` | `google` | Use OpenAI-compat endpoint — see below |
| Groq | `GROQ_API_KEY` | `groq` | |
| Cerebras | `CEREBRAS_API_KEY` | `cerebras` | |
| Mistral | `MISTRAL_API_KEY` | `mistral` | |
| xAI | `XAI_API_KEY` | `xai` | |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter` | Meta-provider; access many models via one key |

For any provider not in this list, add it to `~/.pi-py/models.json`
(see [models.md](models.md)).

### Google Gemini

Google's native API format is not yet natively supported by pi-ai. Use the
OpenAI-compatible endpoint instead:

```json
{
  "providers": {
    "google": {
      "api": "openai-completions",
      "baseUrl": "https://generativelanguage.googleapis.com/v1beta/openai",
      "apiKey": "GEMINI_API_KEY",
      "models": [
        {
          "id": "gemini-2.5-pro",
          "name": "Gemini 2.5 Pro",
          "reasoning": true,
          "input": ["text", "image"],
          "cost": { "input": 1.25, "output": 10, "cacheRead": 0.31, "cacheWrite": 0 },
          "contextWindow": 1048576,
          "maxTokens": 65536,
          "compat": {
            "maxTokensField": "max_tokens",
            "supportsDeveloperRole": false
          }
        },
        {
          "id": "gemini-2.5-flash",
          "name": "Gemini 2.5 Flash",
          "reasoning": true,
          "input": ["text", "image"],
          "cost": { "input": 0.15, "output": 0.6, "cacheRead": 0.037, "cacheWrite": 0 },
          "contextWindow": 1048576,
          "maxTokens": 65536,
          "compat": {
            "maxTokensField": "max_tokens",
            "supportsDeveloperRole": false
          }
        }
      ]
    }
  }
}
```

---

## `~/.pi-py/auth.json`

Store credentials here instead of (or in addition to) environment variables.
Auth file entries take precedence over environment variables.

```json
{
  "anthropic": { "type": "api_key", "apiKey": "sk-ant-..." },
  "openai":    { "type": "api_key", "apiKey": "sk-..." },
  "deepseek":  { "type": "api_key", "apiKey": "sk-..." },
  "google":    { "type": "api_key", "apiKey": "..." },
  "groq":      { "type": "api_key", "apiKey": "..." }
}
```

For Anthropic OAuth tokens (from Claude Pro/Max accounts):

```json
{
  "anthropic": {
    "type": "oauth",
    "access": "sk-ant-oat01-...",
    "refresh": "sk-ant-ort01-...",
    "expires": 1778055166818
  }
}
```

OAuth tokens expire. pi-agent warns when a token is expired but continues —
refresh manually if API calls fail.

The file is read by `pi_agent.load_auth(provider)` and
`pi_agent.make_auth_provider()`.

---

## Cloud providers

Cloud providers (Azure OpenAI, Amazon Bedrock, Google Vertex AI) require more
than just an API key and are not directly supported by pi-py yet. They can be
added via `~/.pi-py/models.json` if the cloud provider exposes an
OpenAI-compatible endpoint.

See [FUTURE_ENHANCEMENTS.md](../FUTURE_ENHANCEMENTS.md) for the planned additions.

---

## Local / custom providers

Any provider that speaks one of the supported APIs can be added via
`~/.pi-py/models.json` without modifying any code. See [models.md](models.md).

Common examples: Ollama, LM Studio, vLLM, SGLang, LocalAI, LiteLLM proxy.
