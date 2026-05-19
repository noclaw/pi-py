# pi-py

Python port of the [pi TypeScript monorepo](https://github.com/earendil-works/pi).  
Source: https://github.com/noclaw/pi-py

## Packages

| Package | PyPI name | Description |
|---|---|---|
| [`packages/ai`](packages/ai/README.md) | `pi-ai` | Unified LLM API — streaming, tool calling, thinking, image generation |
| [`packages/agent`](packages/agent/README.md) | `pi-agent` | Stateful agent loop, session persistence, built-in coding tools, settings |

## Installation

```bash
pip install pi-ai pi-agent
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pi-ai pi-agent
```

## Quick start

```python
import asyncio, pi_agent

async def main():
    # Resolves model from ~/.pi/agent/settings.json automatically
    harness = await pi_agent.create_agent(cwd=".")
    reply = await harness.prompt("What files are in this directory?")
    print(reply.content[0].text)

asyncio.run(main())
```

See [`packages/agent/README.md`](packages/agent/README.md) for the full API and
[`packages/ai/README.md`](packages/ai/README.md) for the low-level LLM interface.

---

## CLI (`pi-py`)

`pi-agent` ships a command-line tool named **`pi-py`** (distinct from the
TypeScript `pi` CLI).

### Setup

The entry point lives in the workspace's `.venv`.  Add an alias so it works from
any directory:

```bash
echo 'alias pi-py="uv run --project /path/to/pi-py pi-py"' >> ~/.zshrc
source ~/.zshrc
```

Or use `uv run pi-py ...` directly from within the repo.

### Commands

```
pi-py prompt TEXT [OPTIONS]    Run a prompt and stream the response
pi-py sessions list            List saved sessions, newest first
pi-py sessions show ID         Print the transcript for a session
pi-py models list              List built-in and custom models
```

Global option: `--settings-dir PATH` (default: `~/.pi/agent`).

### `prompt`

```bash
# Uses model from ~/.pi/agent/settings.json
pi-py prompt "What files are in this directory?"

# Explicit model
pi-py prompt -m anthropic:claude-haiku-4-5 "Explain this codebase"

# No tools — pure LLM call
pi-py prompt --no-tools "What is the capital of France?"

# Save session to disk
pi-py prompt --sessions-dir ~/.pi/sessions "Start a task"

# Resume a saved session
pi-py prompt --sessions-dir ~/.pi/sessions --session abc12345 "Continue"

# Structured JSON output (one AgentEvent per line, for subprocess use)
pi-py prompt --json "Write hello.py" | python3 -m json.tool
```

| Flag | Description |
|---|---|
| `-m`, `--model PROVIDER:ID` | Model override, e.g. `anthropic:claude-sonnet-4-6` |
| `-s`, `--session ID` | Resume existing session by ID |
| `--sessions-dir PATH` | Session storage root (default: `~/.pi/sessions`) |
| `--system TEXT` | Replace the auto-generated system prompt |
| `--cwd PATH` | Working directory for file/shell tools (default: current dir) |
| `--no-tools` | Disable built-in file and shell tools |
| `--json` | Emit newline-delimited JSON events instead of streaming text |

**Text output** streams the assistant response with cyan annotations for tool
calls and a cost/token summary line at the end:

```
  ⚙ $ python hello.py
    hello world
The script ran successfully and printed "hello world".

[claude-haiku-4-5 · 245 in / 38 out · $0.000042]
```

**JSON output** (`--json`) emits one `AgentEvent` object per line, ending with
`{"type":"agent_end","messages":[...]}`.  This is a direct replacement for
`claude --output-format stream-json` in subprocess-based integrations.

### `sessions`

```bash
pi-py sessions list
# ID        CREATED               CWD
# ──────────────────────────────────────────────────────────────────────
# abc12345  2025-05-19 10:30      ~/code/my-project
# def67890  2025-05-18 09:15      ~/code/other-project

pi-py sessions show abc12345
# Session: abc12345abc12345
# Created: 2025-05-19 10:30 PDT
# CWD:     /Users/jeff/code/my-project
#
# user: What does this codebase do?
# assistant: This is a FastAPI service that ...
#   [called: read, bash]
```

### `models`

```bash
pi-py models list
# Built-in models:
#   PROVIDER      MODEL ID                           CTX  REASONING
#   ──────────────────────────────────────────────────────────────────
#   anthropic     claude-sonnet-4-6                   1M  ✓
#   openai        gpt-4o                            128K
#   ...
#
# Custom models (models.json):
#   omlx          Qwen3.6-27B-bf16                  32K   [default]
#
# Default: omlx:Qwen3.6-27B-bf16
```

---

## Settings files

`pi-agent` reads three optional JSON files from `~/.pi/agent/` (or a custom
directory passed as `settings_dir`).

### `~/.pi/agent/settings.json`

Global defaults. A project-level `.pi/settings.json` anywhere on the path from
the working directory up to the filesystem root overrides these values.

```json
{
  "defaultProvider": "anthropic",
  "defaultModel": "claude-sonnet-4-6"
}
```

| Field | Type | Description |
|---|---|---|
| `defaultProvider` | string | Provider name used when `model=None` in `create_agent()` |
| `defaultModel` | string | Model ID within that provider |

### `~/.pi/agent/models.json`

Custom providers and models not in the built-in catalog (local servers, private
endpoints, custom OpenAI-compatible APIs).

```json
{
  "providers": {
    "my-local": {
      "baseUrl": "http://127.0.0.1:8008/v1",
      "api": "openai-completions",
      "apiKey": "secret",
      "authHeader": true,
      "models": [
        {
          "id": "my-model",
          "name": "My Local Model",
          "reasoning": false,
          "input": ["text", "image"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 32768,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `baseUrl` | string | Base URL for the provider API |
| `api` | string | API type — currently `"openai-completions"` |
| `apiKey` | string | API key for this provider |
| `authHeader` | bool | `true` → send key as `Authorization: Bearer`; `false` → provider default |
| `models[].contextWindow` | int | Context window size in tokens |
| `models[].maxTokens` | int | Maximum output tokens |

Loaded models are returned by `pi_agent.load_custom_models()` and automatically
used by `pi_agent.get_default_model()` and `pi_agent.create_agent()`.

### `~/.pi/agent/auth.json`

Per-provider authentication credentials. Supports plain API keys and OAuth tokens.

**API key:**
```json
{
  "my-provider": {
    "type": "api_key",
    "apiKey": "sk-..."
  }
}
```

**OAuth (e.g. Anthropic):**
```json
{
  "anthropic": {
    "type": "oauth",
    "refresh": "sk-ant-ort01-...",
    "access": "sk-ant-oat01-...",
    "expires": 1778055166818
  }
}
```

| Field | Type | Description |
|---|---|---|
| `type` | string | `"api_key"` or `"oauth"` |
| `apiKey` | string | For `api_key` type — the key to use |
| `access` | string | For `oauth` type — the access token |
| `refresh` | string | For `oauth` type — the refresh token (not auto-refreshed by pi-agent) |
| `expires` | int | For `oauth` type — expiry as milliseconds since Unix epoch |

`pi_agent.load_auth(provider)` reads this file and returns `{"apiKey": ..., "headers": {...}}`.
OAuth entries include `Authorization: Bearer` in the headers. Expired tokens trigger a
warning but are returned anyway — refresh them manually if calls fail.

`pi_agent.make_auth_provider()` wraps `load_auth` into a callable suitable for
passing as `get_api_key_and_headers` to `AgentHarness` or `create_agent()`. When
no entry exists for a provider, `pi_ai` falls back to environment variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.).

---

## Running the live tests

`test_live.py` in the repository root exercises both packages end-to-end against
real provider APIs.

```bash
# Install workspace deps
uv sync --all-packages

# Run with environment variables
ANTHROPIC_API_KEY=sk-ant-... uv run python test_live.py

# Or export keys first
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-...
uv run python test_live.py
```

Keys already in `~/.pi/agent/auth.json` are picked up automatically by the
`create_agent` test; other tests rely on environment variables.

### What gets tested

| Test | Provider | What it covers |
|---|---|---|
| `test_text` | Anthropic, OpenAI | Streaming text, token counts, cost |
| `test_tools` | Anthropic, OpenAI | Tool calling, streaming tool args |
| `test_thinking` | Anthropic | Extended thinking / reasoning |
| `test_abort` | Anthropic | Abort signal mid-stream |
| `test_image_generation` | OpenRouter | Image generation |
| `test_agent_streaming_events` | Anthropic, OpenAI | Agent lifecycle event sequence |
| `test_agent_tool_loop` | Anthropic, OpenAI | Multi-turn tool execution loop |
| `test_agent_abort` | Anthropic | Abort an in-flight agent run |
| `test_agent_follow_up` | Anthropic | Follow-up queue resumes after stop |
| `test_agent_harness_session` | Anthropic | Session persistence across turns |
| `test_create_agent` | Anthropic, OpenAI | `create_agent()` with write + bash tools |

Tests for each provider are skipped automatically when the corresponding key is absent.

## Development

```bash
# Clone and install
git clone https://github.com/noclaw/pi-py.git
cd pi-py
uv sync --all-packages

# Run tests
uv run python test_live.py

# Package-specific READMEs
open packages/ai/README.md
open packages/agent/README.md
```
