# pi-py-sdk

[![CI](https://github.com/noclaw/pi-py/actions/workflows/ci.yml/badge.svg)](https://github.com/noclaw/pi-py/actions/workflows/ci.yml)

Python SDK for the [Pi](https://pi.dev) coding agent. It drives `pi-agent-core` — the
well-tested TypeScript agent runtime — over Pi's **RPC mode** (`pi --mode rpc`, strict
JSONL over stdin/stdout), so the agent loop, tool calling, sessions, compaction,
retries, and provider auth all run inside Pi. No agent logic is reimplemented in Python.

It includes the bridge core (transport, strict JSONL framing, id-correlated commands,
streaming), the full RPC command surface, typed events and message models, the
interactive **extension-UI sub-protocol** (tool approvals/dialogs), and a synchronous
facade (`PiAgentSync`). A terminal coding agent (`pi-py`) ships on top. See
[`docs/python-sdk-plan.md`](docs/python-sdk-plan.md) for the design.

There are **two clients, at two levels**:

- **`PiAgent`** drives the **full Pi agent** over `pi --mode rpc` — loop, tools,
  sessions, compaction. Use it to run Pi as-is from Python.
- **`PiModelClient`** exposes just the **raw model layer**: it streams a single
  assistant response (text, thinking, tool calls) from [`@earendil-works/pi-ai`](https://pi.dev),
  with no agent loop or tools. This is the seam for building your *own* agent loop in
  Python while still delegating providers, auth, transports, and local models to pi-ai.
  Neither client reimplements agent logic — with `PiModelClient`, only the LLM call
  crosses the boundary.

## Install

```bash
pip install pi-py-sdk
```

This installs the `pi_py_sdk` library and the `pi-py` agent CLI. You also need the Pi
runtime for live use:

```bash
npm i -g @earendil-works/pi-coding-agent   # provides the `pi` binary
export ANTHROPIC_API_KEY=...               # or another supported provider key
```

If `pi` isn't on `PATH`, `PiAgent` falls back to `npx --yes @earendil-works/pi-coding-agent@<pinned>`.

`PiModelClient` additionally needs **Node** on `PATH` and resolves the bundled
`@earendil-works/pi-ai` package from the global `pi` install (or set `PI_AI_DIR`). It can
authenticate from a provider env var **or** from an existing Pi OAuth login
(`~/.pi/agent/auth.json`, e.g. after `/login` in `pi`).

### Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Usage

```python
import asyncio
from pi_py_sdk import PiAgent, MessageUpdateEvent

async def main():
    async with PiAgent(model="anthropic/claude-sonnet-4-6", cwd=".") as agent:
        async for ev in agent.prompt_stream("List the Python files here"):
            if isinstance(ev, MessageUpdateEvent) and ev.assistantMessageEvent:
                ame = ev.assistantMessageEvent
                if ame.type == "text_delta" and ame.delta:
                    print(ame.delta, end="", flush=True)

asyncio.run(main())
```

A prompt completes on an `agent_end` event with `willRetry == False` (an `agent_end`
with `willRetry == True` is followed by an automatic retry).

### Synchronous use

For non-async code, `PiAgentSync` runs the agent on a background loop and blocks:

```python
from pi_py_sdk import PiAgentSync, message_text

with PiAgentSync(model="anthropic/claude-sonnet-4-6") as agent:
    for event in agent.prompt_stream("hello"):
        ...
    for msg in agent.get_messages():        # typed messages
        print(msg.role, message_text(msg))
```

### Model streaming (low-level)

`PiModelClient` streams a single assistant response straight from pi-ai — no agent loop,
no tools running inside Pi. You provide the context (system prompt + messages + tool
definitions) and own the turn structure; pi-ai handles the provider call. This is the
foundation for building a native-Python agent loop.

```python
import asyncio
from pi_py_sdk import PiModelClient

async def main():
    async with PiModelClient() as client:
        async for ev in client.stream(
            provider="anthropic",
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Say hello", "timestamp": 0}],
            reasoning="low",                # optional thinking level
        ):
            if ev.type == "text_delta":
                print(ev.delta, end="", flush=True)

asyncio.run(main())
```

Every stream ends with a terminal event (`ev.is_terminal`): `done` carries the final
`AssistantMessage` on `ev.final_message`, `error` carries a failed message (rejected
auth, content filtering). A shim-level failure (e.g. unknown model id) raises
`PiModelError`. Tool calls surface as `toolcall_end` events with a parsed `ev.toolCall`.
Other methods: `complete()` (drain to the final message), `list_models()`,
`list_providers()`, and a blocking `PiModelClientSync` facade.

```python
# Pass tool definitions and let the model decide to call one:
tools = [{"name": "get_weather", "description": "Current weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}]
async for ev in client.stream(provider="anthropic", model="claude-sonnet-4-6",
                              messages=messages, tools=tools):
    if ev.type == "toolcall_end":
        print(ev.toolCall.name, ev.toolCall.arguments)
```

### Tool approvals

Extensions request decisions (allow this tool? pick an option? enter a value?) via the
extension-UI sub-protocol. Install a handler with `on_ui_request`; without one, the SDK
safely denies confirmations and cancels other dialogs so the agent never hangs.

```python
def approve(req):
    if req.method == "confirm":
        return True                 # allow
    if req.method == "select":
        return (req.options or [None])[0]
    return None                     # cancel input/editor

agent.on_ui_request(approve)        # see examples/with_approvals.py
```

The full command surface (`set_model`, `bash`, `compact`, `fork`, `get_session_stats`,
steering/follow-up modes, …) is available as async methods on `PiAgent`.

## Running the examples

The [`examples/`](examples/) directory has runnable scripts. Make sure `pi` is on
`PATH` (or available via `npx`) and a provider key is exported first:

```bash
export ANTHROPIC_API_KEY=...   # or another supported provider key
```

Each script takes the prompt as a command-line argument (and falls back to a default
if you omit it):

```bash
python examples/one_shot.py "List the Python files in this directory"
python examples/sync_usage.py "Say hello in one short sentence."
python examples/with_approvals.py "Refactor foo.py and run the tests"
python examples/model_stream.py "Say hello in one short sentence."
```

- **`one_shot.py`** — stream a single prompt's text/thinking/tool events to the
  terminal, with error surfacing (preflight failures, run errors, retries).
- **`sync_usage.py`** — the same, using the blocking `PiAgentSync` facade, then prints
  the typed message history.
- **`with_approvals.py`** — installs an interactive console handler so you can approve
  tool calls and answer dialogs.
- **`model_stream.py`** — the low-level `PiModelClient`: stream a raw model response
  (no agent loop or tools), the building block for a custom Python agent loop.

The examples target `anthropic/claude-sonnet-4-6`; edit the `model=` argument to use a
different model or provider. If a prompt returns blank output, it's usually an
unavailable model id or a missing/invalid provider key — `one_shot.py` will print a
hint in that case.

## The `pi-py` coding agent

The repo also ships `pi_py_agent`, a small terminal coding agent built entirely on the
SDK (the agent loop, tools, and model calls all run inside Pi). Installing the package
provides a `pi-py` command:

```bash
pi-py                                   # interactive REPL
pi-py --print "Run the tests and summarize failures"   # one-shot
pi-py --model anthropic/claude-sonnet-4-6 --no-session
```

It streams assistant text, thinking, and tool activity (with result previews) to the
terminal, answers approval dialogs interactively, and supports slash commands (`/help`,
`/model`, `/models`, `/new`, `/state`, `/compact`, `/clone`, `/fork`, `/exit`). While
the agent is responding you can **steer** it by typing (or `+text` to queue a
follow-up). Ctrl-C aborts the current turn; Ctrl-D exits.

## Tests

```bash
pytest                 # unit tests (no Node required); integration is deselected by default
pytest -m integration  # live tests against a real `pi` (needs the binary on PATH)
```

Most integration tests avoid LLM calls (state, models, bash) and don't need a provider
key; `PiModelClient`'s also need `node` on `PATH`. The live model-call tests
(prompt completion, `PiModelClient.stream`) need a working model/credentials and are
skipped unless `PI_LIVE_LLM=1` is set.

## Releasing

CI (`.github/workflows/ci.yml`) runs the unit suite across Python 3.10–3.13, builds the
wheel, and best-effort-smokes a real `pi` on every push/PR. Publishing
(`.github/workflows/publish.yml`) builds and uploads to PyPI when a GitHub Release is
published — it uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC, no token secret), which must be configured once for the repo.
