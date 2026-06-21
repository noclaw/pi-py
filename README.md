# pi-py-sdk

[![CI](https://github.com/noclaw/pi-py/actions/workflows/ci.yml/badge.svg)](https://github.com/noclaw/pi-py/actions/workflows/ci.yml)

Python SDK for the [Pi](https://pi.dev) coding agent. It drives `pi-agent-core` — the
well-tested TypeScript agent runtime — over Pi's **RPC mode** (`pi --mode rpc`, strict
JSONL over stdin/stdout), so the agent loop, tool calling, sessions, compaction,
retries, and provider auth all run inside Pi. No agent logic is reimplemented in Python.

> Status: the bridge core (transport, JSONL framing, command/response correlation,
> prompt streaming), the full RPC command surface, the richer event model, the
> interactive **extension-UI sub-protocol** (tool approvals/dialogs), typed message-block
> models, and a synchronous facade (`PiAgentSync`). A terminal coding agent (`pi-py`)
> ships on top. See [`docs/python-sdk-plan.md`](docs/python-sdk-plan.md) for the design.

## Install (dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

You also need the Pi runtime for live use:

```bash
npm i -g @earendil-works/pi-coding-agent   # provides the `pi` binary
export ANTHROPIC_API_KEY=...               # or another supported provider key
```

If `pi` isn't on `PATH`, the SDK falls back to `npx --yes @earendil-works/pi-coding-agent@<pinned>`.

## Usage

```python
import asyncio
from pi_py_sdk import PiAgent, MessageUpdateEvent

async def main():
    async with PiAgent(model="anthropic/claude-sonnet-4-20250514", cwd=".") as agent:
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

with PiAgentSync(model="anthropic/claude-sonnet-4-20250514") as agent:
    for event in agent.prompt_stream("hello"):
        ...
    for msg in agent.get_messages():        # typed messages
        print(msg.role, message_text(msg))
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

## The `pi-py` coding agent

The repo also ships `pi_py_agent`, a small terminal coding agent built entirely on the
SDK (the agent loop, tools, and model calls all run inside Pi). Installing the package
provides a `pi-py` command:

```bash
pi-py                                   # interactive REPL
pi-py --print "Run the tests and summarize failures"   # one-shot
pi-py --model anthropic/claude-sonnet-4-20250514 --no-session
```

It streams assistant text, thinking, and tool activity to the terminal, answers
approval dialogs interactively, and supports `/help`, `/model`, `/models`, `/new`,
`/state`, and `/exit`. Ctrl-C aborts the current turn; Ctrl-D exits.

## Tests

```bash
pytest                 # unit tests (no Node required); integration is deselected by default
pytest -m integration  # live tests against a real `pi` (needs the binary on PATH)
```

The integration tests avoid LLM calls (state, models, bash), so they don't need a
provider key. The one prompt-completion test additionally needs a working model and is
skipped unless `PI_LIVE_LLM=1` is set.

## Releasing

CI (`.github/workflows/ci.yml`) runs the unit suite across Python 3.10–3.13, builds the
wheel, and best-effort-smokes a real `pi` on every push/PR. Publishing
(`.github/workflows/publish.yml`) builds and uploads to PyPI when a GitHub Release is
published — it uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC, no token secret), which must be configured once for the repo.
