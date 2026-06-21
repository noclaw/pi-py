# pi-py-sdk

Python SDK for the [Pi](https://pi.dev) coding agent. It drives `pi-agent-core` — the
well-tested TypeScript agent runtime — over Pi's **RPC mode** (`pi --mode rpc`, strict
JSONL over stdin/stdout), so the agent loop, tool calling, sessions, compaction,
retries, and provider auth all run inside Pi. No agent logic is reimplemented in Python.

> Status: **Phase 0** — the bridge core (transport, JSONL framing, command/response
> correlation, prompt streaming). See [`docs/python-sdk-plan.md`](docs/python-sdk-plan.md)
> for the full design and roadmap.

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

## Tests

```bash
pytest                 # unit tests (no Node required)
pytest -m integration  # live tests against a real `pi` (needs the binary + a key)
```
