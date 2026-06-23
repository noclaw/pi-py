# CLAUDE.md

Guidance for working in this repository.

## What this is

`pi-py-sdk` is a Python SDK that drives **Pi**'s agent runtime (`pi-agent-core`, written
in TypeScript) over Pi's **RPC mode** (`pi --mode rpc`, strict JSONL over stdin/stdout),
plus a terminal coding agent (`pi-py`) built on that SDK.

**Core principle: no agent logic is reimplemented in Python.** The agent loop, tool
calling, sessions, compaction, retries, and provider auth all run inside the `pi`
subprocess. This package is a transport + protocol + ergonomics layer over that
subprocess. When extending it, prefer exposing an existing Pi RPC command/event over
adding behavior here.

**Two clients, two levels:**
- `PiAgent` (`client.py`) drives the **full agent** over `pi --mode rpc` — loop, tools,
  sessions, compaction. Use it to run Pi as-is from Python.
- `PiModelClient` (`model.py`) exposes just the **raw model layer**: it spawns
  `_shim/stream.mjs`, a small Node bridge to `@earendil-works/pi-ai`'s `streamSimple`.
  This is the seam for building a *native-Python* agent loop (the loop and tools live in
  the consumer, e.g. the `py-coding-agent` project) while still delegating providers,
  auth, transports, and local models to pi-ai. It still reimplements no agent logic —
  only the LLM call crosses the boundary.

## Layout

```
src/pi_py_sdk/        # the SDK (the bridge)
  jsonl.py            # strict LF-only JSONL framing (mirrors Pi's jsonl.ts)
  transport.py        # async subprocess lifecycle, stdout framing, stderr ring buffer
  protocol.py         # Pydantic models: commands, responses, events, messages, StreamEvent
  client.py           # PiAgent — full-agent async API over `pi --mode rpc`
  model.py            # PiModelClient — raw pi-ai model streaming over the Node shim
  _shim/stream.mjs    # Node bridge: pi-ai streamSimple <-> JSONL (used by PiModelClient)
  sync.py             # PiAgentSync / PiModelClientSync — blocking facades
  config.py           # PiConfig (CLI args + env)
  _discovery.py       # resolve `pi` (PATH -> npx); resolve node + pi-ai dir for the shim
  errors.py           # PiError hierarchy (incl. PiModelError)
src/pi_py_agent/      # the terminal coding agent (consumes the SDK only)
  render.py           # stream events -> terminal (text/thinking/tool/queue)
  app.py              # async REPL + one-shot runner, mid-turn steering, approvals
  cli.py              # `pi-py` console entry point
tests/                # pytest; unit tests need no Node (fake transports)
docs/python-sdk-plan.md   # full design + roadmap
```

## Non-obvious invariants — read before editing the protocol

- **JSONL framing is LF-only.** Split stdout on `\n` only; never on U+2028/U+2029 (valid
  inside JSON strings). Strip a single trailing `\r`. See `jsonl.py`; this must stay
  byte-compatible with Pi's `packages/coding-agent/src/modes/rpc/jsonl.ts`.
- **A `prompt` response means preflight succeeded, not that the turn finished.**
  Completion is the `agent_end` event — and only when `agent_end.willRetry == False`.
  `willRetry == True` is followed by an `auto_retry_*` cycle and another `agent_end`.
  `prompt_stream`/`wait_for_idle` depend on this.
- **Wire field names are camelCase** (`assistantMessageEvent`, `toolCallId`, `willRetry`,
  `modelId`). Models keep those names verbatim. All models use `extra="allow"` so new Pi
  fields/events degrade gracefully — don't tighten this.
- **Auth lives in Pi, not here.** The transport forwards `os.environ`; Pi resolves all
  provider keys/OAuth. Never add key handling in Python.
- **`bash` command vs the bash tool:** the `bash` RPC command stores a
  BashExecutionMessage that is only sent to the LLM on the *next* prompt.
- **`extension_ui_response` is not id-correlated** — it's written directly to stdin, not
  via the request/response (`_send`) path.
- **The model shim imports pi-ai by file path, not bare specifier.** pi-ai's
  `package.json` declares an *import-only* `exports` map, so `require.resolve(...)` /
  bare `import "@earendil-works/pi-ai"` fail. The shim reads pi-ai's `package.json`
  `exports` and imports the resolved `dist/*.js` via `pathToFileURL`. `_discovery.py`
  finds the pi-ai *directory* (bundled under the global `pi` install, or `PI_AI_DIR`).
- **Shim credential order:** caller `apiKey` > provider env var (pi-ai's own
  `getEnvApiKey`) > the coding agent's OAuth login in `~/.pi/agent/auth.json` (refreshed
  via pi-ai's `oauth` module and written back). OAuth tokens authenticate when passed as
  `options.apiKey` (anthropic detects the `sk-ant-oat` prefix). Still "auth lives in Pi."
- **Shim stream lifecycle:** each `stream` request runs concurrently keyed by `id`;
  closing/cancelling a `PiModelClient.stream()` sends `{type:"abort", id}`. pi-ai
  terminates every stream with a `done`/`error` event carrying the final message, so the
  Python side never accumulates deltas. A *thrown* shim error (bad model id) comes back
  as a top-level `stream_error` line and raises `PiModelError`; a model-produced `error`
  *event* is delivered in-band as a terminal `StreamEvent`.

## Dev commands

```bash
pip install -e ".[dev]"
pytest                       # unit tests (no Node; integration deselected via addopts)
pytest -m integration        # live tests; needs `pi` on PATH (no key needed)
PI_LIVE_LLM=1 pytest -m integration   # also runs the prompt-completion test (needs a model)
python -m build              # build wheel + sdist
```

There is no linter/formatter configured. Match the surrounding style: 4-space indent,
type hints, `from __future__ import annotations`, Google-ish docstrings.

## Testing approach

Unit tests inject a **fake transport** (see `tests/test_client_fake.py`,
`test_ui_and_events.py`, `test_commands.py`) or a fake agent (`test_sync.py`) so nothing
spawns Node. Pure helpers (framing, event/message parsing, REPL routing, rendering) are
tested directly. Live behavior goes in `tests/test_integration.py` behind the
`integration` marker (skipped automatically when `pi` is absent).

## Release

Version lives in `pyproject.toml` and `src/pi_py_sdk/__init__.py` (keep them in sync).
CI (`.github/workflows/ci.yml`) runs the unit matrix (3.10–3.13), builds the wheel, and
best-effort-smokes a real `pi`. Publishing to PyPI happens on a GitHub Release via
`.github/workflows/publish.yml` using Trusted Publishing (OIDC, no token). To cut a
release: bump the version, then `gh release create vX.Y.Z --generate-notes`.

## Git

Work on a feature branch and fast-forward into `main`; `main` tracks
`origin` (github.com/noclaw/pi-py). End commit messages with the
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
