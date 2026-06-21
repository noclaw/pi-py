# pi-py: Python SDK for Pi via RPC bridge — Implementation Plan

## Goal

Provide a Python SDK that drives `pi-agent-core` (the well-tested TypeScript agent
runtime) over Pi's **RPC mode**, instead of reimplementing the agent in Python.
This is the foundation for building a Python coding agent on top of Pi without
maintaining a port.

## Why this works (verified against source)

Pi ships a first-class, language-agnostic integration surface designed for exactly
this. Confirmed by reading the source at `earendil-works/pi`:

- `pi --mode rpc` runs a headless, long-lived Node process speaking **strict JSONL**
  over stdin/stdout (`packages/coding-agent/src/modes/rpc/rpc-mode.ts`).
- The TS reference client (`rpc-client.ts`) is a complete blueprint we mirror in Python.
- Commands, responses, events, and message types are fully typed in
  `rpc-types.ts`, `packages/agent/src/types.ts`, and `agent-session.ts`.
- Auth is resolved by Pi itself from env vars / its own auth store — the client only
  forwards environment (`packages/ai/src/env-api-keys.ts`).

We do **not** need raw `pi-ai` (LLM-only) access — the agent via RPC is sufficient.

## Architecture

```
┌─────────────────────────────┐        JSONL over stdin/stdout        ┌──────────────────────────┐
│  Python application          │  ───── commands  (id-correlated) ───► │  pi --mode rpc           │
│  (your coding agent / app)   │                                       │  (Node subprocess)       │
│                              │  ◄──── responses + events ──────────  │  pi-coding-agent         │
│   PiAgent (async API)        │                                       │   └ pi-agent-core        │
│    ├ Transport (subprocess)  │                                       │      └ pi-ai (LLM calls) │
│    ├ JSONL framing           │                                       └──────────────────────────┘
│    ├ Request/response router │
│    ├ Event bus               │
│    └ Pydantic models         │
└─────────────────────────────┘
```

Pi handles the agent loop, tool calling, sessions, compaction, retries, and provider
auth. Python handles process lifecycle, framing, correlation, typing, and ergonomics.

## Package layout

```
pi-py/
  pyproject.toml                 # distribution "pi-py-sdk", importable as `pi_py_sdk`
  src/pi_py_sdk/
    __init__.py                  # public exports: PiAgent, types, exceptions
    transport.py                 # subprocess spawn/kill, stdin write, stdout/stderr read
    jsonl.py                     # strict LF-only line framing (encode/decode)
    protocol.py                  # Pydantic models for commands, responses, events, messages
    client.py                    # PiAgent: high-level async API (mirrors TS RpcClient)
    events.py                    # event bus + typed async-iterator helpers
    errors.py                    # PiError, PiProcessError, PiTimeoutError, PiCommandError
    config.py                    # PiConfig: binary path, provider, model, cwd, env, args
    _discovery.py                # locate `pi`: PATH → npx fallback → explicit override
    sync.py                      # PiAgentSync: blocking facade over a private loop thread
  tests/
    test_jsonl.py                # framing edge cases (\r\n, U+2028 inside strings, partial)
    test_protocol.py             # (de)serialization round-trips for every message type
    test_transport_fake.py       # against a scripted fake subprocess
    test_integration.py          # against a real `pi --mode rpc` (marked, opt-in)
  examples/
    interactive.py               # port of test/rpc-example.ts
    one_shot.py                  # prompt + collect to agent_end
  docs/python-sdk-plan.md        # this file
```

## Wire protocol (as implemented in Pi)

**Launch:** `pi --mode rpc [--provider <p>] [--model <id>] [--no-session] [--session-dir <path>]`
(The reference client spawns `node <cli.js> --mode rpc ...`; we instead invoke the
installed `pi` bin, which is `dist/cli.js`.)

**Framing (`jsonl.py` — must match `rpc/jsonl.ts` exactly):**
- One JSON object per line, terminated by `\n` only.
- Split incoming bytes on `\n` only — **never** on U+2028/U+2029 (they are valid inside
  JSON strings; Node's `readline` and naive splitters are non-compliant).
- Strip a single trailing `\r` from each line before parsing.
- Maintain a UTF-8 decode buffer across chunks for partial lines; flush remainder on EOF.

**Correlation:**
- Each command may carry an optional `id`. We auto-assign `req_<n>` like the TS client.
- The matching reply is `{"type":"response","id":...,"command":...,"success":bool,...}`.
- Any stdout line that is **not** a `type:"response"` with a known pending `id` is an
  **event** (or an `extension_ui_request`) → routed to the event bus.

**Prompt lifecycle (important nuance):**
- `prompt` returns its `response` only after preflight succeeds — it does **not** mean
  the turn is done. Completion is signaled by the **`agent_end` event**.
- `agent_end` carries `willRetry: boolean` (from `AgentSessionEvent`). If `willRetry`
  is true the run is **not** actually finished — an `auto_retry_*` cycle follows.
  Our `wait_for_idle()` must treat `agent_end{willRetry:true}` as *not yet idle*
  (the TS `waitForIdle` ignores this; we improve on it).

## Data model (Pydantic v2)

Mirror the TS unions as discriminated unions on `type`. Source of truth:
`rpc-types.ts` (commands/responses), `types.ts` (AgentEvent + messages),
`agent-session.ts` (AgentSessionEvent extensions).

**Commands (stdin):** `prompt`, `steer`, `follow_up`, `abort`, `new_session`,
`get_state`, `set_model`, `cycle_model`, `get_available_models`, `set_thinking_level`,
`cycle_thinking_level`, `set_steering_mode`, `set_follow_up_mode`, `compact`,
`set_auto_compaction`, `set_auto_retry`, `abort_retry`, `bash`, `abort_bash`,
`get_session_stats`, `export_html`, `switch_session`, `fork`, `clone`,
`get_fork_messages`, `get_last_assistant_text`, `set_session_name`, `get_messages`,
`get_commands`. Plus `extension_ui_response` (`value` | `confirmed` | `cancelled`).

**Events (stdout):** lifecycle `agent_start` / `agent_end{messages,willRetry}`;
`turn_start` / `turn_end{message,toolResults}`; `message_start/update/end`
(`message_update` carries `assistantMessageEvent` with `text_delta`,
`thinking_delta`, `toolcall_start/end`); `tool_execution_start/update/end`;
`queue_update`; `compaction_start/end`; `session_info_changed`;
`thinking_level_changed`; `auto_retry_start/end`; `extension_error`;
and the `extension_ui_request` sub-protocol (`select`/`confirm`/`input`/`editor`/
`notify`/`setStatus`/`setWidget`/`setTitle`/`set_editor_text`).

**Messages:** `UserMessage`, `AssistantMessage` (text/thinking/toolCall blocks +
usage/cost/stopReason), `ToolResultMessage`, `BashExecutionMessage`.

Modeling strategy: hand-write Pydantic models for the message types and the events we
care about; use `model_config = ConfigDict(extra="allow")` so unknown/added fields and
event types degrade gracefully (forward-compat with Pi upgrades) rather than raising.
Pin a known-good Pi version and assert the `session` header `version` on startup.

## Public API (async-first)

```python
from pi_py_sdk import PiAgent, PiConfig

async with PiAgent(model="anthropic/claude-sonnet-4-20250514", cwd=".") as agent:
    # Streaming: async-iterate the events of a single prompt until agent_end
    async for ev in agent.prompt_stream("List the Python files here"):
        if ev.type == "message_update" and ev.assistant_message_event.type == "text_delta":
            print(ev.assistant_message_event.delta, end="", flush=True)

    # Or fire-and-collect
    events = await agent.prompt_and_wait("Now count them")
    print(await agent.get_last_assistant_text())
```

Surface (mirrors `RpcClient`, one method per command):
- Lifecycle: `start()`, `stop()`, `__aenter__/__aexit__`.
- Prompting: `prompt()`, `prompt_stream()`, `prompt_and_wait()`, `steer()`,
  `follow_up()`, `abort()`.
- Subscriptions: `subscribe(listener) -> unsubscribe`, `events()` async-iterator,
  `wait_for_idle(timeout)`, `collect_events(timeout)`.
- State/model/thinking/queue/compaction/retry/bash/session/messages/commands:
  thin typed wrappers over `send()`.
- Tool approvals: `on_ui_request(handler)` where handler returns the
  `extension_ui_response` value/confirmed/cancelled (with default-on-timeout, since
  Pi auto-resolves with the default if we don't reply).

Provide a small **sync facade** (`PiAgentSync`) that runs the async client on a private
event loop thread, for non-async callers. Async is the primary API because the protocol
is inherently concurrent (events interleave with responses).

## Concurrency & lifecycle design (lessons from `rpc-client.ts`)

- **Reader task:** one asyncio task reads stdout, frames lines, parses, and dispatches
  (response → resolve pending future; else → event bus). A second task drains stderr
  into a ring buffer for diagnostics (surfaced in error messages, like the TS client).
- **Request map:** `dict[str, Future]` keyed by `id`; default per-request timeout 30s
  (matches TS); on process exit, reject all pending with a `PiProcessError` containing
  captured stderr.
- **Startup:** spawn, attach readers, wait briefly, then verify the process is still
  alive (TS waits 100 ms and checks `exitCode`); fail fast with stderr if it died.
  Optionally wait for the first `session` header line as a readiness signal.
- **Shutdown:** `stop()` sends SIGTERM, waits up to 1 s, then SIGKILL (matches TS);
  closing stdin also triggers a clean Pi shutdown (`process.stdin "end"`).
- **Backpressure:** respect `stdin.write` drain; the server applies backpressure on its
  side. Keep writes awaited.
- **Abort semantics:** `abort()` is a command; SIGINT-style UX is the host app's job.

## Auth & configuration

- The client forwards `os.environ` (plus any `PiConfig.env` overrides) to the subprocess,
  exactly like the TS client (`env: {...process.env, ...options.env}`). Pi resolves
  provider credentials itself from env keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GEMINI_API_KEY`, OAuth tokens, Vertex ADC, Bedrock profiles, …) or its own auth
  store under `~/.pi`. **No key handling lives in Python.**
- `PiConfig` fields: `bin` (path or "pi"/"npx pi"), `provider`, `model`, `cwd`,
  `env`, `session_dir`, `no_session`, `extra_args`.

## Distribution / the Node dependency

Pi is a Node CLI (`bin: pi -> dist/cli.js`). Options, in recommended order:
1. **Require an installed `pi`** (`npm i -g @earendil-works/pi-coding-agent`), discover
   it on `PATH`; document the prereq and validate version at startup. Lowest complexity.
2. **`npx` fallback:** invoke `npx --yes @earendil-works/pi-coding-agent@<pinned>` when no
   binary is found. Zero global install, slower cold start.
3. **Managed install:** a `pi_sdk install-runtime` helper that pins+installs into a
   cache dir. Most control, most maintenance.

Start with (1)+(2). Pin a tested Pi version range and assert it via `get_state`/the
session header.

## Testing strategy

- **Unit (no Node):** JSONL framing (CRLF, U+2028 inside strings, split-across-chunks,
  EOF remainder); Pydantic round-trips for every command/response/event/message;
  request/response correlation and timeout via a scripted fake transport.
- **Integration (opt-in marker, needs `pi` + a key):** real `pi --mode rpc` —
  prompt→`agent_end`, tool execution events, `bash` semantics (result stored, surfaced
  on next prompt), steer/follow-up, abort, model listing, compaction, session
  fork/clone. Port `test/rpc-example.ts` as a smoke example.
- Borrow scenarios directly from Pi's own `test/rpc*.test.ts` for parity.

## Phased roadmap

- **Phase 0 — Spike:** transport + jsonl + minimal `prompt`/`agent_end`; stream a reply
  end-to-end against real `pi`. Proves the bridge.
- **Phase 1 — Core SDK:** full command set, full event models, event bus, wait/collect,
  errors, lifecycle hardening, sync facade. The shippable SDK.
- **Phase 2 — Approvals & richness:** `extension_ui_request` handler, queue/steer/
  follow-up modes, compaction/retry events, session fork/clone/switch, stats.
- **Phase 3 — Python coding-agent layer:** build the higher-level app on the SDK —
  REPL/TUI, config, custom prompt/skills wiring via Pi commands (`get_commands`).
- **Phase 4 — Packaging/CI:** binary discovery + npx fallback, version pinning,
  integration CI with a provider key, published `pi-sdk` wheel.

## Confirmed decisions

1. **API style:** async-first (`asyncio`) primary, with a thin `PiAgentSync` facade.
2. **Runtime sourcing:** discover installed `pi` on `PATH`; fall back to
   `npx --yes @earendil-works/pi-coding-agent@<pinned>`.
3. **Package name:** distribution `pi-py-sdk`, import `pi_py_sdk`.
4. **Modeling layer:** Pydantic v2 (discriminated unions on `type`, `extra="allow"`
   for forward-compat with Pi upgrades).

A managed-install command (`pi_py_sdk install-runtime`) may be added later as an
opt-in convenience, but is not part of the default resolution path.
```
