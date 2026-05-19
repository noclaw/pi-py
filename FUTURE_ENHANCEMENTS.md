# Future Enhancements

Roadmap for `pi-py`.  Phases 1–4 are complete and removed from this document.

The guiding philosophy: **pi-py is not a port of the full coding-agent application**.
It is a Python library that makes LLM agents understandable and modifiable — so that
developers can build *their own* personal agents rather than configure someone else's.
The TypeScript `pi` / `pi coding-agent` CLI already exists and works well.  pi-py
targets Python-native use cases: scripting, web services, integration into existing
Python codebases, and agents that reason about Python-specific tools.

Reference use case: migrating **noclaw** (`/Users/jeff/code/noclaw`) — a personal
assistant that currently shells out to the `claude` CLI — to use pi-py directly.
That migration drives several of the priorities below.

---

## Phase 5 — CLI (`pi-py`) — next up

A `click`-based command named **`pi-py`** (not `pi`, which conflicts with the
TypeScript CLI already installed).

### Subcommands

```
pi-py prompt "..."            # single-shot: run once and exit (print mode)
pi-py prompt --session <id>   # resume an existing session by ID
pi-py sessions list           # list saved sessions
pi-py sessions show <id>      # show transcript for a session
pi-py models list             # list built-in + custom models
```

### Key flags

| Flag | Description |
|---|---|
| `--model provider:id` | Override default model (e.g. `anthropic:claude-sonnet-4-6`) |
| `--session <id>` | Resume an existing JSONL session by ID |
| `--system "..."` | Append to or replace the system prompt |
| `--cwd <path>` | Working directory (default: current dir) |
| `--sessions-dir <path>` | Where to store sessions (default: `~/.pi/sessions`) |
| `--settings-dir <path>` | Config dir (default: `~/.pi/agent`) |
| `--no-tools` | Run without file/bash tools (reasoning-only) |
| `--json` | Emit newline-delimited JSON events instead of text (for subprocess use) |

### Print mode output

Without `--json`: stream assistant text to stdout, print token/cost summary to
stderr on completion.

With `--json`: emit one JSON object per line matching the `AgentEvent` dict
structure, ending with `{"type":"agent_end", ...}`.  This allows noclaw-style
subprocess integration with structured output — replacing the `claude
--output-format stream-json` pattern without the Claude Code dependency.

### Implementation sketch

```python
# cli.py
import asyncio, click, pi_agent, pi_ai

@click.command()
@click.argument("prompt_text")
@click.option("--model", default=None)
@click.option("--session", default=None)
@click.option("--json", "json_output", is_flag=True)
def prompt(prompt_text, model, session, json_output):
    asyncio.run(_run(prompt_text, model, session, json_output))

async def _run(text, model_str, session_id, json_output):
    harness = await pi_agent.create_agent(
        model=pi_ai.get_model(*model_str.split(":")) if model_str else None,
        session_id=session_id,   # Phase 6
    )
    ...
```

Entry point in `pyproject.toml`:
```toml
[project.scripts]
pi-py = "pi_agent.cli:main"
```

---

## Phase 6 — Session resume in `create_agent()`

`create_agent()` currently always creates a new session.  Adding `session_id`
allows resuming an existing JSONL session — the primary enabler for the noclaw
migration and for stateful CLI use.

```python
harness = await pi_agent.create_agent(
    session_id="abc123",          # open existing session from sessions_dir
    sessions_dir="~/.pi/sessions",
)
```

Implementation:
- Add `session_id: str | None` and `sessions_dir: str | None` to `create_agent()`
- When `session_id` is set: list sessions from `JsonlSessionRepo`, find the one
  with matching ID, open it via `repo.open(metadata)`
- When `session_id` is set but not found: raise `ValueError` with a helpful message
  listing available session IDs

This pairs with `pi-py --session <id>` in Phase 5.

---

## Phase 7 — `noclaw` migration guide

noclaw (`/Users/jeff/code/noclaw`) dispatches tasks to the `claude` CLI as a
subprocess.  pi-py already provides everything needed to replace that pattern.

**What noclaw needs and where it exists in pi-py:**

| noclaw requirement | pi-py equivalent |
|---|---|
| Run agent on a prompt | `AgentHarness.prompt()` |
| Stream structured output | `harness.subscribe()` events |
| Resume session by ID | Phase 6 |
| Append system prompt | `create_agent(system_prompt=...)` |
| Inject memory/vault context | `load_context_files()` or `system_prompt=` callable |
| Progress file updates | `harness.on("turn_end", ...)` hook |
| Token counts + cost | `reply.usage.total_tokens`, `reply.usage.cost.total` |
| Model selection per request | `create_agent(model=pi_ai.get_model(...))` |
| Session timeout | `bash` tool `timeout` param; `AgentLoopConfig` abort signal |
| `--json` subprocess output | Phase 5 `pi-py prompt --json` |

A concrete migration would replace `cli_session.py`'s subprocess launch with a
direct `create_agent()` call and a `subscribe()` listener that writes the same
`.progress/{agent_id}.log` format.  `sdk_session.py` could be replaced entirely.

Capturing structured results:

```python
harness = await pi_agent.create_agent(model=model, session_id=session_id)
reply = await harness.prompt(task_text)

result = {
    "status": "SUCCESS" if reply.stop_reason == "stop" else "ERROR",
    "output": " ".join(b.text for b in reply.content if hasattr(b, "text")),
    "tokens": reply.usage.total_tokens,
    "cost": reply.usage.cost.total,
}
```

---

## Phase 8 — Personal agent examples

Short, self-contained scripts that demonstrate common personal-agent patterns.
Placed in `examples/` at the repo root.

| Example | What it shows |
|---|---|
| `examples/ask.py` | One-shot `create_agent()` prompt, print response |
| `examples/chat.py` | Interactive REPL loop with session persistence |
| `examples/codebase_qa.py` | Load a repo's CLAUDE.md, answer questions about it |
| `examples/journal.py` | Append a daily note to an Obsidian vault via file tools |
| `examples/webhook_agent.py` | FastAPI endpoint → `create_agent()` → JSON response |

The webhook example directly targets the noclaw pattern: replace the subprocess
with a library call, keep the same HTTP interface.

---

## Longer-term ideas (no phase assigned)

### TUI layer (`pi-tui-py`)
A separate package inspired by `packages/tui` in the TypeScript repo.  The TS
`pi-tui` uses differential rendering, synchronized output (CSI 2026), and an
overlay system that Python's `curses`/`blessed`/`rich` don't match.  Could be
built on top of `rich` or `textual` as a starting point.  Not needed for the
core library use case but useful for interactive CLI agents.

### Custom provider registration
Users can already construct `Model(api="openai-completions", base_url=...)` for
local servers.  A higher-level `register_provider()` function that reads a
`models.json`-style dict and installs models into the in-process registry would
remove the need to build `Model` objects manually.

```python
pi_agent.register_providers_from_settings()  # reads ~/.pi/agent/models.json
model = pi_ai.get_model("my-local", "my-model")  # now available
```

### OAuth token refresh
`auth.json` OAuth tokens expire.  A `refresh_oauth_token(provider)` helper that
reads the `refresh` field and calls the provider's token endpoint would remove the
manual refresh step.  Currently the library warns on expiry and continues.

### Agent-to-agent calls
A tool that lets one agent spawn a sub-agent with a separate session and model,
collecting its result as a tool result.  Useful for parallelising work or
delegating to a specialised agent (e.g. a "critic" agent reviewing a "writer"
agent's output).

```python
sub_agent_tool = create_sub_agent_tool(
    name="review",
    model=pi_ai.get_model("anthropic", "claude-opus-4-7"),
    system_prompt="You are a code reviewer.",
)
```

### Streaming to web clients
A helper that bridges `AgentEventStream` to a Server-Sent Events or WebSocket
response — so a FastAPI/Flask endpoint can stream agent progress directly to a
browser without buffering the full response.
