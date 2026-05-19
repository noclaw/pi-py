# Future Enhancements

Roadmap for extending `pi_agent` toward a general-purpose coding agent.
Based on analysis of `/Users/jeff/code/pi/packages/coding-agent`.

---

## Phase 1 — Built-in tools (in progress)

Create `packages/agent/src/pi_agent/tools/` with the 7 tools from the coding-agent
and helper functions. These are the highest-value, most portable piece.

### Tools

| Tool | Schema | Key behavior |
|---|---|---|
| `read` | `path, offset?, limit?` | Truncates to 2000 lines / 50 KB; actionable "use offset=N to continue" messages; image detection |
| `bash` | `command, timeout?` | Streams stdout+stderr combined; tail-truncates; writes overflow to temp file; throws on non-zero exit |
| `edit` | `path, edits: [{old_text, new_text}]` | Multi-edit atomically; unique-match validation; BOM/CRLF handling; returns unified diff |
| `write` | `path, content` | Creates parent dirs; overwrites |
| `grep` | `pattern, path?, glob?, ignore_case?, literal?, context?, limit?` | Tries `rg` first, falls back to Python `re`; per-line 500-char truncation |
| `find` | `pattern, path?, limit?` | Tries `fd` first, falls back to `pathlib`/`glob`; gitignore-aware via `pathspec` |
| `ls` | `path?, limit?` | Sorted entries; shows kind (file/dir/symlink) |

### Helper functions

- `create_tools(env, cwd=None) -> list[AgentTool]` — factory returning all 7 tools
- `build_system_prompt(tools, context="") -> str` — assembles system prompt with tool list and guidelines
- `load_context_files(cwd, filenames=["CLAUDE.md","AGENTS.md"]) -> str` — walks directory tree upward collecting context files

### Usage pattern

```python
from pi_agent.tools import create_tools, load_context_files, build_system_prompt
from pi_agent import AgentHarness, PythonExecutionEnv, InMemorySessionRepo
import pi_ai

env  = PythonExecutionEnv(cwd="/my/project")
tools = create_tools(env)
context = load_context_files("/my/project")
system_prompt = build_system_prompt(tools, context)

repo = InMemorySessionRepo()  # or JsonlSessionRepo for persistence
session = await repo.create()

harness = AgentHarness(
    env=env, session=session,
    model=pi_ai.get_model("anthropic", "claude-sonnet-4-6"),
    tools=tools,
    system_prompt=system_prompt,
)
reply = await harness.prompt("Refactor the auth module to use JWT.")
```

---

## Phase 2 — Auto-compaction trigger ✓ DONE

All the pieces exist (`should_stop_after_turn`, `estimate_context_tokens`, `should_compact`,
`compact()`). Wire them together so the harness automatically compacts when approaching
the context window limit, without the caller having to manage it.

```python
harness = AgentHarness(
    ...,
    auto_compact=True,          # new option
    compact_reserve_tokens=16384,
)
```

Implementation sketch:
- In `_loop_config`, set `should_stop_after_turn` to check
  `should_compact(estimate_context_tokens(messages).tokens, model.context_window, settings)`.
- If true, fire `compact()`, update `context` via `prepare_next_turn`, and continue.
- Surface via a new `auto_compact: bool = False` init param on `AgentHarness`.

---

## Phase 3 — `create_agent()` convenience function ✓ DONE (live-tested)

A one-call setup for the 90% case: `PythonExecutionEnv`, session, tools,
context loading, and `AgentHarness` with sensible defaults.

```python
from pi_agent import create_agent

harness = await create_agent(
    cwd="/my/project",
    model=pi_ai.get_model("anthropic", "claude-sonnet-4-6"),
    session_dir="~/.pi/sessions",   # None = in-memory
    tools="all",                     # or list[AgentTool]
    context_files=["CLAUDE.md"],
    auto_compact=True,
)
reply = await harness.prompt("What does this codebase do?")
```

---

## Phase 4 — Settings management ✓ DONE

Simple JSON config files loaded and deep-merged at startup.

- Global: `~/.pi/settings.json`
- Project: `.pi/settings.json` (searched upward from cwd)

```python
from pi_agent.settings import load_settings, Settings

settings = load_settings(cwd="/my/project")
# settings.model, settings.compaction, settings.tools, etc.
```

Settings schema (initial):
```json
{
  "model": { "provider": "anthropic", "id": "claude-sonnet-4-6" },
  "thinking_level": "off",
  "compaction": { "enabled": true, "reserve_tokens": 16384 },
  "tools": { "bash": { "timeout": 30 } }
}
```

---

## Phase 5 — CLI / print mode

A `click`-based `pi` CLI for non-interactive use. Deferred until there is an
application-level consumer of the library.

```bash
pi prompt "Explain the auth module"      # print mode: single shot
pi prompt --session ~/.pi/sessions/abc "Continue the refactor"
```

Implementation: thin wrapper around `create_agent()` + `harness.prompt()`.

---

## What was explicitly NOT ported from coding-agent

| Component | Reason |
|---|---|
| TUI / interactive mode | App-specific; requires `pi-tui` or equivalent |
| Extension plugin system | `AgentHarness.on()` / `.subscribe()` hooks cover this adequately |
| Auth key management | `pi_ai.get_env_api_key()` covers the common case |
| Session export to HTML | App-specific |
| Image resizing / EXIF | App-specific; too many binary deps |
| Telemetry | App-specific |
| RPC server/client | App-specific |
| Keybindings | App-specific |

## TS source reference

Coding-agent source: `/Users/jeff/code/pi/packages/coding-agent/src/core/tools/`

Each Python tool is a faithful port of the corresponding TS file, minus
the `renderCall`/`renderResult` TUI methods and the `pi-tui` dependency.
