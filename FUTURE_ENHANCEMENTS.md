# Future Enhancements

Longer-term ideas for `pi-py`. Phases 1–8 are complete.

Sources reviewed: `/Users/jeff/code/pi/packages/coding-agent/docs/` and
`/Users/jeff/code/pi/packages/web-ui/`.

---

## Near-term library improvements

### Custom provider registration

Users can already construct `Model(api="openai-completions", base_url=...)` for
local servers. A higher-level `register_provider()` function that reads a
`models.json`-style dict and installs models into the in-process registry would
remove the need to build `Model` objects manually.

```python
pi_agent.register_providers_from_settings()  # reads ~/.pi/agent/models.json
model = pi_ai.get_model("my-local", "my-model")  # now available
```

### OAuth token refresh

`auth.json` OAuth tokens expire. A `refresh_oauth_token(provider)` helper that
reads the `refresh` field and calls the provider's token endpoint would remove the
manual refresh step. Currently the library warns on expiry and continues.

### Agent-to-agent calls

A tool that lets one agent spawn a sub-agent with a separate session and model,
collecting its result as a tool result. Useful for parallelising work or
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
browser without buffering the full response. (Prototype already exists in
`examples/webhook_agent.py`.)

---

## Advanced agent features (from coding-agent)

Features the TypeScript coding-agent offers that pi-py does not yet have.
Listed individually so any subset can be picked up independently.

### Additional providers

The coding-agent supports subscription-based OAuth providers and cloud platforms
that pi-py currently has no built-in support for:

| Provider | Notes |
|---|---|
| Claude Pro/Max OAuth | `anthropic-messages` API, OAuth token flow |
| GitHub Copilot OAuth | Expiring tokens, requires token refresh |
| Azure OpenAI | `openai-completions` API, tenant/deployment URL |
| Amazon Bedrock | Requires AWS credentials, different endpoint format |
| Google Vertex AI | GCP auth, different streaming protocol |
| Google Gemini | Direct API key, OpenAI-compatible endpoint available |
| Mistral, Groq, Cerebras | Already partially covered via `openai-completions` |
| Cloudflare AI Gateway | Proxy layer, no new API needed |

Most cloud providers just need the right `base_url` and auth headers; Bedrock
requires a signing step. Adding them to `models_catalog.py` + `env_keys.py` is
the main work.

### Permission-based tool control

An allowlist/blocklist system surfaced to users, distinct from the current
`before_tool_call` hook (which is programmatic). Useful for:
- Restricting the `bash` tool to specific commands or directories
- Preventing write/edit outside a defined workspace
- User-visible prompting before destructive operations

The coding-agent has per-tool permission prompts and project-level tool policies.

### Pi packages (distributable agent configurations)

The coding-agent supports distributing extensions, skills, prompt templates, and
themes as npm packages. A Python equivalent: pip packages that export skill
directories and tool factories.

```python
# install: pip install pi-pkg-my-skills
import pi_pkg_my_skills
tools = pi_pkg_my_skills.create_tools()
skills = pi_pkg_my_skills.load_skills()
```

The conventions to standardise: directory layout for skills/prompts within a
package, a manifest format, and how `load_skills()` discovers installed packages.

### Session tree branching from the CLI / REPL

`pi-py` sessions are linear (one branch). The coding-agent supports:
- Forking a session at any message
- Navigating between branches
- Labelling sessions and messages

The `Session` class already has `move_to()` and the JSONL format supports trees —
only the user-facing interface (CLI commands and `navigate_tree()` wiring) is
missing.

### Retry and delivery settings

The coding-agent exposes configurable retry behaviour and "message delivery"
options (how many times to re-attempt a failed turn, back-off timing). These map
to `AgentLoopConfig` fields that exist but aren't yet surfaced as named settings.

### Extension system

The coding-agent's TypeScript extension API lets third-party code:
- Register new tools at runtime
- Hook into agent lifecycle events
- Add custom slash commands
- Provide custom UI components (terminal panels)
- Persist custom session entries

The Python equivalent would be importable modules that call into the existing
`AgentHarness.on()` / `subscribe()` hooks and `create_tools()`. A formal
extension protocol (discovery, lifecycle, API surface) would make this pluggable.

### Interactive / TUI mode

A full terminal UI for `pi-py` — session tree navigation, live tool output,
model/thinking-level switching mid-session, theme support. The coding-agent's TUI
is the reference implementation. Python options: `textual` (rich component model,
async-native) or `prompt_toolkit`.

---

## Web UI (React-based, starting from scratch)

If a web UI is built for pi-py, it would be React (not web components / mini-lit).
The following are the features worth implementing, drawn from the `pi-web-ui`
package and the coding-agent. The TS implementation is a reference only — the
React version would be built independently.

### Core chat interface

| Feature | Notes |
|---|---|
| Streaming message display | Text deltas rendered as they arrive via SSE or WebSocket |
| Thinking block display | Collapsible reasoning sections |
| Tool call display | Expandable per-tool panels (bash output, file diffs, etc.) |
| Message history | Scrollable list, human/assistant/tool message types |
| Input editor | Multi-line, supports `@file` mentions, image paste |
| Session resume | Load past sessions by ID |
| Model / thinking selector | Dropdown picker bound to `pi_ai.get_models()` |

### Artifact rendering

The web-ui's most distinctive feature: agent-generated content is rendered
interactively rather than displayed as raw text.

| Artifact type | Render |
|---|---|
| HTML | Sandboxed `<iframe>` (isolated JS execution) |
| SVG | Inline rendering |
| Markdown | Full HTML render |
| Images | Native `<img>` |
| PDF / DOCX / XLSX | Viewer components (pdfjs, docx-preview, xlsx) |
| Text / code | Syntax-highlighted block |

Artifacts are produced by an `artifacts` tool the agent can call. The iframe
sandbox includes a console output capture and file download bridge.

### JavaScript REPL tool

A sandboxed `<iframe>` that executes JavaScript the agent writes. Output
(console.log, errors, returned values) is captured and shown in the chat. Useful
for data analysis, prototyping, and visualisations.

**Python-side**: a `create_js_repl_tool()` that returns an `AgentTool`; the
actual execution happens in the browser sandbox.

### Document extraction tool

Upload a PDF, DOCX, XLSX, or PPTX and extract its text content for the agent to
reason over. The TS web-ui uses client-side libraries; a Python version could use
`pypdf`, `python-docx`, or `openpyxl` on the server side instead.

### Storage and persistence

| Store | Purpose |
|---|---|
| Sessions store | Conversation history with metadata (IndexedDB or API-backed) |
| Settings store | User preferences (model, thinking level, theme, etc.) |
| Provider keys store | API keys per provider (encrypted at rest) |
| Custom providers store | User-defined models.json entries |

For a pi-py web UI, these would be API-backed (FastAPI endpoints) rather than
IndexedDB, since pi-py already has JSONL session persistence on the server.

### Settings and provider management UI

- API key entry per provider with validation
- Custom provider form (base URL, API key, model definitions)
- Thinking level and compaction settings
- Theme / display preferences

### Attachment support

Drag-and-drop files (images, documents) that are converted to `ImageContent` or
extracted text before being sent to the model. Mirrors the `pi-web-ui`
attachment pipeline but server-side extraction replaces browser-side libraries.

### Architecture for a pi-py web UI

```
React frontend
├─ ChatPanel component
│   ├─ MessageList (streaming SSE consumer)
│   ├─ MessageEditor (input + attachments)
│   ├─ ArtifactsPanel (iframe sandbox)
│   └─ Toolbars (model selector, session controls)
└─ API layer
    └─ FastAPI backend (pi_agent + pi_ai)
        ├─ POST /run/stream   → SSE of AgentEvents
        ├─ GET/POST /sessions → JSONL session management
        ├─ GET /models        → pi_ai.get_models()
        └─ POST /upload       → attachment extraction
```

The FastAPI backend already exists as a prototype in `examples/webhook_agent.py`.
The main new work is the React frontend and the iframe artifact sandbox.

---

## TUI layer (`pi-tui-py`)

A separate package for building interactive terminal UIs. The TS `pi-tui` uses
differential rendering, synchronized output (CSI 2026), and an overlay system.
Python options: `rich` (good rendering), `textual` (async-native components,
closer to the TS model), or `prompt_toolkit` (input handling focus).

Not needed for the core library but useful as a building block for the
interactive `pi-py` CLI mode and extension UIs.
