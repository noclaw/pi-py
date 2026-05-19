# Future Enhancements

Longer-term ideas for `pi-py`. Phases 1–8 are complete.

---

## TUI layer (`pi-tui-py`)

A separate package inspired by `packages/tui` in the TypeScript repo. The TS
`pi-tui` uses differential rendering, synchronized output (CSI 2026), and an
overlay system that Python's `curses`/`blessed`/`rich` don't match. Could be
built on top of `rich` or `textual` as a starting point. Not needed for the
core library use case but useful for interactive CLI agents.

## Custom provider registration

Users can already construct `Model(api="openai-completions", base_url=...)` for
local servers. A higher-level `register_provider()` function that reads a
`models.json`-style dict and installs models into the in-process registry would
remove the need to build `Model` objects manually.

```python
pi_agent.register_providers_from_settings()  # reads ~/.pi/agent/models.json
model = pi_ai.get_model("my-local", "my-model")  # now available
```

## OAuth token refresh

`auth.json` OAuth tokens expire. A `refresh_oauth_token(provider)` helper that
reads the `refresh` field and calls the provider's token endpoint would remove the
manual refresh step. Currently the library warns on expiry and continues.

## Agent-to-agent calls

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

## Streaming to web clients

A helper that bridges `AgentEventStream` to a Server-Sent Events or WebSocket
response — so a FastAPI/Flask endpoint can stream agent progress directly to a
browser without buffering the full response.
