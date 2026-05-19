# pi-py — Claude Context

## Project overview

Python port of the TypeScript monorepo at `/Users/jeff/code/pi` (GitHub: https://github.com/earendil-works/pi).

The TS repo has two main packages. Both are candidates for Python porting:

| TS package | Status | Python location |
|---|---|---|
| `packages/ai` | **Complete** | `packages/ai/src/pi_ai/` |
| `packages/agent` | **Not started** | — |

## packages/ai — COMPLETE

Unified LLM API: streaming text, tool calling, thinking/reasoning, image generation.

**Public API** (all from `import pi_ai`):
- `stream()`, `complete()`, `stream_simple()`, `complete_simple()`, `complete_sync()`
- `generate_images()`, `generate_images_sync()`
- `get_model(provider, id)`, `get_models(provider)`, `get_providers()`
- `get_image_model(provider, id)`, `get_image_models(provider)`, `get_image_providers()`
- `validate_tool_call(tools, tool_call)` — JSON Schema validation via `jsonschema`
- `is_context_overflow(message, context_window?)` — provider-neutral overflow detection
- `models_are_equal(a, b)`, `calculate_cost(model, usage)`, `get_env_api_key(provider)`
- `string_enum(values, ...)` — builds cross-provider-safe `{"type":"string","enum":[...]}` schema

**Providers registered:**
- `anthropic-messages` — Anthropic SDK async streaming
- `openai-completions` — OpenAI SDK async streaming; also used for DeepSeek, Groq, Cerebras, Ollama, vLLM, LiteLLM, etc.
- `openrouter-images` — OpenRouter image generation (non-streaming)

**Key design decisions:**
- Pydantic v2 for all types (`UserMessage`, `AssistantMessage`, `ToolResultMessage`, `Model`, `ImagesModel`, `StreamOptions`, etc.)
- Events are plain `dict[str, Any]` — no Pydantic overhead on hot streaming path
- `AssistantMessageEventStream` uses `asyncio.Queue` + `asyncio.Event` for `result()`
- Provider-specific option subclasses (`AnthropicStreamOptions`, `OpenAIStreamOptions`) must be defined at **module level**, not inside functions — locally-defined Pydantic subclasses lose inherited fields with `from __future__ import annotations`
- `signal: Optional[Any] = None` in `StreamOptions` accepts an `asyncio.Event`; set it to abort
- `transform_messages()` is called before every provider conversion: downgrades images for non-vision models, converts cross-model thinking blocks to plain text, inserts synthetic tool results for orphaned calls, normalizes tool call IDs for Anthropic (`[a-zA-Z0-9_-]{0,64}`)
- OpenAI strict mode: `additionalProperties: false` injected recursively into tool schemas
- Anthropic cache: `cache_control: {type: "ephemeral"}` on system prompt and last tool when `cache_retention != "none"`

**File layout:**
```
packages/ai/
├── pyproject.toml           # deps: openai, anthropic, pydantic, json-repair, jsonschema
└── src/pi_ai/
    ├── types.py             # All Pydantic models
    ├── stream.py            # AssistantMessageEventStream
    ├── registry.py          # Text provider registry
    ├── images_registry.py   # Image provider registry
    ├── env_keys.py          # get_env_api_key()
    ├── models.py            # get_model/get_models/get_providers + utilities
    ├── models_catalog.py    # Curated text model definitions
    ├── image_models.py      # get_image_model/get_image_models/get_image_providers
    ├── image_models_catalog.py  # Curated image model definitions
    ├── validation.py        # validate_tool_call, string_enum
    └── providers/
        ├── __init__.py      # Registers text providers
        ├── openai_completions.py
        ├── anthropic_messages.py
        ├── transform_messages.py
        └── images/
            ├── __init__.py  # Registers image providers
            └── openrouter.py
```

**Dev workflow:**
```bash
cd /Users/jeff/code/pi-py
uv sync --all-packages      # install deps
uv run python test_live.py  # live tests (needs API keys)
```

---

## packages/agent — NOT YET PORTED

TS source: `/Users/jeff/code/pi/packages/agent/`

### What it does

A stateful agentic loop built on top of `packages/ai`. Key concepts:

**Low-level loop** (`agentLoop`, `agentLoopContinue`):
- Multi-turn: LLM call → stream events → execute tool calls → repeat until stop
- Tool execution: sequential or parallel (`asyncio.gather` equivalent needed)
- Lifecycle hooks: `beforeToolCall`, `afterToolCall`, `shouldStopAfterTurn`, `prepareNextTurn`
- Steering queue: inject mid-run messages (`getSteeringMessages`)
- Follow-up queue: continue after the agent would otherwise stop (`getFollowUpMessages`)
- Returns an `EventStream<AgentEvent, AgentMessage[]>` — mirrors the `AssistantMessageEventStream` pattern

**High-level `Agent` class** (`agent.ts`):
- Stateful wrapper: holds `model`, `thinkingLevel`, `tools`, `messages`
- `agent.prompt(messages)` / `agent.continue()` — start runs
- `agent.subscribe(event, listener)` — event-based API (asyncio-friendly: use `asyncio.Event` or callbacks)
- `agent.stop()` — abort in-flight run
- Thread-safe pending tool call tracking

**Harness** (`harness/`):
- Context compaction / summarization (prune old messages when context window fills up)
- Session persistence: JSONL file repo or in-memory repo
- System prompt templates, skill definitions
- Shell output helpers, truncation utilities

**`AgentTool`** (extends `Tool`):
```typescript
{
  name, description, parameters,  // inherited from Tool
  label: str,                      // human-readable for UI
  prepareArguments?(args) -> args, // raw arg shim before validation
  execute(toolCallId, params, signal?, onUpdate?) -> AgentToolResult,
  executionMode?: "sequential" | "parallel"
}
```

**`AgentEvent`** union (what the loop emits):
- `agent_start`, `agent_end`
- `turn_start`, `turn_end`
- `message_start`, `message_update`, `message_end`
- `tool_execution_start`, `tool_execution_update`, `tool_execution_end`

**`AgentMessage`** = `Message | CustomAgentMessages[...]` (extensible union)

### Python porting notes

- The low-level loop can follow the same async background-task pattern as `stream()`: fire `asyncio.create_task(_run_loop(...))`, return a stream immediately
- Parallel tool execution → `asyncio.gather(*(execute(tc) for tc in tool_calls))`
- `AgentTool.execute` should be an async method
- Session persistence (JSONL) → plain `aiofiles` + `json` or `jsonlines`
- Compaction/summarization calls back into `pi_ai.complete()` — no extra deps needed
- `Agent.subscribe()` → can use a simple dict of `event_type -> list[Callable]` + `asyncio.gather` to await all listeners
- Custom message types → Python `TypedDict` or Pydantic with discriminated union
- `EventStream[AgentEvent, AgentMessage[]]` → same `AssistantMessageEventStream` pattern, parameterized differently (or reuse the base class from `stream.py`)

### Recommended port order

1. `AgentTool` type + `validate_tool_call` (already done in `validation.py`)
2. Low-level `agent_loop()` — core multi-turn execution
3. `Agent` class — stateful wrapper
4. Compaction — needs a summarization model call
5. Session persistence — JSONL repo
