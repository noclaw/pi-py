# pi-py — Claude Context

## Project overview

Python port of the TypeScript monorepo at https://github.com/earendil-works/pi.  
This repo: https://github.com/noclaw/pi-py

The TS repo has two main packages. Both are candidates for Python porting:

| TS package | Status | Python location |
|---|---|---|
| `packages/ai` | **Complete** | `packages/ai/src/pi_ai/` |
| `packages/agent` | **Complete** | `packages/agent/src/pi_agent/` |

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
    ├── models.py            # get_model/get_models/get_providers + JSON loader
    ├── models.json          # Bundled model catalog (loaded at import; merged with ~/.pi-py/models.json)
    ├── image_models.py      # get_image_model/get_image_models/get_image_providers + JSON loader
    ├── image_models.json    # Bundled image model catalog (merged with ~/.pi-py/image_models.json)
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

## packages/agent — COMPLETE

TS source: `/Users/jeff/code/pi/packages/agent/` (coding-agent tools/settings from `/Users/jeff/code/pi/packages/coding-agent/`)

Stateful agentic loop built on top of `packages/ai`, plus built-in coding tools, settings management, and a one-call factory.

**Public API** (all from `import pi_agent`):

*Core loop:*
- `agent_loop(prompts, context, config, signal?)` → `AgentEventStream` — fire-and-forget background task
- `agent_loop_continue(context, config, signal?)` → `AgentEventStream` — resume from existing context
- `run_agent_loop(...)`, `run_agent_loop_continue(...)` — awaitable versions (no stream wrapper)
- `Agent` — stateful wrapper: holds transcript, tools, queues; `prompt()`, `proceed()`, `abort()`, `subscribe()`
- `AgentHarness(env, session, model, tools?, auto_compact?, ...)` — high-level orchestration: session + compaction + hooks + `prompt()`, `compact()`, `navigate_tree()`

*One-call factory:*
- `create_agent(model?, cwd?, session_dir?, settings_dir?, tools?, context_files?, ...)` → `AgentHarness` — wires env, session, tools, context loading, settings, and auth in one call; `model=None` resolves from `~/.pi-py/settings.json`

*Built-in tools:*
- `create_tools(env, cwd?)` → `list[AgentTool]` — all 7 tools
- `create_read_tool(cwd)`, `create_bash_tool(env, cwd)`, `create_edit_tool(cwd)`, `create_write_tool(cwd)`, `create_grep_tool(cwd)`, `create_find_tool(cwd)`, `create_ls_tool(cwd)`
- `load_context_files(cwd, filenames?)` — walks up directory tree for `CLAUDE.md` / `AGENTS.md`
- `build_system_prompt(tools?, context?)` — assembles system prompt with tool list + guidelines

*Settings:*
- `load_settings(cwd?, settings_dir?)` → `Settings` — merges `~/.pi-py/settings.json` with project `.pi-py/settings.json`
- `load_custom_models(settings_dir?)` → `list[(provider, Model)]` — custom providers from `models.json`
- `find_custom_model(provider, model_id, settings_dir?)` → `Model | None`
- `load_auth(provider, settings_dir?)` → `dict | None` — reads `auth.json`; OAuth tokens get Bearer header
- `get_default_model(cwd?, settings_dir?)` → `Model | None` — resolves `defaultProvider`+`defaultModel`
- `make_auth_provider(settings_dir?)` → callable for `AgentHarness.get_api_key_and_headers`

**Key types:**
- `AgentTool(name, description, parameters, label, execute, prepare_arguments?, execution_mode?)`
- `AgentToolResult(content, details, terminate?)`
- `AgentContext(system_prompt, messages, tools?)`
- `AgentLoopConfig` — callbacks: `convert_to_llm`, `get_api_key`, `get_steering_messages`, `get_follow_up_messages`, `before_tool_call`, `after_tool_call`, `should_stop_after_turn`, `prepare_next_turn`
- `AgentEvent` — plain `dict` with `"type"` key: `agent_start/end`, `turn_start/end`, `message_start/update/end`, `tool_execution_start/update/end`
- `Settings(default_provider, default_model)` — merged settings result

**Session:**
- `InMemorySessionRepo` / `InMemorySessionStorage` — in-process, no disk
- `JsonlSessionRepo` / `JsonlSessionStorage` — JSONL files, survives restarts
- `Session` — tree of entries; `build_context()` reconstructs message history; `move_to()` for branching
- Messages stored via `model_dump()` and deserialized via `model_validate()` on context rebuild — without this, session context is silently dropped before LLM calls

**Harness layer:**
- `AgentHarness(auto_compact=False, compact_reserve_tokens=16384, compact_keep_recent_tokens=20000)` — `auto_compact=True` fires `_maybe_auto_compact()` silently in `prepare_next_turn`; `create_agent()` defaults to `True`
- `get_api_key_and_headers` flows through both compaction AND streaming via `AgentLoopConfig.get_api_key`
- Compaction hooks: `session_before_compact`, `session_compact`; hooks via `on()` / `subscribe()`
- `compact(preparation, model, api_key, ...)`, `prepare_compaction()`, `generate_branch_summary()`
- `load_skills(env, dirs)`, `format_skills_for_system_prompt(skills)`
- `load_prompt_templates(env, paths)`, `format_prompt_template_invocation(template, args)`
- `PythonExecutionEnv(cwd)` — full `ExecutionEnv` using stdlib + `asyncio.to_thread`
- `truncate_head(content, max_lines, max_bytes)`, `truncate_tail(...)` — UTF-8-aware output truncation
- `execute_shell_with_capture(env, command, options)` — streaming with temp-file overflow

**Built-in tools (all in `pi_agent.tools`):**

| Tool | Parameters | Key behavior |
|---|---|---|
| `read` | `path, offset?, limit?` | Truncates to 2000 lines/50 KB; actionable "use offset=N" messages; image → `ImageContent` |
| `bash` | `command, timeout?` | Streams via `execute_shell_with_capture`; throws on non-zero exit with output prepended |
| `edit` | `path, edits:[{old_text,new_text}]` | Multi-edit atomic; unique-match validation; BOM/CRLF normalization; returns unified diff |
| `write` | `path, content` | Creates parent dirs; reports Created vs Updated |
| `grep` | `pattern, path?, glob?, ignore_case?, literal?, context?, limit?` | Tries `rg`, falls back to `re`; 500-char per-line truncation |
| `find` | `pattern, path?, limit?` | Tries `fd`/`fdfind`, falls back to `pathlib.rglob` |
| `ls` | `path?, limit?` | Dirs first (trailing `/`), symlinks marked `@`, sorted |

**Settings files (`~/.pi-py/`):**
- `settings.json` — `defaultProvider`, `defaultModel`; project `.pi-py/settings.json` overrides (walked up from `cwd`)
- `models.json` — custom providers + models; `authHeader: true` → API key stored as `Authorization: Bearer` in `model.headers`
- `auth.json` — per-provider credentials; `type: "oauth"` → access token + `Authorization: Bearer` header + expiry warning

**Key design decisions:**
- `AbortSignal` → `asyncio.Event` (`.is_set()` = abort), consistent with `pi_ai`
- `agent.continue()` → `agent.proceed()` (Python keyword conflict)
- `AgentTool` is a `@dataclass` (not Pydantic) — stores async callables
- Events are plain `dict[str, Any]` — same hot-path design as `pi_ai`'s `AssistantMessageEvent`
- `asyncio.coroutine` (removed in 3.11) → `async def`; `Promise.all` → `asyncio.gather`
- Extra deps: `pyyaml>=6.0` (YAML frontmatter), `pathspec>=0.12` (gitignore patterns)

**File layout:**
```
packages/agent/
├── pyproject.toml           # deps: pi-ai, pyyaml, pathspec
└── src/pi_agent/
    ├── types.py             # AgentTool, AgentContext, AgentLoopConfig, AgentEvent, hook types
    ├── agent_loop.py        # agent_loop(), run_agent_loop(), AgentEventStream, tool execution
    ├── agent.py             # Agent class
    ├── create_agent.py      # create_agent() one-call factory
    ├── settings.py          # load_settings(), load_custom_models(), load_auth(), get_default_model()
    ├── tools/
    │   ├── __init__.py      # create_tools(), load_context_files(), build_system_prompt()
    │   ├── read.py          # read tool
    │   ├── bash.py          # bash tool
    │   ├── edit.py          # edit tool (multi-edit, diff)
    │   ├── write.py         # write tool
    │   ├── grep.py          # grep tool (rg / re fallback)
    │   ├── find.py          # find tool (fd / pathlib fallback)
    │   └── ls.py            # ls tool
    └── harness/
        ├── types.py         # Result/Ok/Err, error classes, FileSystem/Shell ABCs, session types
        ├── messages.py      # BashExecutionMessage, CustomMessage, convert_to_llm()
        ├── skills.py        # load_skills(), format_skill_invocation()
        ├── system_prompt.py # format_skills_for_system_prompt()
        ├── prompt_templates.py  # load_prompt_templates(), format_prompt_template_invocation()
        ├── agent_harness.py # AgentHarness class (auto_compact, _do_compact, _maybe_auto_compact)
        ├── session/
        │   ├── uuid.py              # uuidv7()
        │   ├── memory_storage.py    # InMemorySessionStorage
        │   ├── jsonl_storage.py     # JsonlSessionStorage
        │   ├── session.py           # Session, build_session_context(), _deserialize_message()
        │   ├── memory_repo.py       # InMemorySessionRepo
        │   ├── jsonl_repo.py        # JsonlSessionRepo
        │   └── repo_utils.py        # get_entries_to_fork(), get_fs_result_or_throw()
        ├── compaction/
        │   ├── utils.py             # FileOperations, serialize_conversation()
        │   ├── compaction.py        # prepare_compaction(), compact(), estimate_tokens()
        │   └── branch_summarization.py  # generate_branch_summary()
        ├── env/
        │   └── python.py            # PythonExecutionEnv
        └── utils/
            ├── truncate.py          # truncate_head(), truncate_tail()
            └── shell_output.py      # execute_shell_with_capture()
```
