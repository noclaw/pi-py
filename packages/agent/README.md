# pi-agent

Stateful agent with tool execution and event streaming. Built on `pi-ai`.

## Installation

```bash
pip install pi-agent
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pi-agent
```

Dependencies: `pi-ai`, `pyyaml`, `pathspec`.

## Quick Start

```python
import asyncio
import pi_agent
import pi_ai

async def main():
    model = pi_ai.get_model("anthropic", "claude-sonnet-4-6")

    agent = pi_agent.Agent(
        model=model,
        system_prompt="You are a helpful assistant.",
    )

    agent.subscribe(lambda event, _: (
        print(event["message"].content[0].text
              if event["type"] == "message_update"
              and hasattr(event.get("message"), "content") else ""),
    ))

    await agent.prompt("Hello!")

asyncio.run(main())
```

## Core Concepts

### AgentMessage vs LLM Message

The agent works with `AgentMessage`, a flexible type that can include:
- Standard LLM messages (`user`, `assistant`, `toolResult`)
- Custom app-specific message types (harness message types such as `BashExecutionMessage`, `CompactionSummaryMessage`, etc.)

LLMs only understand `user`, `assistant`, and `toolResult`. The `convert_to_llm` function bridges this gap by filtering and transforming messages before each LLM call.

### Message Flow

```
AgentMessage[] → transform_context() → AgentMessage[] → convert_to_llm() → Message[] → LLM
                     (optional)                           (required)
```

1. **transform_context**: Prune old messages, inject external context
2. **convert_to_llm**: Filter out UI-only messages, convert custom types to LLM format

## Event Flow

The agent emits events for UI updates. Understanding the event sequence helps build responsive interfaces.

### prompt() Event Sequence

When you call `await agent.prompt("Hello")`:

```
prompt("Hello")
├─ agent_start
├─ turn_start
├─ message_start   { message: user_message }       # Your prompt
├─ message_end     { message: user_message }
├─ message_start   { message: assistant_message }  # LLM starts responding
├─ message_update  { message: partial... }         # Streaming chunks
├─ message_update  { message: partial... }
├─ message_end     { message: assistant_message }  # Complete response
├─ turn_end        { message, tool_results: [] }
└─ agent_end       { messages: [...] }
```

### With Tool Calls

If the assistant calls tools, the loop continues:

```
prompt("Read config.json")
├─ agent_start
├─ turn_start
├─ message_start/end  { user_message }
├─ message_start      { assistant_message with tool_call }
├─ message_update...
├─ message_end        { assistant_message }
├─ tool_execution_start   { tool_call_id, tool_name, args }
├─ tool_execution_update  { partial_result }          # If tool streams
├─ tool_execution_end     { tool_call_id, result }
├─ message_start/end  { tool_result_message }
├─ turn_end           { message, tool_results: [tool_result] }
│
├─ turn_start                                         # Next turn
├─ message_start      { assistant_message }           # LLM responds
├─ message_update...
├─ message_end
├─ turn_end
└─ agent_end
```

Tool execution mode is configurable:

- `parallel` (default): preflight tool calls sequentially, execute allowed tools concurrently, emit `tool_execution_end` as soon as each tool is finalized, then emit `toolResult` messages and `turn_end.tool_results` in assistant source order
- `sequential`: execute tool calls one by one

The mode can be set globally via `tool_execution` in `AgentLoopConfig` or the `Agent` constructor, or per-tool via `execution_mode` on `AgentTool`. If any tool call in a batch targets a tool with `execution_mode="sequential"`, the entire batch executes sequentially regardless of the global setting.

The `before_tool_call` hook runs after `tool_execution_start` and validated argument parsing. It can block execution. The `after_tool_call` hook runs after tool execution finishes and before `tool_execution_end` and final tool result message events are emitted.

Tools can return `terminate=True` to hint that the automatic follow-up LLM call should be skipped. The loop only stops early when every finalized tool result in that batch sets `terminate=True`.

`should_stop_after_turn` (on `AgentLoopConfig`) runs after `turn_end` is emitted. If it returns `True`, the loop emits `agent_end` and exits without starting another LLM call.

### proceed() — Resume from Current Context

`proceed()` resumes from existing context without adding a new message. Use it for retries after errors.

```python
# After an error, retry from current state
await agent.proceed()
```

The last message in context must be `user` or `toolResult` (not `assistant`). `proceed()` is the Python equivalent of TypeScript's `agent.continue()` — `continue` is a reserved keyword in Python.

### Event Types

| Event | Description |
|-------|-------------|
| `agent_start` | Agent begins processing |
| `agent_end` | Final event for the run |
| `turn_start` | New turn begins (one LLM call + tool executions) |
| `turn_end` | Turn completes with assistant message and tool results |
| `message_start` | Any message begins (user, assistant, toolResult) |
| `message_update` | **Assistant only.** Includes `assistant_message_event` with streaming delta |
| `message_end` | Message completes |
| `tool_execution_start` | Tool begins |
| `tool_execution_update` | Tool streams progress |
| `tool_execution_end` | Tool completes |

`Agent.subscribe()` listeners are called in registration order. `agent_end` means no more loop events will be emitted.

## Agent Options

```python
agent = pi_agent.Agent(
    model=model,

    # System prompt sent with each request
    system_prompt="You are a helpful assistant.",

    # Initial thinking level
    thinking_level="off",  # "off" | "minimal" | "low" | "medium" | "high" | "xhigh"

    # Initial tools
    tools=[my_tool],

    # Initial transcript
    messages=[],

    # Convert AgentMessage list to LLM Message list.
    # Filter out custom message types here.
    convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role")],

    # Transform context before convert_to_llm (for pruning, compaction)
    transform_context=lambda msgs, signal: prune_old_messages(msgs),

    # Steering mode: "one-at-a-time" (default) or "all"
    steering_mode="one-at-a-time",

    # Follow-up mode: "one-at-a-time" (default) or "all"
    follow_up_mode="one-at-a-time",

    # Session ID for provider caching
    session_id="session-123",

    # Dynamic API key resolution (for expiring OAuth tokens)
    get_api_key=lambda provider: refresh_token(),

    # Tool execution mode: "parallel" (default) or "sequential"
    tool_execution="parallel",

    # Preflight each tool call after args are validated. Can block execution.
    before_tool_call=my_before_tool_call,

    # Postprocess each tool result before final tool events are emitted.
    after_tool_call=my_after_tool_call,

    # Custom thinking token budgets
    thinking_budgets={"minimal": 128, "low": 512, "medium": 1024},
)
```

## Agent State

```python
agent.model             # Active model
agent.thinking_level    # Current thinking level
agent.tools             # List of AgentTool; assigning copies the list
agent.messages          # Transcript; assigning copies the list
agent.is_streaming      # True while a run is active
agent.streaming_message # Current partial assistant message, or None
agent.pending_tool_calls  # frozenset of in-flight tool_call_ids
agent.error_message     # Error from the most recent failed turn, or None
```

Assign `agent.tools = [...]` or `agent.messages = [...]` to replace; both copy the provided list. Mutating the returned list mutates current agent state.

## Methods

### Prompting

```python
# Text prompt
await agent.prompt("Hello")

# With images
await agent.prompt("What's in this image?", images=[
    pi_ai.ImageContent(data=base64_data, mime_type="image/jpeg")
])

# AgentMessage directly
from pi_ai import UserMessage
await agent.prompt(UserMessage(content="Hello"))

# Resume from current context (last message must be user or toolResult)
await agent.proceed()
```

### State Management

```python
agent.system_prompt = "New prompt"
agent.model = pi_ai.get_model("openai", "gpt-4o")
agent.thinking_level = "medium"
agent.tools = [my_tool]
agent.tool_execution = "sequential"
agent.before_tool_call = my_before_hook
agent.after_tool_call = my_after_hook
agent.messages = new_messages    # top-level list is copied
agent.messages.append(message)
agent.reset()
```

### Control

```python
agent.abort()            # Cancel current operation (sets the abort asyncio.Event)
await agent.wait_for_idle()   # Not implemented on Agent; use proceed() and prompt() sequentially
```

### Events

```python
def my_listener(event: dict, signal) -> None:
    if event["type"] == "agent_end":
        print("Run complete, messages:", len(event["messages"]))

unsubscribe = agent.subscribe(my_listener)
unsubscribe()  # Remove the listener
```

## Steering and Follow-up

Steering messages let you inject guidance after the current turn's tools finish but before the next LLM call. Follow-up messages let you queue work to run after the agent would otherwise stop.

```python
agent.steering_mode = "one-at-a-time"
agent.follow_up_mode = "one-at-a-time"

# Inject after current tool batch completes
agent.steer(pi_ai.UserMessage(content="Focus on the error in line 42."))

# Run after agent would otherwise stop
agent.follow_up(pi_ai.UserMessage(content="Now summarise the result."))

agent.clear_steering_queue()
agent.clear_follow_up_queue()
agent.clear_all_queues()
```

When steering messages are detected after a turn completes, all tool calls from the current assistant message have already finished. The messages are injected and the LLM responds on the next turn.

Follow-up messages are checked only when there are no more tool calls and no steering messages. If any are queued, they are injected and another turn runs.

## Tools

```python
from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent

read_file_tool = AgentTool(
    name="read_file",
    label="Read File",         # Human-readable label for UI
    description="Read a file's contents",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
        },
        "required": ["path"],
    },
    # Per-tool execution mode override (optional).
    # "sequential" forces the entire batch to run sequentially.
    # "parallel" allows concurrent execution with other calls.
    # If omitted, the global tool_execution config applies.
    execution_mode="sequential",
    execute=read_file,
)

async def read_file(tool_call_id: str, params: dict, signal=None, on_update=None) -> AgentToolResult:
    content = open(params["path"]).read()

    # Optional: stream progress
    if on_update:
        on_update(AgentToolResult(content=[TextContent(text="Reading...")], details={}))

    # Optional: set terminate=True to skip the automatic follow-up LLM call
    # when every finalized tool result in the batch does the same.
    return AgentToolResult(
        content=[TextContent(text=content)],
        details={"path": params["path"], "size": len(content)},
    )

agent.tools = [read_file_tool]
```

### Error Handling

**Raise an exception** when a tool fails. Do not encode errors as content.

```python
async def read_file(tool_call_id, params, signal=None, on_update=None) -> AgentToolResult:
    if not os.path.exists(params["path"]):
        raise FileNotFoundError(f"File not found: {params['path']}")
    return AgentToolResult(content=[TextContent(text=open(params["path"]).read())], details={})
```

Raised exceptions are caught by the agent and reported to the LLM as tool errors with `is_error=True`.

### Argument Preparation

Use `prepare_arguments` as a raw argument shim before JSON Schema validation — useful when a provider returns arguments in a slightly different shape than your schema expects:

```python
tool = AgentTool(
    name="my_tool",
    ...
    prepare_arguments=lambda args: {**args, "mode": args.get("mode", "default")},
    execute=my_execute,
)
```

## Low-Level API

For direct control without the `Agent` class:

```python
import asyncio
import pi_agent
import pi_ai

context = pi_agent.AgentContext(
    system_prompt="You are helpful.",
    messages=[],
    tools=[],
)

config = pi_agent.AgentLoopConfig(
    model=pi_ai.get_model("anthropic", "claude-haiku-4-5"),
    convert_to_llm=lambda msgs: [m for m in msgs if hasattr(m, "role")
                                  and m.role in ("user", "assistant", "toolResult")],
    tool_execution="parallel",
)

user_message = pi_ai.UserMessage(content="Hello")

async def run():
    async for event in pi_agent.agent_loop([user_message], context, config):
        print(event["type"])

    # Continue from existing context
    async for event in pi_agent.agent_loop_continue(context, config):
        print(event["type"])

asyncio.run(run())
```

Or use the awaitable form (no stream wrapper):

```python
messages = await pi_agent.run_agent_loop([user_message], context, config, emit=my_emit)
```

The low-level loop does not wait for your async event handling to settle before later producer phases continue. If you need message processing to act as a barrier before tool preflight, use the `Agent` class.

## AgentHarness

`AgentHarness` is a higher-level orchestration layer built on the low-level loop. It adds session persistence, context compaction, hooks, and resource management.

```python
from pi_agent import (
    AgentHarness, PythonExecutionEnv,
    InMemorySessionRepo,   # or JsonlSessionRepo for disk persistence
)
import pi_ai

async def main():
    env = PythonExecutionEnv(cwd="/path/to/project")
    repo = InMemorySessionRepo()
    session = await repo.create()
    model = pi_ai.get_model("anthropic", "claude-sonnet-4-6")

    harness = AgentHarness(
        env=env,
        session=session,
        model=model,
        system_prompt="You are a helpful assistant.",
        tools=[my_tool],
    )

    reply = await harness.prompt("Hello!")
    print(reply.content[0].text)

asyncio.run(main())
```

### Session Persistence

Use `JsonlSessionRepo` (and `PythonExecutionEnv`) for disk-backed sessions that survive restarts:

```python
from pi_agent import JsonlSessionRepo, PythonExecutionEnv

env = PythonExecutionEnv(cwd="/path/to/project")
repo = JsonlSessionRepo(fs=env, sessions_root="~/.mysessions")

# Create a new session
session = await repo.create(cwd="/path/to/project")

# List existing sessions
sessions = await repo.list(cwd="/path/to/project")

# Reopen an existing session
session = await repo.open(sessions[0])
```

### Context Compaction

When the context window fills up, compact older history into a summary:

```python
# Explicit compaction
result = await harness.compact()
print(f"Compacted {result['tokensBefore']} tokens, summary written")

# Check whether compaction is needed
from pi_agent import prepare_compaction, should_compact, DEFAULT_COMPACTION_SETTINGS
branch = await session.get_branch()
prep = prepare_compaction(branch, DEFAULT_COMPACTION_SETTINGS)
if prep.ok and prep.value:
    await harness.compact()
```

### Hooks

```python
# Subscribe to all events (agent lifecycle + harness-specific)
unsubscribe = harness.subscribe(lambda event, signal: print(event["type"]))

# Hook into specific harness events with return values
harness.on("tool_call", lambda event: (
    {"block": True, "reason": "Not allowed"} if event["toolName"] == "bash" else None
))

harness.on("tool_result", lambda event: (
    {"terminate": True} if event["toolName"] == "done" else None
))

harness.on("context", lambda event: (
    {"messages": event["messages"][-50:]}  # Keep last 50 messages
))
```

### Skills and Prompt Templates

```python
from pi_agent import load_skills, load_prompt_templates, format_skills_for_system_prompt

# Load SKILL.md files from a directory tree
result = await load_skills(env, "/path/to/skills")
skills = result["skills"]

# Load .md prompt templates from a directory
result = await load_prompt_templates(env, "/path/to/templates")
templates = result["promptTemplates"]

# Build a system prompt block listing available skills
prompt_block = format_skills_for_system_prompt(skills)

harness = AgentHarness(
    env=env,
    session=session,
    model=model,
    resources=pi_agent.AgentHarnessResources(skills=skills, prompt_templates=templates),
    system_prompt=lambda **kwargs: f"You are helpful.\n\n{format_skills_for_system_prompt(kwargs['resources'].skills or [])}",
)

# Invoke a skill by name
await harness.skill("my-skill", "Additional context")

# Invoke a prompt template with positional arguments
await harness.prompt_from_template("summarise", args=["file.py"])
```

### Utilities

**Truncation** — keep output within line and byte limits (UTF-8-aware):

```python
from pi_agent import truncate_head, truncate_tail

# Keep the first N lines/bytes (for file reads)
result = truncate_head(content, max_lines=500, max_bytes=50_000)

# Keep the last N lines/bytes (for command output — tail has the errors)
result = truncate_tail(content, max_lines=500, max_bytes=50_000)

print(f"truncated={result.truncated}, by={result.truncated_by}")
```

**Shell capture** — run a command and stream output with automatic truncation:

```python
from pi_agent import execute_shell_with_capture, ShellCaptureOptions

result = await execute_shell_with_capture(
    env, "pytest tests/", ShellCaptureOptions(timeout=60.0)
)
if result.ok:
    capture = result.value
    print(capture.output)
    print(f"exit_code={capture.exit_code}, truncated={capture.truncated}")
```

## Abort

```python
# Agent class — sets an asyncio.Event
agent.abort()

# AgentHarness — clears queues and signals the in-flight run
result = await harness.abort()
print("cleared steer:", len(result["clearedSteer"]))
print("cleared follow_up:", len(result["clearedFollowUp"]))
await harness.wait_for_idle()
```

## Custom Message Types

Pass any object with a `role` attribute through `AgentMessage`. Handle custom types in `convert_to_llm`:

```python
from dataclasses import dataclass

@dataclass
class NotificationMessage:
    role: str = "notification"
    text: str = ""
    timestamp: int = 0

def my_convert_to_llm(messages):
    result = []
    for m in messages:
        if isinstance(m, NotificationMessage):
            pass  # Filter out — UI only
        elif hasattr(m, "role") and m.role in ("user", "assistant", "toolResult"):
            result.append(m)
    return result

agent = pi_agent.Agent(model=model, convert_to_llm=my_convert_to_llm)
agent.messages.append(NotificationMessage(text="Task started", timestamp=0))
```

The harness ships with built-in custom message types in `pi_agent.harness.messages`: `BashExecutionMessage`, `CustomMessage`, `BranchSummaryMessage`, `CompactionSummaryMessage`. Its `convert_to_llm()` converts these to user messages automatically.

## Credits

This package is a Python port of the [`@earendil-works/pi-agent-core`](https://github.com/earendil-works/pi/tree/main/packages/agent) TypeScript library. The original TypeScript implementation is the authoritative reference for the agent loop protocol, event sequence guarantees, tool execution semantics, session tree structure, and harness orchestration. The Python version is a rewrite targeting the same public API surface using `asyncio`, Pydantic v2, and Python stdlib in place of Node.js primitives.

## License

MIT
