# pi-ai

Unified LLM API for Python with streaming, tool calling, thinking/reasoning, image generation, and token and cost tracking.

## Table of Contents

- [Supported Providers](#supported-providers)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tools](#tools)
  - [Defining Tools](#defining-tools)
  - [Handling Tool Calls](#handling-tool-calls)
  - [Validating Tool Arguments](#validating-tool-arguments)
  - [Streaming Tool Calls with Partial JSON](#streaming-tool-calls-with-partial-json)
  - [Complete Event Reference](#complete-event-reference)
- [Image Input](#image-input)
- [Image Generation](#image-generation)
- [Thinking/Reasoning](#thinkingreasoning)
  - [Unified Interface](#unified-interface-stream_simplecomplete_simple)
  - [Streaming Thinking Content](#streaming-thinking-content)
- [Stop Reasons](#stop-reasons)
- [Error Handling](#error-handling)
  - [Aborting Requests](#aborting-requests)
  - [Context Overflow Detection](#context-overflow-detection)
- [APIs, Models, and Providers](#apis-models-and-providers)
  - [Providers and Models](#providers-and-models)
  - [Querying Providers and Models](#querying-providers-and-models)
  - [Custom Models](#custom-models)
  - [OpenAI Compatibility Settings](#openai-compatibility-settings)
- [Context Serialization](#context-serialization)
- [Environment Variables](#environment-variables)
- [Synchronous Usage](#synchronous-usage)
- [License](#license)

## Supported Providers

**Text generation:**
- **OpenAI** — GPT-4o, GPT-4.1, o3, o4-mini, and more
- **Anthropic** — Claude Opus, Sonnet, Haiku (4.x series)
- **DeepSeek** — V4 Flash, V4 Pro (via OpenAI-compatible API)
- **Groq** — Llama 3.x (via OpenAI-compatible API)
- **Cerebras** — Llama 3.1 (via OpenAI-compatible API)
- **Any OpenAI-compatible API** — Ollama, vLLM, LM Studio, LiteLLM, etc.

**Image generation:**
- **OpenRouter** — FLUX.2, Gemini image models, GPT Image 1, Recraft V3, and more

## Installation

```bash
pip install pi-ai
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pi-ai
```

Dependencies: `openai`, `anthropic`, `pydantic`, `json-repair`, `jsonschema`.

## Quick Start

```python
import asyncio
import pi_ai

async def main():
    # Look up a model from the built-in catalog
    model = pi_ai.get_model("openai", "gpt-4o-mini")

    # Define tools using plain JSON Schema dicts
    tools = [
        pi_ai.Tool(
            name="get_time",
            description="Get the current time",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "e.g. America/New_York"}
                },
            },
        )
    ]

    # Build a conversation context
    context = pi_ai.Context(
        system_prompt="You are a helpful assistant.",
        messages=[pi_ai.UserMessage(content="What time is it?")],
        tools=tools,
    )

    # Option 1: Stream all events
    s = pi_ai.stream(model, context)

    async for event in s:
        match event["type"]:
            case "text_delta":
                print(event["delta"], end="", flush=True)
            case "toolcall_end":
                tc = event["tool_call"]
                print(f"\nTool called: {tc.name}({tc.arguments})")
            case "done":
                msg = event["message"]
                print(f"\nStop: {event['reason']}, "
                      f"tokens: {msg.usage.input}in/{msg.usage.output}out, "
                      f"cost: ${msg.usage.cost.total:.6f}")
            case "error":
                print(f"\nError: {event['error'].error_message}")

    # Get the final AssistantMessage and add it to context
    final = await s.result()
    context.messages.append(final)

    # Handle tool calls and continue
    tool_calls = [b for b in final.content if isinstance(b, pi_ai.ToolCall)]
    for call in tool_calls:
        from datetime import datetime
        result = datetime.now().isoformat()
        context.messages.append(
            pi_ai.ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[pi_ai.TextContent(text=result)],
                is_error=False,
            )
        )

    if tool_calls:
        continuation = await pi_ai.complete(model, context)
        print(continuation.content[0].text if continuation.content else "")

    # Option 2: Get a complete response without streaming
    response = await pi_ai.complete(model, pi_ai.Context(
        messages=[pi_ai.UserMessage(content="Hello!")]
    ))
    for block in response.content:
        if isinstance(block, pi_ai.TextContent):
            print(block.text)


asyncio.run(main())
```

## Tools

### Defining Tools

Tools use plain JSON Schema dicts for their `parameters` field. Use `string_enum()` for enum fields — it generates a `{"type": "string", "enum": [...]}` schema compatible with all providers (some reject the `anyOf/const` pattern that TypeScript tools may produce):

```python
weather_tool = pi_ai.Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or coordinates"},
            "units": pi_ai.string_enum(["celsius", "fahrenheit"], default="celsius"),
        },
        "required": ["location"],
    },
)

book_meeting_tool = pi_ai.Tool(
    name="book_meeting",
    description="Schedule a meeting",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "start_time": {"type": "string", "format": "date-time"},
            "attendees": {"type": "array", "items": {"type": "string", "format": "email"}},
        },
        "required": ["title", "start_time", "attendees"],
    },
)
```

### Handling Tool Calls

Tool results support both text and image content:

```python
import base64

context = pi_ai.Context(
    messages=[pi_ai.UserMessage(content="What's the weather in London?")],
    tools=[weather_tool],
)

response = await pi_ai.complete(model, context)

for block in response.content:
    if isinstance(block, pi_ai.ToolCall):
        result = await execute_weather_api(block.arguments)
        context.messages.append(
            pi_ai.ToolResultMessage(
                tool_call_id=block.id,
                tool_name=block.name,
                content=[pi_ai.TextContent(text=str(result))],
                is_error=False,
            )
        )

# Tool results can include images (for vision-capable models)
chart_bytes = open("chart.png", "rb").read()
context.messages.append(
    pi_ai.ToolResultMessage(
        tool_call_id="tool_xyz",
        tool_name="generate_chart",
        content=[
            pi_ai.TextContent(text="Generated chart showing temperature trends"),
            pi_ai.ImageContent(
                data=base64.b64encode(chart_bytes).decode(),
                mime_type="image/png",
            ),
        ],
        is_error=False,
    )
)
```

### Validating Tool Arguments

Use `validate_tool_call()` to validate arguments against a tool's JSON Schema before executing. It raises `ValueError` with a descriptive message on failure — return that message to the model as an error result so it can retry:

```python
for block in response.content:
    if isinstance(block, pi_ai.ToolCall):
        try:
            args = pi_ai.validate_tool_call(context.tools, block)
            result = await execute_tool(block.name, args)
            context.messages.append(pi_ai.ToolResultMessage(
                tool_call_id=block.id,
                tool_name=block.name,
                content=[pi_ai.TextContent(text=str(result))],
                is_error=False,
            ))
        except ValueError as e:
            # Return the validation error to the model so it can retry
            context.messages.append(pi_ai.ToolResultMessage(
                tool_call_id=block.id,
                tool_name=block.name,
                content=[pi_ai.TextContent(text=str(e))],
                is_error=True,
            ))
```

### Streaming Tool Calls with Partial JSON

During streaming, tool arguments are progressively parsed as they arrive:

```python
async for event in pi_ai.stream(model, context):
    if event["type"] == "toolcall_delta":
        idx = event["content_index"]
        block = event["partial"].content[idx]
        if isinstance(block, pi_ai.ToolCall) and block.arguments:
            # Be defensive — arguments may be incomplete during streaming
            if block.name == "write_file" and block.arguments.get("path"):
                print(f"Writing to: {block.arguments['path']}")

    if event["type"] == "toolcall_end":
        tc = event["tool_call"]
        print(f"Tool complete: {tc.name}({tc.arguments})")
```

**Important notes:**
- During `toolcall_delta`, `arguments` contains the best-effort parse of partial JSON
- Fields may be missing or incomplete — always check before use
- At minimum, `arguments` is `{}`, never `None`
- At `toolcall_end`, arguments are fully parsed

### Complete Event Reference

| Event type | Description | Key fields |
|---|---|---|
| `start` | Stream begins | `partial`: initial `AssistantMessage` |
| `text_start` | Text block starts | `content_index` |
| `text_delta` | Text chunk received | `delta`, `content_index` |
| `text_end` | Text block complete | `content`, `content_index` |
| `thinking_start` | Thinking block starts | `content_index` |
| `thinking_delta` | Thinking chunk received | `delta`, `content_index` |
| `thinking_end` | Thinking block complete | `content`, `content_index` |
| `toolcall_start` | Tool call begins | `content_index` |
| `toolcall_delta` | Tool arguments streaming | `delta`, `content_index` |
| `toolcall_end` | Tool call complete | `tool_call` (fully parsed `ToolCall`) |
| `done` | Stream complete | `reason`, `message` |
| `error` | Error occurred | `reason` (`"error"` or `"aborted"`), `error` |

Use `content_index` to associate `*_start`/`*_delta`/`*_end` events with their block in `partial.content`. Events for different blocks may interleave.

## Image Input

Models with vision capabilities can process images. Check `model.input` to see if `"image"` is supported:

```python
import base64

model = pi_ai.get_model("openai", "gpt-4o-mini")

if "image" in model.input:
    print("Model supports vision")

image_bytes = open("image.png", "rb").read()

response = await pi_ai.complete(model, pi_ai.Context(
    messages=[
        pi_ai.UserMessage(content=[
            pi_ai.TextContent(text="What is in this image?"),
            pi_ai.ImageContent(
                data=base64.b64encode(image_bytes).decode(),
                mime_type="image/png",
            ),
        ])
    ]
))

for block in response.content:
    if isinstance(block, pi_ai.TextContent):
        print(block.text)
```

## Image Generation

Image generation uses a separate API surface. Use `get_image_model()` to look up a model and `generate_images()` to call it. Unlike `stream()`, this is a one-shot async call — no streaming.

```python
import asyncio
import pi_ai

async def main():
    # Discover available image providers and models
    print(pi_ai.get_image_providers())          # ['openrouter']
    for m in pi_ai.get_image_models("openrouter"):
        print(m.id, "input:", m.input, "output:", m.output)

    model = pi_ai.get_image_model("openrouter", "google/gemini-2.5-flash-image")

    result = await pi_ai.generate_images(
        model,
        pi_ai.ImagesContext(
            input=[pi_ai.TextContent(text="A small red circle on a white background.")]
        ),
        pi_ai.ImagesOptions(api_key="your-openrouter-key"),
    )

    print("stop_reason:", result.stop_reason)
    for block in result.output:
        if isinstance(block, pi_ai.TextContent):
            print("text:", block.text)
        elif isinstance(block, pi_ai.ImageContent):
            # block.data is base64-encoded, block.mime_type is e.g. "image/png"
            import base64
            open("output.png", "wb").write(base64.b64decode(block.data))
            print(f"saved image ({block.mime_type})")

    if result.usage:
        print(f"cost: ${result.usage.cost.total:.6f}")

asyncio.run(main())
```

Some models accept image input for editing or variation:

```python
import base64

input_bytes = open("input.png", "rb").read()
result = await pi_ai.generate_images(
    model,
    pi_ai.ImagesContext(input=[
        pi_ai.TextContent(text="Change the background to blue"),
        pi_ai.ImageContent(
            data=base64.b64encode(input_bytes).decode(),
            mime_type="image/png",
        ),
    ]),
)
```

Check `model.input` and `model.output` to see what a model accepts and produces:

```python
model = pi_ai.get_image_model("openrouter", "google/gemini-2.5-flash-image")
print(model.input)   # ['text', 'image']
print(model.output)  # ['image', 'text']
```

Use `generate_images_sync()` from non-async code:

```python
result = pi_ai.generate_images_sync(model, context, options)
```

## Thinking/Reasoning

Many models support extended thinking. Check `model.reasoning` to see if the model supports it.

### Unified Interface (`stream_simple`/`complete_simple`)

```python
model = pi_ai.get_model("anthropic", "claude-sonnet-4-6")

if model.reasoning:
    print("Model supports reasoning/thinking")

response = await pi_ai.complete_simple(
    model,
    pi_ai.Context(messages=[pi_ai.UserMessage(content="Solve: 2x + 5 = 13")]),
    pi_ai.SimpleStreamOptions(reasoning="medium"),
    # reasoning levels: "minimal" | "low" | "medium" | "high" | "xhigh"
)

for block in response.content:
    if isinstance(block, pi_ai.ThinkingContent):
        print("Thinking:", block.thinking)
    elif isinstance(block, pi_ai.TextContent):
        print("Response:", block.text)
```

Supported reasoning levels vary by model. Use `get_supported_thinking_levels(model)` and `clamp_thinking_level(model, level)` to query and clamp to the nearest available level.

### Streaming Thinking Content

```python
s = pi_ai.stream_simple(
    model,
    context,
    pi_ai.SimpleStreamOptions(reasoning="high"),
)

async for event in s:
    match event["type"]:
        case "thinking_start":
            print("[Model started thinking]")
        case "thinking_delta":
            print(event["delta"], end="", flush=True)
        case "thinking_end":
            print("\n[Thinking complete]")
        case "text_delta":
            print(event["delta"], end="", flush=True)
```

## Stop Reasons

Every `AssistantMessage` includes a `stop_reason` field:

- `"stop"` — Normal completion
- `"length"` — Output hit the max token limit
- `"toolUse"` — Model is calling tools and expects results
- `"error"` — An error occurred during generation
- `"aborted"` — Request was cancelled

`AssistantMessage` may also include `response_id`, a provider-specific identifier when the upstream API exposes one.

## Error Handling

Errors are surfaced as an `"error"` event; partial content received before the error is preserved:

```python
async for event in pi_ai.stream(model, context):
    if event["type"] == "error":
        # event["reason"] is "error" or "aborted"
        print(f"Error ({event['reason']}): {event['error'].error_message}")
        print("Partial content:", event["error"].content)

message = await s.result()
if message.stop_reason in ("error", "aborted"):
    print("Request failed:", message.error_message)
    print("Partial tokens:", message.usage.input, message.usage.output)
```

### Aborting Requests

Pass an `asyncio.Event` as `signal` — set it to abort:

```python
import asyncio

abort = asyncio.Event()

async def cancel_after(seconds: float) -> None:
    await asyncio.sleep(seconds)
    abort.set()

asyncio.create_task(cancel_after(2.0))

s = pi_ai.stream(
    model,
    pi_ai.Context(messages=[pi_ai.UserMessage(content="Write a long story")]),
    pi_ai.StreamOptions(signal=abort),
)

async for event in s:
    if event["type"] == "text_delta":
        print(event["delta"], end="", flush=True)
    elif event["type"] == "error":
        print(f"\n{event['reason']}: {event['error'].error_message}")
```

### Context Overflow Detection

`is_context_overflow()` detects when a request failed because the input exceeded the model's context window, using provider-neutral error message patterns (Anthropic, OpenAI, Groq, Google, and many others):

```python
message = await pi_ai.complete(model, context)

if pi_ai.is_context_overflow(message):
    print("Context too long — truncate or summarise earlier messages")
elif message.stop_reason == "error":
    print("Other error:", message.error_message)

# Pass context_window to also detect silent overflow (provider accepted
# the request but usage.input exceeded the window)
if pi_ai.is_context_overflow(message, context_window=model.context_window):
    print("Silent overflow detected via token count")
```

## APIs, Models, and Providers

Built-in API implementations:

- **`anthropic-messages`** — Anthropic Messages API
- **`openai-completions`** — OpenAI Chat Completions API (also used for OpenAI-compatible endpoints)
- **`openrouter-images`** — OpenRouter image generation API

### Providers and Models

A **provider** offers models through a specific API:
- **Anthropic** models use `anthropic-messages`
- **OpenAI** models use `openai-completions`
- **DeepSeek, Groq, Cerebras** use `openai-completions` (OpenAI-compatible)
- **OpenRouter** image models use `openrouter-images`

### Querying Providers and Models

```python
import pi_ai

# Text providers and models
print(pi_ai.get_providers())   # ['openai', 'anthropic', 'deepseek', 'groq', 'cerebras']

for model in pi_ai.get_models("anthropic"):
    print(f"{model.id}: {model.name}")
    print(f"  API: {model.api}, context: {model.context_window}, reasoning: {model.reasoning}")

model = pi_ai.get_model("openai", "gpt-4o-mini")
print(f"Using {model.name} via {model.api}")

# Image providers and models
print(pi_ai.get_image_providers())  # ['openrouter']

for m in pi_ai.get_image_models("openrouter"):
    print(f"{m.id}: input={m.input} output={m.output}")

# Check if two model references point to the same model
m1 = pi_ai.get_model("anthropic", "claude-haiku-4-5")
m2 = pi_ai.get_model("anthropic", "claude-haiku-4-5")
print(pi_ai.models_are_equal(m1, m2))  # True
```

### Custom Models

Create custom `Model` objects for local inference servers or custom endpoints:

```python
from pi_ai import Model, ModelCost, OpenAICompletionsCompat

# Ollama (OpenAI-compatible)
ollama_model = Model(
    id="llama3.1:8b",
    name="Llama 3.1 8B (Ollama)",
    api="openai-completions",
    provider="ollama",
    base_url="http://localhost:11434/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(input=0, output=0),
    context_window=128_000,
    max_tokens=32_000,
)

# LiteLLM proxy with explicit compat overrides
litellm_model = Model(
    id="gpt-4o",
    name="GPT-4o (via LiteLLM)",
    api="openai-completions",
    provider="litellm",
    base_url="http://localhost:4000/v1",
    reasoning=False,
    input=["text", "image"],
    cost=ModelCost(input=2.5, output=10),
    context_window=128_000,
    max_tokens=16_384,
    compat=OpenAICompletionsCompat(
        supports_store=False,
        max_tokens_field="max_tokens",
    ),
)

response = await pi_ai.complete(ollama_model, context, pi_ai.StreamOptions(api_key="dummy"))
```

For reasoning models on OpenAI-compatible servers that do not support `developer` role or `reasoning_effort`:

```python
custom_reasoning_model = Model(
    id="my-reasoning-model",
    name="Custom Reasoning Model",
    api="openai-completions",
    provider="my-server",
    base_url="http://localhost:8000/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(input=0, output=0),
    context_window=131_072,
    max_tokens=32_000,
    thinking_level_map={
        "minimal": None, "low": None, "medium": None,
        "high": "high",  # maps to provider's own value
        "xhigh": None,
    },
    compat=OpenAICompletionsCompat(
        supports_developer_role=False,
        supports_reasoning_effort=False,
        max_tokens_field="max_tokens",
    ),
)
```

### OpenAI Compatibility Settings

Auto-detected from `base_url` for known providers. Override via `compat` for custom endpoints:

```python
class OpenAICompletionsCompat(BaseModel):
    supports_store: bool                   # Whether provider accepts `store` field
    supports_developer_role: bool          # `developer` role vs `system` for system prompt
    supports_reasoning_effort: bool        # Whether provider accepts `reasoning_effort`
    supports_usage_in_streaming: bool      # Whether to request `stream_options.include_usage`
    supports_strict_mode: bool             # Whether to include `strict: true` in tool definitions
    max_tokens_field: str                  # "max_completion_tokens" or "max_tokens"
    requires_tool_result_name: bool        # Whether tool results need a `name` field
    requires_thinking_as_text: bool        # Convert thinking blocks to <thinking> tagged text
    requires_reasoning_content_on_assistant_messages: bool
    thinking_format: str | None            # "openai" | "openrouter" | "deepseek" | "together" | ...
```

## Context Serialization

`Context` and all message types are Pydantic models and fully JSON-serializable:

```python
import json
import pi_ai

context = pi_ai.Context(
    system_prompt="You are a helpful assistant.",
    messages=[pi_ai.UserMessage(content="What is Python?")],
)

model = pi_ai.get_model("openai", "gpt-4o-mini")
response = await pi_ai.complete(model, context)
context.messages.append(response)

# Serialize to JSON string
serialized = context.model_dump_json()
open("conversation.json", "w").write(serialized)

# Restore and continue with any model
data = json.loads(open("conversation.json").read())
restored = pi_ai.Context.model_validate(data)
restored.messages.append(pi_ai.UserMessage(content="Tell me more"))

new_model = pi_ai.get_model("anthropic", "claude-haiku-4-5")
continuation = await pi_ai.complete(new_model, restored)
```

Cross-provider handoffs work automatically — thinking blocks from a different model are converted to plain text, tool call IDs are normalized to meet each provider's format requirements, and images are replaced with placeholder text when sent to non-vision models.

## Environment Variables

Set these to avoid passing `api_key` explicitly in every call:

| Provider | Environment variable(s) |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` or `ANTHROPIC_OAUTH_TOKEN` |
| Google (Gemini) | `GEMINI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Cerebras | `CEREBRAS_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| xAI | `XAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

```python
# Uses OPENAI_API_KEY from environment automatically
model = pi_ai.get_model("openai", "gpt-4o-mini")
response = await pi_ai.complete(model, context)

# Override with an explicit key
response = await pi_ai.complete(model, context, pi_ai.StreamOptions(api_key="sk-other-key"))

# Check whether a key is configured
key = pi_ai.get_env_api_key("openai")  # returns None if not set
```

When `ANTHROPIC_OAUTH_TOKEN` is set, the Anthropic provider uses `Authorization: Bearer` instead of `x-api-key`.

## Synchronous Usage

```python
import pi_ai

model = pi_ai.get_model("anthropic", "claude-haiku-4-5")
context = pi_ai.Context(messages=[pi_ai.UserMessage(content="Hello!")])

# Text — creates a temporary event loop; do not call inside an existing one
response = pi_ai.complete_sync(model, context)
print(response.content[0].text)

# Image generation
image_model = pi_ai.get_image_model("openrouter", "google/gemini-2.5-flash-image")
result = pi_ai.generate_images_sync(image_model, pi_ai.ImagesContext(
    input=[pi_ai.TextContent(text="A blue triangle")]
))
```

For streaming from synchronous code, wrap in `asyncio.run()`:

```python
import asyncio
import pi_ai

async def _run():
    model = pi_ai.get_model("openai", "gpt-4o-mini")
    ctx = pi_ai.Context(messages=[pi_ai.UserMessage(content="Hello!")])
    async for event in pi_ai.stream(model, ctx):
        if event["type"] == "text_delta":
            print(event["delta"], end="", flush=True)

asyncio.run(_run())
```

## Credits

This package is a Python port of the [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai) TypeScript library. The original TypeScript implementation is the authoritative reference for provider behavior, streaming event protocol, cross-provider message transformation, and model catalog. The Python version is a simplified rewrite targeting text generation, tool calling, thinking/reasoning, and image generation via the same unified interface.

## License

MIT
