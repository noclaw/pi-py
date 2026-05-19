"""Quick live test — run with: uv run python test_live.py"""
import asyncio
import pi_ai


async def test_text(model_id: str, provider: str = "anthropic") -> None:
    model = pi_ai.get_model(provider, model_id)
    ctx = pi_ai.Context(
        messages=[pi_ai.UserMessage(content="Count from 1 to 5, one number per line.")]
    )
    print(f"\n=== {provider}/{model_id} — streaming text ===")
    async for event in pi_ai.stream(model, ctx):
        if event["type"] == "text_delta":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "done":
            print(f"\n→ stop_reason={event['reason']}, "
                  f"tokens={event['message'].usage.input}in/{event['message'].usage.output}out, "
                  f"cost=${event['message'].usage.cost.total:.6f}")
        elif event["type"] == "error":
            print(f"\n✗ ERROR: {event['error'].error_message}")


async def test_tools(model_id: str, provider: str = "anthropic") -> None:
    model = pi_ai.get_model(provider, model_id)
    tool = pi_ai.Tool(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    )
    ctx = pi_ai.Context(
        messages=[pi_ai.UserMessage(content="What's the weather in Tokyo?")],
        tools=[tool],
    )
    print(f"\n=== {provider}/{model_id} — tool calling ===")
    async for event in pi_ai.stream(model, ctx):
        if event["type"] == "toolcall_end":
            tc = event["tool_call"]
            print(f"→ tool call: {tc.name}({tc.arguments})")
        elif event["type"] == "done":
            print(f"→ stop_reason={event['reason']}")
        elif event["type"] == "error":
            print(f"✗ ERROR: {event['error'].error_message}")


async def test_thinking(model_id: str, provider: str = "anthropic") -> None:
    model = pi_ai.get_model(provider, model_id)
    ctx = pi_ai.Context(
        messages=[pi_ai.UserMessage(content="What is 17 * 23? Think step by step.")]
    )
    opts = pi_ai.SimpleStreamOptions(reasoning="low")
    print(f"\n=== {provider}/{model_id} — thinking (low) ===")
    async for event in pi_ai.stream_simple(model, ctx, opts):
        if event["type"] == "thinking_delta":
            print(f"[think] {event['delta']}", end="", flush=True)
        elif event["type"] == "text_delta":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "done":
            print(f"\n→ stop_reason={event['reason']}")
        elif event["type"] == "error":
            print(f"\n✗ ERROR: {event['error'].error_message}")


async def test_abort(model_id: str, provider: str = "anthropic") -> None:
    model = pi_ai.get_model(provider, model_id)
    signal = asyncio.Event()

    async def _cancel() -> None:
        await asyncio.sleep(0.15)
        signal.set()

    asyncio.create_task(_cancel())
    ctx = pi_ai.Context(
        messages=[pi_ai.UserMessage(content="Write a 500-word essay on the ocean.")]
    )
    print(f"\n=== {provider}/{model_id} — abort signal ===")
    s = pi_ai.stream(model, ctx, pi_ai.StreamOptions(signal=signal))
    async for event in s:
        if event["type"] == "text_delta":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "error":
            print(f"\n→ aborted: {event['error'].error_message[:60]}")
    msg = await s.result()
    print(f"→ stop_reason={msg.stop_reason}")
    assert msg.stop_reason == "aborted", f"Expected aborted, got {msg.stop_reason}"
    print("✓ abort test passed")


async def test_image_generation(model_id: str, provider: str = "openrouter") -> None:
    model = pi_ai.get_image_model(provider, model_id)
    ctx = pi_ai.ImagesContext(
        input=[pi_ai.TextContent(text="A small red circle on a plain white background.")]
    )
    print(f"\n=== {provider}/{model_id} — image generation ===")
    result = await pi_ai.generate_images(model, ctx)
    print(f"→ stop_reason={result.stop_reason}")
    if result.error_message:
        print(f"✗ ERROR: {result.error_message}")
        return
    for block in result.output:
        if isinstance(block, pi_ai.TextContent):
            print(f"→ text: {block.text[:80]}")
        elif isinstance(block, pi_ai.ImageContent):
            print(f"→ image: {block.mime_type}, {len(block.data)} base64 chars")
    if result.usage:
        print(f"→ cost: ${result.usage.cost.total:.6f}")


async def main() -> None:
    import os
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_OAUTH_TOKEN"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if not has_anthropic and not has_openai:
        print("Set ANTHROPIC_API_KEY, ANTHROPIC_OAUTH_TOKEN, or OPENAI_API_KEY to run live tests.")
        return

    if has_anthropic:
        await test_text("claude-haiku-4-5", "anthropic")
        await test_tools("claude-haiku-4-5", "anthropic")
        await test_thinking("claude-haiku-4-5", "anthropic")
        await test_abort("claude-haiku-4-5", "anthropic")

    if has_openai:
        await test_text("gpt-4o-mini", "openai")
        await test_tools("gpt-4o-mini", "openai")

    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    if has_openrouter:
        await test_image_generation("google/gemini-2.5-flash-image", "openrouter")
    else:
        print("\nSet OPENROUTER_API_KEY to test image generation")


if __name__ == "__main__":
    asyncio.run(main())
