"""Quick live test — run with: uv run python test_live.py"""
import asyncio
import pi_ai
import pi_agent


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


### ── Agent live tests ────────────────────────────────────────────────────────

async def test_agent_tool_loop(model_id: str, provider: str = "anthropic") -> None:
    """Agent runs a tool call, receives the result, and produces a final answer."""
    model = pi_ai.get_model(provider, model_id)

    call_count = 0

    tool = pi_agent.AgentTool(
        name="add",
        description="Add two integers.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        label="Add",
        execute=_add_tool,
    )

    async def _count_calls(event: dict, _) -> None:
        nonlocal call_count
        if event["type"] == "tool_execution_end":
            call_count += 1

    agent = pi_agent.Agent(
        model=model,
        system_prompt="You are a calculator assistant. Use the add tool when asked to add numbers.",
        tools=[tool],
    )
    agent.subscribe(_count_calls)

    print(f"\n=== {provider}/{model_id} — agent tool loop ===")
    await agent.prompt("What is 17 + 25?")

    msgs = agent.messages
    final_assistant = next((m for m in reversed(msgs) if isinstance(m, pi_ai.AssistantMessage)), None)
    assert final_assistant is not None, "No assistant message"
    assert final_assistant.stop_reason in ("stop", "toolUse", "length"), f"bad stop: {final_assistant.stop_reason}"
    assert call_count >= 1, "Expected at least one tool call"
    text = " ".join(b.text for b in final_assistant.content if hasattr(b, "text"))
    print(f"→ tool calls: {call_count}, final reply: {text[:80]}")
    print(f"→ stop_reason={final_assistant.stop_reason}  messages={len(msgs)}")
    print("✓ agent tool loop passed")


async def _add_tool(_, params: dict, *_args) -> pi_agent.AgentToolResult:
    result = params["a"] + params["b"]
    return pi_agent.AgentToolResult(
        content=[pi_ai.TextContent(text=str(result))],
        details={"sum": result},
    )


async def test_agent_streaming_events(model_id: str, provider: str = "anthropic") -> None:
    """Verify agent_loop yields the expected event sequence for a no-tool prompt."""
    model = pi_ai.get_model(provider, model_id)

    agent = pi_agent.Agent(model=model, system_prompt="You are concise.")

    events: list[str] = []

    def _record(event: dict, *_) -> None:
        events.append(event["type"])

    agent.subscribe(_record)

    print(f"\n=== {provider}/{model_id} — agent event sequence ===")
    await agent.prompt("Say exactly: hello")

    expected_sequence = ["agent_start", "turn_start", "message_start", "message_end", "turn_end", "agent_end"]
    for expected in expected_sequence:
        assert expected in events, f"Missing event: {expected} (got {events})"
    assert events[0] == "agent_start", f"First event should be agent_start, got {events[0]}"
    assert events[-1] == "agent_end", f"Last event should be agent_end, got {events[-1]}"
    print(f"→ event sequence: {events}")
    print("✓ agent event sequence passed")


async def test_agent_abort(model_id: str, provider: str = "anthropic") -> None:
    """Abort an in-flight agent run and verify the stream stops cleanly."""
    model = pi_ai.get_model(provider, model_id)
    agent = pi_agent.Agent(model=model, system_prompt="You are verbose.")

    ended = asyncio.Event()

    def _on_event(event: dict, *_) -> None:
        if event["type"] == "agent_end":
            ended.set()

    agent.subscribe(_on_event)

    print(f"\n=== {provider}/{model_id} — agent abort ===")

    async def _prompt_and_abort() -> None:
        prompt_task = asyncio.create_task(
            agent.prompt("Write a 500-word essay on mountains.")
        )
        await asyncio.sleep(0.2)
        agent.abort()
        await prompt_task

    await _prompt_and_abort()
    await asyncio.wait_for(ended.wait(), timeout=10)

    final = next((m for m in reversed(agent.messages) if isinstance(m, pi_ai.AssistantMessage)), None)
    assert final is not None
    assert final.stop_reason == "aborted", f"Expected aborted, got {final.stop_reason}"
    assert not agent.is_streaming
    print(f"→ stop_reason={final.stop_reason}, messages={len(agent.messages)}")
    print("✓ agent abort passed")


async def test_agent_harness_session(model_id: str, provider: str = "anthropic") -> None:
    """AgentHarness persists messages to session and rebuilds context correctly."""
    import tempfile
    model = pi_ai.get_model(provider, model_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        env = pi_agent.PythonExecutionEnv(cwd=tmpdir)
        repo = pi_agent.InMemorySessionRepo()
        session = await repo.create()

        harness = pi_agent.AgentHarness(
            env=env,
            session=session,
            model=model,
            system_prompt="You are a helpful assistant. Keep replies short.",
        )

        print(f"\n=== {provider}/{model_id} — AgentHarness session persistence ===")

        # First turn
        reply1 = await harness.prompt("My name is Alice. Say hi.")
        text1 = " ".join(b.text for b in reply1.content if hasattr(b, "text"))
        print(f"→ turn 1: {text1[:80]}")
        assert reply1.stop_reason in ("stop", "length")

        # Second turn — model must recall the name from session context
        reply2 = await harness.prompt("What is my name?")
        text2 = " ".join(b.text for b in reply2.content if hasattr(b, "text"))
        print(f"→ turn 2: {text2[:80]}")
        assert reply2.stop_reason in ("stop", "length")
        assert "alice" in text2.lower(), f"Expected 'Alice' in reply, got: {text2[:120]}"

        # Verify session stored the messages
        branch = await session.get_branch()
        msg_entries = [e for e in branch if e.get("type") == "message"]
        assert len(msg_entries) >= 4, f"Expected ≥4 messages in session, got {len(msg_entries)}"

        print(f"→ session entries: {len(branch)}, message entries: {len(msg_entries)}")
        print("✓ AgentHarness session persistence passed")


async def test_agent_follow_up(model_id: str, provider: str = "anthropic") -> None:
    """Follow-up queue resumes agent after it would otherwise stop."""
    model = pi_ai.get_model(provider, model_id)

    agent = pi_agent.Agent(model=model, system_prompt="You are concise.")
    agent.follow_up_mode = "one-at-a-time"

    turn_count = 0

    def _on_event(event: dict, *_) -> None:
        nonlocal turn_count
        if event["type"] == "turn_start":
            turn_count += 1

    agent.subscribe(_on_event)

    # Queue a follow-up before starting — agent will process it after the first stop
    agent.follow_up(pi_ai.UserMessage(content="Now say: goodbye"))

    print(f"\n=== {provider}/{model_id} — agent follow-up queue ===")
    await agent.prompt("Say: hello")

    assert turn_count >= 2, f"Expected ≥2 turns (initial + follow-up), got {turn_count}"
    msgs = agent.messages
    final = next((m for m in reversed(msgs) if isinstance(m, pi_ai.AssistantMessage)), None)
    assert final is not None
    print(f"→ turns: {turn_count}, total messages: {len(msgs)}")
    print(f"→ final reply: {' '.join(b.text for b in final.content if hasattr(b, 'text'))[:80]}")
    print("✓ agent follow-up queue passed")


async def test_create_agent(model_id: str, provider: str = "anthropic") -> None:
    """create_agent() wires tools, context, and session; agent uses them in one prompt."""
    import tempfile
    from pathlib import Path

    model = pi_ai.get_model(provider, model_id)

    print(f"\n=== {provider}/{model_id} — create_agent() end-to-end ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Plant a CLAUDE.md so context loading is exercised
        Path(tmpdir, "CLAUDE.md").write_text(
            "# Test Project\nAll scripts must print their output to stdout."
        )

        harness = await pi_agent.create_agent(
            model=model,
            cwd=tmpdir,
            system_prompt=(
                "You are a coding assistant. Complete tasks using the available tools. "
                "Be concise — use tools directly without explanation."
            ),
        )

        # Verify wiring
        assert len(harness._tools) == 7, f"Expected 7 tools, got {len(harness._tools)}"
        assert harness._auto_compact is True

        tool_calls_made: list[str] = []

        def _record(event: dict, *_) -> None:
            if event["type"] == "tool_execution_end":
                tool_calls_made.append(event["tool_name"])

        harness.subscribe(_record)

        # Task requires: write → bash (verify execution)
        reply = await harness.prompt(
            "Create a file called hello.py containing exactly:\n\n"
            "    print('hello world')\n\n"
            "Then run it with bash and tell me what it printed."
        )

        text = " ".join(
            b.text for b in reply.content if hasattr(b, "text")
        ).lower()

        assert reply.stop_reason in ("stop", "length"), f"bad stop: {reply.stop_reason}"
        assert "write" in tool_calls_made, f"Expected 'write' tool call, got: {tool_calls_made}"
        assert "bash" in tool_calls_made, f"Expected 'bash' tool call, got: {tool_calls_made}"
        assert "hello world" in text, f"Expected 'hello world' in reply, got: {text[:200]}"

        # Verify session persisted the conversation
        branch = await harness._session.get_branch()
        msg_entries = [e for e in branch if e.get("type") == "message"]
        assert len(msg_entries) >= 2, f"Expected ≥2 session entries, got {len(msg_entries)}"

        print(f"→ tools used: {tool_calls_made}")
        print(f"→ stop_reason={reply.stop_reason}, session entries={len(msg_entries)}")
        print(f"→ reply: {text[:100]}")
        print("✓ create_agent() end-to-end passed")


async def main() -> None:
    import os
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_OAUTH_TOKEN"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))

    if not has_anthropic and not has_openai and not has_openrouter:
        print("Set ANTHROPIC_API_KEY, ANTHROPIC_OAUTH_TOKEN, OPENAI_API_KEY, or OPENROUTER_API_KEY to run live tests.")
        return

    if has_anthropic:
        await test_text("claude-haiku-4-5", "anthropic")
        await test_tools("claude-haiku-4-5", "anthropic")
        await test_thinking("claude-haiku-4-5", "anthropic")
        await test_abort("claude-haiku-4-5", "anthropic")
        # Agent tests
        await test_agent_streaming_events("claude-haiku-4-5", "anthropic")
        await test_agent_tool_loop("claude-haiku-4-5", "anthropic")
        await test_agent_abort("claude-haiku-4-5", "anthropic")
        await test_agent_follow_up("claude-haiku-4-5", "anthropic")
        await test_agent_harness_session("claude-haiku-4-5", "anthropic")
        await test_create_agent("claude-haiku-4-5", "anthropic")

    if has_openai:
        await test_text("gpt-4o-mini", "openai")
        await test_tools("gpt-4o-mini", "openai")
        # Agent tests
        await test_agent_streaming_events("gpt-4o-mini", "openai")
        await test_agent_tool_loop("gpt-4o-mini", "openai")
        await test_create_agent("gpt-4o-mini", "openai")

    if has_openrouter:
        await test_image_generation("google/gemini-2.5-flash-image", "openrouter")
    else:
        print("\nSet OPENROUTER_API_KEY to test image generation")


if __name__ == "__main__":
    asyncio.run(main())
