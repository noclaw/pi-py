from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from pi_ai import stream_simple
from pi_ai.types import (
    AssistantMessage,
    Context,
    SimpleStreamOptions,
    Tool,
    ToolResultMessage,
)
from pi_ai.validation import validate_tool_call as _pi_validate

from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
)


# ── Event stream ───────────────────────────────────────────────────────────────

class AgentEventStream:
    """Async-iterable stream of agent lifecycle events.

    Events are plain dicts with a ``type`` key. Iterate with ``async for``,
    or skip iteration and call ``await stream.result()`` to get the final
    message list directly.

    Must be used from within a running asyncio event loop.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._result: list[AgentMessage] = []
        self._result_ready = asyncio.Event()

    def push(self, event: dict[str, Any]) -> None:
        """Enqueue an event. Non-blocking."""
        self._queue.put_nowait(event)

    def end(self, result: list[AgentMessage]) -> None:
        """Close the stream with a final message list."""
        self._result = result
        self._result_ready.set()
        self._queue.put_nowait(None)  # sentinel

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def result(self) -> list[AgentMessage]:
        """Await and return the final message list."""
        await self._result_ready.wait()
        return self._result


AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]


# ── Public loop entry points ───────────────────────────────────────────────────

def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
) -> AgentEventStream:
    """Start an agent loop with new prompt messages.

    The prompts are appended to the context and events are emitted for them.
    Returns an :class:`AgentEventStream` immediately; the loop runs as a
    background asyncio task.
    """
    stream = AgentEventStream()

    async def _run() -> None:
        messages = await run_agent_loop(prompts, context, config, stream.push, signal)
        stream.end(messages)

    asyncio.create_task(_run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None = None,
) -> AgentEventStream:
    """Continue an agent loop from the current context without adding new messages.

    The last message in context must convert to a ``user`` or ``toolResult``
    message via ``convert_to_llm``.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    last = context.messages[-1]
    if isinstance(last, AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")

    stream = AgentEventStream()

    async def _run() -> None:
        messages = await run_agent_loop_continue(context, config, stream.push, signal)
        stream.end(messages)

    asyncio.create_task(_run())
    return stream


# ── Async loop implementations (testable without the stream wrapper) ───────────

async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None = None,
) -> list[AgentMessage]:
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=context.tools,
    )

    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})
    for prompt in prompts:
        await _emit(emit, {"type": "message_start", "message": prompt})
        await _emit(emit, {"type": "message_end", "message": prompt})

    await _run_loop(current_context, new_messages, config, signal, emit)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None = None,
) -> list[AgentMessage]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    last = context.messages[-1]
    if isinstance(last, AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await _emit(emit, {"type": "agent_start"})
    await _emit(emit, {"type": "turn_start"})

    await _run_loop(current_context, new_messages, config, signal, emit)
    return new_messages


# ── Core loop ──────────────────────────────────────────────────────────────────

async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> None:
    current_context = initial_context
    config = initial_config
    first_turn = True

    pending_messages: list[AgentMessage] = await _call(config.get_steering_messages) or []

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await _emit(emit, {"type": "turn_start"})
            else:
                first_turn = False

            if pending_messages:
                for msg in pending_messages:
                    await _emit(emit, {"type": "message_start", "message": msg})
                    await _emit(emit, {"type": "message_end", "message": msg})
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                pending_messages = []

            message = await _stream_assistant_response(current_context, config, signal, emit)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await _emit(emit, {"type": "turn_end", "message": message, "tool_results": []})
                await _emit(emit, {"type": "agent_end", "messages": new_messages})
                return

            tool_calls = [c for c in message.content if c.type == "toolCall"]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False

            if tool_calls:
                batch = await _execute_tool_calls(current_context, message, config, signal, emit)
                tool_results = batch.messages
                has_more_tool_calls = not batch.terminate
                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(emit, {"type": "turn_end", "message": message, "tool_results": tool_results})

            next_turn_ctx = ShouldStopAfterTurnContext(
                message=message,
                tool_results=tool_results,
                context=current_context,
                new_messages=new_messages,
            )

            if config.prepare_next_turn:
                update: AgentLoopTurnUpdate | None = await _call(config.prepare_next_turn, next_turn_ctx)
                if update:
                    if update.context is not None:
                        current_context = update.context
                    if update.model is not None or update.thinking_level is not None:
                        config = copy.copy(config)
                        if update.model is not None:
                            config.model = update.model
                        if update.thinking_level is not None:
                            config.reasoning = (
                                None if update.thinking_level == "off"
                                else update.thinking_level
                            )

            if config.should_stop_after_turn:
                stop: bool = await _call(config.should_stop_after_turn, next_turn_ctx)
                if stop:
                    await _emit(emit, {"type": "agent_end", "messages": new_messages})
                    return

            pending_messages = await _call(config.get_steering_messages) or []

        follow_up: list[AgentMessage] = await _call(config.get_follow_up_messages) or []
        if follow_up:
            pending_messages = follow_up
            continue
        break

    await _emit(emit, {"type": "agent_end", "messages": new_messages})


# ── LLM response streaming ─────────────────────────────────────────────────────

_STREAMING_EVENT_TYPES = frozenset({
    "text_start", "text_delta", "text_end",
    "thinking_start", "thinking_delta", "thinking_end",
    "toolcall_start", "toolcall_delta", "toolcall_end",
})


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> AssistantMessage:
    messages = context.messages
    if config.transform_context:
        messages = await _call(config.transform_context, messages, signal)

    llm_messages = await _call(config.convert_to_llm, messages)

    pi_tools: list[Tool] | None = None
    if context.tools:
        pi_tools = [
            Tool(name=t.name, description=t.description, parameters=t.parameters)
            for t in context.tools
        ]

    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=pi_tools,
    )

    api_key = config.api_key
    if config.get_api_key:
        resolved = await _call(config.get_api_key, config.model.provider)
        api_key = resolved or api_key

    options = SimpleStreamOptions(
        reasoning=config.reasoning,
        api_key=api_key,
        signal=signal,
        cache_retention=config.cache_retention,
        session_id=config.session_id,
        on_payload=config.on_payload,
        on_response=config.on_response,
        headers=config.headers,
        timeout_ms=config.timeout_ms,
        max_retries=config.max_retries,
        metadata=config.metadata,
        thinking_budgets=config.thinking_budgets,
    )

    response = stream_simple(config.model, llm_context, options)

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        etype = event.get("type")
        if etype == "start":
            partial_message = event["partial"]
            context.messages.append(partial_message)
            added_partial = True
            await _emit(emit, {"type": "message_start", "message": partial_message})
        elif etype in _STREAMING_EVENT_TYPES:
            if partial_message is not None:
                partial_message = event["partial"]
                context.messages[-1] = partial_message
                await _emit(emit, {
                    "type": "message_update",
                    "message": partial_message,
                    "assistant_message_event": event,
                })
        elif etype in ("done", "error"):
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await _emit(emit, {"type": "message_start", "message": final_message})
            await _emit(emit, {"type": "message_end", "message": final_message})
            return final_message

    # Fallback — loop exhausted without a terminal event (shouldn't normally happen)
    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(emit, {"type": "message_start", "message": final_message})
    await _emit(emit, {"type": "message_end", "message": final_message})
    return final_message


# ── Tool execution ─────────────────────────────────────────────────────────────

@dataclass
class _ToolBatch:
    messages: list[ToolResultMessage]
    terminate: bool


@dataclass
class _Prepared:
    tool_call: AgentToolCall
    tool: AgentTool
    args: Any


@dataclass
class _Immediate:
    result: AgentToolResult
    is_error: bool


@dataclass
class _Finalized:
    tool_call: AgentToolCall
    result: AgentToolResult
    is_error: bool


def _should_terminate(finalized: list[_Finalized]) -> bool:
    return bool(finalized) and all(f.result.terminate is True for f in finalized)


async def _execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> _ToolBatch:
    tool_calls = [c for c in assistant_message.content if c.type == "toolCall"]
    has_sequential = any(
        next((t for t in (context.tools or []) if t.name == tc.name), None) is not None
        and next((t for t in (context.tools or []) if t.name == tc.name), None).execution_mode == "sequential"  # type: ignore[union-attr]
        for tc in tool_calls
    )
    if config.tool_execution == "sequential" or has_sequential:
        return await _execute_sequential(context, assistant_message, tool_calls, config, signal, emit)
    return await _execute_parallel(context, assistant_message, tool_calls, config, signal, emit)


async def _execute_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> _ToolBatch:
    finalized_list: list[_Finalized] = []
    messages: list[ToolResultMessage] = []

    for tc in tool_calls:
        await _emit(emit, {
            "type": "tool_execution_start",
            "tool_call_id": tc.id, "tool_name": tc.name, "args": tc.arguments,
        })

        prep = await _prepare(context, assistant_message, tc, config, signal)
        if isinstance(prep, _Immediate):
            finalized = _Finalized(tool_call=tc, result=prep.result, is_error=prep.is_error)
        else:
            executed = await _run_tool(prep, signal, emit)
            finalized = await _finalize(context, assistant_message, prep, executed, config, signal)

        await _emit_end(finalized, emit)
        msg = _make_result_message(finalized)
        await _emit(emit, {"type": "message_start", "message": msg})
        await _emit(emit, {"type": "message_end", "message": msg})
        finalized_list.append(finalized)
        messages.append(msg)

    return _ToolBatch(messages=messages, terminate=_should_terminate(finalized_list))


async def _execute_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> _ToolBatch:
    # Phase 1: emit start + prepare all tool calls sequentially
    prepared_entries: list[_Finalized | _Prepared] = []
    for tc in tool_calls:
        await _emit(emit, {
            "type": "tool_execution_start",
            "tool_call_id": tc.id, "tool_name": tc.name, "args": tc.arguments,
        })
        prep = await _prepare(context, assistant_message, tc, config, signal)
        if isinstance(prep, _Immediate):
            f = _Finalized(tool_call=tc, result=prep.result, is_error=prep.is_error)
            await _emit_end(f, emit)
            prepared_entries.append(f)
        else:
            prepared_entries.append(prep)

    # Phase 2: execute prepared calls in parallel
    async def run_one(p: _Prepared) -> _Finalized:
        executed = await _run_tool(p, signal, emit)
        f = await _finalize(context, assistant_message, p, executed, config, signal)
        await _emit_end(f, emit)
        return f

    pending_indices = [i for i, e in enumerate(prepared_entries) if isinstance(e, _Prepared)]
    if pending_indices:
        results = await asyncio.gather(
            *(run_one(prepared_entries[i])  # type: ignore[arg-type]
              for i in pending_indices)
        )
        for i, result in zip(pending_indices, results):
            prepared_entries[i] = result

    resolved = [e for e in prepared_entries]  # all now _Finalized

    messages: list[ToolResultMessage] = []
    for f in resolved:
        assert isinstance(f, _Finalized)
        msg = _make_result_message(f)
        await _emit(emit, {"type": "message_start", "message": msg})
        await _emit(emit, {"type": "message_end", "message": msg})
        messages.append(msg)

    return _ToolBatch(messages=messages, terminate=_should_terminate(resolved))  # type: ignore[arg-type]


async def _prepare(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tc: AgentToolCall,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> _Prepared | _Immediate:
    tool = next((t for t in (context.tools or []) if t.name == tc.name), None)
    if tool is None:
        return _Immediate(result=_error_result(f"Tool {tc.name!r} not found"), is_error=True)

    try:
        prepared_tc = tc
        if tool.prepare_arguments:
            new_args = tool.prepare_arguments(tc.arguments)
            if new_args is not tc.arguments:
                prepared_tc = tc.model_copy(update={"arguments": new_args})

        pi_tool = Tool(name=tool.name, description=tool.description, parameters=tool.parameters)
        args = _pi_validate([pi_tool], prepared_tc)

        if config.before_tool_call:
            before_ctx = BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tc,
                args=args,
                context=context,
            )
            before_result: BeforeToolCallResult | None = await _call(
                config.before_tool_call, before_ctx, signal
            )
            if before_result and before_result.block:
                return _Immediate(
                    result=_error_result(before_result.reason or "Tool execution was blocked"),
                    is_error=True,
                )

        return _Prepared(tool_call=prepared_tc, tool=tool, args=args)
    except Exception as exc:
        return _Immediate(result=_error_result(str(exc)), is_error=True)


async def _run_tool(
    prepared: _Prepared,
    signal: asyncio.Event | None,
    emit: AgentEventSink,
) -> tuple[AgentToolResult, bool]:
    """Execute a prepared tool call. Returns (result, is_error)."""
    update_tasks: list[asyncio.Task] = []

    def on_update(partial_result: AgentToolResult) -> None:
        update_tasks.append(asyncio.create_task(
            _emit(emit, {
                "type": "tool_execution_update",
                "tool_call_id": prepared.tool_call.id,
                "tool_name": prepared.tool_call.name,
                "args": prepared.tool_call.arguments,
                "partial_result": partial_result,
            })
        ))

    try:
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            signal,
            on_update,
        )
        if update_tasks:
            await asyncio.gather(*update_tasks)
        return result, False
    except Exception as exc:
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)
        return _error_result(str(exc)), True


async def _finalize(
    context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _Prepared,
    executed: tuple[AgentToolResult, bool],
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> _Finalized:
    result, is_error = executed

    if config.after_tool_call:
        after_ctx = AfterToolCallContext(
            assistant_message=assistant_message,
            tool_call=prepared.tool_call,
            args=prepared.args,
            result=result,
            is_error=is_error,
            context=context,
        )
        try:
            override: AfterToolCallResult | None = await _call(
                config.after_tool_call, after_ctx, signal
            )
            if override:
                result = AgentToolResult(
                    content=override.content if override.content is not None else result.content,
                    details=override.details if override.details is not None else result.details,
                    terminate=override.terminate if override.terminate is not None else result.terminate,
                )
                if override.is_error is not None:
                    is_error = override.is_error
        except Exception as exc:
            result = _error_result(str(exc))
            is_error = True

    return _Finalized(tool_call=prepared.tool_call, result=result, is_error=is_error)


async def _emit_end(finalized: _Finalized, emit: AgentEventSink) -> None:
    await _emit(emit, {
        "type": "tool_execution_end",
        "tool_call_id": finalized.tool_call.id,
        "tool_name": finalized.tool_call.name,
        "result": finalized.result,
        "is_error": finalized.is_error,
    })


def _make_result_message(finalized: _Finalized) -> ToolResultMessage:
    import time as _time
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content,
        details=finalized.result.details,
        is_error=finalized.is_error,
        timestamp=int(_time.time() * 1000),
    )


def _error_result(message: str) -> AgentToolResult:
    from pi_ai.types import TextContent
    return AgentToolResult(content=[TextContent(text=message)], details={})


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _emit(sink: AgentEventSink, event: AgentEvent) -> None:
    result = sink(event)
    if asyncio.iscoroutine(result):
        await result


async def _call(fn: Callable | None, *args: Any) -> Any:
    """Call fn(*args) if fn is not None, awaiting if it returns a coroutine."""
    if fn is None:
        return None
    result = fn(*args)
    if asyncio.iscoroutine(result):
        return await result
    return result
