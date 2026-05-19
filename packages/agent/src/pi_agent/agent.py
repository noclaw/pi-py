from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ToolResultMessage,
    Usage,
    UsageCost,
)

from .agent_loop import run_agent_loop, run_agent_loop_continue
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    PrepareNextTurnContext,
    QueueMode,
    ThinkingLevel,
    ToolExecutionMode,
)


# ── Defaults ───────────────────────────────────────────────────────────────────

def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    return [
        m for m in messages
        if isinstance(m, (dict,)) and m.get("role") in ("user", "assistant", "toolResult")
        or isinstance(m, AssistantMessage)
        or isinstance(m, ToolResultMessage)
        or (hasattr(m, "role") and m.role in ("user", "assistant", "toolResult"))
    ]


_EMPTY_USAGE = Usage(
    input=0, output=0, cache_read=0, cache_write=0, total_tokens=0,
    cost=UsageCost(input=0, output=0, cache_read=0, cache_write=0, total=0),
)


# ── Message queue ──────────────────────────────────────────────────────────────

class _PendingMessageQueue:
    def __init__(self, mode: QueueMode) -> None:
        self.mode: QueueMode = mode
        self._messages: list[AgentMessage] = []

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return bool(self._messages)

    def drain(self) -> list[AgentMessage]:
        if self.mode == "all":
            drained = list(self._messages)
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        self._messages = []


# ── Agent ──────────────────────────────────────────────────────────────────────

class Agent:
    """Stateful wrapper around the low-level agent loop.

    Owns the transcript, emits lifecycle events to subscribers, executes tools,
    and exposes queue APIs for steering and follow-up messages.

    Usage::

        agent = Agent(model=my_model, system_prompt="You are helpful.")
        agent.tools = [my_tool]
        await agent.prompt("Hello")

    Events are delivered to subscribers registered via :meth:`subscribe`.
    """

    def __init__(
        self,
        *,
        model: Model | None = None,
        system_prompt: str = "",
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool] | None = None,
        messages: list[AgentMessage] | None = None,
        convert_to_llm: Callable[[list[AgentMessage]], Awaitable[list[Message]] | list[Message]] | None = None,
        transform_context: Callable[[list[AgentMessage], Any], Awaitable[list[AgentMessage]]] | None = None,
        get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = None,
        on_payload: Any = None,
        on_response: Any = None,
        before_tool_call: Callable[[BeforeToolCallContext, Any], Awaitable[BeforeToolCallResult | None]] | None = None,
        after_tool_call: Callable[[AfterToolCallContext, Any], Awaitable[AfterToolCallResult | None]] | None = None,
        prepare_next_turn: Callable[[PrepareNextTurnContext], Awaitable[AgentLoopTurnUpdate | None] | AgentLoopTurnUpdate | None] | None = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        session_id: str | None = None,
        thinking_budgets: dict[str, int] | None = None,
        tool_execution: ToolExecutionMode = "parallel",
    ) -> None:
        from pi_ai.models import get_model

        # Mutable state
        self.system_prompt = system_prompt
        self.model: Model = model or _placeholder_model()
        self.thinking_level: ThinkingLevel = thinking_level
        self._tools: list[AgentTool] = list(tools) if tools else []
        self._messages: list[AgentMessage] = list(messages) if messages else []

        # Runtime state
        self._is_streaming = False
        self._streaming_message: AgentMessage | None = None
        self._pending_tool_calls: set[str] = set()
        self._error_message: str | None = None

        # Hooks & options
        self.convert_to_llm = convert_to_llm or _default_convert_to_llm
        self.transform_context = transform_context
        self.get_api_key = get_api_key
        self.on_payload = on_payload
        self.on_response = on_response
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.prepare_next_turn = prepare_next_turn
        self.session_id = session_id
        self.thinking_budgets = thinking_budgets
        self.tool_execution: ToolExecutionMode = tool_execution

        # Queues
        self._steering_queue = _PendingMessageQueue(steering_mode)
        self._follow_up_queue = _PendingMessageQueue(follow_up_mode)

        # Active run bookkeeping
        self._active_signal: Any | None = None  # asyncio.Event set = abort
        self._active_run_resolve: Callable[[], None] | None = None
        self._listeners: list[Callable[[AgentEvent, Any], Awaitable[None] | None]] = []

    # ── State accessors ────────────────────────────────────────────────────────

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, value: list[AgentMessage]) -> None:
        self._messages = list(value)

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def streaming_message(self) -> AgentMessage | None:
        return self._streaming_message

    @property
    def pending_tool_calls(self) -> frozenset[str]:
        return frozenset(self._pending_tool_calls)

    @property
    def error_message(self) -> str | None:
        return self._error_message

    # ── Queue API ──────────────────────────────────────────────────────────────

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    def steer(self, message: AgentMessage) -> None:
        """Queue a message to be injected after the current assistant turn."""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """Queue a message to run only after the agent would otherwise stop."""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    # ── Abort ──────────────────────────────────────────────────────────────────

    def abort(self) -> None:
        """Abort the current run, if one is active."""
        if self._active_signal is not None:
            self._active_signal.set()

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear transcript, runtime state, and queued messages."""
        self._messages = []
        self._is_streaming = False
        self._streaming_message = None
        self._pending_tool_calls = set()
        self._error_message = None
        self.clear_all_queues()

    # ── Event subscriptions ────────────────────────────────────────────────────

    def subscribe(
        self,
        listener: Callable[[AgentEvent, Any], Awaitable[None] | None],
    ) -> Callable[[], None]:
        """Subscribe to agent lifecycle events.

        Returns an unsubscribe callable. Listeners receive the event and the
        active abort signal (an ``asyncio.Event``).
        """
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    # ── Prompt / continue ──────────────────────────────────────────────────────

    async def prompt(
        self,
        message: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Start a new prompt from text, a single message, or a batch of messages."""
        if self._is_streaming:
            raise RuntimeError(
                "Agent is already processing. Use steer() or follow_up() to queue messages, "
                "or wait for completion."
            )
        messages = self._normalize_input(message, images)
        await self._run_prompts(messages)

    async def proceed(self) -> None:
        """Continue from the current transcript.

        The last message must be a user or tool-result message (equivalent to
        TypeScript's ``agent.continue()``).
        """
        if self._is_streaming:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        if not self._messages:
            raise RuntimeError("No messages to continue from")

        last = self._messages[-1]
        if isinstance(last, AssistantMessage):
            queued = self._steering_queue.drain()
            if queued:
                await self._run_prompts(queued, skip_initial_steering_poll=True)
                return
            queued = self._follow_up_queue.drain()
            if queued:
                await self._run_prompts(queued)
                return
            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _normalize_input(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(input, list):
            return input
        if isinstance(input, str):
            from pi_ai.types import UserMessage
            content: list[Any] = [TextContent(text=input)]
            if images:
                content.extend(images)
            return [UserMessage(content=content, timestamp=int(time.time() * 1000))]
        return [input]

    def _context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self.system_prompt,
            messages=list(self._messages),
            tools=list(self._tools),
        )

    def _loop_config(self, *, skip_initial_steering_poll: bool = False) -> AgentLoopConfig:
        _skip = [skip_initial_steering_poll]  # mutable cell

        def get_steering() -> list[AgentMessage]:
            if _skip[0]:
                _skip[0] = False
                return []
            return self._steering_queue.drain()

        return AgentLoopConfig(
            model=self.model,
            reasoning=None if self.thinking_level == "off" else self.thinking_level,
            session_id=self.session_id,
            on_payload=self.on_payload,
            on_response=self.on_response,
            thinking_budgets=self.thinking_budgets,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            prepare_next_turn=self.prepare_next_turn,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering,
            get_follow_up_messages=self._follow_up_queue.drain,
        )

    async def _run_prompts(
        self,
        messages: list[AgentMessage],
        *,
        skip_initial_steering_poll: bool = False,
    ) -> None:
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop(
                messages,
                self._context_snapshot(),
                self._loop_config(skip_initial_steering_poll=skip_initial_steering_poll),
                self._process_event,
                signal,
            )
        )

    async def _run_continuation(self) -> None:
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop_continue(
                self._context_snapshot(),
                self._loop_config(),
                self._process_event,
                signal,
            )
        )

    async def _run_with_lifecycle(
        self,
        executor: Callable[[Any], Awaitable[Any]],
    ) -> None:
        import asyncio

        if self._is_streaming:
            raise RuntimeError("Agent is already processing.")

        signal = asyncio.Event()
        self._active_signal = signal
        self._is_streaming = True
        self._streaming_message = None
        self._error_message = None

        try:
            await executor(signal)
        except Exception as exc:
            await self._handle_run_failure(exc, signal.is_set())
        finally:
            self._is_streaming = False
            self._streaming_message = None
            self._pending_tool_calls = set()
            self._active_signal = None

    async def _handle_run_failure(self, error: Exception, aborted: bool) -> None:
        failure: AgentMessage = AssistantMessage(
            content=[TextContent(text="")],
            api=self.model.api,
            provider=self.model.provider,
            model=self.model.id,
            usage=_EMPTY_USAGE,
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            timestamp=int(time.time() * 1000),
        )
        await self._process_event({"type": "message_start", "message": failure})
        await self._process_event({"type": "message_end", "message": failure})
        await self._process_event({"type": "turn_end", "message": failure, "tool_results": []})
        await self._process_event({"type": "agent_end", "messages": [failure]})

    async def _process_event(self, event: AgentEvent) -> None:
        etype = event.get("type")

        if etype == "message_start":
            self._streaming_message = event["message"]
        elif etype == "message_update":
            self._streaming_message = event["message"]
        elif etype == "message_end":
            self._streaming_message = None
            self._messages.append(event["message"])
        elif etype == "tool_execution_start":
            self._pending_tool_calls.add(event["tool_call_id"])
        elif etype == "tool_execution_end":
            self._pending_tool_calls.discard(event["tool_call_id"])
        elif etype == "turn_end":
            msg = event.get("message")
            if isinstance(msg, AssistantMessage) and msg.error_message:
                self._error_message = msg.error_message
        elif etype == "agent_end":
            self._streaming_message = None

        signal = self._active_signal
        for listener in list(self._listeners):
            result = listener(event, signal)
            if hasattr(result, "__await__"):
                await result


# ── Placeholder model (used when no model is supplied yet) ─────────────────────

def _placeholder_model() -> Model:
    from pi_ai.types import ModelCost
    return Model(
        id="unknown", name="unknown", api="unknown", provider="unknown",
        base_url="", reasoning=False,
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=0, max_tokens=0,
    )
