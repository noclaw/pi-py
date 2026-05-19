from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Union

from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

# ── Literals ───────────────────────────────────────────────────────────────────

ToolExecutionMode = Literal["sequential", "parallel"]
QueueMode = Literal["all", "one-at-a-time"]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

# ── Core aliases ───────────────────────────────────────────────────────────────

# A tool call content block as emitted by the assistant.
AgentToolCall = ToolCall

# AgentMessage is any LLM message; apps can treat custom message types as Any
# and filter them out in convert_to_llm.
AgentMessage = Union[UserMessage, AssistantMessage, ToolResultMessage, Any]


# ── Tool result ────────────────────────────────────────────────────────────────

@dataclass
class AgentToolResult:
    """Final or partial result produced by a tool."""
    content: list[Union[TextContent, ImageContent]]
    details: Any = None
    terminate: bool | None = None


# ── Context ────────────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    """Context snapshot passed into the agent loop. The messages list is mutated during a run."""
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] | None = None


# ── Tool definition ────────────────────────────────────────────────────────────

@dataclass
class AgentTool:
    """Tool definition used by the agent runtime."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    label: str
    execute: Callable[..., Awaitable[AgentToolResult]]
    prepare_arguments: Callable[[Any], Any] | None = None
    execution_mode: ToolExecutionMode | None = None


# ── Hook contexts ──────────────────────────────────────────────────────────────

@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    context: AgentContext


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: AgentToolCall
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[AgentMessage]


PrepareNextTurnContext = ShouldStopAfterTurnContext


# ── Hook results ───────────────────────────────────────────────────────────────

@dataclass
class AgentLoopTurnUpdate:
    context: AgentContext | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None


@dataclass
class AfterToolCallResult:
    content: list[Union[TextContent, ImageContent]] | None = None
    details: Any = None
    is_error: bool | None = None
    terminate: bool | None = None


# ── Agent loop config ──────────────────────────────────────────────────────────

@dataclass
class AgentLoopConfig:
    """Configuration for a single agent loop run."""

    model: Model

    # Required: converts AgentMessage[] to LLM-compatible Message[] before each call.
    # Filter out custom message types here; never throw.
    convert_to_llm: Callable[[list[AgentMessage]], Awaitable[list[Message]] | list[Message]] = field(
        default=None  # type: ignore[assignment]
    )

    # Optional thinking level (None or "off" means no reasoning).
    reasoning: str | None = None

    # Optional pre-LLM context transform (prune, inject, etc.). Never throw.
    transform_context: (
        Callable[[list[AgentMessage], Any], Awaitable[list[AgentMessage]]] | None
    ) = None

    # Dynamic API key resolver (e.g. for expiring OAuth tokens).
    get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = None

    # Return True to stop after the current turn.
    should_stop_after_turn: (
        Callable[[ShouldStopAfterTurnContext], Awaitable[bool] | bool] | None
    ) = None

    # Return context/model updates before the next LLM call.
    prepare_next_turn: (
        Callable[
            [PrepareNextTurnContext],
            Awaitable[AgentLoopTurnUpdate | None] | AgentLoopTurnUpdate | None,
        ]
        | None
    ) = None

    # Return messages to inject after the current turn (mid-run steering).
    get_steering_messages: (
        Callable[[], Awaitable[list[AgentMessage]]] | None
    ) = None

    # Return messages to continue after the agent would otherwise stop.
    get_follow_up_messages: (
        Callable[[], Awaitable[list[AgentMessage]]] | None
    ) = None

    tool_execution: ToolExecutionMode = "parallel"

    before_tool_call: (
        Callable[[BeforeToolCallContext, Any], Awaitable[BeforeToolCallResult | None]] | None
    ) = None

    after_tool_call: (
        Callable[[AfterToolCallContext, Any], Awaitable[AfterToolCallResult | None]] | None
    ) = None

    # Stream options forwarded to stream_simple
    api_key: str | None = None
    session_id: str | None = None
    on_payload: Any = None
    on_response: Any = None
    cache_retention: str = "short"
    headers: dict[str, str] | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    thinking_budgets: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


# ── Events ─────────────────────────────────────────────────────────────────────

# Agent events are plain dicts (same design philosophy as AssistantMessageEvent).
# Discriminated by the "type" key.
#
# agent_start      — {}
# agent_end        — {"messages": list[AgentMessage]}
# turn_start       — {}
# turn_end         — {"message": AgentMessage, "tool_results": list[ToolResultMessage]}
# message_start    — {"message": AgentMessage}
# message_update   — {"message": AgentMessage, "assistant_message_event": dict}
# message_end      — {"message": AgentMessage}
# tool_execution_start  — {"tool_call_id": str, "tool_name": str, "args": Any}
# tool_execution_update — {"tool_call_id": str, "tool_name": str, "args": Any, "partial_result": Any}
# tool_execution_end    — {"tool_call_id": str, "tool_name": str, "result": Any, "is_error": bool}

AgentEvent = dict[str, Any]
