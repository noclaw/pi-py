"""Pydantic models for the Pi RPC wire protocol.

Phase 0 scope: enough typing to send commands and consume the streaming events of a
prompt through to ``agent_end``. Field names intentionally match the wire format
(camelCase) for fidelity with Pi's ``rpc-types.ts`` / agent ``types.ts``.

All models allow extra fields (``extra="allow"``) so that newer Pi versions adding
fields — or event types we don't yet model — degrade gracefully instead of raising.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class _Wire(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Responses (stdout, correlated by ``id``)
# ---------------------------------------------------------------------------


class Response(_Wire):
    type: str = "response"
    id: str | None = None
    command: str
    success: bool
    data: Any = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Events (stdout, uncorrelated)
# ---------------------------------------------------------------------------


class Event(_Wire):
    """Base event. ``type`` discriminates; unknown types fall back to this class."""

    type: str


class AssistantMessageEvent(_Wire):
    """The ``assistantMessageEvent`` nested in a ``message_update``.

    Notable ``type`` values: ``text_delta``, ``thinking_delta``, ``toolcall_start``,
    ``toolcall_end``. ``delta`` is present for the ``*_delta`` variants.
    """

    type: str
    delta: str | None = None


class MessageUpdateEvent(Event):
    assistantMessageEvent: AssistantMessageEvent | None = None
    message: Any = None


class AgentEndEvent(Event):
    """Signals the end of an agent run.

    ``willRetry`` is the critical field: when true, the run is *not* actually finished —
    an ``auto_retry_*`` cycle (and a subsequent ``agent_end``) follows.
    """

    messages: list[Any] = []
    willRetry: bool = False


class ToolExecutionStartEvent(Event):
    toolCallId: str | None = None
    toolName: str | None = None
    args: Any = None


class ToolExecutionEndEvent(Event):
    toolCallId: str | None = None
    toolName: str | None = None
    result: Any = None
    isError: bool = False


_EVENT_MODELS: dict[str, type[Event]] = {
    "message_update": MessageUpdateEvent,
    "agent_end": AgentEndEvent,
    "tool_execution_start": ToolExecutionStartEvent,
    "tool_execution_end": ToolExecutionEndEvent,
}


def parse_event(data: dict[str, Any]) -> Event:
    """Parse a decoded stdout object into the most specific Event model available."""
    model = _EVENT_MODELS.get(data.get("type", ""), Event)
    return model.model_validate(data)
