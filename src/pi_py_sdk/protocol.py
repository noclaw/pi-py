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


class ToolExecutionUpdateEvent(Event):
    toolCallId: str | None = None
    toolName: str | None = None
    args: Any = None
    partialResult: Any = None


class AgentStartEvent(Event):
    pass


class TurnEndEvent(Event):
    message: Any = None
    toolResults: list[Any] = []


class MessageStartEvent(Event):
    message: Any = None


class MessageEndEvent(Event):
    message: Any = None


class QueueUpdateEvent(Event):
    """Pending steering / follow-up queues changed."""

    steering: list[str] = []
    followUp: list[str] = []


class CompactionStartEvent(Event):
    reason: str | None = None  # "manual" | "threshold" | "overflow"


class CompactionEndEvent(Event):
    reason: str | None = None
    result: Any = None
    aborted: bool = False
    willRetry: bool = False
    errorMessage: str | None = None


class AutoRetryStartEvent(Event):
    attempt: int | None = None
    maxAttempts: int | None = None
    delayMs: int | None = None
    errorMessage: str | None = None


class AutoRetryEndEvent(Event):
    success: bool = False
    attempt: int | None = None
    finalError: str | None = None


class SessionInfoChangedEvent(Event):
    name: str | None = None


class ThinkingLevelChangedEvent(Event):
    level: str | None = None


class ExtensionErrorEvent(Event):
    extensionPath: str | None = None
    event: str | None = None
    error: Any = None


_EVENT_MODELS: dict[str, type[Event]] = {
    "agent_start": AgentStartEvent,
    "agent_end": AgentEndEvent,
    "turn_end": TurnEndEvent,
    "message_start": MessageStartEvent,
    "message_update": MessageUpdateEvent,
    "message_end": MessageEndEvent,
    "tool_execution_start": ToolExecutionStartEvent,
    "tool_execution_update": ToolExecutionUpdateEvent,
    "tool_execution_end": ToolExecutionEndEvent,
    "queue_update": QueueUpdateEvent,
    "compaction_start": CompactionStartEvent,
    "compaction_end": CompactionEndEvent,
    "auto_retry_start": AutoRetryStartEvent,
    "auto_retry_end": AutoRetryEndEvent,
    "session_info_changed": SessionInfoChangedEvent,
    "thinking_level_changed": ThinkingLevelChangedEvent,
    "extension_error": ExtensionErrorEvent,
}


def parse_event(data: dict[str, Any]) -> Event:
    """Parse a decoded stdout object into the most specific Event model available."""
    model = _EVENT_MODELS.get(data.get("type", ""), Event)
    return model.model_validate(data)


# ---------------------------------------------------------------------------
# Extension UI sub-protocol (interactive dialogs / approvals)
# ---------------------------------------------------------------------------

#: Methods that block the agent until the client sends an extension_ui_response.
DIALOG_METHODS = frozenset({"select", "confirm", "input", "editor"})


class ExtensionUIRequest(_Wire):
    """A request emitted when an extension needs user interaction.

    ``method`` discriminates. Dialog methods (see ``DIALOG_METHODS``) expect a reply
    via ``extension_ui_response``; the rest (notify/setStatus/setWidget/setTitle/
    set_editor_text) are fire-and-forget.
    """

    type: str = "extension_ui_request"
    id: str
    method: str
    title: str | None = None
    message: str | None = None
    options: list[str] | None = None
    placeholder: str | None = None
    prefill: str | None = None
    timeout: int | None = None
    notifyType: str | None = None
