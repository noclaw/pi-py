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


# ---------------------------------------------------------------------------
# Messages (returned by get_messages and embedded in events)
# ---------------------------------------------------------------------------
#
# Field names match pi-ai's types.ts. Content blocks are kept as raw dicts (with
# extra="allow" on the message) rather than parsed into models, so unknown block types
# survive; use ``message_text()`` to extract readable text.


class TextContent(_Wire):
    type: str = "text"
    text: str = ""


class ThinkingContent(_Wire):
    type: str = "thinking"
    thinking: str = ""


class ImageContent(_Wire):
    type: str = "image"
    data: str = ""
    mimeType: str = ""


class ToolCall(_Wire):
    type: str = "toolCall"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = {}


class UserMessage(_Wire):
    role: str = "user"
    content: Any = ""  # str | list[TextContent | ImageContent]
    timestamp: int | None = None


class AssistantMessage(_Wire):
    role: str = "assistant"
    content: list[Any] = []  # (TextContent | ThinkingContent | ToolCall)[]
    provider: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    stopReason: str | None = None
    errorMessage: str | None = None
    timestamp: int | None = None


class ToolResultMessage(_Wire):
    role: str = "toolResult"
    toolCallId: str = ""
    toolName: str = ""
    content: list[Any] = []  # (TextContent | ImageContent)[]
    isError: bool = False
    timestamp: int | None = None


class BashExecutionMessage(_Wire):
    role: str = "bashExecution"
    command: str = ""
    output: str = ""
    exitCode: int | None = None
    cancelled: bool = False
    truncated: bool = False
    timestamp: int | None = None


_MESSAGE_MODELS: dict[str, type[_Wire]] = {
    "user": UserMessage,
    "assistant": AssistantMessage,
    "toolResult": ToolResultMessage,
    "bashExecution": BashExecutionMessage,
}


def parse_message(data: dict[str, Any]) -> _Wire:
    """Parse one message dict into the model matching its ``role`` (raw dict fallback)."""
    model = _MESSAGE_MODELS.get(data.get("role", ""))
    return model.model_validate(data) if model else _Wire.model_validate(data)


def parse_messages(items: list[Any]) -> list[_Wire]:
    return [parse_message(item) if isinstance(item, dict) else item for item in items]


def _block_text(block: Any) -> str:
    block = block if isinstance(block, dict) else getattr(block, "__dict__", {})
    return block["text"] if block.get("type") == "text" and isinstance(block.get("text"), str) else ""


def message_text(message: Any) -> str:
    """Concatenate the text content of a message (handles str or block-list content)."""
    content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_block_text(block) for block in content)
    return ""


# ---------------------------------------------------------------------------
# Model streaming (pi-ai's AssistantMessageEvent, surfaced by PiModelClient)
# ---------------------------------------------------------------------------
#
# These come from the low-level ``_shim/stream.mjs`` bridge to ``@earendil-works/pi-ai``
# (not the ``pi --mode rpc`` agent). One ``StreamEvent`` per pi-ai event; the stream is
# terminated by a ``done`` or ``error`` event carrying the final assistant message.

#: Event ``type`` values that terminate a model stream.
STREAM_TERMINAL_TYPES = frozenset({"done", "error"})


class StreamEvent(_Wire):
    """One event from a model stream.

    A single permissive model (like :class:`AssistantMessageEvent`) covers the whole
    pi-ai union, discriminated on ``type``:

    * ``start`` — stream opened; ``partial`` holds the (empty) assistant message.
    * ``text_start`` / ``text_delta`` / ``text_end`` — assistant text; ``delta`` for the
      incremental piece, ``content`` for the completed block.
    * ``thinking_start`` / ``thinking_delta`` / ``thinking_end`` — reasoning, same shape.
    * ``toolcall_start`` / ``toolcall_delta`` / ``toolcall_end`` — a tool call;
      ``toolCall`` holds the finished call on ``toolcall_end``, ``delta`` carries partial
      JSON arguments on ``toolcall_delta``.
    * ``done`` — success; ``message`` holds the final :class:`AssistantMessage`.
    * ``error`` — failure (``reason`` is "error" or "aborted"); ``error`` holds the final
      message with ``stopReason``/``errorMessage`` set.

    ``partial`` (present on the non-terminal events) is always the full message so far,
    so consumers never need to accumulate deltas themselves.
    """

    type: str
    contentIndex: int | None = None
    delta: str | None = None
    content: str | None = None
    reason: str | None = None
    partial: AssistantMessage | None = None
    toolCall: ToolCall | None = None
    message: AssistantMessage | None = None
    error: AssistantMessage | None = None

    @property
    def is_terminal(self) -> bool:
        """True for the ``done``/``error`` event that ends the stream."""
        return self.type in STREAM_TERMINAL_TYPES

    @property
    def final_message(self) -> AssistantMessage | None:
        """The final assistant message on a terminal event (``message`` or ``error``)."""
        if self.type == "done":
            return self.message
        if self.type == "error":
            return self.error
        return None


def parse_stream_event(data: dict[str, Any]) -> StreamEvent:
    """Parse a decoded pi-ai stream event into a :class:`StreamEvent`."""
    return StreamEvent.model_validate(data)
