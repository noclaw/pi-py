"""pi-py-sdk: Python SDK for the Pi coding agent over its RPC bridge.

Drives ``pi --mode rpc`` (the well-tested ``pi-agent-core`` runtime) as a subprocess,
exposing an async Python API. No agent logic is reimplemented in Python.
"""

from __future__ import annotations

from .client import PiAgent, UiHandler, UiResult
from .config import PiConfig
from .errors import (
    PiCommandError,
    PiError,
    PiNotStartedError,
    PiProcessError,
    PiTimeoutError,
)
from .protocol import (
    DIALOG_METHODS,
    AgentEndEvent,
    AgentStartEvent,
    AssistantMessageEvent,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    Event,
    ExtensionErrorEvent,
    ExtensionUIRequest,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    QueueUpdateEvent,
    Response,
    SessionInfoChangedEvent,
    ThinkingLevelChangedEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    parse_event,
)

__version__ = "0.0.1"

__all__ = [
    "PiAgent",
    "PiConfig",
    "UiHandler",
    "UiResult",
    "PiError",
    "PiNotStartedError",
    "PiProcessError",
    "PiTimeoutError",
    "PiCommandError",
    "Event",
    "AgentStartEvent",
    "AgentEndEvent",
    "AssistantMessageEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "TurnEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
    "QueueUpdateEvent",
    "CompactionStartEvent",
    "CompactionEndEvent",
    "AutoRetryStartEvent",
    "AutoRetryEndEvent",
    "SessionInfoChangedEvent",
    "ThinkingLevelChangedEvent",
    "ExtensionErrorEvent",
    "ExtensionUIRequest",
    "DIALOG_METHODS",
    "Response",
    "parse_event",
    "__version__",
]
