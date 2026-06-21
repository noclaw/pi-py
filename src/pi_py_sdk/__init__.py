"""pi-py-sdk: Python SDK for the Pi coding agent over its RPC bridge.

Drives ``pi --mode rpc`` (the well-tested ``pi-agent-core`` runtime) as a subprocess,
exposing an async Python API. No agent logic is reimplemented in Python.
"""

from __future__ import annotations

from .client import PiAgent
from .config import PiConfig
from .errors import (
    PiCommandError,
    PiError,
    PiNotStartedError,
    PiProcessError,
    PiTimeoutError,
)
from .protocol import (
    AgentEndEvent,
    AssistantMessageEvent,
    Event,
    MessageUpdateEvent,
    Response,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    parse_event,
)

__version__ = "0.0.1"

__all__ = [
    "PiAgent",
    "PiConfig",
    "PiError",
    "PiNotStartedError",
    "PiProcessError",
    "PiTimeoutError",
    "PiCommandError",
    "Event",
    "AgentEndEvent",
    "AssistantMessageEvent",
    "MessageUpdateEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionEndEvent",
    "Response",
    "parse_event",
    "__version__",
]
