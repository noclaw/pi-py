# Register built-in providers on import
from . import providers as _providers  # noqa: F401

from .registry import get_api_provider
from .stream import AssistantMessageEventStream
from .types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    SimpleStreamOptions,
    StopReason,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from .models import (
    calculate_cost,
    clamp_thinking_level,
    get_model,
    get_models,
    get_providers,
    get_supported_thinking_levels,
)


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Start streaming a model response.

    Returns an :class:`AssistantMessageEventStream` immediately and schedules
    the network call in the background. Must be called from within a running
    asyncio event loop (i.e. from ``async`` code or inside ``asyncio.run()``).

    Iterate events with ``async for event in stream(...):``, or skip iteration
    and call ``await stream(...).result()`` to get the final message.
    """
    provider = get_api_provider(model.api)
    if not provider:
        raise ValueError(f"No provider registered for api: {model.api!r}")
    return provider.stream(model, context, options)


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """Return the final :class:`AssistantMessage`, consuming the stream internally."""
    return await stream(model, context, options).result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Like :func:`stream` but accepts unified ``reasoning`` level via
    :class:`SimpleStreamOptions` instead of provider-specific options."""
    provider = get_api_provider(model.api)
    if not provider:
        raise ValueError(f"No provider registered for api: {model.api!r}")
    return provider.stream_simple(model, context, options)


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """Like :func:`complete` but accepts unified ``reasoning`` level."""
    return await stream_simple(model, context, options).result()


def complete_sync(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """Synchronous wrapper around :func:`complete`. Creates a new event loop.

    Use only from non-async code. Do not call from inside an existing event loop.
    """
    import asyncio
    return asyncio.run(complete(model, context, options))


__all__ = [
    # Core API
    "stream",
    "complete",
    "stream_simple",
    "complete_simple",
    "complete_sync",
    # Model registry
    "get_model",
    "get_models",
    "get_providers",
    "calculate_cost",
    "clamp_thinking_level",
    "get_supported_thinking_levels",
    # Types
    "AssistantMessage",
    "AssistantMessageEventStream",
    "Context",
    "ImageContent",
    "Model",
    "SimpleStreamOptions",
    "StopReason",
    "StreamOptions",
    "TextContent",
    "ThinkingContent",
    "ThinkingLevel",
    "Tool",
    "ToolCall",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
]
