from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

from .stream import AssistantMessageEventStream
from .types import Context, Model, SimpleStreamOptions, StreamOptions


class ApiProvider(Protocol):
    """Protocol every API provider must satisfy."""

    api: str

    def stream(
        self,
        model: Model,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream: ...


# Internal registry: api string -> provider
_registry: dict[str, Any] = {}


def register_api_provider(provider: Any) -> None:
    """Register a provider object (must have .api, .stream, .stream_simple)."""
    _registry[provider.api] = provider


def get_api_provider(api: str) -> Optional[Any]:
    return _registry.get(api)


def get_api_providers() -> list[Any]:
    return list(_registry.values())


def unregister_api_providers(api: str) -> None:
    _registry.pop(api, None)


def clear_api_providers() -> None:
    _registry.clear()
