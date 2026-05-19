from __future__ import annotations

from typing import Any, Optional

from .types import AssistantImages, ImagesContext, ImagesModel, ImagesOptions

_registry: dict[str, Any] = {}


def register_images_api_provider(provider: Any) -> None:
    """Register an image generation provider (must have .api and .generate_images)."""
    _registry[provider.api] = provider


def get_images_api_provider(api: str) -> Optional[Any]:
    return _registry.get(api)


def get_images_api_providers() -> list[Any]:
    return list(_registry.values())
