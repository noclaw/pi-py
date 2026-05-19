from __future__ import annotations

from .image_models_catalog import IMAGE_MODELS
from .types import ImagesModel


def get_image_model(provider: str, model_id: str) -> ImagesModel:
    """Return the ImagesModel for the given provider and model ID."""
    provider_models = IMAGE_MODELS.get(provider)
    if not provider_models:
        raise KeyError(f"Unknown image provider: {provider!r}")
    model = provider_models.get(model_id)
    if not model:
        raise KeyError(f"Unknown image model {model_id!r} for provider {provider!r}")
    return model


def get_image_models(provider: str) -> list[ImagesModel]:
    """Return all image models for the given provider."""
    return list(IMAGE_MODELS.get(provider, {}).values())


def get_image_providers() -> list[str]:
    """Return all known image provider names."""
    return list(IMAGE_MODELS.keys())
