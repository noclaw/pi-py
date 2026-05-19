from __future__ import annotations

from .models_catalog import MODELS
from .types import Model, ModelThinkingLevel, Usage, THINKING_LEVELS


def get_model(provider: str, model_id: str) -> Model:
    """Return the Model for the given provider and model ID."""
    provider_models = MODELS.get(provider)
    if not provider_models:
        raise KeyError(f"Unknown provider: {provider!r}")
    model = provider_models.get(model_id)
    if not model:
        raise KeyError(f"Unknown model {model_id!r} for provider {provider!r}")
    return model


def get_models(provider: str) -> list[Model]:
    """Return all models for the given provider."""
    return list(MODELS.get(provider, {}).values())


def get_providers() -> list[str]:
    """Return all known provider names."""
    return list(MODELS.keys())


def calculate_cost(model: Model, usage: Usage) -> Usage:
    """Compute and populate ``usage.cost`` in-place, then return ``usage``."""
    m = 1_000_000
    usage.cost.input = (model.cost.input / m) * usage.input
    usage.cost.output = (model.cost.output / m) * usage.output
    usage.cost.cache_read = (model.cost.cache_read / m) * usage.cache_read
    usage.cost.cache_write = (model.cost.cache_write / m) * usage.cache_write
    usage.cost.total = (
        usage.cost.input
        + usage.cost.output
        + usage.cost.cache_read
        + usage.cost.cache_write
    )
    return usage


def get_supported_thinking_levels(model: Model) -> list[ModelThinkingLevel]:
    """Return the thinking levels the model supports.

    Mirrors TS logic: a level mapped to None is unsupported; xhigh requires
    an explicit non-None mapping; all other levels default to supported.
    """
    if not model.reasoning:
        return ["off"]
    level_map = model.thinking_level_map or {}
    result: list[ModelThinkingLevel] = []
    for level in THINKING_LEVELS:
        mapped = level_map.get(level, "UNSET")
        if mapped is None:
            continue  # explicitly None = unsupported
        if level == "xhigh" and mapped == "UNSET":
            continue  # xhigh only included when explicitly mapped
        result.append(level)
    return result


def clamp_thinking_level(model: Model, level: ModelThinkingLevel) -> ModelThinkingLevel:
    """Return the nearest supported thinking level, clamping up then down."""
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    idx = THINKING_LEVELS.index(level) if level in THINKING_LEVELS else -1
    if idx == -1:
        return available[0] if available else "off"
    # Try higher levels first, then lower
    for candidate in THINKING_LEVELS[idx:]:
        if candidate in available:
            return candidate
    for candidate in reversed(THINKING_LEVELS[:idx]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"
