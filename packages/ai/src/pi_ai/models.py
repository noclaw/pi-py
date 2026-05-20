from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Optional

from .types import (
    AnthropicMessagesCompat,
    AssistantMessage,
    Model,
    ModelCost,
    ModelThinkingLevel,
    OpenAICompletionsCompat,
    Usage,
    THINKING_LEVELS,
)

# ── JSON catalog loader ────────────────────────────────────────────────────────

def _compat_from_dict(api: str, data: dict) -> OpenAICompletionsCompat | AnthropicMessagesCompat | None:
    if not data:
        return None
    if api == "openai-completions":
        return OpenAICompletionsCompat(
            supports_store=data.get("supportsStore", False),
            supports_developer_role=data.get("supportsDeveloperRole", False),
            supports_reasoning_effort=data.get("supportsReasoningEffort", False),
            supports_usage_in_streaming=data.get("supportsUsageInStreaming", True),
            max_tokens_field=data.get("maxTokensField", "max_completion_tokens"),
            requires_tool_result_name=data.get("requiresToolResultName", False),
            requires_assistant_after_tool_result=data.get("requiresAssistantAfterToolResult", False),
            requires_thinking_as_text=data.get("requiresThinkingAsText", False),
            requires_reasoning_content_on_assistant_messages=data.get(
                "requiresReasoningContentOnAssistantMessages", False
            ),
            thinking_format=data.get("thinkingFormat"),
            supports_strict_mode=data.get("supportsStrictMode", True),
        )
    if api == "anthropic-messages":
        return AnthropicMessagesCompat(
            supports_eager_tool_input_streaming=data.get("supportsEagerToolInputStreaming", True),
            supports_long_cache_retention=data.get("supportsLongCacheRetention", True),
            send_session_affinity_headers=data.get("sendSessionAffinityHeaders", False),
            supports_cache_control_on_tools=data.get("supportsCacheControlOnTools", True),
        )
    return None


def _models_from_json(data: dict) -> dict[str, dict[str, Model]]:
    """Parse a models.json dict into the {provider: {model_id: Model}} registry."""
    result: dict[str, dict[str, Model]] = {}
    for provider_name, pdata in (data.get("providers") or {}).items():
        api: str = pdata.get("api", "openai-completions")
        base_url: str = pdata.get("baseUrl", "")
        api_key: str | None = pdata.get("apiKey") or None
        auth_header: bool = bool(pdata.get("authHeader", False))
        provider_compat_data: dict = pdata.get("compat") or {}

        default_headers: dict[str, str] | None = (
            {"Authorization": f"Bearer {api_key}"} if auth_header and api_key else None
        )

        result[provider_name] = {}
        for m in pdata.get("models") or []:
            if not m.get("id"):
                continue
            # Model-level compat overrides provider-level compat
            compat_data = {**provider_compat_data, **(m.get("compat") or {})}
            compat = _compat_from_dict(api, compat_data) if compat_data else None
            cost_raw = m.get("cost") or {}
            result[provider_name][m["id"]] = Model(
                id=m["id"],
                name=m.get("name", m["id"]),
                api=api,
                provider=provider_name,
                base_url=base_url,
                reasoning=bool(m.get("reasoning", False)),
                input=m.get("input", ["text"]),
                cost=ModelCost(
                    input=float(cost_raw.get("input", 0)),
                    output=float(cost_raw.get("output", 0)),
                    cache_read=float(cost_raw.get("cacheRead", 0)),
                    cache_write=float(cost_raw.get("cacheWrite", 0)),
                ),
                context_window=int(m.get("contextWindow", 4096)),
                max_tokens=int(m.get("maxTokens", 4096)),
                thinking_level_map=m.get("thinkingLevelMap") or None,
                headers=m.get("headers") or default_headers,
                compat=compat,
                hint=m.get("hint") or None,
            )
    return result


def _load_catalog() -> dict[str, dict[str, Model]]:
    """Load model catalog from JSON.

    Always loads the bundled ``pi_ai/models.json`` as the base catalog.
    If ``~/.pi-py/models.json`` exists, its providers are merged on top:
    new providers are added; existing providers are replaced wholesale.
    This lets users add local servers without touching the built-in entries.
    """
    catalog: dict[str, dict[str, Model]] = {}

    # Base: bundled catalog
    try:
        pkg_file = importlib.resources.files("pi_ai").joinpath("models.json")
        catalog = _models_from_json(json.loads(pkg_file.read_text(encoding="utf-8")))
    except Exception:
        pass

    # Overlay: user additions / overrides from ~/.pi-py/models.json
    user_path = Path("~/.pi-py/models.json").expanduser()
    if user_path.is_file():
        try:
            user_models = _models_from_json(json.loads(user_path.read_text(encoding="utf-8")))
            catalog.update(user_models)  # per-provider override
        except Exception:
            pass

    return catalog


MODELS: dict[str, dict[str, Model]] = _load_catalog()


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


def models_are_equal(
    a: Optional[Model],
    b: Optional[Model],
) -> bool:
    """Return True when two models share the same provider and ID."""
    if not a or not b:
        return False
    return a.id == b.id and a.provider == b.provider


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


# ── Context overflow detection ─────────────────────────────────────────────────

import re as _re

_OVERFLOW_PATTERNS = [
    _re.compile(r"prompt is too long", _re.I),
    _re.compile(r"request_too_large", _re.I),
    _re.compile(r"input is too long for requested model", _re.I),
    _re.compile(r"exceeds the context window", _re.I),
    _re.compile(r"exceeds (?:the )?(?:model'?s )?maximum context length of [\d,]+ tokens?", _re.I),
    _re.compile(r"input token count.*exceeds the maximum", _re.I),
    _re.compile(r"maximum prompt length is \d+", _re.I),
    _re.compile(r"reduce the length of the messages", _re.I),
    _re.compile(r"maximum context length is \d+ tokens", _re.I),
    _re.compile(r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)", _re.I),
    _re.compile(r"exceeds the limit of \d+", _re.I),
    _re.compile(r"exceeds the available context size", _re.I),
    _re.compile(r"greater than the context length", _re.I),
    _re.compile(r"context window exceeds limit", _re.I),
    _re.compile(r"exceeded model token limit", _re.I),
    _re.compile(r"too large for model with \d+ maximum context length", _re.I),
    _re.compile(r"model_context_window_exceeded", _re.I),
    _re.compile(r"prompt too long; exceeded (?:max )?context length", _re.I),
    _re.compile(r"context[_ ]length[_ ]exceeded", _re.I),
    _re.compile(r"too many tokens", _re.I),
    _re.compile(r"token limit exceeded", _re.I),
    _re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", _re.I),
]

_NON_OVERFLOW_PATTERNS = [
    _re.compile(r"^(Throttling error|Service unavailable):", _re.I),
    _re.compile(r"rate limit", _re.I),
    _re.compile(r"too many requests", _re.I),
]


def is_context_overflow(message: AssistantMessage, context_window: Optional[int] = None) -> bool:
    """Return True when the message indicates a context-window overflow.

    Handles three cases:
    1. **Error overflow** — ``stop_reason == "error"`` with a provider error message
       matching known overflow patterns (Anthropic, OpenAI, Groq, Google, etc.).
    2. **Silent overflow** — provider accepted the request but ``usage.input``
       exceeds ``context_window`` (e.g. z.ai).
    3. **Length-stop overflow** — provider truncated input to fit the window,
       leaving no room for output: ``stop_reason == "length"`` + ``usage.output == 0``
       + input fills ≥ 99 % of the window (e.g. Xiaomi MiMo).
    """
    if message.stop_reason == "error" and message.error_message:
        is_non_overflow = any(p.search(message.error_message) for p in _NON_OVERFLOW_PATTERNS)
        if not is_non_overflow and any(p.search(message.error_message) for p in _OVERFLOW_PATTERNS):
            return True

    if context_window and message.stop_reason == "stop":
        if (message.usage.input + message.usage.cache_read) > context_window:
            return True

    if context_window and message.stop_reason == "length" and message.usage.output == 0:
        if (message.usage.input + message.usage.cache_read) >= context_window * 0.99:
            return True

    return False
