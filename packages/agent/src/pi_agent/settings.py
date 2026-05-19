"""Settings management for pi-agent.

Reads ``~/.pi/agent/settings.json``, ``~/.pi/agent/models.json``, and
``~/.pi/agent/auth.json``.  Project-level overrides can live in ``.pi/settings.json``
anywhere on the path from ``cwd`` up to the filesystem root.

Typical usage::

    from pi_agent.settings import load_settings, get_default_model, make_auth_provider

    settings = load_settings(cwd="/my/project")
    model    = get_default_model(cwd="/my/project")      # None if unresolvable
    auth     = make_auth_provider()                       # callable for AgentHarness

    harness = AgentHarness(
        ...,
        get_api_key_and_headers=auth,
    )

Or via the one-call factory::

    harness = await create_agent(
        model=model or pi_ai.get_model("anthropic", "claude-sonnet-4-6"),
        settings_dir="~/.pi/agent",
    )
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pi_ai.types import Model, ModelCost, OpenAICompletionsCompat

# ── Paths ──────────────────────────────────────────────────────────────────────

GLOBAL_SETTINGS_DIR = Path("~/.pi/agent").expanduser()
PROJECT_SETTINGS_DIRNAME = ".pi"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Settings:
    """Merged global + project settings."""
    default_provider: str | None = None
    default_model: str | None = None


@dataclass
class _CustomModelDef:
    id: str
    name: str
    reasoning: bool
    input: list[str]
    cost: dict[str, float]
    context_window: int
    max_tokens: int


@dataclass
class _CustomProviderDef:
    base_url: str
    api: str
    api_key: str | None
    auth_header: bool
    models: list[_CustomModelDef] = field(default_factory=list)


# ── File I/O helpers ───────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _settings_dir(settings_dir: str | Path | None) -> Path:
    if settings_dir is None:
        return GLOBAL_SETTINGS_DIR
    return Path(settings_dir).expanduser()


# ── settings.json ──────────────────────────────────────────────────────────────

def load_settings(
    cwd: str | None = None,
    settings_dir: str | Path | None = None,
) -> Settings:
    """Load and merge global + project settings.

    Global settings come from ``~/.pi/agent/settings.json`` (or *settings_dir*).
    Project settings are found by walking up from *cwd* looking for
    ``.pi/settings.json``.  Project values win over global values.
    """
    global_data = _read_json(_settings_dir(settings_dir) / "settings.json")
    project_data: dict = {}

    if cwd:
        current = Path(cwd).resolve()
        while True:
            candidate = current / PROJECT_SETTINGS_DIRNAME / "settings.json"
            if candidate.is_file():
                project_data = _read_json(candidate)
                break
            parent = current.parent
            if parent == current:
                break
            current = parent

    merged = {**global_data, **project_data}
    return Settings(
        default_provider=merged.get("defaultProvider") or None,
        default_model=merged.get("defaultModel") or None,
    )


# ── models.json ────────────────────────────────────────────────────────────────

def _parse_provider(name: str, data: dict) -> _CustomProviderDef | None:
    if not isinstance(data, dict):
        return None
    raw_models = data.get("models", [])
    models: list[_CustomModelDef] = []
    for m in raw_models:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        cost = m.get("cost") or {}
        models.append(_CustomModelDef(
            id=m["id"],
            name=m.get("name", m["id"]),
            reasoning=bool(m.get("reasoning", False)),
            input=m.get("input", ["text"]),
            cost={
                "input": float(cost.get("input", 0)),
                "output": float(cost.get("output", 0)),
                "cacheRead": float(cost.get("cacheRead", 0)),
                "cacheWrite": float(cost.get("cacheWrite", 0)),
            },
            context_window=int(m.get("contextWindow", 4096)),
            max_tokens=int(m.get("maxTokens", 4096)),
        ))
    return _CustomProviderDef(
        base_url=data.get("baseUrl", ""),
        api=data.get("api", "openai-completions"),
        api_key=data.get("apiKey") or None,
        auth_header=bool(data.get("authHeader", False)),
        models=models,
    )


def _provider_to_models(name: str, provider: _CustomProviderDef) -> list[tuple[str, Model]]:
    """Convert a custom provider definition to a list of (provider_name, Model) pairs."""
    results: list[tuple[str, Model]] = []

    # When authHeader=true, the key goes into Authorization: Bearer.
    # Store it in Model.headers so the provider's SDK uses it automatically.
    default_headers: dict[str, str] | None = None
    if provider.auth_header and provider.api_key:
        default_headers = {"Authorization": f"Bearer {provider.api_key}"}

    for m in provider.models:
        compat: OpenAICompletionsCompat | None = None
        if provider.api == "openai-completions":
            compat = OpenAICompletionsCompat(
                max_tokens_field="max_tokens",
                supports_store=False,
                supports_developer_role=False,
            )

        results.append((name, Model(
            id=m.id,
            name=m.name,
            api=provider.api,
            provider=name,
            base_url=provider.base_url,
            reasoning=m.reasoning,
            input=m.input,
            cost=ModelCost(
                input=m.cost["input"],
                output=m.cost["output"],
                cache_read=m.cost["cacheRead"],
                cache_write=m.cost["cacheWrite"],
            ),
            context_window=m.context_window,
            max_tokens=m.max_tokens,
            headers=default_headers,
            compat=compat,
        )))
    return results


def load_custom_models(
    settings_dir: str | Path | None = None,
) -> list[tuple[str, Model]]:
    """Return all custom models defined in models.json.

    Each entry is ``(provider_name, Model)``.  Models are ready to pass
    directly to :func:`pi_ai.stream`, :func:`pi_ai.complete_simple`, etc.
    """
    data = _read_json(_settings_dir(settings_dir) / "models.json")
    providers_raw = data.get("providers") or {}
    results: list[tuple[str, Model]] = []
    for name, raw in providers_raw.items():
        provider = _parse_provider(name, raw)
        if provider:
            results.extend(_provider_to_models(name, provider))
    return results


def find_custom_model(
    provider: str,
    model_id: str,
    settings_dir: str | Path | None = None,
) -> Model | None:
    """Look up a specific custom model by provider and model id."""
    for pname, model in load_custom_models(settings_dir):
        if pname == provider and model.id == model_id:
            return model
    return None


# ── auth.json ──────────────────────────────────────────────────────────────────

def _load_auth_data(settings_dir: str | Path | None = None) -> dict:
    return _read_json(_settings_dir(settings_dir) / "auth.json")


def load_auth(
    provider: str,
    settings_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return auth credentials for *provider* from auth.json.

    Returns a dict with ``apiKey`` and optionally ``headers``, or ``None``
    if no valid credentials are found.

    For OAuth entries, the access token is returned as ``apiKey`` and also
    included as an ``Authorization: Bearer`` header so providers that
    require bearer auth work correctly.  Expired tokens are returned with
    a warning but not refreshed (token refresh requires an OAuth dance).
    """
    data = _load_auth_data(settings_dir)
    entry = data.get(provider)
    if not entry or not isinstance(entry, dict):
        return None

    auth_type = entry.get("type", "api_key")

    if auth_type == "oauth":
        access = entry.get("access") or ""
        expires_ms = entry.get("expires", 0)
        if not access:
            return None
        now_ms = int(time.time() * 1000)
        if expires_ms and expires_ms < now_ms:
            import warnings
            warnings.warn(
                f"OAuth token for {provider!r} expired "
                f"{(now_ms - expires_ms) // 1000}s ago. "
                "Attempting to use it anyway — refresh if calls fail.",
                stacklevel=2,
            )
        return {
            "apiKey": access,
            # Include Bearer header so providers that require it work correctly
            # even when the environment variable isn't set.
            "headers": {"Authorization": f"Bearer {access}"},
        }

    if auth_type == "api_key":
        key = entry.get("apiKey") or entry.get("key") or ""
        if not key:
            return None
        return {"apiKey": key}

    return None


# ── Default model resolution ───────────────────────────────────────────────────

def get_default_model(
    cwd: str | None = None,
    settings_dir: str | Path | None = None,
) -> Model | None:
    """Resolve the default model from settings + models catalog.

    Resolution order:
    1. Project-level ``.pi/settings.json`` defaultProvider + defaultModel
    2. Global ``~/.pi/agent/settings.json`` defaultProvider + defaultModel
    3. Custom models from models.json
    4. Built-in pi_ai catalog (via ``pi_ai.get_model``)

    Returns ``None`` when no default is configured or the model cannot be found.
    """
    settings = load_settings(cwd=cwd, settings_dir=settings_dir)
    if not settings.default_provider or not settings.default_model:
        return None

    provider = settings.default_provider
    model_id = settings.default_model

    # Try custom models first (they may override built-in catalog entries)
    custom = find_custom_model(provider, model_id, settings_dir)
    if custom:
        return custom

    # Fall back to built-in catalog
    try:
        from pi_ai import get_model
        return get_model(provider, model_id)
    except Exception:
        return None


# ── Auth provider factory ──────────────────────────────────────────────────────

def _models_json_api_key(provider: str, settings_dir: str | Path | None = None) -> str | None:
    """Return the raw apiKey for *provider* from models.json, or None."""
    data = _read_json(_settings_dir(settings_dir) / "models.json")
    provider_data = (data.get("providers") or {}).get(provider) or {}
    return provider_data.get("apiKey") or None


def make_auth_provider(
    settings_dir: str | Path | None = None,
) -> Any:
    """Return a ``get_api_key_and_headers`` callable for use with AgentHarness.

    The callable accepts a ``Model`` and returns ``{"apiKey": ..., "headers": ...}``.

    Resolution order:

    1. ``auth.json`` entry for the provider (OAuth tokens, explicitly managed keys).
    2. ``apiKey`` from the provider's ``models.json`` entry (custom local providers).
    3. ``None`` — lets pi_ai fall back to environment variables.

    Usage::

        harness = AgentHarness(
            ...,
            get_api_key_and_headers=make_auth_provider(),
        )
    """
    def _get_auth(model: Model) -> dict[str, Any] | None:
        # auth.json takes precedence
        auth = load_auth(model.provider, settings_dir)
        if auth:
            return auth
        # Fall back to the apiKey embedded in models.json for custom providers.
        # This covers local/private servers where the key lives in models.json
        # rather than auth.json.
        key = _models_json_api_key(model.provider, settings_dir)
        if key:
            return {"apiKey": key}
        return None

    return _get_auth
