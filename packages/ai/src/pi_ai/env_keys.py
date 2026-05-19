from __future__ import annotations

import os

_PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "cerebras": ["CEREBRAS_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "huggingface": ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
    "github-copilot": ["COPILOT_GITHUB_TOKEN", "GITHUB_TOKEN"],
}


def get_env_api_key(provider: str) -> str | None:
    """Return the first non-empty env var for the given provider, or None."""
    for key in _PROVIDER_ENV_KEYS.get(provider, []):
        value = os.environ.get(key)
        if value:
            return value
    return None
