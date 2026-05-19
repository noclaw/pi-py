"""Register all built-in API providers."""
from ..registry import register_api_provider
from .openai_completions import openai_completions_provider
from .anthropic_messages import anthropic_messages_provider

register_api_provider(openai_completions_provider)
register_api_provider(anthropic_messages_provider)
