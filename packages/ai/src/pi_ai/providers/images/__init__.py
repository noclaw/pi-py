"""Register all built-in image generation providers."""
from ...images_registry import register_images_api_provider
from .openrouter import openrouter_images_provider

register_images_api_provider(openrouter_images_provider)
