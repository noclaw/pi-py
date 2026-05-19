"""Curated image-generation model catalog (OpenRouter images API)."""
from __future__ import annotations

from .types import ImagesModel, ModelCost

# fmt: off
IMAGE_MODELS: dict[str, dict[str, ImagesModel]] = {
    "openrouter": {
        "black-forest-labs/flux.2-flex": ImagesModel(
            id="black-forest-labs/flux.2-flex",
            name="FLUX.2 Flex",
            api="openrouter-images", provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            input=["text", "image"], output=["image"],
            cost=ModelCost(input=0, output=0),
        ),
        "black-forest-labs/flux.2-pro": ImagesModel(
            id="black-forest-labs/flux.2-pro",
            name="FLUX.2 Pro",
            api="openrouter-images", provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            input=["text", "image"], output=["image"],
            cost=ModelCost(input=0, output=0),
        ),
        "google/gemini-2.5-flash-image": ImagesModel(
            id="google/gemini-2.5-flash-image",
            name="Gemini 2.5 Flash Image",
            api="openrouter-images", provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            input=["text", "image"], output=["image", "text"],
            cost=ModelCost(input=0.3, output=2.5, cache_read=0.03, cache_write=0.0833),
        ),
        "google/gemini-3-pro-image-preview": ImagesModel(
            id="google/gemini-3-pro-image-preview",
            name="Gemini 3 Pro Image Preview",
            api="openrouter-images", provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            input=["text", "image"], output=["image", "text"],
            cost=ModelCost(input=2, output=12, cache_read=0.2, cache_write=0.375),
        ),
        "google/gemini-3.1-flash-image-preview": ImagesModel(
            id="google/gemini-3.1-flash-image-preview",
            name="Gemini 3.1 Flash Image Preview",
            api="openrouter-images", provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            input=["text", "image"], output=["image", "text"],
            cost=ModelCost(input=0, output=0),
        ),
        "openai/gpt-image-1": ImagesModel(
            id="openai/gpt-image-1",
            name="GPT Image 1 (via OpenRouter)",
            api="openrouter-images", provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            input=["text", "image"], output=["image"],
            cost=ModelCost(input=0, output=0),
        ),
        "recraft-ai/recraft-v3": ImagesModel(
            id="recraft-ai/recraft-v3",
            name="Recraft V3",
            api="openrouter-images", provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            input=["text"], output=["image"],
            cost=ModelCost(input=0, output=0),
        ),
    },
}
# fmt: on
