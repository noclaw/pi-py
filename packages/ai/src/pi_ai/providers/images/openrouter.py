"""OpenRouter image generation provider.

Uses the OpenAI chat completions API (non-streaming) with OpenRouter's
image-capable models. The response carries generated images in
``choices[0].message.images`` as base64 data URLs alongside optional
text content in ``choices[0].message.content``.
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional

from openai import AsyncOpenAI

from ...env_keys import get_env_api_key
from ...types import (
    AssistantImages,
    ImageContent,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
    TextContent,
    Usage,
    UsageCost,
)


def _parse_usage(raw: Any, model: ImagesModel) -> Usage:
    prompt = getattr(raw, "prompt_tokens", 0) or 0
    completion = getattr(raw, "completion_tokens", 0) or 0
    details = getattr(raw, "prompt_tokens_details", None)
    cache_write = getattr(details, "cache_write_tokens", 0) or 0 if details else 0
    reported_cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
    cache_read = max(0, reported_cached - cache_write) if cache_write > 0 else reported_cached
    inp = max(0, prompt - cache_read - cache_write)
    out = completion
    m = 1_000_000
    cost = UsageCost(
        input=(model.cost.input / m) * inp,
        output=(model.cost.output / m) * out,
        cache_read=(model.cost.cache_read / m) * cache_read,
        cache_write=(model.cost.cache_write / m) * cache_write,
    )
    cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write
    return Usage(
        input=inp, output=out, cache_read=cache_read, cache_write=cache_write,
        total_tokens=inp + out + cache_read + cache_write, cost=cost,
    )


async def generate_images_openrouter(
    model: ImagesModel,
    context: ImagesContext,
    options: Optional[ImagesOptions] = None,
) -> AssistantImages:
    output = AssistantImages(
        api=model.api,
        provider=model.provider,
        model=model.id,
        timestamp=int(time.time() * 1000),
    )
    try:
        api_key = (options.api_key if options else None) or get_env_api_key(model.provider) or ""
        if not api_key:
            raise ValueError(f"No API key available for provider: {model.provider!r}")

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": model.base_url,
            "max_retries": (options.max_retries if options and options.max_retries is not None else 2),
        }
        if options and options.timeout_ms is not None:
            client_kwargs["timeout"] = options.timeout_ms / 1000
        if model.headers:
            client_kwargs["default_headers"] = model.headers

        client = AsyncOpenAI(**client_kwargs)

        # Build content parts from context
        parts: list[dict[str, Any]] = []
        for item in context.input:
            if isinstance(item, TextContent):
                parts.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
                })

        modalities = ["image", "text"] if "text" in model.output else ["image"]
        params: dict[str, Any] = {
            "model": model.id,
            "messages": [{"role": "user", "content": parts}],
            "stream": False,
            "modalities": modalities,
        }

        if options and options.on_payload:
            result = await options.on_payload(params, model)
            if result is not None:
                params = result

        extra_headers: dict[str, str] = options.headers if options and options.headers else {}

        response = await client.chat.completions.create(**params, extra_headers=extra_headers)
        output.response_id = getattr(response, "id", None)

        if response.usage:
            output.usage = _parse_usage(response.usage, model)

        choice = response.choices[0] if response.choices else None
        if choice:
            # Text content from message.content
            content = choice.message.content
            if isinstance(content, str) and content.strip():
                output.output.append(TextContent(text=content))

            # Images from message.images (OpenRouter-specific extension)
            msg_dict = choice.message.model_dump() if hasattr(choice.message, "model_dump") else {}
            msg_extra = getattr(choice.message, "model_extra", {}) or {}
            images = msg_dict.get("images") or msg_extra.get("images") or []

            for img in images:
                # img is {"image_url": "data:...;base64,..."} or {"image_url": {"url": "data:..."}}
                if isinstance(img, dict):
                    raw_url = img.get("image_url")
                    if isinstance(raw_url, dict):
                        raw_url = raw_url.get("url", "")
                    raw_url = raw_url or ""
                else:
                    continue

                if not raw_url.startswith("data:"):
                    continue
                m = re.match(r"^data:([^;]+);base64,(.+)$", raw_url, re.DOTALL)
                if not m:
                    continue
                output.output.append(ImageContent(mime_type=m.group(1), data=m.group(2)))

    except Exception as exc:
        signal = getattr(options, "signal", None)
        aborted = bool(signal and getattr(signal, "is_set", lambda: False)())
        output.stop_reason = "aborted" if aborted else "error"
        output.error_message = str(exc)

    return output


class _OpenRouterImagesProvider:
    api = "openrouter-images"

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: Optional[ImagesOptions] = None,
    ) -> AssistantImages:
        return await generate_images_openrouter(model, context, options)


openrouter_images_provider = _OpenRouterImagesProvider()
