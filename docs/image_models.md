# Image Generation Models

pi-py supports image generation through a separate model type (`ImagesModel`)
and API (`generate_images()`), distinct from the text-generation models covered
in [models.md](models.md).

---

## Quick start

```python
import asyncio
import pi_ai

async def main():
    model = pi_ai.get_image_model("openrouter", "google/gemini-2.5-flash-image")

    result = await pi_ai.generate_images(
        model,
        pi_ai.ImagesContext(
            input=[pi_ai.TextContent(text="A small red circle on a white background.")]
        ),
    )

    print("stop_reason:", result.stop_reason)
    for block in result.output:
        if isinstance(block, pi_ai.ImageContent):
            import base64
            open("output.png", "wb").write(base64.b64decode(block.data))
            print(f"saved image ({block.mime_type})")
        elif isinstance(block, pi_ai.TextContent):
            print("text:", block.text)

asyncio.run(main())
```

---

## Built-in image models

Image models are loaded from `packages/ai/src/pi_ai/image_models.json` (bundled)
and merged with `~/.pi-py/image_models.json` (user additions).

```bash
python -c "import pi_ai; [print(p, m.id) for p in pi_ai.get_image_providers() for m in pi_ai.get_image_models(p)]"
```

Current bundled models (all via OpenRouter):

| Model ID | Name | Input | Output |
|---|---|---|---|
| `black-forest-labs/flux.2-flex` | FLUX.2 Flex | text, image | image |
| `black-forest-labs/flux.2-pro` | FLUX.2 Pro | text, image | image |
| `google/gemini-2.5-flash-image` | Gemini 2.5 Flash Image | text, image | image, text |
| `google/gemini-3-pro-image-preview` | Gemini 3 Pro Image Preview | text, image | image, text |
| `google/gemini-3.1-flash-image-preview` | Gemini 3.1 Flash Image Preview | text, image | image, text |
| `openai/gpt-image-1` | GPT Image 1 (via OpenRouter) | text, image | image |
| `recraft-ai/recraft-v3` | Recraft V3 | text | image |

---

## Image input (editing / variation)

Models that accept `"image"` in their `input` list support image editing:

```python
import base64

image_bytes = open("input.png", "rb").read()

result = await pi_ai.generate_images(
    model,
    pi_ai.ImagesContext(input=[
        pi_ai.TextContent(text="Change the background to blue"),
        pi_ai.ImageContent(
            data=base64.b64encode(image_bytes).decode(),
            mime_type="image/png",
        ),
    ]),
)
```

Check `model.input` before passing images:

```python
if "image" in model.input:
    print("Model supports image input")
```

---

## API key

All current built-in image models go through OpenRouter.
Set `OPENROUTER_API_KEY` or add an `openrouter` entry to `~/.pi-py/auth.json`.

---

## Adding custom image models

Add image models from any provider to `~/.pi-py/image_models.json`.
The file has the same provider/models structure as `~/.pi-py/models.json`
but uses image-specific fields.

### `~/.pi-py/image_models.json` format

```json
{
  "providers": {
    "PROVIDER_NAME": {
      "api": "openrouter-images",
      "baseUrl": "https://openrouter.ai/api/v1",
      "models": [
        {
          "id": "model-id",
          "name": "Human-readable name",
          "input": ["text"],
          "output": ["image"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

### Provider fields

| Field | Description |
|---|---|
| `api` | API type — currently only `"openrouter-images"` is supported |
| `baseUrl` | API endpoint |
| `models` | Model definitions (see below) |

### Model fields

| Field | Default | Description |
|---|---|---|
| `id` | required | Model identifier sent to the API |
| `name` | `id` | Human-readable label |
| `input` | `["text"]` | Input modalities: `"text"` and/or `"image"` |
| `output` | `["image"]` | Output modalities: `"image"` and/or `"text"` |
| `cost` | all zeros | `{ "input", "output", "cacheRead", "cacheWrite" }` — $/million tokens |

### Example: add a model to the OpenRouter provider

```json
{
  "providers": {
    "openrouter": {
      "api": "openrouter-images",
      "baseUrl": "https://openrouter.ai/api/v1",
      "models": [
        {
          "id": "stability-ai/stable-diffusion-xl",
          "name": "Stable Diffusion XL",
          "input": ["text"],
          "output": ["image"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

When you include an existing provider (like `openrouter`), your definition
**replaces** the built-in one entirely — include all models you want, not just
the new ones.

---

## Synchronous usage

```python
result = pi_ai.generate_images_sync(model, context, options)
```

Creates a temporary event loop. Do not call from inside an existing event loop.

---

## Result structure

`generate_images()` returns an `AssistantImages` object:

```python
result.stop_reason   # "stop" | "error" | "aborted"
result.error_message # set when stop_reason is "error"
result.output        # list[TextContent | ImageContent]
result.usage         # token/cost info (may be None)
result.model         # model ID that ran
result.provider      # provider name
```

`ImageContent.data` is base64-encoded. `ImageContent.mime_type` is e.g. `"image/png"`.
