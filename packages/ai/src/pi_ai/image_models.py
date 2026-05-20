from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

from .types import ImagesModel, ModelCost


def _image_models_from_json(data: dict) -> dict[str, dict[str, ImagesModel]]:
    result: dict[str, dict[str, ImagesModel]] = {}
    for provider_name, pdata in (data.get("providers") or {}).items():
        api: str = pdata.get("api", "openrouter-images")
        base_url: str = pdata.get("baseUrl", "")
        result[provider_name] = {}
        for m in pdata.get("models") or []:
            if not m.get("id"):
                continue
            cost_raw = m.get("cost") or {}
            result[provider_name][m["id"]] = ImagesModel(
                id=m["id"],
                name=m.get("name", m["id"]),
                api=m.get("api", api),
                provider=provider_name,
                base_url=m.get("baseUrl", base_url),
                input=m.get("input", ["text"]),
                output=m.get("output", ["image"]),
                cost=ModelCost(
                    input=float(cost_raw.get("input", 0)),
                    output=float(cost_raw.get("output", 0)),
                    cache_read=float(cost_raw.get("cacheRead", 0)),
                    cache_write=float(cost_raw.get("cacheWrite", 0)),
                ),
                headers=m.get("headers") or None,
            )
    return result


def _load_image_catalog() -> dict[str, dict[str, ImagesModel]]:
    """Load image model catalog from JSON.

    Loads the bundled ``pi_ai/image_models.json`` as the base, then merges
    ``~/.pi-py/image_models.json`` on top (new providers added; existing
    providers replaced).
    """
    catalog: dict[str, dict[str, ImagesModel]] = {}

    try:
        pkg_file = importlib.resources.files("pi_ai").joinpath("image_models.json")
        catalog = _image_models_from_json(json.loads(pkg_file.read_text(encoding="utf-8")))
    except Exception:
        pass

    user_path = Path("~/.pi-py/image_models.json").expanduser()
    if user_path.is_file():
        try:
            catalog.update(_image_models_from_json(json.loads(user_path.read_text(encoding="utf-8"))))
        except Exception:
            pass

    return catalog


IMAGE_MODELS: dict[str, dict[str, ImagesModel]] = _load_image_catalog()


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
