"""Curated model catalog.

OpenAI models are mapped to ``openai-completions`` (Chat Completions API)
rather than ``openai-responses`` used in the TS catalog, since the Python
OpenAI SDK exposes chat completions as the primary streaming interface.
"""
from __future__ import annotations

from .types import Model, ModelCost, OpenAICompletionsCompat, AnthropicMessagesCompat

# fmt: off
MODELS: dict[str, dict[str, Model]] = {

    # ── OpenAI ────────────────────────────────────────────────────────────────
    "openai": {
        "gpt-4o": Model(
            id="gpt-4o", name="GPT-4o",
            api="openai-completions", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=False, input=["text", "image"],
            cost=ModelCost(input=2.5, output=10, cache_read=1.25, cache_write=0),
            context_window=128_000, max_tokens=16_384,
            compat=OpenAICompletionsCompat(
                supports_store=True, supports_developer_role=True,
                max_tokens_field="max_completion_tokens",
            ),
        ),
        "gpt-4o-mini": Model(
            id="gpt-4o-mini", name="GPT-4o mini",
            api="openai-completions", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=False, input=["text", "image"],
            cost=ModelCost(input=0.15, output=0.6, cache_read=0.08, cache_write=0),
            context_window=128_000, max_tokens=16_384,
            compat=OpenAICompletionsCompat(
                supports_store=True, supports_developer_role=True,
                max_tokens_field="max_completion_tokens",
            ),
        ),
        "gpt-4.1": Model(
            id="gpt-4.1", name="GPT-4.1",
            api="openai-completions", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=False, input=["text", "image"],
            cost=ModelCost(input=2, output=8, cache_read=0.5, cache_write=0),
            context_window=1_047_576, max_tokens=32_768,
            compat=OpenAICompletionsCompat(
                supports_store=True, supports_developer_role=True,
                max_tokens_field="max_completion_tokens",
            ),
        ),
        "gpt-4.1-mini": Model(
            id="gpt-4.1-mini", name="GPT-4.1 mini",
            api="openai-completions", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=False, input=["text", "image"],
            cost=ModelCost(input=0.4, output=1.6, cache_read=0.1, cache_write=0),
            context_window=1_047_576, max_tokens=32_768,
            compat=OpenAICompletionsCompat(
                supports_store=True, supports_developer_role=True,
                max_tokens_field="max_completion_tokens",
            ),
        ),
        "o3": Model(
            id="o3", name="o3",
            api="openai-completions", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=True, input=["text", "image"],
            cost=ModelCost(input=2, output=8, cache_read=0.5, cache_write=0),
            context_window=200_000, max_tokens=100_000,
            compat=OpenAICompletionsCompat(
                supports_store=True, supports_developer_role=True,
                supports_reasoning_effort=True,
                max_tokens_field="max_completion_tokens",
            ),
        ),
        "o3-mini": Model(
            id="o3-mini", name="o3-mini",
            api="openai-completions", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=True, input=["text"],
            cost=ModelCost(input=1.1, output=4.4, cache_read=0.55, cache_write=0),
            context_window=200_000, max_tokens=100_000,
            compat=OpenAICompletionsCompat(
                supports_store=True, supports_developer_role=True,
                supports_reasoning_effort=True,
                max_tokens_field="max_completion_tokens",
            ),
        ),
        "o4-mini": Model(
            id="o4-mini", name="o4-mini",
            api="openai-completions", provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=True, input=["text", "image"],
            cost=ModelCost(input=1.1, output=4.4, cache_read=0.28, cache_write=0),
            context_window=200_000, max_tokens=100_000,
            compat=OpenAICompletionsCompat(
                supports_store=True, supports_developer_role=True,
                supports_reasoning_effort=True,
                max_tokens_field="max_completion_tokens",
            ),
        ),
    },

    # ── Anthropic ─────────────────────────────────────────────────────────────
    "anthropic": {
        "claude-opus-4-7": Model(
            id="claude-opus-4-7", name="Claude Opus 4.7",
            api="anthropic-messages", provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True, input=["text", "image"],
            cost=ModelCost(input=5, output=25, cache_read=0.5, cache_write=6.25),
            context_window=1_000_000, max_tokens=128_000,
            thinking_level_map={"xhigh": "xhigh"},
        ),
        "claude-opus-4-5": Model(
            id="claude-opus-4-5", name="Claude Opus 4.5 (latest)",
            api="anthropic-messages", provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True, input=["text", "image"],
            cost=ModelCost(input=5, output=25, cache_read=0.5, cache_write=6.25),
            context_window=200_000, max_tokens=64_000,
        ),
        "claude-sonnet-4-6": Model(
            id="claude-sonnet-4-6", name="Claude Sonnet 4.6",
            api="anthropic-messages", provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True, input=["text", "image"],
            cost=ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
            context_window=1_000_000, max_tokens=64_000,
        ),
        "claude-sonnet-4-5": Model(
            id="claude-sonnet-4-5", name="Claude Sonnet 4.5 (latest)",
            api="anthropic-messages", provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True, input=["text", "image"],
            cost=ModelCost(input=3, output=15, cache_read=0.3, cache_write=3.75),
            context_window=200_000, max_tokens=64_000,
        ),
        "claude-haiku-4-5": Model(
            id="claude-haiku-4-5", name="Claude Haiku 4.5 (latest)",
            api="anthropic-messages", provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True, input=["text", "image"],
            cost=ModelCost(input=1, output=5, cache_read=0.1, cache_write=1.25),
            context_window=200_000, max_tokens=64_000,
        ),
        "claude-haiku-4-5-20251001": Model(
            id="claude-haiku-4-5-20251001", name="Claude Haiku 4.5",
            api="anthropic-messages", provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True, input=["text", "image"],
            cost=ModelCost(input=1, output=5, cache_read=0.1, cache_write=1.25),
            context_window=200_000, max_tokens=64_000,
        ),
    },

    # ── DeepSeek ──────────────────────────────────────────────────────────────
    "deepseek": {
        "deepseek-v4-flash": Model(
            id="deepseek-v4-flash", name="DeepSeek V4 Flash",
            api="openai-completions", provider="deepseek",
            base_url="https://api.deepseek.com",
            reasoning=True, input=["text"],
            cost=ModelCost(input=0.14, output=0.28, cache_read=0.0028, cache_write=0),
            context_window=1_000_000, max_tokens=384_000,
            compat=OpenAICompletionsCompat(
                thinking_format="deepseek",
                requires_reasoning_content_on_assistant_messages=True,
                max_tokens_field="max_tokens",
                supports_usage_in_streaming=True,
            ),
        ),
        "deepseek-v4-pro": Model(
            id="deepseek-v4-pro", name="DeepSeek V4 Pro",
            api="openai-completions", provider="deepseek",
            base_url="https://api.deepseek.com",
            reasoning=True, input=["text"],
            cost=ModelCost(input=0.435, output=0.87, cache_read=0.003625, cache_write=0),
            context_window=1_000_000, max_tokens=384_000,
            compat=OpenAICompletionsCompat(
                thinking_format="deepseek",
                requires_reasoning_content_on_assistant_messages=True,
                max_tokens_field="max_tokens",
                supports_usage_in_streaming=True,
            ),
        ),
    },

    # ── Groq ──────────────────────────────────────────────────────────────────
    "groq": {
        "llama-3.3-70b-versatile": Model(
            id="llama-3.3-70b-versatile", name="Llama 3.3 70B Versatile",
            api="openai-completions", provider="groq",
            base_url="https://api.groq.com/openai/v1",
            reasoning=False, input=["text"],
            cost=ModelCost(input=0.59, output=0.79, cache_read=0, cache_write=0),
            context_window=131_072, max_tokens=32_768,
            compat=OpenAICompletionsCompat(
                max_tokens_field="max_tokens",
                supports_store=False,
            ),
        ),
        "llama-3.1-8b-instant": Model(
            id="llama-3.1-8b-instant", name="Llama 3.1 8B Instant",
            api="openai-completions", provider="groq",
            base_url="https://api.groq.com/openai/v1",
            reasoning=False, input=["text"],
            cost=ModelCost(input=0.05, output=0.08, cache_read=0, cache_write=0),
            context_window=131_072, max_tokens=131_072,
            compat=OpenAICompletionsCompat(
                max_tokens_field="max_tokens",
                supports_store=False,
            ),
        ),
    },

    # ── Cerebras ──────────────────────────────────────────────────────────────
    "cerebras": {
        "llama3.1-8b": Model(
            id="llama3.1-8b", name="Llama 3.1 8B",
            api="openai-completions", provider="cerebras",
            base_url="https://api.cerebras.ai/v1",
            reasoning=False, input=["text"],
            cost=ModelCost(input=0.1, output=0.1, cache_read=0, cache_write=0),
            context_window=32_000, max_tokens=8_000,
            compat=OpenAICompletionsCompat(
                max_tokens_field="max_tokens",
                supports_store=False,
                supports_strict_mode=False,
            ),
        ),
    },
}
# fmt: on
