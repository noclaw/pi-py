"""OpenAI Chat Completions provider (also used for OpenAI-compatible APIs)."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from openai import AsyncOpenAI

from ..env_keys import get_env_api_key
from ..models import calculate_cost, clamp_thinking_level
from ..stream import AssistantMessageEventStream
from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    OpenAICompletionsCompat,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


# ── Compat resolution ─────────────────────────────────────────────────────────

_DEFAULTS = OpenAICompletionsCompat()


def _resolve_compat(model: Model) -> OpenAICompletionsCompat:
    """Return resolved compat, auto-detecting from base_url when not explicit."""
    if isinstance(model.compat, OpenAICompletionsCompat):
        return model.compat

    base = model.base_url.lower()
    compat = OpenAICompletionsCompat()

    if "api.openai.com" in base:
        compat.supports_store = True
        compat.supports_developer_role = True
        compat.supports_reasoning_effort = True
        compat.max_tokens_field = "max_completion_tokens"
    elif "api.deepseek.com" in base:
        compat.thinking_format = "deepseek"
        compat.max_tokens_field = "max_tokens"
    elif "api.groq.com" in base:
        compat.max_tokens_field = "max_tokens"
        compat.supports_store = False
    elif "api.cerebras.ai" in base:
        compat.max_tokens_field = "max_tokens"
        compat.supports_store = False
        compat.supports_strict_mode = False
    elif "openrouter.ai" in base:
        compat.thinking_format = "openrouter"

    return compat


# ── Message conversion ────────────────────────────────────────────────────────

def _convert_messages(
    context: Context,
    compat: OpenAICompletionsCompat,
) -> list[dict[str, Any]]:
    system_role = "developer" if compat.supports_developer_role else "system"
    messages: list[dict[str, Any]] = []

    if context.system_prompt:
        messages.append({"role": system_role, "content": context.system_prompt})

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                messages.append({"role": "user", "content": msg.content})
            else:
                parts: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                        })
                messages.append({"role": "user", "content": parts})

        elif isinstance(msg, AssistantMessage):
            text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
            thinking_blocks = [b for b in msg.content if isinstance(b, ThinkingContent)]
            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]

            text = "".join(b.text for b in text_blocks)

            assistant_msg: dict[str, Any] = {"role": "assistant"}

            if thinking_blocks:
                if compat.requires_thinking_as_text:
                    prefix = "\n".join(
                        f"<thinking>{b.thinking}</thinking>" for b in thinking_blocks
                    )
                    text = prefix + ("\n" + text if text else "")
                elif compat.requires_reasoning_content_on_assistant_messages:
                    assistant_msg["reasoning_content"] = "".join(
                        b.thinking for b in thinking_blocks
                    )

            assistant_msg["content"] = text or None

            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]

            messages.append(assistant_msg)

        elif isinstance(msg, ToolResultMessage):
            text = "".join(
                b.text for b in msg.content if isinstance(b, TextContent)
            )
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": text,
            }
            if compat.requires_tool_result_name:
                tool_msg["name"] = msg.tool_name
            messages.append(tool_msg)

    return messages


def _convert_tools(
    tools: list,
    compat: OpenAICompletionsCompat,
) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        func: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.supports_strict_mode:
            func["strict"] = True
        result.append({"type": "function", "function": func})
    return result


# ── Usage / stop-reason helpers ───────────────────────────────────────────────

def _parse_stop_reason(finish_reason: str) -> tuple[str, Optional[str]]:
    mapping: dict[str, tuple[str, Optional[str]]] = {
        "stop": ("stop", None),
        "length": ("length", None),
        "tool_calls": ("toolUse", None),
        "function_call": ("toolUse", None),
        "content_filter": ("error", "Content was filtered"),
    }
    return mapping.get(finish_reason, ("stop", None))


def _parse_usage(chunk_usage: Any, model: Model) -> Usage:
    usage = Usage(
        input=getattr(chunk_usage, "prompt_tokens", 0) or 0,
        output=getattr(chunk_usage, "completion_tokens", 0) or 0,
    )
    details = getattr(chunk_usage, "prompt_tokens_details", None)
    if details:
        usage.cache_read = getattr(details, "cached_tokens", 0) or 0
    usage.total_tokens = usage.input + usage.output
    calculate_cost(model, usage)
    return usage


def _parse_streaming_json(partial: str) -> dict[str, Any]:
    if not partial:
        return {}
    try:
        return json.loads(partial)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json  # type: ignore[import]
            return json.loads(repair_json(partial))
        except Exception:
            return {}


# ── Core streaming function ───────────────────────────────────────────────────

def stream_openai_completions(
    model: Model,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessageEventStream:
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
        )
        try:
            api_key = (
                (options.api_key if options else None)
                or get_env_api_key(model.provider)
                or ""
            )
            compat = _resolve_compat(model)

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

            messages = _convert_messages(context, compat)

            params: dict[str, Any] = {
                "model": model.id,
                "messages": messages,
                "stream": True,
            }

            if compat.supports_usage_in_streaming:
                params["stream_options"] = {"include_usage": True}
            if compat.supports_store:
                params["store"] = False

            if options and options.max_tokens:
                field = compat.max_tokens_field
                params[field] = options.max_tokens

            if options and options.temperature is not None:
                params["temperature"] = options.temperature

            if context.tools:
                params["tools"] = _convert_tools(context.tools, compat)

            # Reasoning effort (OpenAI o-series style)
            reasoning_effort: Optional[str] = getattr(options, "reasoning_effort", None)
            if reasoning_effort and model.reasoning and compat.supports_reasoning_effort:
                mapped = (model.thinking_level_map or {}).get(reasoning_effort, reasoning_effort)
                params["reasoning_effort"] = mapped

            # DeepSeek thinking format
            if compat.thinking_format == "deepseek" and model.reasoning:
                enabled = bool(reasoning_effort)
                params["thinking"] = {"type": "enabled" if enabled else "disabled"}
                if enabled and compat.supports_reasoning_effort:
                    params["reasoning_effort"] = reasoning_effort

            # OpenRouter reasoning format
            if compat.thinking_format == "openrouter" and model.reasoning:
                if reasoning_effort:
                    mapped = (model.thinking_level_map or {}).get(reasoning_effort, reasoning_effort)
                    params["reasoning"] = {"effort": mapped}

            # on_payload hook
            if options and options.on_payload:
                result = await options.on_payload(params, model)
                if result is not None:
                    params = result

            extra_headers: dict[str, str] = {}
            if options and options.headers:
                extra_headers = options.headers

            raw_stream = await client.chat.completions.create(
                **params,
                extra_headers=extra_headers,
            )

            event_stream.push({"type": "start", "partial": output})

            # ── Streaming state ────────────────────────────────────────────────
            text_block: Optional[TextContent] = None
            thinking_block: Optional[ThinkingContent] = None
            # index -> (ToolCall block, accumulated partial_args)
            tool_call_blocks: dict[int, tuple[ToolCall, str]] = {}
            has_finish_reason = False

            def content_index(block: Any) -> int:
                return output.content.index(block)

            def ensure_text() -> TextContent:
                nonlocal text_block
                if text_block is None:
                    text_block = TextContent(text="")
                    output.content.append(text_block)
                    event_stream.push({
                        "type": "text_start",
                        "content_index": content_index(text_block),
                        "partial": output,
                    })
                return text_block

            def ensure_thinking(signature: str) -> ThinkingContent:
                nonlocal thinking_block
                if thinking_block is None:
                    thinking_block = ThinkingContent(thinking="", thinking_signature=signature)
                    output.content.append(thinking_block)
                    event_stream.push({
                        "type": "thinking_start",
                        "content_index": content_index(thinking_block),
                        "partial": output,
                    })
                return thinking_block

            def get_or_create_tool_call(tc_delta: Any, idx: int) -> ToolCall:
                if idx in tool_call_blocks:
                    return tool_call_blocks[idx][0]
                block = ToolCall(
                    id=tc_delta.id or "",
                    name=(tc_delta.function.name if tc_delta.function else "") or "",
                    arguments={},
                )
                output.content.append(block)
                tool_call_blocks[idx] = (block, "")
                event_stream.push({
                    "type": "toolcall_start",
                    "content_index": content_index(block),
                    "partial": output,
                })
                return block

            async for chunk in raw_stream:
                output.response_id = output.response_id or chunk.id
                if chunk.model and chunk.model != model.id:
                    output.response_model = output.response_model or chunk.model
                if chunk.usage:
                    output.usage = _parse_usage(chunk.usage, model)

                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                if choice.finish_reason:
                    stop_reason, err_msg = _parse_stop_reason(choice.finish_reason)
                    output.stop_reason = stop_reason  # type: ignore[assignment]
                    if err_msg:
                        output.error_message = err_msg
                    has_finish_reason = True

                delta = choice.delta
                if not delta:
                    continue

                # Text
                if delta.content:
                    block = ensure_text()
                    block.text += delta.content
                    event_stream.push({
                        "type": "text_delta",
                        "content_index": content_index(block),
                        "delta": delta.content,
                        "partial": output,
                    })

                # Reasoning / thinking (provider-specific extra fields)
                extra: dict[str, Any] = {}
                if hasattr(delta, "model_extra") and delta.model_extra:
                    extra = delta.model_extra
                for field in ("reasoning_content", "reasoning", "reasoning_text"):
                    reasoning_val = getattr(delta, field, None) or extra.get(field)
                    if isinstance(reasoning_val, str) and reasoning_val:
                        block2 = ensure_thinking(field)
                        block2.thinking += reasoning_val
                        event_stream.push({
                            "type": "thinking_delta",
                            "content_index": content_index(block2),
                            "delta": reasoning_val,
                            "partial": output,
                        })
                        break

                # Tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index if tc_delta.index is not None else 0
                        block3 = get_or_create_tool_call(tc_delta, idx)

                        if not block3.id and tc_delta.id:
                            block3.id = tc_delta.id
                        if not block3.name and tc_delta.function and tc_delta.function.name:
                            block3.name = tc_delta.function.name

                        delta_args = ""
                        if tc_delta.function and tc_delta.function.arguments:
                            delta_args = tc_delta.function.arguments
                            old_block, old_args = tool_call_blocks[idx]
                            new_args = old_args + delta_args
                            tool_call_blocks[idx] = (old_block, new_args)
                            block3.arguments = _parse_streaming_json(new_args)

                        event_stream.push({
                            "type": "toolcall_delta",
                            "content_index": content_index(block3),
                            "delta": delta_args,
                            "partial": output,
                        })

            # ── Finish all open blocks ─────────────────────────────────────────
            if text_block:
                event_stream.push({
                    "type": "text_end",
                    "content_index": content_index(text_block),
                    "content": text_block.text,
                    "partial": output,
                })
            if thinking_block:
                event_stream.push({
                    "type": "thinking_end",
                    "content_index": content_index(thinking_block),
                    "content": thinking_block.thinking,
                    "partial": output,
                })
            for idx, (block3, partial_args) in tool_call_blocks.items():
                block3.arguments = _parse_streaming_json(partial_args)
                event_stream.push({
                    "type": "toolcall_end",
                    "content_index": content_index(block3),
                    "tool_call": block3,
                    "partial": output,
                })

            if not has_finish_reason:
                raise RuntimeError("Stream ended without finish_reason")
            if output.stop_reason == "error":
                raise RuntimeError(output.error_message or "Provider returned an error stop reason")

            event_stream.push({"type": "done", "reason": output.stop_reason, "message": output})
            event_stream.end()

        except Exception as exc:
            output.stop_reason = "error"
            output.error_message = str(exc)
            event_stream.push({"type": "error", "reason": "error", "error": output})
            event_stream.end()

    asyncio.create_task(_run())
    return event_stream


# ── stream_simple wrapper ─────────────────────────────────────────────────────

def stream_simple_openai_completions(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    reasoning_effort: Optional[str] = None
    if options and options.reasoning and model.reasoning:
        level = clamp_thinking_level(model, options.reasoning)
        if level != "off":
            reasoning_effort = level

    # Build provider options by merging base StreamOptions fields
    base = options.model_dump(exclude={"reasoning", "thinking_budgets"}) if options else {}

    class _Options(StreamOptions):
        reasoning_effort: Optional[str] = None

    provider_opts = _Options(**base, reasoning_effort=reasoning_effort)
    return stream_openai_completions(model, context, provider_opts)


# ── Provider object ───────────────────────────────────────────────────────────

class _OpenAICompletionsProvider:
    api = "openai-completions"

    def stream(
        self,
        model: Model,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        return stream_openai_completions(model, context, options)

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream:
        return stream_simple_openai_completions(model, context, options)


openai_completions_provider = _OpenAICompletionsProvider()
