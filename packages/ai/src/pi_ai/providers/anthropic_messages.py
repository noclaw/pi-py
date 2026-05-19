"""Anthropic Messages provider."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import anthropic as anthropic_sdk

from ..env_keys import get_env_api_key
from ..models import calculate_cost, clamp_thinking_level
from ..stream import AssistantMessageEventStream
from ..types import (
    AnthropicMessagesCompat,
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


# ── Compat ────────────────────────────────────────────────────────────────────

def _get_compat(model: Model) -> AnthropicMessagesCompat:
    if isinstance(model.compat, AnthropicMessagesCompat):
        return model.compat
    return AnthropicMessagesCompat()


# ── Message conversion ────────────────────────────────────────────────────────

def _user_content(msg: UserMessage) -> Any:
    if isinstance(msg.content, str):
        return msg.content
    parts: list[dict[str, Any]] = []
    for block in msg.content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.mime_type,
                    "data": block.data,
                },
            })
    return parts or msg.content


def _convert_messages(context: Context) -> list[dict[str, Any]]:
    """Convert Context messages to Anthropic API message format."""
    messages: list[dict[str, Any]] = []

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            messages.append({"role": "user", "content": _user_content(msg)})

        elif isinstance(msg, AssistantMessage):
            content_blocks: list[dict[str, Any]] = []

            for block in msg.content:
                if isinstance(block, TextContent):
                    content_blocks.append({"type": "text", "text": block.text})

                elif isinstance(block, ThinkingContent):
                    if block.redacted and block.thinking_signature:
                        # Redacted: pass back the encrypted payload for continuity
                        content_blocks.append({
                            "type": "thinking",
                            "thinking": "",
                            "signature": block.thinking_signature,
                        })
                    elif block.thinking_signature:
                        content_blocks.append({
                            "type": "thinking",
                            "thinking": block.thinking,
                            "signature": block.thinking_signature,
                        })
                    # No signature = thinking from a different model; skip it

                elif isinstance(block, ToolCall):
                    content_blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    })

            if content_blocks:
                messages.append({"role": "assistant", "content": content_blocks})

        elif isinstance(msg, ToolResultMessage):
            # Anthropic tool results go into user messages
            result_content: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    result_content.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageContent):
                    result_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.mime_type,
                            "data": block.data,
                        },
                    })

            tool_result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": result_content or "",
                "is_error": msg.is_error,
            }
            messages.append({"role": "user", "content": [tool_result]})

    return messages


def _convert_tools(tools: list, compat: AnthropicMessagesCompat) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
        for tool in tools
    ]


# ── Usage helper ──────────────────────────────────────────────────────────────

def _parse_stop_reason(anthropic_reason: Optional[str]) -> str:
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "toolUse",
        "stop_sequence": "stop",
    }
    return mapping.get(anthropic_reason or "", "stop")


# ── Core streaming function ───────────────────────────────────────────────────

def stream_anthropic(
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
            compat = _get_compat(model)

            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "base_url": model.base_url,
                "max_retries": (options.max_retries if options and options.max_retries is not None else 2),
            }
            if options and options.timeout_ms is not None:
                client_kwargs["timeout"] = options.timeout_ms / 1000
            if model.headers:
                client_kwargs["default_headers"] = model.headers

            client = anthropic_sdk.AsyncAnthropic(**client_kwargs)

            messages = _convert_messages(context)

            # Provider-specific thinking options
            thinking_enabled: bool = getattr(options, "thinking_enabled", False)
            thinking_budget: Optional[int] = getattr(options, "thinking_budget_tokens", None)
            effort: Optional[str] = getattr(options, "effort", None)

            max_tokens = (options.max_tokens if options and options.max_tokens else None) or model.max_tokens

            params: dict[str, Any] = {
                "model": model.id,
                "max_tokens": max_tokens,
                "messages": messages,
            }

            if context.system_prompt:
                params["system"] = context.system_prompt

            if context.tools:
                params["tools"] = _convert_tools(context.tools, compat)

            if options and options.temperature is not None:
                params["temperature"] = options.temperature

            if options and options.metadata:
                params["metadata"] = options.metadata

            # Thinking configuration
            if thinking_enabled or effort:
                if effort:
                    # Adaptive thinking (Claude 4)
                    params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget or 10000}
                    # effort maps to betas or model-specific params — simplified here
                else:
                    params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget or 10000}
            elif thinking_budget:
                params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

            # on_payload hook
            if options and options.on_payload:
                result = await options.on_payload(params, model)
                if result is not None:
                    params = result

            extra_headers: dict[str, str] = {}
            if options and options.headers:
                extra_headers = options.headers

            # Stream via messages.stream() context manager
            async with client.messages.stream(
                **params,
                extra_headers=extra_headers,
            ) as stream_mgr:
                event_stream.push({"type": "start", "partial": output})

                # Track content blocks by Anthropic index
                # Maps Anthropic block index -> our content block (or None to skip)
                block_map: dict[int, Any] = {}
                # For tool calls: track partial JSON per Anthropic block index
                tool_partial_args: dict[int, str] = {}

                async for event in stream_mgr:
                    etype = event.type

                    if etype == "message_start":
                        msg_usage = event.message.usage
                        output.usage.input = msg_usage.input_tokens
                        output.response_id = event.message.id

                    elif etype == "content_block_start":
                        idx = event.index
                        cb = event.content_block
                        cb_type = cb.type

                        if cb_type == "text":
                            block = TextContent(text="")
                            output.content.append(block)
                            block_map[idx] = block
                            event_stream.push({
                                "type": "text_start",
                                "content_index": output.content.index(block),
                                "partial": output,
                            })

                        elif cb_type == "thinking":
                            block = ThinkingContent(
                                thinking="",
                                thinking_signature=getattr(cb, "signature", None) or "",
                            )
                            output.content.append(block)
                            block_map[idx] = block
                            event_stream.push({
                                "type": "thinking_start",
                                "content_index": output.content.index(block),
                                "partial": output,
                            })

                        elif cb_type == "tool_use":
                            block = ToolCall(
                                id=cb.id,
                                name=cb.name,
                                arguments={},
                            )
                            output.content.append(block)
                            block_map[idx] = block
                            tool_partial_args[idx] = ""
                            event_stream.push({
                                "type": "toolcall_start",
                                "content_index": output.content.index(block),
                                "partial": output,
                            })

                        else:
                            block_map[idx] = None  # unknown block type, skip

                    elif etype == "content_block_delta":
                        idx = event.index
                        block = block_map.get(idx)
                        delta = event.delta
                        delta_type = delta.type

                        if delta_type == "text_delta" and isinstance(block, TextContent):
                            block.text += delta.text
                            event_stream.push({
                                "type": "text_delta",
                                "content_index": output.content.index(block),
                                "delta": delta.text,
                                "partial": output,
                            })

                        elif delta_type == "thinking_delta" and isinstance(block, ThinkingContent):
                            block.thinking += delta.thinking
                            event_stream.push({
                                "type": "thinking_delta",
                                "content_index": output.content.index(block),
                                "delta": delta.thinking,
                                "partial": output,
                            })

                        elif delta_type == "signature_delta" and isinstance(block, ThinkingContent):
                            block.thinking_signature = (block.thinking_signature or "") + delta.signature

                        elif delta_type == "input_json_delta" and isinstance(block, ToolCall):
                            partial = (tool_partial_args.get(idx) or "") + delta.partial_json
                            tool_partial_args[idx] = partial
                            try:
                                block.arguments = json.loads(partial)
                            except json.JSONDecodeError:
                                pass
                            event_stream.push({
                                "type": "toolcall_delta",
                                "content_index": output.content.index(block),
                                "delta": delta.partial_json,
                                "partial": output,
                            })

                    elif etype == "content_block_stop":
                        idx = event.index
                        block = block_map.get(idx)

                        if isinstance(block, TextContent):
                            event_stream.push({
                                "type": "text_end",
                                "content_index": output.content.index(block),
                                "content": block.text,
                                "partial": output,
                            })

                        elif isinstance(block, ThinkingContent):
                            event_stream.push({
                                "type": "thinking_end",
                                "content_index": output.content.index(block),
                                "content": block.thinking,
                                "partial": output,
                            })

                        elif isinstance(block, ToolCall):
                            # Final parse with repair
                            partial = tool_partial_args.get(idx) or ""
                            try:
                                block.arguments = json.loads(partial)
                            except json.JSONDecodeError:
                                try:
                                    from json_repair import repair_json  # type: ignore[import]
                                    block.arguments = json.loads(repair_json(partial))
                                except Exception:
                                    block.arguments = {}
                            event_stream.push({
                                "type": "toolcall_end",
                                "content_index": output.content.index(block),
                                "tool_call": block,
                                "partial": output,
                            })

                    elif etype == "message_delta":
                        output.stop_reason = _parse_stop_reason(event.delta.stop_reason)  # type: ignore[assignment]
                        if event.usage:
                            output.usage.output = event.usage.output_tokens
                            output.usage.total_tokens = output.usage.input + output.usage.output
                            calculate_cost(model, output.usage)

                    elif etype == "message_stop":
                        pass  # nothing useful here

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

_DEFAULT_THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
}


def stream_simple_anthropic(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    level: Optional[str] = None
    effort: Optional[str] = None
    thinking_budget: Optional[int] = None

    if options and options.reasoning and model.reasoning:
        clamped = clamp_thinking_level(model, options.reasoning)
        if clamped != "off":
            level = clamped
            budgets = (options.thinking_budgets or {}) if options else {}
            thinking_budget = budgets.get(clamped) or _DEFAULT_THINKING_BUDGETS.get(clamped)
            effort = clamped  # Anthropic uses effort string directly for Claude 4

    base = options.model_dump(exclude={"reasoning", "thinking_budgets"}) if options else {}

    class _AnthropicOptions(StreamOptions):
        thinking_enabled: bool = False
        thinking_budget_tokens: Optional[int] = None
        effort: Optional[str] = None

    provider_opts = _AnthropicOptions(
        **base,
        thinking_enabled=bool(level),
        thinking_budget_tokens=thinking_budget,
        effort=effort,
    )
    return stream_anthropic(model, context, provider_opts)


# ── Provider object ───────────────────────────────────────────────────────────

class _AnthropicMessagesProvider:
    api = "anthropic-messages"

    def stream(
        self,
        model: Model,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AssistantMessageEventStream:
        return stream_anthropic(model, context, options)

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: Optional[SimpleStreamOptions] = None,
    ) -> AssistantMessageEventStream:
        return stream_simple_anthropic(model, context, options)


anthropic_messages_provider = _AnthropicMessagesProvider()
