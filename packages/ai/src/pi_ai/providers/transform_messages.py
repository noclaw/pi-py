"""Cross-provider message transformation.

Called before converting our message types to a provider's wire format.
Handles image downgrade, thinking block compatibility, orphaned tool call
patching, and tool call ID normalization.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

from ..types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"

# Anthropic tool call IDs must match [a-zA-Z0-9_-], max 64 chars.
_ANTHROPIC_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")
_ANTHROPIC_ID_MAX = 64


def normalize_tool_call_id_for_anthropic(tool_call_id: str) -> str:
    """Replace illegal characters and truncate to Anthropic's ID constraints."""
    normalized = _ANTHROPIC_ID_RE.sub("_", tool_call_id)
    return normalized[:_ANTHROPIC_ID_MAX]


def _replace_images(
    content: list,
    placeholder: str,
) -> list:
    """Replace ImageContent blocks with a single placeholder TextContent."""
    result = []
    last_was_placeholder = False
    for block in content:
        if isinstance(block, ImageContent):
            if not last_was_placeholder:
                result.append(TextContent(text=placeholder))
            last_was_placeholder = True
        else:
            result.append(block)
            last_was_placeholder = getattr(block, "text", None) == placeholder
    return result


def _downgrade_images(messages: list[Message], model: Model) -> list[Message]:
    if "image" in model.input:
        return messages
    result: list[Message] = []
    for msg in messages:
        if isinstance(msg, UserMessage) and isinstance(msg.content, list):
            new_content = _replace_images(msg.content, _USER_IMAGE_PLACEHOLDER)
            result.append(msg.model_copy(update={"content": new_content}))
        elif isinstance(msg, ToolResultMessage):
            new_content = _replace_images(msg.content, _TOOL_IMAGE_PLACEHOLDER)
            result.append(msg.model_copy(update={"content": new_content}))
        else:
            result.append(msg)
    return result


def _transform_assistant(
    msg: AssistantMessage,
    model: Model,
    normalize_id: Optional[Callable[[str], str]],
    tool_id_map: dict[str, str],
) -> AssistantMessage:
    """Transform a single AssistantMessage for replay with `model`."""
    is_same = (
        msg.provider == model.provider
        and msg.api == model.api
        and msg.model == model.id
    )
    new_content = []
    for block in msg.content:
        if isinstance(block, ThinkingContent):
            if block.redacted:
                # Encrypted payload only valid for the exact same model
                if is_same:
                    new_content.append(block)
                # else: drop it entirely
            elif is_same and block.thinking_signature:
                new_content.append(block)
            elif not block.thinking or not block.thinking.strip():
                pass  # empty block — drop
            elif is_same:
                new_content.append(block)
            else:
                # Cross-model: convert to plain text
                new_content.append(TextContent(text=block.thinking))

        elif isinstance(block, TextContent):
            new_content.append(TextContent(text=block.text))

        elif isinstance(block, ToolCall):
            tc = block
            # Strip Google-specific thought_signature when crossing providers
            if not is_same and tc.thought_signature:
                tc = tc.model_copy(update={"thought_signature": None})
            # Normalize ID if a normalizer is provided and we're crossing providers
            if normalize_id and not is_same:
                new_id = normalize_id(tc.id)
                if new_id != tc.id:
                    tool_id_map[tc.id] = new_id
                    tc = tc.model_copy(update={"id": new_id})
            new_content.append(tc)

        else:
            new_content.append(block)

    return msg.model_copy(update={"content": new_content})


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Optional[Callable[[str], str]] = None,
) -> list[Message]:
    """Prepare messages for replay with `model`.

    1. Downgrade images the model cannot process to placeholder text.
    2. Transform thinking blocks — preserve for the same model, convert to
       text for cross-model replay, drop encrypted/redacted blocks.
    3. Skip errored or aborted assistant messages entirely.
    4. Insert synthetic ``toolResult`` messages for any tool calls that were
       never answered (prevents API validation errors).
    5. Normalize tool call IDs via ``normalize_tool_call_id`` (e.g. to satisfy
       Anthropic's ``[a-zA-Z0-9_-]{0,64}`` constraint) and propagate the
       mapping to matching ``toolResult`` messages.
    """
    # Map from original ID -> normalized ID; populated during assistant pass
    tool_id_map: dict[str, str] = {}

    # Step 1: image downgrade
    image_aware = _downgrade_images(messages, model)

    # Step 2 & 3: transform assistant messages, collect tool call state
    transformed: list[Message] = []
    for msg in image_aware:
        if isinstance(msg, AssistantMessage):
            # Skip error/aborted turns — they represent incomplete states
            if msg.stop_reason in ("error", "aborted"):
                continue
            transformed.append(
                _transform_assistant(msg, model, normalize_tool_call_id, tool_id_map)
            )
        elif isinstance(msg, ToolResultMessage):
            # Propagate normalized ID if the original was remapped
            normalized_call_id = tool_id_map.get(msg.tool_call_id, msg.tool_call_id)
            if normalized_call_id != msg.tool_call_id:
                transformed.append(msg.model_copy(update={"tool_call_id": normalized_call_id}))
            else:
                transformed.append(msg)
        else:
            transformed.append(msg)

    # Step 4: insert synthetic tool results for orphaned tool calls
    result: list[Message] = []
    pending_calls: list[ToolCall] = []
    answered_ids: set[str] = set()

    def _flush_orphans() -> None:
        for tc in pending_calls:
            if tc.id not in answered_ids:
                result.append(
                    ToolResultMessage(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[TextContent(text="No result provided")],
                        is_error=True,
                        timestamp=int(time.time() * 1000),
                    )
                )
        pending_calls.clear()
        answered_ids.clear()

    for msg in transformed:
        if isinstance(msg, AssistantMessage):
            _flush_orphans()
            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            if tool_calls:
                pending_calls.extend(tool_calls)
            result.append(msg)
        elif isinstance(msg, ToolResultMessage):
            answered_ids.add(msg.tool_call_id)
            result.append(msg)
        elif isinstance(msg, UserMessage):
            _flush_orphans()
            result.append(msg)
        else:
            result.append(msg)

    _flush_orphans()
    return result
