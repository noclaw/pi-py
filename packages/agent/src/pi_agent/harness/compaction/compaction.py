"""Context compaction: token estimation, cut-point selection, summary generation."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from pi_ai import complete_simple
from pi_ai.types import Context, SimpleStreamOptions, UserMessage, TextContent

from ..messages import convert_to_llm
from ..session.session import build_session_context
from ..types import (
    CompactionError,
    CompactionPreparation,
    CompactionSettings,
    Err,
    Ok,
    Result,
    err,
    ok,
)
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)


DEFAULT_COMPACTION_SETTINGS = CompactionSettings(
    enabled=True,
    reserve_tokens=16384,
    keep_recent_tokens=20000,
)


# ── Token estimation ───────────────────────────────────────────────────────────

def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value) or "undefined"
    except Exception:
        return "[unserializable]"


def calculate_context_tokens(usage: Any) -> int:
    if hasattr(usage, "total_tokens"):
        return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)
    if isinstance(usage, dict):
        return usage.get("totalTokens") or (
            usage.get("input", 0) + usage.get("output", 0)
            + usage.get("cacheRead", 0) + usage.get("cacheWrite", 0)
        )
    return 0


def estimate_tokens(message: Any) -> int:
    role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
    chars = 0
    if role == "user":
        content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for b in content:
                btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                if btype == "text":
                    chars += len(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
    elif role == "assistant":
        content = getattr(message, "content", []) or (message.get("content", []) if isinstance(message, dict) else [])
        for b in content:
            btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
            if btype == "text":
                chars += len(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
            elif btype == "thinking":
                chars += len(b.get("thinking", "") if isinstance(b, dict) else getattr(b, "thinking", ""))
            elif btype == "toolCall":
                args = b.get("arguments") if isinstance(b, dict) else getattr(b, "arguments", {})
                name = b.get("name", "") if isinstance(b, dict) else getattr(b, "name", "")
                chars += len(name) + len(_safe_json(args))
    elif role in ("custom", "toolResult"):
        content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for b in content:
                btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                if btype == "text":
                    chars += len(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                elif btype == "image":
                    chars += 4800
    elif role == "bashExecution":
        chars = len(getattr(message, "command", "")) + len(getattr(message, "output", ""))
    elif role in ("branchSummary", "compactionSummary"):
        chars = len(getattr(message, "summary", ""))
    return (chars + 3) // 4


@dataclass
class ContextUsageEstimate:
    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def estimate_context_tokens(messages: list[Any]) -> ContextUsageEstimate:
    last_usage_idx: int | None = None
    last_usage: Any = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        role = getattr(msg, "role", None)
        if role == "assistant":
            usage = getattr(msg, "usage", None)
            stop = getattr(msg, "stop_reason", None)
            if usage and stop not in ("aborted", "error"):
                last_usage_idx = i
                last_usage = usage
                break

    if last_usage_idx is None:
        total = sum(estimate_tokens(m) for m in messages)
        return ContextUsageEstimate(
            tokens=total, usage_tokens=0, trailing_tokens=total, last_usage_index=None
        )

    usage_tokens = calculate_context_tokens(last_usage)
    trailing = sum(estimate_tokens(messages[i]) for i in range(last_usage_idx + 1, len(messages)))
    return ContextUsageEstimate(
        tokens=usage_tokens + trailing,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing,
        last_usage_index=last_usage_idx,
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


# ── Cut-point selection ────────────────────────────────────────────────────────

def _find_valid_cut_points(entries: list[dict], start: int, end: int) -> list[int]:
    cut_points: list[int] = []
    for i in range(start, end):
        entry = entries[i]
        t = entry.get("type")
        if t == "message":
            role = (entry.get("message") or {}).get("role", "")
            if role in ("bashExecution", "custom", "branchSummary", "compactionSummary", "user", "assistant"):
                cut_points.append(i)
        elif t in ("branch_summary", "custom_message"):
            cut_points.append(i)
    return cut_points


def find_turn_start_index(entries: list[dict], entry_index: int, start_index: int) -> int:
    for i in range(entry_index, start_index - 1, -1):
        entry = entries[i]
        t = entry.get("type")
        if t in ("branch_summary", "custom_message"):
            return i
        if t == "message":
            role = (entry.get("message") or {}).get("role", "")
            if role in ("user", "bashExecution"):
                return i
    return -1


@dataclass
class CutPointResult:
    first_kept_entry_index: int
    turn_start_index: int
    is_split_turn: bool


def find_cut_point(
    entries: list[dict],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    cut_points = _find_valid_cut_points(entries, start_index, end_index)
    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)

    cut_index = cut_points[0]
    accumulated = 0

    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if entry.get("type") != "message":
            continue
        msg = entry.get("message")
        if not msg:
            continue
        # Convert if needed
        from ..messages import (
            create_branch_summary_message, create_compaction_summary_message, create_custom_message
        )
        accumulated += estimate_tokens(msg)
        if accumulated >= keep_recent_tokens:
            for c in cut_points:
                if c >= i:
                    cut_index = c
                    break
            break

    while cut_index > start_index:
        prev = entries[cut_index - 1]
        if prev.get("type") in ("compaction", "message"):
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_msg = (
        cut_entry.get("type") == "message"
        and (cut_entry.get("message") or {}).get("role") == "user"
    )
    turn_start = -1 if is_user_msg else find_turn_start_index(entries, cut_index, start_index)

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start,
        is_split_turn=not is_user_msg and turn_start != -1,
    )


# ── Summary prompts ────────────────────────────────────────────────────────────

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation between "
    "a user and an AI coding assistant, then produce a structured summary following the exact "
    "format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in "
    "the conversation. ONLY output the structured summary."
)

_SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

_TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


async def generate_summary(
    current_messages: list[Any],
    model: Any,
    reserve_tokens: int,
    api_key: str,
    headers: dict | None = None,
    signal: asyncio.Event | None = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: str | None = None,
) -> Result:
    max_tokens = min(
        int(0.8 * reserve_tokens),
        model.max_tokens if model.max_tokens > 0 else 10**9,
    )
    base_prompt = _UPDATE_SUMMARIZATION_PROMPT if previous_summary else _SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    llm_messages = convert_to_llm(current_messages)
    conv_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conv_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    opts = SimpleStreamOptions(
        max_tokens=max_tokens,
        api_key=api_key,
        headers=headers,
        signal=signal,
        reasoning=thinking_level if (thinking_level and thinking_level != "off" and model.reasoning) else None,
    )

    try:
        response = await complete_simple(
            model,
            Context(
                system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
                messages=[UserMessage(content=[TextContent(text=prompt_text)])],
            ),
            opts,
        )
    except Exception as e:
        return err(CompactionError("summarization_failed", f"Summarization request failed: {e}", e))

    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Summarization aborted"))
    if response.stop_reason == "error":
        return err(CompactionError("summarization_failed", f"Summarization failed: {response.error_message or 'Unknown error'}"))

    text = "\n".join(
        b.text for b in response.content if hasattr(b, "type") and b.type == "text"
    )
    return ok(text)


# ── Compaction helpers ─────────────────────────────────────────────────────────

def _get_message_from_entry_for_compact(entry: dict) -> Any | None:
    t = entry.get("type")
    if t == "compaction":
        return None
    if t == "message":
        return entry.get("message")
    from ..messages import create_custom_message, create_branch_summary_message, create_compaction_summary_message
    if t == "custom_message":
        return create_custom_message(
            entry.get("customType", ""), entry.get("content", ""),
            entry.get("display", True), entry.get("details"), entry.get("timestamp", "")
        )
    if t == "branch_summary":
        return create_branch_summary_message(entry.get("summary", ""), entry.get("fromId", ""), entry.get("timestamp", ""))
    return None


def _extract_file_operations(
    messages: list[Any],
    entries: list[dict],
    prev_compaction_index: int,
) -> FileOperations:
    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        prev = entries[prev_compaction_index]
        if not prev.get("fromHook") and prev.get("details"):
            details = prev["details"]
            if isinstance(details, dict):
                for f in (details.get("readFiles") or []):
                    file_ops.read.add(f)
                for f in (details.get("modifiedFiles") or []):
                    file_ops.edited.add(f)
    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)
    return file_ops


def prepare_compaction(
    path_entries: list[dict],
    settings: CompactionSettings,
) -> Result:
    if not path_entries or path_entries[-1].get("type") == "compaction":
        return ok(None)

    prev_idx = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if path_entries[i].get("type") == "compaction":
            prev_idx = i
            break

    prev_summary: str | None = None
    boundary_start = 0
    if prev_idx >= 0:
        prev = path_entries[prev_idx]
        prev_summary = prev.get("summary")
        first_kept_id = prev.get("firstKeptEntryId")
        idx = next((i for i, e in enumerate(path_entries) if e.get("id") == first_kept_id), -1)
        boundary_start = idx if idx >= 0 else prev_idx + 1

    boundary_end = len(path_entries)
    ctx = build_session_context(path_entries)
    tokens_before = estimate_context_tokens(ctx["messages"]).tokens
    cut = find_cut_point(path_entries, boundary_start, boundary_end, settings.keep_recent_tokens)
    first_kept_entry = path_entries[cut.first_kept_entry_index] if cut.first_kept_entry_index < len(path_entries) else None
    if not first_kept_entry or not first_kept_entry.get("id"):
        return err(CompactionError("invalid_session", "First kept entry has no UUID - session may need migration"))
    first_kept_entry_id = first_kept_entry["id"]

    history_end = cut.turn_start_index if cut.is_split_turn else cut.first_kept_entry_index
    messages_to_summarize = [
        m for m in (
            _get_message_from_entry_for_compact(path_entries[i])
            for i in range(boundary_start, history_end)
        ) if m is not None
    ]
    turn_prefix_messages: list[Any] = []
    if cut.is_split_turn:
        turn_prefix_messages = [
            m for m in (
                _get_message_from_entry_for_compact(path_entries[i])
                for i in range(cut.turn_start_index, cut.first_kept_entry_index)
            ) if m is not None
        ]

    file_ops = _extract_file_operations(messages_to_summarize, path_entries, prev_idx)
    if cut.is_split_turn:
        for msg in turn_prefix_messages:
            extract_file_ops_from_message(msg, file_ops)

    return ok(CompactionPreparation(
        first_kept_entry_id=first_kept_entry_id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=prev_summary,
        file_ops=file_ops,
        settings=settings,
    ))


async def _generate_turn_prefix_summary(
    messages: list[Any],
    model: Any,
    reserve_tokens: int,
    api_key: str,
    headers: dict | None,
    signal: asyncio.Event | None,
    thinking_level: str | None,
) -> Result:
    max_tokens = min(int(0.5 * reserve_tokens), model.max_tokens if model.max_tokens > 0 else 10**9)
    llm_messages = convert_to_llm(messages)
    conv_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conv_text}\n</conversation>\n\n{_TURN_PREFIX_SUMMARIZATION_PROMPT}"
    opts = SimpleStreamOptions(
        max_tokens=max_tokens, api_key=api_key, headers=headers, signal=signal,
        reasoning=thinking_level if (thinking_level and thinking_level != "off" and model.reasoning) else None,
    )
    try:
        response = await complete_simple(
            model,
            Context(
                system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
                messages=[UserMessage(content=[TextContent(text=prompt_text)])],
            ),
            opts,
        )
    except Exception as e:
        return err(CompactionError("summarization_failed", f"Turn prefix summarization failed: {e}", e))
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Turn prefix summarization aborted"))
    if response.stop_reason == "error":
        return err(CompactionError("summarization_failed", f"Turn prefix summarization failed: {response.error_message or 'Unknown'}"))
    text = "\n".join(b.text for b in response.content if hasattr(b, "type") and b.type == "text")
    return ok(text)


async def _no_history_coro() -> Result:
    return ok("No prior history.")


async def compact(
    preparation: CompactionPreparation,
    model: Any,
    api_key: str,
    headers: dict | None = None,
    custom_instructions: str | None = None,
    signal: asyncio.Event | None = None,
    thinking_level: str | None = None,
) -> Result:
    p = preparation
    if not p.first_kept_entry_id:
        return err(CompactionError("invalid_session", "First kept entry has no UUID"))

    if p.is_split_turn and p.turn_prefix_messages:
        history_coro = (
            generate_summary(
                p.messages_to_summarize, model, p.settings.reserve_tokens,
                api_key, headers, signal, custom_instructions, p.previous_summary, thinking_level,
            )
            if p.messages_to_summarize
            else _no_history_coro()
        )
        history_result, prefix_result = await asyncio.gather(
            history_coro,
            _generate_turn_prefix_summary(
                p.turn_prefix_messages, model, p.settings.reserve_tokens,
                api_key, headers, signal, thinking_level,
            ),
        )
        if not history_result.ok:
            return history_result
        if not prefix_result.ok:
            return prefix_result
        summary = f"{history_result.value}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_result.value}"
    else:
        summary_result = await generate_summary(
            p.messages_to_summarize, model, p.settings.reserve_tokens,
            api_key, headers, signal, custom_instructions, p.previous_summary, thinking_level,
        )
        if not summary_result.ok:
            return summary_result
        summary = summary_result.value

    file_lists = compute_file_lists(p.file_ops)
    summary += format_file_operations(file_lists["readFiles"], file_lists["modifiedFiles"])

    return ok({
        "summary": summary,
        "firstKeptEntryId": p.first_kept_entry_id,
        "tokensBefore": p.tokens_before,
        "details": {"readFiles": file_lists["readFiles"], "modifiedFiles": file_lists["modifiedFiles"]},
    })
