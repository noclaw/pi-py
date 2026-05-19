"""Branch summarization for session tree navigation."""
from __future__ import annotations

import asyncio
from typing import Any

from pi_ai import complete_simple
from pi_ai.types import Context, SimpleStreamOptions, TextContent, UserMessage

from ..messages import convert_to_llm, create_branch_summary_message, create_compaction_summary_message, create_custom_message
from ..types import BranchSummaryError, Result, err, ok
from .compaction import SUMMARIZATION_SYSTEM_PROMPT, estimate_tokens
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)


# ── Branch entry collection ────────────────────────────────────────────────────

async def collect_entries_for_branch_summary(
    session: Any,  # Session
    old_leaf_id: str | None,
    target_id: str,
) -> dict:
    if not old_leaf_id:
        return {"entries": [], "commonAncestorId": None}

    old_path_set = {e["id"] for e in await session.get_branch(old_leaf_id)}
    target_path = await session.get_branch(target_id)
    common_ancestor_id: str | None = None
    for entry in reversed(target_path):
        if entry["id"] in old_path_set:
            common_ancestor_id = entry["id"]
            break

    entries: list[dict] = []
    current: str | None = old_leaf_id
    while current and current != common_ancestor_id:
        entry = await session.get_entry(current)
        if not entry:
            from ..types import SessionError
            raise SessionError("invalid_session", f"Entry {current} not found")
        entries.append(entry)
        current = entry.get("parentId")
    entries.reverse()

    return {"entries": entries, "commonAncestorId": common_ancestor_id}


# ── Branch entry message helpers ───────────────────────────────────────────────

def _get_message_from_entry(entry: dict) -> Any | None:
    t = entry.get("type")
    if t == "message":
        msg = entry.get("message", {})
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "toolResult":
            return None
        return msg
    if t == "custom_message":
        return create_custom_message(
            entry.get("customType", ""), entry.get("content", ""),
            entry.get("display", True), entry.get("details"), entry.get("timestamp", ""),
        )
    if t == "branch_summary":
        return create_branch_summary_message(
            entry.get("summary", ""), entry.get("fromId", ""), entry.get("timestamp", "")
        )
    if t == "compaction":
        return create_compaction_summary_message(
            entry.get("summary", ""), entry.get("tokensBefore", 0), entry.get("timestamp", "")
        )
    return None


def prepare_branch_entries(entries: list[dict], token_budget: int = 0) -> dict:
    messages: list[Any] = []
    file_ops = create_file_ops()
    total_tokens = 0

    for entry in entries:
        if entry.get("type") == "branch_summary" and not entry.get("fromHook") and entry.get("details"):
            details = entry["details"]
            if isinstance(details, dict):
                for f in (details.get("readFiles") or []):
                    file_ops.read.add(f)
                for f in (details.get("modifiedFiles") or []):
                    file_ops.edited.add(f)

    for entry in reversed(entries):
        msg = _get_message_from_entry(entry)
        if msg is None:
            continue
        extract_file_ops_from_message(msg, file_ops)
        tokens = estimate_tokens(msg)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            t = entry.get("type")
            if t in ("compaction", "branch_summary") and total_tokens < token_budget * 0.9:
                messages.insert(0, msg)
                total_tokens += tokens
            break
        messages.insert(0, msg)
        total_tokens += tokens

    return {"messages": messages, "fileOps": file_ops, "totalTokens": total_tokens}


# ── Summary prompts ────────────────────────────────────────────────────────────

_BRANCH_SUMMARY_PREAMBLE = "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n"

_BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_branch_summary(
    entries: list[dict],
    model: Any,
    api_key: str,
    headers: dict | None = None,
    signal: asyncio.Event | None = None,
    custom_instructions: str | None = None,
    replace_instructions: bool = False,
    reserve_tokens: int = 16384,
) -> Result:
    context_window = getattr(model, "context_window", None) or 128000
    token_budget = context_window - reserve_tokens

    branch = prepare_branch_entries(entries, token_budget)
    messages = branch["messages"]
    file_ops: FileOperations = branch["fileOps"]

    if not messages:
        return ok({"summary": "No content to summarize", "readFiles": [], "modifiedFiles": []})

    llm_messages = convert_to_llm(messages)
    conv_text = serialize_conversation(llm_messages)

    if replace_instructions and custom_instructions:
        instructions = custom_instructions
    elif custom_instructions:
        instructions = f"{_BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}"
    else:
        instructions = _BRANCH_SUMMARY_PROMPT

    prompt_text = f"<conversation>\n{conv_text}\n</conversation>\n\n{instructions}"
    opts = SimpleStreamOptions(api_key=api_key, headers=headers, signal=signal, max_tokens=2048)

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
        return err(BranchSummaryError("summarization_failed", f"Branch summary request failed: {e}", e))

    if response.stop_reason == "aborted":
        return err(BranchSummaryError("aborted", response.error_message or "Branch summary aborted"))
    if response.stop_reason == "error":
        return err(BranchSummaryError("summarization_failed", f"Branch summary failed: {response.error_message or 'Unknown'}"))

    summary = "\n".join(b.text for b in response.content if hasattr(b, "type") and b.type == "text")
    summary = _BRANCH_SUMMARY_PREAMBLE + summary
    file_lists = compute_file_lists(file_ops)
    summary += format_file_operations(file_lists["readFiles"], file_lists["modifiedFiles"])

    return ok({
        "summary": summary or "No summary generated",
        "readFiles": file_lists["readFiles"],
        "modifiedFiles": file_lists["modifiedFiles"],
    })
