"""Custom harness message types and convert_to_llm."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Union

from pi_ai.types import ImageContent, Message, TextContent, UserMessage


COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
)
BRANCH_SUMMARY_SUFFIX = "</summary>"


# ── Custom message types ───────────────────────────────────────────────────────

@dataclass
class BashExecutionMessage:
    role: str = "bashExecution"
    command: str = ""
    output: str = ""
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    exclude_from_context: bool = False


@dataclass
class CustomMessage:
    role: str = "custom"
    custom_type: str = ""
    content: Union[str, list[Union[TextContent, ImageContent]]] = ""
    display: bool = True
    details: Any = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class BranchSummaryMessage:
    role: str = "branchSummary"
    summary: str = ""
    from_id: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class CompactionSummaryMessage:
    role: str = "compactionSummary"
    summary: str = ""
    tokens_before: int = 0
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


# ── Factory helpers ────────────────────────────────────────────────────────────

def bash_execution_to_text(msg: BashExecutionMessage) -> str:
    text = f"Ran `{msg.command}`\n"
    if msg.output:
        text += f"```\n{msg.output}\n```"
    else:
        text += "(no output)"
    if msg.cancelled:
        text += "\n\n(command cancelled)"
    elif msg.exit_code is not None and msg.exit_code != 0:
        text += f"\n\nCommand exited with code {msg.exit_code}"
    if msg.truncated and msg.full_output_path:
        text += f"\n\n[Output truncated. Full output: {msg.full_output_path}]"
    return text


def create_branch_summary_message(summary: str, from_id: str, timestamp: str) -> BranchSummaryMessage:
    import datetime
    ts = int(datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
    return BranchSummaryMessage(summary=summary, from_id=from_id, timestamp=ts)


def create_compaction_summary_message(summary: str, tokens_before: int, timestamp: str) -> CompactionSummaryMessage:
    import datetime
    ts = int(datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
    return CompactionSummaryMessage(summary=summary, tokens_before=tokens_before, timestamp=ts)


def create_custom_message(
    custom_type: str,
    content: Union[str, list[Union[TextContent, ImageContent]]],
    display: bool,
    details: Any,
    timestamp: str,
) -> CustomMessage:
    import datetime
    ts = int(datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
    return CustomMessage(
        custom_type=custom_type, content=content, display=display, details=details, timestamp=ts
    )


# ── convert_to_llm ─────────────────────────────────────────────────────────────

def convert_to_llm(messages: list[Any]) -> list[Message]:
    """Convert AgentMessages (including harness custom types) to LLM-compatible Message list."""
    result: list[Message] = []
    for m in messages:
        if isinstance(m, BashExecutionMessage):
            if m.exclude_from_context:
                continue
            result.append(UserMessage(
                content=[TextContent(text=bash_execution_to_text(m))],
                timestamp=m.timestamp,
            ))
        elif isinstance(m, CustomMessage):
            content = (
                [TextContent(text=m.content)] if isinstance(m.content, str) else m.content
            )
            result.append(UserMessage(content=content, timestamp=m.timestamp))
        elif isinstance(m, BranchSummaryMessage):
            result.append(UserMessage(
                content=[TextContent(text=BRANCH_SUMMARY_PREFIX + m.summary + BRANCH_SUMMARY_SUFFIX)],
                timestamp=m.timestamp,
            ))
        elif isinstance(m, CompactionSummaryMessage):
            result.append(UserMessage(
                content=[TextContent(text=COMPACTION_SUMMARY_PREFIX + m.summary + COMPACTION_SUMMARY_SUFFIX)],
                timestamp=m.timestamp,
            ))
        elif hasattr(m, "role") and m.role in ("user", "assistant", "toolResult"):
            result.append(m)
    return result
