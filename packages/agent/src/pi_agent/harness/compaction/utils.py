"""File-operation tracking and conversation serialization utilities for compaction."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileOperations:
    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def create_file_ops() -> FileOperations:
    return FileOperations()


def extract_file_ops_from_message(message: Any, file_ops: FileOperations) -> None:
    role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
    if role != "assistant":
        return
    content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            args = block.get("arguments") or {}
            name = block.get("name")
        elif hasattr(block, "type"):
            block_type = block.type
            args = getattr(block, "arguments", {}) or {}
            name = getattr(block, "name", None)
        else:
            continue
        if block_type != "toolCall":
            continue
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not isinstance(path, str):
            continue
        if name == "read":
            file_ops.read.add(path)
        elif name == "write":
            file_ops.written.add(path)
        elif name == "edit":
            file_ops.edited.add(path)


def compute_file_lists(file_ops: FileOperations) -> dict:
    modified = file_ops.edited | file_ops.written
    read_only = sorted(f for f in file_ops.read if f not in modified)
    modified_files = sorted(modified)
    return {"readFiles": read_only, "modifiedFiles": modified_files}


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    sections: list[str] = []
    if read_files:
        sections.append(f"<read-files>\n{chr(10).join(read_files)}\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n{chr(10).join(modified_files)}\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


_TOOL_RESULT_MAX_CHARS = 2000


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value) or "undefined"
    except Exception:
        return "[unserializable]"


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[... {len(text) - max_chars} more characters truncated]"


def serialize_conversation(messages: list[Any]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "user":
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                    for b in content
                    if (b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "text"
                )
            else:
                text = ""
            if text:
                parts.append(f"[User]: {text}")
        elif role == "assistant":
            content = getattr(msg, "content", []) or (msg.get("content", []) if isinstance(msg, dict) else [])
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[str] = []
            for block in content:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if btype == "text":
                    t = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                    text_parts.append(t or "")
                elif btype == "thinking":
                    t = block.get("thinking") if isinstance(block, dict) else getattr(block, "thinking", "")
                    thinking_parts.append(t or "")
                elif btype == "toolCall":
                    args = block.get("arguments") if isinstance(block, dict) else getattr(block, "arguments", {})
                    name = block.get("name") if isinstance(block, dict) else getattr(block, "name", "")
                    args_str = ", ".join(f"{k}={_safe_json(v)}" for k, v in (args or {}).items())
                    tool_calls.append(f"{name}({args_str})")
            if thinking_parts:
                parts.append(f"[Assistant thinking]: {chr(10).join(thinking_parts)}")
            if text_parts:
                parts.append(f"[Assistant]: {chr(10).join(text_parts)}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")
        elif role == "toolResult":
            content = getattr(msg, "content", []) or (msg.get("content", []) if isinstance(msg, dict) else [])
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                for b in content
                if (b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "text"
            )
            if text:
                parts.append(f"[Tool result]: {_truncate_for_summary(text, _TOOL_RESULT_MAX_CHARS)}")
    return "\n\n".join(parts)
