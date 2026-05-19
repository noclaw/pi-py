"""Edit tool — apply one or more targeted replacements to a file."""
from __future__ import annotations

import asyncio
import difflib
import os
from pathlib import Path
from typing import Any

from pi_ai.types import TextContent

from ..types import AgentTool, AgentToolResult

DESCRIPTION = (
    "Edit a file by replacing exact text. Each edit specifies old_text (must be unique in the file) "
    "and new_text. Multiple edits in one call are applied atomically. "
    "Use write to create new files. Prefer edit over write for existing files."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to edit (relative or absolute)",
        },
        "edits": {
            "type": "array",
            "description": (
                "One or more replacements. Each edit is matched against the original file "
                "(not incrementally). Do not include overlapping edits; merge them into one instead."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace. Must appear exactly once in the file.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["old_text", "new_text"],
                "additionalProperties": False,
            },
            "minItems": 1,
        },
    },
    "required": ["path", "edits"],
}


# ── Line-ending and BOM helpers ───────────────────────────────────────────────

def _strip_bom(text: str) -> tuple[str, bool]:
    if text.startswith("﻿"):
        return text[1:], True
    return text, False


def _detect_line_ending(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(text: str, ending: str) -> str:
    if ending == "\n":
        return text
    return text.replace("\n", ending)


# ── Edit application ──────────────────────────────────────────────────────────

def _apply_edits(content: str, edits: list[dict[str, str]]) -> str:
    """Apply edits in file order. Each old_text must be unique."""
    positions: list[tuple[int, int, dict]] = []  # (start, end, edit)

    for i, edit in enumerate(edits):
        old = edit["old_text"]
        if not old:
            raise ValueError(f"Edit {i + 1}: old_text must not be empty")
        first = content.find(old)
        if first == -1:
            preview = repr(old[:120]) + ("…" if len(old) > 120 else "")
            raise ValueError(
                f"Edit {i + 1}: old_text not found in file.\n"
                f"Looking for: {preview}"
            )
        if content.find(old, first + 1) != -1:
            preview = repr(old[:120]) + ("…" if len(old) > 120 else "")
            raise ValueError(
                f"Edit {i + 1}: old_text is not unique (found multiple occurrences). "
                f"Add more surrounding context to make it unique.\n"
                f"Pattern: {preview}"
            )
        positions.append((first, first + len(old), edit))

    # Sort by start position
    positions.sort(key=lambda x: x[0])

    # Check for overlaps
    for j in range(len(positions) - 1):
        _, end1, _ = positions[j]
        start2, _, edit2 = positions[j + 1]
        if end1 > start2:
            raise ValueError(
                f"Edits overlap: one edit ends at position {end1} but the next starts at {start2}. "
                "Merge overlapping edits into a single edit."
            )

    # Apply from end to start to preserve positions
    result = content
    for start, end, edit in reversed(positions):
        result = result[:start] + edit["new_text"] + result[end:]

    return result


def _prepare_arguments(args: Any) -> dict:
    """Support both array edits and legacy single old_text/new_text."""
    if not isinstance(args, dict):
        return args

    # Some models send edits as a JSON string
    edits = args.get("edits", [])
    if isinstance(edits, str):
        import json
        try:
            edits = json.loads(edits)
        except Exception:
            edits = []

    # Promote legacy top-level old_text/new_text into edits array
    old_text = args.get("old_text")
    new_text = args.get("new_text")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits = list(edits) + [{"old_text": old_text, "new_text": new_text}]

    return {**args, "edits": edits}


def _resolve(cwd: str, path: str) -> str:
    return path if os.path.isabs(path) else str(Path(cwd) / path)


async def _execute(
    cwd: str,
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None,
    on_update: Any,
) -> AgentToolResult:
    params = _prepare_arguments(params)
    path: str = params["path"]
    edits: list[dict] = params.get("edits") or []

    if not edits:
        raise ValueError("edits must contain at least one replacement")

    abs_path = _resolve(cwd, path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.access(abs_path, os.R_OK | os.W_OK):
        raise PermissionError(f"File is not readable/writable: {path}")

    original_bytes = await asyncio.to_thread(Path(abs_path).read_bytes)
    original_text = original_bytes.decode("utf-8", errors="replace")

    stripped, had_bom = _strip_bom(original_text)
    line_ending = _detect_line_ending(stripped)
    normalized = _normalize_to_lf(stripped)

    # Normalize old_text line endings too so edits work regardless of source
    normalized_edits = [
        {
            "old_text": _normalize_to_lf(e["old_text"]),
            "new_text": _normalize_to_lf(e["new_text"]),
        }
        for e in edits
    ]

    modified = _apply_edits(normalized, normalized_edits)
    restored = _restore_line_endings(modified, line_ending)
    final = ("﻿" + restored) if had_bom else restored

    await asyncio.to_thread(Path(abs_path).write_text, final, encoding="utf-8")

    # Generate unified diff for the model
    diff = "".join(difflib.unified_diff(
        normalized.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    ))

    n_edits = len(edits)
    summary = f"Applied {n_edits} edit{'s' if n_edits != 1 else ''} to {path}"
    output = f"{summary}\n\n{diff}" if diff else summary

    return AgentToolResult(
        content=[TextContent(text=output)],
        details={"path": abs_path, "diff": diff, "edits_applied": n_edits},
    )


def create_edit_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return await _execute(cwd, tool_call_id, params, signal, on_update)

    return AgentTool(
        name="edit",
        label="Edit",
        description=DESCRIPTION,
        parameters=PARAMETERS,
        execute=execute,
        prepare_arguments=_prepare_arguments,
    )
