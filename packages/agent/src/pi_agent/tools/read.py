"""Read tool — read a file with optional offset/limit and smart truncation."""
from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from pi_ai.types import ImageContent, TextContent

from ..harness.types import ExecutionEnv
from ..harness.utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
)
from ..types import AgentTool, AgentToolResult

_SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

DESCRIPTION = (
    f"Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "
    f"For text files, output is truncated to {DEFAULT_MAX_LINES} lines or "
    f"{DEFAULT_MAX_BYTES // 1024}KB (whichever comes first). "
    "Use offset/limit for large files. When you need the full file, continue with offset until complete."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to read (relative or absolute)",
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (1-indexed)",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read",
        },
    },
    "required": ["path"],
}


def _detect_image_mime(path: str) -> str | None:
    mime, _ = mimetypes.guess_type(path)
    return mime if mime in _SUPPORTED_IMAGE_TYPES else None


def _resolve(cwd: str, path: str) -> str:
    return path if os.path.isabs(path) else str(Path(cwd) / path)


async def _execute(
    cwd: str,
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None,
    on_update: Any,
) -> AgentToolResult:
    path: str = params["path"]
    offset: int | None = params.get("offset")
    limit: int | None = params.get("limit")

    abs_path = _resolve(cwd, path)

    if signal and signal.is_set():
        raise RuntimeError("Operation aborted")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.access(abs_path, os.R_OK):
        raise PermissionError(f"File is not readable: {path}")

    mime = _detect_image_mime(abs_path)
    if mime:
        raw = await asyncio.to_thread(Path(abs_path).read_bytes)
        b64 = base64.b64encode(raw).decode()
        note = f"Read image file [{mime}]"
        return AgentToolResult(
            content=[TextContent(text=note), ImageContent(data=b64, mime_type=mime)],
            details={"path": abs_path, "mime_type": mime},
        )

    # Text file
    text = await asyncio.to_thread(Path(abs_path).read_text, encoding="utf-8", errors="replace")
    all_lines = text.split("\n")
    total_file_lines = len(all_lines)

    start_idx = max(0, (offset - 1) if offset else 0)
    if offset and start_idx >= len(all_lines):
        raise ValueError(f"Offset {offset} is beyond end of file ({total_file_lines} lines total)")

    start_display = start_idx + 1

    if limit is not None:
        end_idx = min(start_idx + limit, len(all_lines))
        selected = "\n".join(all_lines[start_idx:end_idx])
        user_limited_lines: int | None = end_idx - start_idx
    else:
        selected = "\n".join(all_lines[start_idx:])
        user_limited_lines = None

    trunc = truncate_head(selected)

    if trunc.first_line_exceeds_limit:
        first_line_size = format_size(len(all_lines[start_idx].encode("utf-8")))
        output = (
            f"[Line {start_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
            f"Use bash: sed -n '{start_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
        )
    elif trunc.truncated:
        end_display = start_display + trunc.output_lines - 1
        next_offset = end_display + 1
        output = trunc.content
        if trunc.truncated_by == "lines":
            output += f"\n\n[Showing lines {start_display}-{end_display} of {total_file_lines}. Use offset={next_offset} to continue.]"
        else:
            output += (
                f"\n\n[Showing lines {start_display}-{end_display} of {total_file_lines} "
                f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
            )
    elif user_limited_lines is not None and start_idx + user_limited_lines < len(all_lines):
        remaining = len(all_lines) - (start_idx + user_limited_lines)
        next_offset = start_idx + user_limited_lines + 1
        output = f"{trunc.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
    else:
        output = trunc.content

    return AgentToolResult(
        content=[TextContent(text=output)],
        details={"path": abs_path, "truncated": trunc.truncated},
    )


def create_read_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return await _execute(cwd, tool_call_id, params, signal, on_update)

    return AgentTool(
        name="read",
        label="Read",
        description=DESCRIPTION,
        parameters=PARAMETERS,
        execute=execute,
    )
