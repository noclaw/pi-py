"""Ls tool — list directory contents."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from pi_ai.types import TextContent

from ..harness.utils.truncate import DEFAULT_MAX_BYTES, truncate_head
from ..types import AgentTool, AgentToolResult

_DEFAULT_LIMIT = 500

DESCRIPTION = (
    "List the contents of a directory. "
    "Shows files and subdirectories sorted by name. "
    f"Returns up to {_DEFAULT_LIMIT} entries by default."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory to list (default: current directory)",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of entries to return (default: {_DEFAULT_LIMIT})",
        },
    },
}


def _resolve(cwd: str, path: str) -> str:
    return path if os.path.isabs(path) else str(Path(cwd) / path)


async def _execute(
    cwd: str,
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None,
    on_update: Any,
) -> AgentToolResult:
    path: str = params.get("path", "")
    limit: int = params.get("limit", _DEFAULT_LIMIT)

    target = _resolve(cwd, path) if path else cwd

    if not os.path.exists(target):
        raise FileNotFoundError(f"Path not found: {path or '.'}")
    if not os.path.isdir(target):
        raise ValueError(f"Path is not a directory: {path or '.'}")

    def _list() -> list[tuple[str, str]]:
        entries = []
        with os.scandir(target) as it:
            for entry in sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower())):
                if entry.is_symlink():
                    kind = "symlink"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "dir"
                else:
                    kind = "file"
                entries.append((entry.name, kind))
        return entries

    all_entries = await asyncio.to_thread(_list)
    limit_reached = len(all_entries) > limit
    shown = all_entries[:limit]

    # Format: dirs first (already sorted above), then files
    lines = []
    for name, kind in shown:
        suffix = "/" if kind == "dir" else ("@" if kind == "symlink" else "")
        lines.append(f"{name}{suffix}")

    combined = "\n".join(lines)
    trunc = truncate_head(combined, max_bytes=DEFAULT_MAX_BYTES)

    display_path = path or "."
    header = f"{display_path}: {len(shown)} entries"
    if limit_reached:
        header += f" (showing {limit} of {len(all_entries)})"
    if trunc.truncated:
        header += " (output truncated)"

    output = f"{header}\n\n{trunc.content}"

    return AgentToolResult(
        content=[TextContent(text=output)],
        details={"path": target, "count": len(shown), "total": len(all_entries)},
    )


def create_ls_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return await _execute(cwd, tool_call_id, params, signal, on_update)

    return AgentTool(
        name="ls",
        label="List",
        description=DESCRIPTION,
        parameters=PARAMETERS,
        execute=execute,
    )
