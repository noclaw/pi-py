"""Find tool — locate files by glob pattern with fd or pathlib fallback."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pi_ai.types import TextContent

from ..harness.utils.truncate import DEFAULT_MAX_BYTES, truncate_head
from ..types import AgentTool, AgentToolResult

_DEFAULT_LIMIT = 1000

DESCRIPTION = (
    "Find files matching a glob pattern. "
    "Uses fd when available (respects .gitignore), falls back to pathlib. "
    f"Returns up to {_DEFAULT_LIMIT} results by default, sorted by path."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match files, e.g. '*.py', '**/*.json', 'src/**/*.spec.py'",
        },
        "path": {
            "type": "string",
            "description": "Directory to search (default: current directory)",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of results (default: {_DEFAULT_LIMIT})",
        },
    },
    "required": ["pattern"],
}


def _resolve(cwd: str, path: str) -> str:
    return path if os.path.isabs(path) else str(Path(cwd) / path)


def _has_fd() -> bool:
    return shutil.which("fd") is not None or shutil.which("fdfind") is not None


def _fd_binary() -> str:
    return "fd" if shutil.which("fd") else "fdfind"


def _run_fd(pattern: str, search_path: str, limit: int) -> list[str]:
    # fd uses its own glob syntax; --glob enables shell glob mode
    # Convert ** glob to fd's regex equivalent or use --glob flag
    cmd = [_fd_binary(), "--glob", pattern, ".", search_path, "--max-results", str(limit), "--color=never"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return [l for l in result.stdout.splitlines() if l.strip()]


def _python_find(pattern: str, search_path: str, limit: int) -> tuple[list[str], bool]:
    root = Path(search_path)
    # pathlib.rglob handles ** patterns
    try:
        matches = sorted(str(p.relative_to(root)) for p in root.rglob(pattern) if p.is_file())
    except Exception:
        matches = []
    limit_reached = len(matches) > limit
    return matches[:limit], limit_reached


async def _execute(
    cwd: str,
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None,
    on_update: Any,
) -> AgentToolResult:
    pattern: str = params["pattern"]
    path: str = params.get("path", "")
    limit: int = params.get("limit", _DEFAULT_LIMIT)

    search_path = _resolve(cwd, path) if path else cwd

    if not os.path.exists(search_path):
        raise FileNotFoundError(f"Path not found: {path or '.'}")
    if not os.path.isdir(search_path):
        raise ValueError(f"Path is not a directory: {path or '.'}")

    limit_reached = False

    if _has_fd():
        results = await asyncio.to_thread(_run_fd, pattern, search_path, limit)
        limit_reached = len(results) >= limit
    else:
        results, limit_reached = await asyncio.to_thread(_python_find, pattern, search_path, limit)

    if not results:
        return AgentToolResult(
            content=[TextContent(text=f"No files found matching: {pattern}")],
            details={"count": 0},
        )

    combined = "\n".join(results)
    trunc = truncate_head(combined, max_bytes=DEFAULT_MAX_BYTES)

    header = f"{len(results)} file{'s' if len(results) != 1 else ''}"
    if limit_reached:
        header += f" (limit {limit} reached)"
    if trunc.truncated:
        header += f" (output truncated)"

    output = f"{header}\n\n{trunc.content}"

    return AgentToolResult(
        content=[TextContent(text=output)],
        details={"count": len(results), "limit_reached": limit_reached},
    )


def create_find_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return await _execute(cwd, tool_call_id, params, signal, on_update)

    return AgentTool(
        name="find",
        label="Find",
        description=DESCRIPTION,
        parameters=PARAMETERS,
        execute=execute,
    )
