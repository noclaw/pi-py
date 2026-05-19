"""Write tool — create or overwrite a file."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from pi_ai.types import TextContent

from ..types import AgentTool, AgentToolResult

DESCRIPTION = (
    "Create or overwrite a file with the given content. "
    "Parent directories are created automatically. "
    "For modifying existing files prefer edit (targeted replacements). "
    "Use write for new files or full rewrites."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file to write (relative or absolute)",
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file",
        },
    },
    "required": ["path", "content"],
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
    path: str = params["path"]
    content: str = params["content"]

    abs_path = _resolve(cwd, path)

    existed = os.path.exists(abs_path)

    def _write() -> None:
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        Path(abs_path).write_text(content, encoding="utf-8")

    await asyncio.to_thread(_write)

    verb = "Updated" if existed else "Created"
    lines = content.count("\n") + 1
    size = len(content.encode("utf-8"))
    summary = f"{verb} {path} ({lines} lines, {size} bytes)"

    return AgentToolResult(
        content=[TextContent(text=summary)],
        details={"path": abs_path, "created": not existed, "lines": lines, "bytes": size},
    )


def create_write_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return await _execute(cwd, tool_call_id, params, signal, on_update)

    return AgentTool(
        name="write",
        label="Write",
        description=DESCRIPTION,
        parameters=PARAMETERS,
        execute=execute,
    )
