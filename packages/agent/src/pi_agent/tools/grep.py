"""Grep tool — search file contents with ripgrep or Python re fallback."""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pi_ai.types import TextContent

from ..harness.utils.truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    truncate_head,
    truncate_line,
)
from ..types import AgentTool, AgentToolResult

_DEFAULT_LIMIT = 100

DESCRIPTION = (
    "Search file contents using a regex or literal pattern. "
    "Uses ripgrep (rg) when available, falls back to Python re. "
    f"Returns up to {_DEFAULT_LIMIT} matches by default. "
    "Each match line is truncated to 500 characters."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
        "path": {
            "type": "string",
            "description": "Directory or file to search (default: current directory)",
        },
        "glob": {
            "type": "string",
            "description": "Filter files by glob pattern, e.g. '*.py' or '**/*.spec.py'",
        },
        "ignore_case": {
            "type": "boolean",
            "description": "Case-insensitive search (default: false)",
        },
        "literal": {
            "type": "boolean",
            "description": "Treat pattern as literal string instead of regex (default: false)",
        },
        "context": {
            "type": "integer",
            "description": "Lines of context before and after each match (default: 0)",
        },
        "limit": {
            "type": "integer",
            "description": f"Maximum number of matches to return (default: {_DEFAULT_LIMIT})",
        },
    },
    "required": ["pattern"],
}


def _resolve(cwd: str, path: str) -> str:
    return path if os.path.isabs(path) else str(Path(cwd) / path)


# ── ripgrep backend ────────────────────────────────────────────────────────────

def _has_rg() -> bool:
    return shutil.which("rg") is not None


def _run_rg(
    pattern: str,
    search_path: str,
    *,
    glob: str | None,
    ignore_case: bool,
    literal: bool,
    context: int,
    limit: int,
) -> str:
    cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
    if ignore_case:
        cmd.append("--ignore-case")
    if literal:
        cmd.append("--fixed-strings")
    if context:
        cmd.extend(["-C", str(context)])
    if glob:
        cmd.extend(["--glob", glob])
    cmd.extend(["-m", str(limit)])
    cmd.extend(["--", pattern, search_path])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout


# ── Python re fallback ────────────────────────────────────────────────────────

def _python_grep(
    pattern: str,
    search_path: str,
    *,
    glob: str | None,
    ignore_case: bool,
    literal: bool,
    context_lines: int,
    limit: int,
) -> tuple[list[str], bool]:
    flags = re.IGNORECASE if ignore_case else 0
    compiled = re.compile(re.escape(pattern) if literal else pattern, flags)

    # Collect files to search
    root = Path(search_path)
    if root.is_file():
        files = [root]
    else:
        if glob:
            files = sorted(root.rglob(glob))
        else:
            files = sorted(f for f in root.rglob("*") if f.is_file())

    lines_out: list[str] = []
    limit_reached = False

    for filepath in files:
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        file_lines = text.splitlines()
        matches_in_file: list[int] = []
        for i, line in enumerate(file_lines):
            if compiled.search(line):
                matches_in_file.append(i)

        for match_idx in matches_in_file:
            start = max(0, match_idx - context_lines)
            end = min(len(file_lines), match_idx + context_lines + 1)
            for j in range(start, end):
                sep = ":" if j == match_idx else "-"
                rel = filepath.relative_to(root) if root.is_dir() else filepath
                lines_out.append(f"{rel}{sep}{j + 1}{sep}{file_lines[j]}")
            if context_lines:
                lines_out.append("--")

            if len(matches_in_file) + len(lines_out) >= limit:
                limit_reached = True
                break
        if limit_reached:
            break

    return lines_out, limit_reached


# ── Main execute ───────────────────────────────────────────────────────────────

async def _execute(
    cwd: str,
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None,
    on_update: Any,
) -> AgentToolResult:
    pattern: str = params["pattern"]
    path: str = params.get("path", "")
    glob_pat: str | None = params.get("glob")
    ignore_case: bool = params.get("ignore_case", False)
    literal: bool = params.get("literal", False)
    context: int = params.get("context", 0)
    limit: int = params.get("limit", _DEFAULT_LIMIT)

    search_path = _resolve(cwd, path) if path else cwd

    if not os.path.exists(search_path):
        raise FileNotFoundError(f"Path not found: {path or '.'}")

    raw_output: str
    limit_reached = False

    if _has_rg():
        raw_output = await asyncio.to_thread(
            _run_rg, pattern, search_path,
            glob=glob_pat, ignore_case=ignore_case, literal=literal,
            context=context, limit=limit,
        )
        lines = [truncate_line(l)["text"] for l in raw_output.splitlines() if l]
        limit_reached = len(lines) >= limit
    else:
        match_lines, limit_reached = await asyncio.to_thread(
            _python_grep, pattern, search_path,
            glob=glob_pat, ignore_case=ignore_case, literal=literal,
            context_lines=context, limit=limit,
        )
        lines = [truncate_line(l)["text"] for l in match_lines]

    if not lines:
        return AgentToolResult(
            content=[TextContent(text=f"No matches found for pattern: {pattern!r}")],
            details={"matches": 0},
        )

    combined = "\n".join(lines)
    trunc = truncate_head(combined, max_bytes=DEFAULT_MAX_BYTES)

    header = f"{len(lines)} match{'es' if len(lines) != 1 else ''}"
    if limit_reached:
        header += f" (limit {limit} reached; refine your search)"
    if trunc.truncated:
        header += f" (output truncated to {trunc.output_lines} lines)"

    output = f"{header}\n\n{trunc.content}"

    return AgentToolResult(
        content=[TextContent(text=output)],
        details={"matches": len(lines), "limit_reached": limit_reached, "truncated": trunc.truncated},
    )


def create_grep_tool(cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return await _execute(cwd, tool_call_id, params, signal, on_update)

    return AgentTool(
        name="grep",
        label="Grep",
        description=DESCRIPTION,
        parameters=PARAMETERS,
        execute=execute,
    )
