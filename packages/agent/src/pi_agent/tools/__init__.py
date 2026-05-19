"""Built-in tools for a general-purpose coding agent.

Usage::

    from pi_agent.tools import create_tools, load_context_files, build_system_prompt
    from pi_agent import AgentHarness, PythonExecutionEnv, InMemorySessionRepo
    import pi_ai

    env   = PythonExecutionEnv(cwd="/my/project")
    tools = create_tools(env)
    ctx   = load_context_files("/my/project")

    harness = AgentHarness(
        env=env,
        session=await InMemorySessionRepo().create(),
        model=pi_ai.get_model("anthropic", "claude-sonnet-4-6"),
        tools=tools,
        system_prompt=build_system_prompt(tools, ctx),
    )
    reply = await harness.prompt("Explain the codebase structure.")
"""
from __future__ import annotations

import os
from pathlib import Path

from ..harness.types import ExecutionEnv
from ..types import AgentTool
from .bash import create_bash_tool
from .edit import create_edit_tool
from .find import create_find_tool
from .grep import create_grep_tool
from .ls import create_ls_tool
from .read import create_read_tool
from .write import create_write_tool


# ── Factory ────────────────────────────────────────────────────────────────────

def create_tools(env: ExecutionEnv, cwd: str | None = None) -> list[AgentTool]:
    """Return all 7 built-in tools configured for the given working directory.

    :param env: Execution environment (used by bash for shell I/O).
    :param cwd: Working directory; defaults to ``env.cwd``.
    """
    resolved_cwd = cwd or env.cwd
    return [
        create_read_tool(resolved_cwd),
        create_bash_tool(env, resolved_cwd),
        create_edit_tool(resolved_cwd),
        create_write_tool(resolved_cwd),
        create_grep_tool(resolved_cwd),
        create_find_tool(resolved_cwd),
        create_ls_tool(resolved_cwd),
    ]


# ── Context file loading ───────────────────────────────────────────────────────

_DEFAULT_CONTEXT_FILES = ["CLAUDE.md", "AGENTS.md"]


def load_context_files(
    cwd: str,
    filenames: list[str] | None = None,
) -> str:
    """Walk up the directory tree from *cwd* collecting context files.

    Files found closer to the root are listed first (least-specific to
    most-specific), matching the convention that project-level CLAUDE.md
    files refine global instructions.

    :param cwd: Starting directory.
    :param filenames: Context file names to look for.
                      Defaults to ``["CLAUDE.md", "AGENTS.md"]``.
    :returns: Combined content of all found files, separated by blank lines,
              or an empty string if none are found.
    """
    names = filenames if filenames is not None else _DEFAULT_CONTEXT_FILES
    current = Path(cwd).resolve()
    found: list[tuple[Path, str]] = []

    while True:
        for name in names:
            candidate = current / name
            if candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8").strip()
                    if content:
                        found.append((candidate, content))
                except OSError:
                    pass
        parent = current.parent
        if parent == current:
            break
        current = parent

    if not found:
        return ""

    # Reverse so root-level files come first, project-level files last
    parts = [f"<!-- {path} -->\n{content}" for path, content in reversed(found)]
    return "\n\n".join(parts)


# ── System prompt builder ──────────────────────────────────────────────────────

_GUIDELINES = [
    "Use `read` to examine files instead of `bash` with `cat` or `sed`.",
    "Use `edit` for targeted changes to existing files. Prefer it over `write` for modifications.",
    "Use `write` only to create new files or completely replace a file's contents.",
    "Use `bash` to run commands, tests, or any task requiring a shell.",
    "When reading large files, use `offset` and `limit` to navigate incrementally.",
    "Prefer `grep` over `bash` for text search — it respects output limits automatically.",
    "Use `find` to locate files by name or glob pattern.",
    "Use `ls` to understand directory structure before diving into files.",
    "Think step by step before making changes. Read relevant files first.",
    "Verify changes by reading back the modified file or running tests via `bash`.",
]


def build_system_prompt(
    tools: list[AgentTool] | None = None,
    context: str = "",
) -> str:
    """Build a system prompt for a general-purpose coding agent.

    :param tools: Tools to list in the prompt. If None, the full built-in
                  tool set is assumed (descriptions are generated from those passed).
    :param context: Optional extra context to append (e.g. from :func:`load_context_files`).
    :returns: A complete system prompt string.
    """
    parts: list[str] = [
        "You are a helpful coding assistant. "
        "You have access to tools for reading, editing, and executing code."
    ]

    if tools:
        tool_lines = "\n".join(f"- **{t.name}**: {t.description}" for t in tools)
        parts += ["", "## Available Tools", tool_lines]

    guideline_block = "\n".join(f"- {g}" for g in _GUIDELINES)
    parts += ["", "## Guidelines", guideline_block]

    if context:
        parts += ["", "## Project Context", context]

    return "\n".join(parts)


__all__ = [
    "create_tools",
    "load_context_files",
    "build_system_prompt",
    "create_read_tool",
    "create_bash_tool",
    "create_edit_tool",
    "create_write_tool",
    "create_grep_tool",
    "create_find_tool",
    "create_ls_tool",
]
