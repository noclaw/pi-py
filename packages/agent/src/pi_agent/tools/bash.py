"""Bash tool — run a shell command and return its output."""
from __future__ import annotations

import asyncio
from typing import Any

from pi_ai.types import TextContent

from ..harness.types import ExecutionEnv
from ..harness.utils.shell_output import ShellCaptureOptions, execute_shell_with_capture
from ..harness.utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size
from ..types import AgentTool, AgentToolResult

DESCRIPTION = (
    f"Execute a bash command in the current working directory. "
    f"Returns combined stdout and stderr. "
    f"Output is truncated to the last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
    "(whichever comes first); if truncated, full output is saved to a temp file. "
    "Provide timeout (seconds) for long-running commands. "
    "Non-zero exit codes raise an error so the model can retry or adjust."
)

PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Bash command to execute",
        },
        "timeout": {
            "type": "number",
            "description": "Timeout in seconds (optional, no default)",
        },
    },
    "required": ["command"],
}


async def _execute(
    env: ExecutionEnv,
    cwd: str,
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None,
    on_update: Any,
) -> AgentToolResult:
    command: str = params["command"]
    timeout: float | None = params.get("timeout")

    if signal and signal.is_set():
        raise RuntimeError("aborted")

    # Stream partial output so subscribers can see progress.
    partial_chunks: list[str] = []

    def on_chunk(chunk: str) -> None:
        partial_chunks.append(chunk)
        if on_update:
            on_update(AgentToolResult(
                content=[TextContent(text="".join(partial_chunks))],
                details={},
            ))

    opts = ShellCaptureOptions(
        cwd=cwd,
        timeout=timeout,
        abort_signal=signal,
        on_chunk=on_chunk,
    )

    result = await execute_shell_with_capture(env, command, opts)

    if not result.ok:
        err = result.error
        code = getattr(err, "code", "unknown")
        if code == "aborted":
            raise RuntimeError("Command aborted")
        if code == "timeout":
            current_output = "".join(partial_chunks)
            msg = current_output + (f"\n\nCommand timed out after {timeout} seconds" if current_output else f"Command timed out after {timeout} seconds")
            raise RuntimeError(msg)
        raise RuntimeError(str(err))

    capture = result.value
    output = capture.output or "(no output)"

    if capture.cancelled:
        raise RuntimeError((output.rstrip() + "\n\nCommand aborted").lstrip("\n"))

    suffix = ""
    if capture.full_output_path:
        suffix += f"\n\n[Output truncated. Full output: {capture.full_output_path}]"

    if capture.exit_code is not None and capture.exit_code != 0:
        raise RuntimeError(f"{output}{suffix}\n\nCommand exited with code {capture.exit_code}".lstrip("\n"))

    return AgentToolResult(
        content=[TextContent(text=output + suffix)],
        details={
            "exit_code": capture.exit_code,
            "truncated": capture.truncated,
            "full_output_path": capture.full_output_path,
        },
    )


def create_bash_tool(env: ExecutionEnv, cwd: str) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return await _execute(env, cwd, tool_call_id, params, signal, on_update)

    return AgentTool(
        name="bash",
        label="Bash",
        description=DESCRIPTION,
        parameters=PARAMETERS,
        execute=execute,
    )
