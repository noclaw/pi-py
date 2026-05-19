"""Shell output capture with streaming, truncation, and temp-file overflow."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ..types import ExecutionEnv, ExecutionError, ExecOptions, err, ok, Result, to_error
from .truncate import DEFAULT_MAX_BYTES, truncate_tail


@dataclass
class ShellCaptureOptions:
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout: float | None = None
    abort_signal: asyncio.Event | None = None
    on_chunk: Any | None = None  # Callable[[str], None]


@dataclass
class ShellCaptureResult:
    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    full_output_path: str | None = None


def sanitize_binary_output(s: str) -> str:
    result: list[str] = []
    for char in s:
        code = ord(char)
        if code in (0x09, 0x0A, 0x0D):
            result.append(char)
        elif code <= 0x1F:
            pass  # strip control chars
        elif 0xFFF9 <= code <= 0xFFFB:
            pass  # strip interlinear annotation chars
        else:
            result.append(char)
    return "".join(result)


async def execute_shell_with_capture(
    env: ExecutionEnv,
    command: str,
    options: ShellCaptureOptions | None = None,
) -> Result:
    opts = options or ShellCaptureOptions()
    output_chunks: list[str] = []
    output_bytes = 0
    max_output_bytes = DEFAULT_MAX_BYTES * 2
    total_bytes = 0
    full_output_path: list[str | None] = [None]  # mutable cell
    capture_error: list[Exception | None] = [None]
    write_lock = asyncio.Lock()
    write_chain: list[Any] = [None]  # current write task

    async def ensure_full_output_file(initial: str) -> None:
        if full_output_path[0] or capture_error[0]:
            return
        async with write_lock:
            if full_output_path[0]:
                return
            result = await env.create_temp_file(prefix="bash-", suffix=".log")
            if not result.ok:
                capture_error[0] = to_error(result.error)
                return
            full_output_path[0] = result.value
            append_result = await env.append_file(full_output_path[0], initial)
            if not append_result.ok:
                capture_error[0] = to_error(append_result.error)

    async def append_full_output(text: str) -> None:
        if not full_output_path[0] or capture_error[0]:
            return
        async with write_lock:
            if not full_output_path[0] or capture_error[0]:
                return
            result = await env.append_file(full_output_path[0], text)
            if not result.ok:
                capture_error[0] = to_error(result.error)

    pending_appends: list[asyncio.Task] = []

    def on_chunk(chunk: str) -> None:
        nonlocal total_bytes, output_bytes
        try:
            total_bytes += len(chunk.encode("utf-8"))
            text = sanitize_binary_output(chunk).replace("\r", "")
            if total_bytes > DEFAULT_MAX_BYTES and not full_output_path[0]:
                pending_appends.append(
                    asyncio.create_task(ensure_full_output_file("".join(output_chunks) + text))
                )
            else:
                if full_output_path[0]:
                    pending_appends.append(asyncio.create_task(append_full_output(text)))
            output_chunks.append(text)
            output_bytes += len(text)
            while output_bytes > max_output_bytes and len(output_chunks) > 1:
                removed = output_chunks.pop(0)
                output_bytes -= len(removed)
            if opts.on_chunk:
                opts.on_chunk(text)
        except Exception as e:
            capture_error[0] = e

    exec_options = ExecOptions(
        cwd=opts.cwd,
        env=opts.env,
        timeout=opts.timeout,
        abort_signal=opts.abort_signal,
        on_stdout=on_chunk,
        on_stderr=on_chunk,
    )

    try:
        exec_result = await env.exec(command, exec_options)
    except Exception as e:
        return err(ExecutionError("unknown", str(e), to_error(e)))

    tail_output = "".join(output_chunks)
    trunc = truncate_tail(tail_output)
    if trunc.truncated and not full_output_path[0]:
        pending_appends.append(asyncio.create_task(ensure_full_output_file(tail_output)))

    if pending_appends:
        await asyncio.gather(*pending_appends, return_exceptions=True)

    if capture_error[0]:
        return err(ExecutionError("unknown", str(capture_error[0]), capture_error[0]))

    output = trunc.content if trunc.truncated else tail_output
    cancelled = opts.abort_signal is not None and opts.abort_signal.is_set()

    if not exec_result.ok:
        if exec_result.error.code == "aborted" or cancelled:
            return ok(ShellCaptureResult(
                output=output, exit_code=None, cancelled=True,
                truncated=trunc.truncated, full_output_path=full_output_path[0],
            ))
        return err(exec_result.error)

    return ok(ShellCaptureResult(
        output=output,
        exit_code=None if cancelled else exec_result.value.get("exitCode"),
        cancelled=cancelled,
        truncated=trunc.truncated,
        full_output_path=full_output_path[0],
    ))
