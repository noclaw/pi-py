"""Async subprocess transport for the Pi RPC bridge.

Owns the child process lifecycle and the byte-level plumbing:
* spawns ``pi --mode rpc`` with piped stdio,
* frames stdout into JSONL lines (delivered to an ``on_line`` callback),
* accumulates stderr into a bounded ring buffer for diagnostics,
* writes commands as JSONL to stdin,
* shuts down with SIGTERM then SIGKILL (mirroring Pi's reference client).
"""

from __future__ import annotations

import asyncio
import collections
import signal
from typing import Any, Callable

from .errors import PiNotStartedError, PiProcessError
from .jsonl import JsonlDecoder, serialize_line

_READ_CHUNK = 1 << 16
_STDERR_RING = 256  # retain the most recent stderr lines for error messages


class Transport:
    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str],
        on_line: Callable[[str], None],
    ) -> None:
        self._argv = argv
        self._cwd = cwd
        self._env = env
        self._on_line = on_line
        self._proc: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr: collections.deque[str] = collections.deque(maxlen=_STDERR_RING)

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode if self._proc else None

    def stderr_text(self) -> str:
        return "".join(self._stderr)

    async def start(self) -> None:
        if self._proc is not None:
            raise PiProcessError("Transport already started")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
                env=self._env,
            )
        except OSError as exc:
            raise PiProcessError(f"Failed to spawn {self._argv[0]!r}: {exc}") from exc

        self._stdout_task = asyncio.create_task(self._read_stdout(), name="pi-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="pi-stderr")

        # Give the process a moment to fail fast (bad flag, missing runtime, etc.).
        await asyncio.sleep(0.1)
        if self._proc.returncode is not None:
            raise PiProcessError(
                f"`pi` exited immediately (code={self._proc.returncode})",
                stderr=self.stderr_text(),
            )

    async def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        decoder = JsonlDecoder()
        stream = self._proc.stdout
        while True:
            chunk = await stream.read(_READ_CHUNK)
            if not chunk:
                for line in decoder.flush():
                    self._on_line(line)
                return
            for line in decoder.feed(chunk):
                self._on_line(line)

    async def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        stream = self._proc.stderr
        while True:
            raw = await stream.readline()
            if not raw:
                return
            self._stderr.append(raw.decode("utf-8", errors="replace"))

    async def write_line(self, obj: Any) -> None:
        if not self._proc or not self._proc.stdin:
            raise PiNotStartedError("Transport not started")
        if self._proc.returncode is not None:
            raise PiProcessError(
                f"`pi` has exited (code={self._proc.returncode})", stderr=self.stderr_text()
            )
        self._proc.stdin.write(serialize_line(obj))
        await self._proc.stdin.drain()

    async def stop(self, timeout: float = 1.0) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.returncode is None:
            # Closing stdin asks Pi to shut down cleanly; SIGTERM nudges it along.
            try:
                if proc.stdin and not proc.stdin.is_closing():
                    proc.stdin.close()
            except (OSError, RuntimeError):
                pass
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

        for task in (self._stdout_task, self._stderr_task):
            if task:
                task.cancel()
        self._proc = None
