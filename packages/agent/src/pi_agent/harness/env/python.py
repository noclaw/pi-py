"""Python/asyncio execution environment (equivalent to Node.js NodeExecutionEnv)."""
from __future__ import annotations

import asyncio
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

from ..types import (
    ExecOptions,
    ExecutionEnv,
    ExecutionError,
    FileError,
    FileInfo,
    Result,
    err,
    ok,
    to_error,
)


def _resolve(cwd: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return str(Path(cwd) / path)


def _to_file_error(exc: Exception, path: str | None = None) -> FileError:
    if isinstance(exc, FileError):
        return exc
    if isinstance(exc, asyncio.CancelledError):
        return FileError("aborted", "aborted", path, exc)
    if isinstance(exc, OSError):
        errno_map = {
            2: "not_found",     # ENOENT
            13: "permission_denied",  # EACCES
            1: "permission_denied",   # EPERM
            20: "not_directory",      # ENOTDIR
            21: "is_directory",       # EISDIR
            22: "invalid",            # EINVAL
        }
        code = errno_map.get(exc.errno, "unknown")
        return FileError(code, exc.strerror or str(exc), path, exc)
    return FileError("unknown", str(exc), path, exc)


def _to_exec_error(exc: Exception) -> ExecutionError:
    if isinstance(exc, ExecutionError):
        return exc
    if isinstance(exc, asyncio.CancelledError):
        return ExecutionError("aborted", "aborted", exc)
    return ExecutionError("unknown", str(exc), exc)


class PythonExecutionEnv(ExecutionEnv):
    """Filesystem and shell execution environment using Python stdlib."""

    def __init__(self, cwd: str, shell: str | None = None) -> None:
        self._cwd = cwd
        self._shell = shell  # custom shell path

    @property
    def cwd(self) -> str:
        return self._cwd

    # ── Path operations ────────────────────────────────────────────────────────

    async def absolute_path(self, path: str, abort_signal: asyncio.Event | None = None) -> Result:
        try:
            return ok(_resolve(self._cwd, path))
        except Exception as e:
            return err(_to_file_error(e))

    async def join_path(self, parts: list[str], abort_signal: asyncio.Event | None = None) -> Result:
        try:
            return ok(str(Path(*parts)))
        except Exception as e:
            return err(_to_file_error(e))

    # ── File reads ─────────────────────────────────────────────────────────────

    async def read_text_file(self, path: str, abort_signal: asyncio.Event | None = None) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            def _read():
                with open(resolved, encoding="utf-8") as f:
                    return f.read()
            content = await asyncio.to_thread(_read)
            return ok(content)
        except Exception as e:
            return err(_to_file_error(e, resolved))

    async def read_text_lines(
        self,
        path: str,
        max_lines: int | None = None,
        abort_signal: asyncio.Event | None = None,
    ) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            def _read():
                lines: list[str] = []
                with open(resolved, encoding="utf-8") as f:
                    for line in f:
                        lines.append(line.rstrip("\n").rstrip("\r"))
                        if max_lines is not None and len(lines) >= max_lines:
                            break
                return lines
            lines = await asyncio.to_thread(_read)
            return ok(lines)
        except Exception as e:
            return err(_to_file_error(e, resolved))

    async def read_binary_file(self, path: str, abort_signal: asyncio.Event | None = None) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            data = await asyncio.to_thread(lambda: Path(resolved).read_bytes())
            return ok(data)
        except Exception as e:
            return err(_to_file_error(e, resolved))

    # ── File writes ────────────────────────────────────────────────────────────

    async def write_file(self, path: str, content: str | bytes, abort_signal: asyncio.Event | None = None) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            def _write():
                Path(resolved).parent.mkdir(parents=True, exist_ok=True)
                mode = "w" if isinstance(content, str) else "wb"
                with open(resolved, mode, encoding="utf-8" if isinstance(content, str) else None) as f:
                    f.write(content)
            await asyncio.to_thread(_write)
            return ok(None)
        except Exception as e:
            return err(_to_file_error(e, resolved))

    async def append_file(self, path: str, content: str | bytes, abort_signal: asyncio.Event | None = None) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            def _append():
                Path(resolved).parent.mkdir(parents=True, exist_ok=True)
                mode = "a" if isinstance(content, str) else "ab"
                with open(resolved, mode, encoding="utf-8" if isinstance(content, str) else None) as f:
                    f.write(content)
            await asyncio.to_thread(_append)
            return ok(None)
        except Exception as e:
            return err(_to_file_error(e, resolved))

    # ── File info / directory ops ──────────────────────────────────────────────

    async def file_info(self, path: str, abort_signal: asyncio.Event | None = None) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            def _stat():
                s = os.lstat(resolved)
                if stat.S_ISREG(s.st_mode):
                    kind = "file"
                elif stat.S_ISDIR(s.st_mode):
                    kind = "directory"
                elif stat.S_ISLNK(s.st_mode):
                    kind = "symlink"
                else:
                    raise FileError("invalid", "Unsupported file type", resolved)
                return FileInfo(
                    name=os.path.basename(resolved),
                    path=resolved,
                    kind=kind,
                    size=s.st_size,
                    mtime_ms=s.st_mtime * 1000,
                )
            info = await asyncio.to_thread(_stat)
            return ok(info)
        except FileError as e:
            return err(e)
        except Exception as e:
            return err(_to_file_error(e, resolved))

    async def list_dir(self, path: str, abort_signal: asyncio.Event | None = None) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            def _list():
                infos: list[FileInfo] = []
                for entry in os.scandir(resolved):
                    s = entry.stat(follow_symlinks=False)
                    if stat.S_ISREG(s.st_mode):
                        kind: str = "file"
                    elif stat.S_ISDIR(s.st_mode):
                        kind = "directory"
                    elif stat.S_ISLNK(s.st_mode):
                        kind = "symlink"
                    else:
                        continue
                    infos.append(FileInfo(
                        name=entry.name,
                        path=entry.path,
                        kind=kind,
                        size=s.st_size,
                        mtime_ms=s.st_mtime * 1000,
                    ))
                return infos
            return ok(await asyncio.to_thread(_list))
        except Exception as e:
            return err(_to_file_error(e, resolved))

    async def canonical_path(self, path: str, abort_signal: asyncio.Event | None = None) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            canon = await asyncio.to_thread(os.path.realpath, resolved)
            return ok(canon)
        except Exception as e:
            return err(_to_file_error(e, resolved))

    async def exists(self, path: str, abort_signal: asyncio.Event | None = None) -> Result:
        info = await self.file_info(path)
        if info.ok:
            return ok(True)
        if info.error.code == "not_found":
            return ok(False)
        return err(info.error)

    async def create_dir(
        self,
        path: str,
        recursive: bool = True,
        abort_signal: asyncio.Event | None = None,
    ) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            await asyncio.to_thread(lambda: Path(resolved).mkdir(parents=recursive, exist_ok=True))
            return ok(None)
        except Exception as e:
            return err(_to_file_error(e, resolved))

    async def remove(
        self,
        path: str,
        recursive: bool = False,
        force: bool = False,
        abort_signal: asyncio.Event | None = None,
    ) -> Result:
        resolved = _resolve(self._cwd, path)
        try:
            def _rm():
                if not os.path.lexists(resolved):
                    if force:
                        return
                    raise FileNotFoundError(f"No such file: {resolved}")
                if os.path.isdir(resolved) and not os.path.islink(resolved):
                    if recursive:
                        shutil.rmtree(resolved)
                    else:
                        os.rmdir(resolved)
                else:
                    os.remove(resolved)
            await asyncio.to_thread(_rm)
            return ok(None)
        except Exception as e:
            return err(_to_file_error(e, resolved))

    async def create_temp_dir(self, prefix: str = "tmp-", abort_signal: asyncio.Event | None = None) -> Result:
        try:
            path = await asyncio.to_thread(lambda: tempfile.mkdtemp(prefix=prefix))
            return ok(path)
        except Exception as e:
            return err(_to_file_error(e))

    async def create_temp_file(
        self,
        prefix: str = "",
        suffix: str = "",
        abort_signal: asyncio.Event | None = None,
    ) -> Result:
        try:
            def _mk():
                fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
                os.close(fd)
                return path
            path = await asyncio.to_thread(_mk)
            return ok(path)
        except Exception as e:
            return err(_to_file_error(e))

    async def cleanup(self) -> None:
        pass  # nothing to clean up

    # ── Shell execution ────────────────────────────────────────────────────────

    async def exec(self, command: str, options: ExecOptions | None = None) -> Result:
        opts = options or ExecOptions()

        if opts.abort_signal is not None and opts.abort_signal.is_set():
            return err(ExecutionError("aborted", "aborted"))

        cwd = _resolve(self._cwd, opts.cwd) if opts.cwd else self._cwd
        shell = self._shell or "/bin/bash"
        if not os.path.exists(shell):
            shell = shutil.which("bash") or shutil.which("sh") or "sh"

        env = {**os.environ, **(opts.env or {})}

        try:
            proc = await asyncio.create_subprocess_exec(
                shell, "-c", command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            return err(ExecutionError("spawn_error", str(e), to_error(e)))

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        callback_error: list[Exception | None] = [None]

        async def read_stream(stream: asyncio.StreamReader, parts: list[str], callback: Any) -> None:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                parts.append(text)
                if callback and not callback_error[0]:
                    try:
                        callback(text)
                    except Exception as e:
                        callback_error[0] = e
                        proc.kill()

        abort_task: asyncio.Task | None = None

        async def watch_abort() -> None:
            if opts.abort_signal is None:
                return
            while not opts.abort_signal.is_set():
                await asyncio.sleep(0.1)
            proc.kill()

        read_tasks = [
            asyncio.create_task(read_stream(proc.stdout, stdout_parts, opts.on_stdout)),
            asyncio.create_task(read_stream(proc.stderr, stderr_parts, opts.on_stderr)),
        ]
        if opts.abort_signal is not None:
            abort_task = asyncio.create_task(watch_abort())

        try:
            if opts.timeout is not None:
                try:
                    await asyncio.wait_for(asyncio.gather(*read_tasks), timeout=opts.timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await asyncio.gather(*read_tasks, return_exceptions=True)
                    return err(ExecutionError("timeout", f"timeout:{opts.timeout}"))
            else:
                await asyncio.gather(*read_tasks)

            await proc.wait()
        except asyncio.CancelledError:
            proc.kill()
            return err(ExecutionError("aborted", "aborted"))
        finally:
            if abort_task:
                abort_task.cancel()
                try:
                    await abort_task
                except asyncio.CancelledError:
                    pass

        if callback_error[0]:
            return err(ExecutionError("callback_error", str(callback_error[0]), callback_error[0]))

        if opts.abort_signal is not None and opts.abort_signal.is_set():
            return err(ExecutionError("aborted", "aborted"))

        return ok({
            "stdout": "".join(stdout_parts),
            "stderr": "".join(stderr_parts),
            "exitCode": proc.returncode or 0,
        })
