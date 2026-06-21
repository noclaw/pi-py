"""Exception hierarchy for the Pi SDK."""

from __future__ import annotations


class PiError(Exception):
    """Base class for all SDK errors."""


class PiNotStartedError(PiError):
    """Raised when an operation requires a started client but it isn't running."""


class PiProcessError(PiError):
    """The underlying `pi` subprocess failed to start, exited, or became unusable.

    Captured stderr (if any) is included to aid debugging.
    """

    def __init__(self, message: str, *, stderr: str | None = None) -> None:
        self.stderr = stderr
        if stderr:
            message = f"{message}\nStderr:\n{stderr}"
        super().__init__(message)


class PiTimeoutError(PiError):
    """A request or wait exceeded its allotted time."""


class PiCommandError(PiError):
    """The agent returned a `success: false` response for a command."""

    def __init__(self, command: str, error: str) -> None:
        self.command = command
        self.error = error
        super().__init__(f"command {command!r} failed: {error}")
