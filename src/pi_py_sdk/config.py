"""Configuration for launching a Pi RPC subprocess."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PiConfig:
    """How to launch and configure ``pi --mode rpc``.

    Attributes:
        bin: Explicit path/name of the ``pi`` binary. When ``None``, the binary is
            discovered on ``PATH`` with an ``npx`` fallback (see ``_discovery``).
        provider: Optional provider override (``--provider``).
        model: Optional model id override (``--model``).
        cwd: Working directory for the agent. Defaults to the current directory.
        env: Extra environment variables layered on top of the current environment.
            Provider credentials are resolved by Pi itself from this environment.
        session_dir: Optional session directory (``--session-dir``).
        no_session: Disable session persistence (``--no-session``).
        extra_args: Any additional raw CLI arguments to pass through.
    """

    bin: str | None = None
    provider: str | None = None
    model: str | None = None
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    session_dir: str | None = None
    no_session: bool = False
    extra_args: list[str] = field(default_factory=list)

    def mode_args(self) -> list[str]:
        """Build the argument list that follows the resolved program prefix."""
        args = ["--mode", "rpc"]
        if self.provider:
            args += ["--provider", self.provider]
        if self.model:
            args += ["--model", self.model]
        if self.session_dir:
            args += ["--session-dir", self.session_dir]
        if self.no_session:
            args.append("--no-session")
        args += self.extra_args
        return args

    def build_env(self) -> dict[str, str]:
        """Full environment for the subprocess: current env plus overrides."""
        return {**os.environ, **self.env}
