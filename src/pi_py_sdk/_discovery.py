"""Locate the Pi runtime.

Resolution order (confirmed project decision):
  1. An explicit ``bin`` (path or command name), if provided.
  2. ``pi`` discovered on ``PATH``.
  3. ``npx --yes @earendil-works/pi-coding-agent@<pinned>`` as a fallback.
"""

from __future__ import annotations

import shutil

from .errors import PiProcessError

#: Pinned Pi package + version this SDK was developed and tested against.
PINNED_PI_PACKAGE = "@earendil-works/pi-coding-agent"
PINNED_PI_VERSION = "0.79.9"


def resolve_pi_command(bin: str | None = None) -> list[str]:
    """Return the argv prefix that launches Pi (before ``--mode rpc`` args).

    Raises:
        PiProcessError: if neither an explicit/`PATH` binary nor ``npx`` is available.
    """
    if bin:
        resolved = shutil.which(bin) or bin
        return [resolved]

    on_path = shutil.which("pi")
    if on_path:
        return [on_path]

    npx = shutil.which("npx")
    if npx:
        return [npx, "--yes", f"{PINNED_PI_PACKAGE}@{PINNED_PI_VERSION}"]

    raise PiProcessError(
        "Could not find the `pi` binary on PATH and `npx` is unavailable. "
        f"Install it with `npm i -g {PINNED_PI_PACKAGE}` (or ensure Node/npx is installed)."
    )
