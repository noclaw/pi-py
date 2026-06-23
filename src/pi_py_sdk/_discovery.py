"""Locate the Pi runtime.

Resolution order (confirmed project decision):
  1. An explicit ``bin`` (path or command name), if provided.
  2. ``pi`` discovered on ``PATH``.
  3. ``npx --yes @earendil-works/pi-coding-agent@<pinned>`` as a fallback.

The model-streaming client (:mod:`pi_py_sdk.model`) additionally needs ``node`` and
the ``@earendil-works/pi-ai`` package directory, resolved by :func:`resolve_node` and
:func:`resolve_pi_ai_dir`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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


def resolve_node(node: str | None = None) -> str:
    """Return a path to the ``node`` executable for running the streaming shim.

    Order: an explicit ``node`` argument, the ``PI_NODE`` env var, then ``node`` on PATH.

    Raises:
        PiProcessError: if Node cannot be found.
    """
    candidate = node or os.environ.get("PI_NODE") or "node"
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    if node or os.environ.get("PI_NODE"):
        return candidate  # honor an explicit path even if `which` can't confirm it
    raise PiProcessError(
        "Could not find `node` on PATH. Install Node.js (https://nodejs.org) "
        "or set PI_NODE to the node executable."
    )


def shim_path() -> str:
    """Absolute path to the bundled model-streaming shim (``_shim/stream.mjs``)."""
    return str(Path(__file__).parent / "_shim" / "stream.mjs")


def resolve_pi_ai_dir(bin: str | None = None) -> str:
    """Return the ``@earendil-works/pi-ai`` package directory the shim imports.

    Order:
      1. ``PI_AI_DIR`` env var, if set (must point at the pi-ai package directory).
      2. Derived from the ``pi`` binary: pi is installed as
         ``@earendil-works/pi-coding-agent`` which bundles pi-ai under its
         ``node_modules`` — walk up from the resolved binary to find it.

    Raises:
        PiProcessError: if pi-ai cannot be located (e.g. the npx fallback is in use, or
            pi is not installed locally). Set ``PI_AI_DIR`` to fix this.
    """
    env_dir = os.environ.get("PI_AI_DIR")
    if env_dir:
        return env_dir

    pi = bin or shutil.which("pi")
    if pi:
        real = Path(pi).resolve()
        for parent in [real, *real.parents]:
            candidate = parent / "node_modules" / "@earendil-works" / "pi-ai"
            if (candidate / "package.json").exists():
                return str(candidate)

    raise PiProcessError(
        "Could not locate the `@earendil-works/pi-ai` package. It ships inside the "
        f"`pi` install ({PINNED_PI_PACKAGE}); install pi with "
        f"`npm i -g {PINNED_PI_PACKAGE}`, or set PI_AI_DIR to the pi-ai package directory."
    )
