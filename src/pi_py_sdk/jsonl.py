"""Strict JSONL framing for the Pi RPC protocol.

Framing rules mirror Pi's ``packages/coding-agent/src/modes/rpc/jsonl.ts`` exactly:

* Records are delimited by ``\\n`` (LF) **only**. We must never split on other
  Unicode line separators (e.g. U+2028 / U+2029) because those are valid *inside*
  JSON string payloads. (Python's ``str.splitlines`` and ``io`` text wrappers split
  on them, so we frame on raw bytes and split on the newline byte instead.)
* A single trailing ``\\r`` is stripped from each line before parsing.
* Any non-empty remainder left in the buffer at end-of-stream is emitted as a final
  line.
"""

from __future__ import annotations

import json
from typing import Any

_LF = 0x0A  # b"\n" — the sole record delimiter


def serialize_line(value: Any) -> bytes:
    """Serialize a single value as one strict JSONL record (UTF-8 bytes incl. the LF)."""
    # separators avoid incidental whitespace; ensure_ascii=False keeps payloads compact.
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class JsonlDecoder:
    """Incremental decoder turning arbitrary byte chunks into complete text lines.

    Splitting happens on the LF *byte*. UTF-8 continuation bytes are never 0x0A, so a
    multibyte character can never be split at a delimiter; partial trailing bytes are
    retained in the buffer until the rest of the line arrives.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list[str]:
        """Append a chunk and return any newly completed lines."""
        self._buf.extend(chunk)
        lines: list[str] = []
        while True:
            idx = self._buf.find(_LF)
            if idx == -1:
                break
            raw = bytes(self._buf[:idx])
            del self._buf[: idx + 1]
            lines.append(self._decode(raw))
        return lines

    def flush(self) -> list[str]:
        """Emit any non-empty buffered remainder at end-of-stream."""
        if not self._buf:
            return []
        raw = bytes(self._buf)
        self._buf.clear()
        line = self._decode(raw)
        return [line] if line else []

    @staticmethod
    def _decode(raw: bytes) -> str:
        line = raw.decode("utf-8")
        if line.endswith("\r"):
            line = line[:-1]
        return line
