"""UUIDv7 generator — time-ordered, monotonic."""
from __future__ import annotations

import os
import time

_last_timestamp: float = float("-inf")
_sequence: int = 0


def uuidv7() -> str:
    global _last_timestamp, _sequence

    random_bytes = bytearray(os.urandom(16))
    timestamp_ms = int(time.time() * 1000)

    if timestamp_ms > _last_timestamp:
        _sequence = (
            (random_bytes[6] << 24)
            | (random_bytes[7] << 16)
            | (random_bytes[8] << 8)
            | random_bytes[9]
        )
        _last_timestamp = timestamp_ms
    else:
        _sequence = (_sequence + 1) & 0xFFFFFFFF
        if _sequence == 0:
            _last_timestamp += 1
        timestamp_ms = int(_last_timestamp)

    b = bytearray(16)
    ts = int(_last_timestamp)
    b[0] = (ts >> 40) & 0xFF
    b[1] = (ts >> 32) & 0xFF
    b[2] = (ts >> 24) & 0xFF
    b[3] = (ts >> 16) & 0xFF
    b[4] = (ts >> 8) & 0xFF
    b[5] = ts & 0xFF
    b[6] = 0x70 | ((_sequence >> 28) & 0x0F)
    b[7] = (_sequence >> 20) & 0xFF
    b[8] = 0x80 | ((_sequence >> 14) & 0x3F)
    b[9] = (_sequence >> 6) & 0xFF
    b[10] = ((_sequence & 0x3F) << 2) | (random_bytes[10] & 0x03)
    b[11] = random_bytes[11]
    b[12] = random_bytes[12]
    b[13] = random_bytes[13]
    b[14] = random_bytes[14]
    b[15] = random_bytes[15]

    hex_bytes = b.hex()
    return (
        f"{hex_bytes[0:8]}-{hex_bytes[8:12]}-"
        f"{hex_bytes[12:16]}-{hex_bytes[16:20]}-{hex_bytes[20:32]}"
    )
