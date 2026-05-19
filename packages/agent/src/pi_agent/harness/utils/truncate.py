"""Output truncation utilities (head and tail), UTF-8-aware."""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50 KB
GREP_MAX_LINE_LENGTH = 500


@dataclass
class TruncationResult:
    content: str
    truncated: bool
    truncated_by: str | None       # "lines" | "bytes" | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int


def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def format_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_}B"
    if bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.1f}KB"
    return f"{bytes_ / (1024 * 1024):.1f}MB"


def truncate_head(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    total_bytes = _utf8_len(content)
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content, truncated=False, truncated_by=None,
            total_lines=total_lines, total_bytes=total_bytes,
            output_lines=total_lines, output_bytes=total_bytes,
            last_line_partial=False, first_line_exceeds_limit=False,
            max_lines=max_lines, max_bytes=max_bytes,
        )

    first_line_bytes = _utf8_len(lines[0]) if lines else 0
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="", truncated=True, truncated_by="bytes",
            total_lines=total_lines, total_bytes=total_bytes,
            output_lines=0, output_bytes=0,
            last_line_partial=False, first_line_exceeds_limit=True,
            max_lines=max_lines, max_bytes=max_bytes,
        )

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by = "lines"

    for i, line in enumerate(lines):
        if i >= max_lines:
            break
        line_bytes = _utf8_len(line) + (1 if i > 0 else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        out_lines.append(line)
        out_bytes += line_bytes

    if len(out_lines) >= max_lines and out_bytes <= max_bytes:
        truncated_by = "lines"

    out_content = "\n".join(out_lines)
    return TruncationResult(
        content=out_content, truncated=True, truncated_by=truncated_by,
        total_lines=total_lines, total_bytes=total_bytes,
        output_lines=len(out_lines), output_bytes=_utf8_len(out_content),
        last_line_partial=False, first_line_exceeds_limit=False,
        max_lines=max_lines, max_bytes=max_bytes,
    )


def truncate_tail(content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES) -> TruncationResult:
    total_bytes = _utf8_len(content)
    lines = content.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines.pop()
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content, truncated=False, truncated_by=None,
            total_lines=total_lines, total_bytes=total_bytes,
            output_lines=total_lines, output_bytes=total_bytes,
            last_line_partial=False, first_line_exceeds_limit=False,
            max_lines=max_lines, max_bytes=max_bytes,
        )

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by = "lines"
    last_line_partial = False

    for i in range(len(lines) - 1, -1, -1):
        if len(out_lines) >= max_lines:
            break
        line = lines[i]
        line_bytes = _utf8_len(line) + (1 if out_lines else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not out_lines:
                truncated_line = _truncate_bytes_from_end(line, max_bytes)
                out_lines.insert(0, truncated_line)
                out_bytes = _utf8_len(truncated_line)
                last_line_partial = True
            break
        out_lines.insert(0, line)
        out_bytes += line_bytes

    if len(out_lines) >= max_lines and out_bytes <= max_bytes:
        truncated_by = "lines"

    out_content = "\n".join(out_lines)
    return TruncationResult(
        content=out_content, truncated=True, truncated_by=truncated_by,
        total_lines=total_lines, total_bytes=total_bytes,
        output_lines=len(out_lines), output_bytes=_utf8_len(out_content),
        last_line_partial=last_line_partial, first_line_exceeds_limit=False,
        max_lines=max_lines, max_bytes=max_bytes,
    )


def _truncate_bytes_from_end(s: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    truncated = encoded[-max_bytes:]
    return truncated.decode("utf-8", errors="replace")


def truncate_line(
    line: str,
    max_chars: int = GREP_MAX_LINE_LENGTH,
) -> dict:
    if len(line) <= max_chars:
        return {"text": line, "wasTruncated": False}
    return {"text": f"{line[:max_chars]}... [truncated]", "wasTruncated": True}
