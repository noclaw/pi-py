"""Render streamed agent events to a terminal.

Pure and testable: construct with a ``write`` callable and a ``color`` flag, then feed
it :class:`~pi_py_sdk.Event` objects. State is tracked only to insert sensible line
breaks between assistant text, thinking, and tool activity.
"""

from __future__ import annotations

from typing import Any, Callable

from pi_py_sdk import Event

_RESET = "\033[0m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_RED = "\033[31m"

# Keys whose value best summarizes a tool call, in priority order.
_ARG_KEYS = ("command", "path", "file_path", "pattern", "query", "url")


def _summarize_args(args: Any, limit: int = 80) -> str:
    if isinstance(args, dict):
        for key in _ARG_KEYS:
            if key in args and args[key] is not None:
                value = str(args[key]).replace("\n", " ")
                return value if len(value) <= limit else value[: limit - 1] + "…"
    return ""


class Renderer:
    def __init__(self, write: Callable[[str], Any] | None = None, *, color: bool = True):
        if write is None:
            import sys

            write = sys.stdout.write
        self._write = write
        self.color = color
        self._mode: str | None = None  # "text" | "thinking" | None

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if self.color else text

    def handle(self, event: Event) -> None:
        kind = event.type
        if kind == "message_update":
            self._on_message_update(event)
        elif kind == "tool_execution_start":
            self._break()
            name = getattr(event, "toolName", None) or "tool"
            summary = _summarize_args(getattr(event, "args", None))
            line = self._c(_CYAN, f"→ {name}") + (f" {summary}" if summary else "")
            self._write(line + "\n")
        elif kind == "tool_execution_end":
            mark = self._c(_RED, "✗") if getattr(event, "isError", False) else self._c(_GREEN, "✓")
            self._write(f"  {mark}\n")
        elif kind == "auto_retry_start":
            self._break()
            a, m = getattr(event, "attempt", "?"), getattr(event, "maxAttempts", "?")
            self._write(self._c(_YELLOW, f"[retrying {a}/{m}…]") + "\n")
        elif kind == "compaction_start":
            self._break()
            self._write(self._c(_YELLOW, "[compacting context…]") + "\n")
        elif kind == "agent_end":
            self._break()

    def _on_message_update(self, event: Event) -> None:
        ame = getattr(event, "assistantMessageEvent", None)
        if ame is None or not getattr(ame, "delta", None):
            return
        if ame.type == "text_delta":
            if self._mode != "text":
                if self._mode == "thinking":
                    self._write("\n")
                self._mode = "text"
            self._write(ame.delta)
        elif ame.type == "thinking_delta":
            if self._mode != "thinking":
                if self._mode == "text":
                    self._write("\n")
                self._write(self._c(_DIM, "(thinking) "))
                self._mode = "thinking"
            self._write(self._c(_DIM, ame.delta))

    def _break(self) -> None:
        if self._mode is not None:
            self._write("\n")
            self._mode = None
