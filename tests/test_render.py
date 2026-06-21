"""Tests for the terminal renderer (pure, no subprocess)."""

from __future__ import annotations

from pi_py_agent.render import Renderer
from pi_py_sdk import parse_event


def _render(events, color=False) -> str:
    out: list[str] = []
    r = Renderer(write=out.append, color=color)
    for ev in events:
        r.handle(parse_event(ev))
    return "".join(out)


def test_text_deltas_concatenate():
    s = _render(
        [
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "Hello"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": " world"}},
            {"type": "agent_end", "messages": [], "willRetry": False},
        ]
    )
    assert "Hello world" in s


def test_no_ansi_when_color_disabled():
    s = _render(
        [
            {"type": "tool_execution_start", "toolName": "bash", "args": {"command": "echo hi"}},
            {"type": "tool_execution_end", "toolName": "bash", "isError": False},
        ]
    )
    assert "\033[" not in s  # no escape codes
    assert "bash" in s and "echo hi" in s
    assert "✓" in s


def test_tool_error_marker():
    s = _render([{"type": "tool_execution_end", "toolName": "bash", "isError": True}])
    assert "✗" in s


def test_thinking_then_text_separated():
    s = _render(
        [
            {"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "delta": "hmm"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "answer"}},
        ]
    )
    assert "(thinking) hmm" in s
    assert "answer" in s
    # a newline separates the thinking block from the text answer
    assert "hmm\nanswer" in s


def test_color_emits_ansi():
    s = _render(
        [{"type": "tool_execution_start", "toolName": "read", "args": {"path": "/x"}}],
        color=True,
    )
    assert "\033[" in s


def test_long_arg_is_truncated():
    long_cmd = "x" * 200
    s = _render([{"type": "tool_execution_start", "toolName": "bash", "args": {"command": long_cmd}}])
    assert "…" in s
    assert long_cmd not in s
