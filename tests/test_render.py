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


def test_tool_result_preview_rendered():
    s = _render(
        [{"type": "tool_execution_end", "toolName": "bash", "isError": False,
          "result": {"output": "line1\nline2"}}]
    )
    assert "│ line1" in s
    assert "│ line2" in s


def test_tool_result_preview_truncates_many_lines():
    body = "\n".join(f"l{i}" for i in range(20))
    s = _render([{"type": "tool_execution_end", "toolName": "bash", "result": {"output": body}}])
    assert "more lines)" in s
    assert "l0" in s and "l19" not in s


def test_tool_result_preview_from_content_blocks():
    s = _render(
        [{"type": "tool_execution_end", "toolName": "read",
          "result": {"content": [{"type": "text", "text": "hello from blocks"}]}}]
    )
    assert "│ hello from blocks" in s


def test_queue_update_rendered():
    s = _render([{"type": "queue_update", "steering": ["a"], "followUp": ["b", "c"]}])
    assert "steering=1" in s and "follow-up=2" in s


def test_empty_result_shows_only_mark():
    s = _render([{"type": "tool_execution_end", "toolName": "bash", "isError": False, "result": None}])
    assert "✓" in s
    assert "│" not in s
