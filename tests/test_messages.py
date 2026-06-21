"""Tests for message-block models and text extraction."""

from __future__ import annotations

from pi_py_sdk import (
    AssistantMessage,
    BashExecutionMessage,
    ToolResultMessage,
    UserMessage,
    message_text,
    parse_message,
    parse_messages,
)


def test_user_message_str_content():
    m = parse_message({"role": "user", "content": "hello", "timestamp": 1})
    assert isinstance(m, UserMessage)
    assert message_text(m) == "hello"


def test_assistant_message_with_blocks():
    m = parse_message(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "hi "},
                {"type": "text", "text": "there"},
                {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "ls"}},
            ],
            "model": "claude",
            "stopReason": "stop",
        }
    )
    assert isinstance(m, AssistantMessage)
    assert message_text(m) == "hi there"  # only text blocks, thinking/toolCall excluded
    assert m.model == "claude"
    assert m.stopReason == "stop"


def test_tool_result_message():
    m = parse_message(
        {
            "role": "toolResult",
            "toolCallId": "c1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "output"}],
            "isError": False,
        }
    )
    assert isinstance(m, ToolResultMessage)
    assert m.toolName == "bash"
    assert message_text(m) == "output"


def test_bash_execution_message():
    m = parse_message(
        {"role": "bashExecution", "command": "ls", "output": "a\nb", "exitCode": 0,
         "cancelled": False, "truncated": False}
    )
    assert isinstance(m, BashExecutionMessage)
    assert m.command == "ls"
    assert m.exitCode == 0


def test_unknown_role_falls_back_and_keeps_extra():
    m = parse_message({"role": "mystery", "weird": 7})
    assert m.model_extra is not None
    assert m.model_extra["weird"] == 7


def test_message_text_accepts_dicts():
    assert message_text({"role": "user", "content": "hi"}) == "hi"
    assert message_text({"role": "user", "content": [{"type": "text", "text": "x"}]}) == "x"


def test_parse_messages_roundtrip_and_passthrough():
    out = parse_messages(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
            "not-a-dict",  # passthrough unchanged
        ]
    )
    assert message_text(out[0]) == "a"
    assert message_text(out[1]) == "b"
    assert out[2] == "not-a-dict"
