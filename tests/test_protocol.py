"""Protocol model tests: event parsing and response handling."""

from __future__ import annotations

from pi_py_sdk.protocol import (
    AgentEndEvent,
    Event,
    MessageUpdateEvent,
    Response,
    ToolExecutionEndEvent,
    parse_event,
)


def test_parse_message_update_text_delta():
    ev = parse_event(
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "Hello"},
        }
    )
    assert isinstance(ev, MessageUpdateEvent)
    assert ev.assistantMessageEvent is not None
    assert ev.assistantMessageEvent.type == "text_delta"
    assert ev.assistantMessageEvent.delta == "Hello"


def test_parse_agent_end_will_retry_default_false():
    ev = parse_event({"type": "agent_end", "messages": []})
    assert isinstance(ev, AgentEndEvent)
    assert ev.willRetry is False


def test_parse_agent_end_will_retry_true():
    ev = parse_event({"type": "agent_end", "messages": [], "willRetry": True})
    assert isinstance(ev, AgentEndEvent)
    assert ev.willRetry is True


def test_parse_tool_execution_end():
    ev = parse_event(
        {"type": "tool_execution_end", "toolName": "bash", "result": {"ok": 1}, "isError": False}
    )
    assert isinstance(ev, ToolExecutionEndEvent)
    assert ev.toolName == "bash"
    assert ev.isError is False


def test_unknown_event_falls_back_to_base_with_extra_preserved():
    ev = parse_event({"type": "something_new", "futureField": 42})
    assert isinstance(ev, Event)
    assert ev.type == "something_new"
    # extra="allow" keeps unmodeled fields accessible.
    assert ev.model_extra is not None
    assert ev.model_extra["futureField"] == 42


def test_response_success_with_data():
    resp = Response.model_validate(
        {"type": "response", "id": "req_1", "command": "get_state", "success": True, "data": {"x": 1}}
    )
    assert resp.success is True
    assert resp.data == {"x": 1}


def test_response_error():
    resp = Response.model_validate(
        {"type": "response", "id": "req_2", "command": "set_model", "success": False, "error": "nope"}
    )
    assert resp.success is False
    assert resp.error == "nope"
