"""Phase 2 tests: extension-UI dialog handling, observation, and richer events."""

from __future__ import annotations

import json

import pytest

from pi_py_sdk import (
    AutoRetryStartEvent,
    CompactionEndEvent,
    ExtensionUIRequest,
    PiAgent,
    QueueUpdateEvent,
    parse_event,
)


class UIScriptTransport:
    """Fake transport: emits a scripted UI request on prompt, then finishes the run
    once the client answers it (or, for fire-and-forget, immediately)."""

    def __init__(self, agent: PiAgent, request: dict, *, expect_response: bool = True):
        self._agent = agent
        self._request = request
        self._expect_response = expect_response
        self.written: list[dict] = []

    def stderr_text(self) -> str:
        return ""

    async def stop(self, timeout: float = 1.0) -> None:  # noqa: ARG002
        pass

    def _emit(self, obj: dict) -> None:
        self._agent._handle_line(json.dumps(obj))

    async def write_line(self, obj: dict) -> None:
        self.written.append(obj)
        t = obj.get("type")
        if t == "prompt":
            self._emit({"type": "response", "id": obj["id"], "command": "prompt", "success": True})
            self._emit(self._request)
            if not self._expect_response:
                self._emit({"type": "agent_end", "messages": [], "willRetry": False})
        elif t == "extension_ui_response":
            self._emit({"type": "agent_end", "messages": [], "willRetry": False})


def _ui_responses(transport: UIScriptTransport) -> list[dict]:
    return [w for w in transport.written if w.get("type") == "extension_ui_response"]


@pytest.mark.asyncio
async def test_confirm_handler_sends_confirmed():
    agent = PiAgent()
    agent._transport = UIScriptTransport(  # type: ignore[assignment]
        agent, {"type": "extension_ui_request", "id": "u1", "method": "confirm", "title": "Allow?"}
    )
    seen: list[str] = []
    agent.on_ui_request(lambda req: seen.append(req.method) or True)

    await agent.prompt_and_wait("go")

    resp = _ui_responses(agent._transport)  # type: ignore[arg-type]
    assert resp == [{"type": "extension_ui_response", "id": "u1", "confirmed": True}]
    assert seen == ["confirm"]


@pytest.mark.asyncio
async def test_default_handler_denies_confirm():
    agent = PiAgent()
    agent._transport = UIScriptTransport(  # type: ignore[assignment]
        agent, {"type": "extension_ui_request", "id": "u2", "method": "confirm", "title": "Allow?"}
    )
    await agent.prompt_and_wait("go")
    assert _ui_responses(agent._transport)[0]["confirmed"] is False  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_async_select_handler_returns_value():
    agent = PiAgent()
    agent._transport = UIScriptTransport(  # type: ignore[assignment]
        agent,
        {"type": "extension_ui_request", "id": "u3", "method": "select",
         "title": "Pick", "options": ["A", "B"]},
    )

    async def handler(req: ExtensionUIRequest) -> str:
        assert req.options == ["A", "B"]
        return "B"

    agent.on_ui_request(handler)
    await agent.prompt_and_wait("go")
    assert _ui_responses(agent._transport)[0] == {"type": "extension_ui_response", "id": "u3", "value": "B"}  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_handler_exception_cancels_dialog():
    agent = PiAgent()
    agent._transport = UIScriptTransport(  # type: ignore[assignment]
        agent, {"type": "extension_ui_request", "id": "u4", "method": "input", "title": "Name?"}
    )

    def boom(req: ExtensionUIRequest):
        raise RuntimeError("handler failed")

    agent.on_ui_request(boom)
    await agent.prompt_and_wait("go")
    assert _ui_responses(agent._transport)[0] == {"type": "extension_ui_response", "id": "u4", "cancelled": True}  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fire_and_forget_notify_is_observed_not_answered():
    agent = PiAgent()
    agent._transport = UIScriptTransport(  # type: ignore[assignment]
        agent,
        {"type": "extension_ui_request", "id": "u5", "method": "notify",
         "message": "heads up", "notifyType": "info"},
        expect_response=False,
    )
    observed: list[ExtensionUIRequest] = []
    agent.observe_ui_requests(observed.append)

    await agent.prompt_and_wait("go")

    assert [r.method for r in observed] == ["notify"]
    assert _ui_responses(agent._transport) == []  # type: ignore[arg-type]  # no reply for fire-and-forget


def test_parse_queue_update():
    ev = parse_event({"type": "queue_update", "steering": ["a"], "followUp": ["b", "c"]})
    assert isinstance(ev, QueueUpdateEvent)
    assert ev.steering == ["a"]
    assert ev.followUp == ["b", "c"]


def test_parse_compaction_end():
    ev = parse_event({"type": "compaction_end", "reason": "threshold", "aborted": False, "willRetry": False})
    assert isinstance(ev, CompactionEndEvent)
    assert ev.reason == "threshold"


def test_parse_auto_retry_start():
    ev = parse_event(
        {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 500, "errorMessage": "boom"}
    )
    assert isinstance(ev, AutoRetryStartEvent)
    assert ev.attempt == 1
    assert ev.maxAttempts == 3
