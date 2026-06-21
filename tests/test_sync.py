"""Unit tests for the PiAgentSync facade using a fake async agent (no subprocess)."""

from __future__ import annotations

import pytest

from pi_py_sdk import PiAgentSync


class _Event:
    def __init__(self, type: str, delta: str | None = None):
        self.type = type
        self.delta = delta


class FakeAgent:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.bashed: list[str] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def get_state(self) -> dict:
        return {"sessionId": "abc"}

    async def bash(self, command: str) -> dict:
        self.bashed.append(command)
        return {"exitCode": 0, "output": command}

    async def prompt_stream(self, message, *, images=None, timeout=300):
        for delta in ("He", "llo"):
            yield _Event("message_update", delta)
        yield _Event("agent_end")

    def subscribe(self, listener):
        return lambda: None

    def on_ui_request(self, handler):
        return lambda: None


def test_lifecycle_and_coroutine_delegation():
    fake = FakeAgent()
    with PiAgentSync(agent=fake) as agent:
        assert fake.started is True
        assert agent.get_state() == {"sessionId": "abc"}
        assert agent.bash("echo hi") == {"exitCode": 0, "output": "echo hi"}
    assert fake.stopped is True
    assert fake.bashed == ["echo hi"]


def test_prompt_stream_is_sync_iterator():
    fake = FakeAgent()
    with PiAgentSync(agent=fake) as agent:
        events = list(agent.prompt_stream("hi"))
    assert [e.type for e in events] == ["message_update", "message_update", "agent_end"]
    assert "".join(e.delta for e in events if e.delta) == "Hello"


def test_prompt_and_wait_collects():
    with PiAgentSync(agent=FakeAgent()) as agent:
        events = agent.prompt_and_wait("hi")
    assert len(events) == 3


def test_unknown_attribute_raises():
    agent = PiAgentSync(agent=FakeAgent())
    try:
        with pytest.raises(AttributeError):
            _ = agent.does_not_exist
    finally:
        agent.stop()
