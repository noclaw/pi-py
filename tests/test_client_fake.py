"""Client logic tests against a scripted fake transport (no Node subprocess).

Validates request/response correlation, prompt streaming to completion, and the
``agent_end.willRetry`` semantics — the bits we wrote on top of the protocol.
"""

from __future__ import annotations

import json

import pytest

from pi_py_sdk import AgentEndEvent, MessageUpdateEvent, PiAgent
from pi_py_sdk.jsonl import serialize_line


class FakeTransport:
    """Stand-in transport that scripts server replies for each prompt.

    On each ``prompt`` command it emits a success response (for correlation) followed
    by the lines in ``script`` (raw wire dicts), simulating streamed events.
    """

    def __init__(self, agent: PiAgent, script: list[dict]):
        self._agent = agent
        self._script = script
        self.written: list[dict] = []

    def stderr_text(self) -> str:
        return ""

    async def stop(self, timeout: float = 1.0) -> None:  # noqa: ARG002
        pass

    async def write_line(self, obj: dict) -> None:
        self.written.append(obj)
        # Round-trip through real framing so we exercise serialize/parse too.
        decoded = json.loads(serialize_line(obj).decode("utf-8"))
        if decoded.get("type") == "prompt":
            self._emit({"type": "response", "id": decoded["id"], "command": "prompt", "success": True})
            for line in self._script:
                self._emit(line)

    def _emit(self, obj: dict) -> None:
        self._agent._handle_line(json.dumps(obj))


@pytest.mark.asyncio
async def test_prompt_stream_completes_on_agent_end():
    agent = PiAgent()
    script = [
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "Hi"}},
        {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "!"}},
        {"type": "agent_end", "messages": [], "willRetry": False},
    ]
    agent._transport = FakeTransport(agent, script)  # type: ignore[assignment]

    events = await agent.prompt_and_wait("hello")

    deltas = [
        e.assistantMessageEvent.delta
        for e in events
        if isinstance(e, MessageUpdateEvent) and e.assistantMessageEvent
    ]
    assert "".join(deltas) == "Hi!"
    assert isinstance(events[-1], AgentEndEvent)
    # The correlated prompt command carried an id.
    assert agent._transport.written[0]["type"] == "prompt"  # type: ignore[union-attr]
    assert agent._transport.written[0]["id"] == "req_1"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_will_retry_agent_end_does_not_terminate_stream():
    agent = PiAgent()
    script = [
        {"type": "agent_end", "messages": [], "willRetry": True},
        {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 10, "errorMessage": "x"},
        {"type": "agent_end", "messages": [], "willRetry": False},
    ]
    agent._transport = FakeTransport(agent, script)  # type: ignore[assignment]

    events = await agent.prompt_and_wait("hello")

    agent_ends = [e for e in events if isinstance(e, AgentEndEvent)]
    assert len(agent_ends) == 2  # streamed through the retry to the final end
    assert agent_ends[0].willRetry is True
    assert agent_ends[-1].willRetry is False
