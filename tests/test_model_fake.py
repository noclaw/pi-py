"""PiModelClient logic tests against a scripted fake transport (no Node subprocess).

Validates stream routing/termination, shim-level error surfacing, request/response
correlation, and the early-break abort, mirroring ``test_client_fake.py``.
"""

from __future__ import annotations

import json

import pytest

from pi_py_sdk import PiModelClient, PiModelError, StreamEvent
from pi_py_sdk.jsonl import serialize_line


class FakeModelTransport:
    """Scripts shim replies for the model client.

    On a ``stream`` command it emits either ``stream_event`` lines (one per entry in
    ``stream_events``) or a single ``stream_error``. Correlated commands (list_*/ping)
    get a success ``response`` carrying ``responses[command]``.
    """

    def __init__(
        self,
        client: PiModelClient,
        *,
        stream_events: list[dict] | None = None,
        stream_error: str | None = None,
        responses: dict[str, dict] | None = None,
    ) -> None:
        self._client = client
        self._stream_events = stream_events or []
        self._stream_error = stream_error
        self._responses = responses or {}
        self.written: list[dict] = []

    def stderr_text(self) -> str:
        return ""

    async def stop(self, timeout: float = 1.0) -> None:  # noqa: ARG002
        pass

    async def write_line(self, obj: dict) -> None:
        self.written.append(obj)
        decoded = json.loads(serialize_line(obj).decode("utf-8"))
        kind = decoded.get("type")
        if kind == "stream":
            sid = decoded["id"]
            if self._stream_error is not None:
                self._emit({"type": "stream_error", "id": sid, "error": self._stream_error})
            else:
                for event in self._stream_events:
                    self._emit({"type": "stream_event", "id": sid, "event": event})
        elif kind == "abort":
            pass  # acked silently in the fake
        else:
            self._emit(
                {
                    "type": "response",
                    "id": decoded["id"],
                    "command": kind,
                    "success": True,
                    "data": self._responses.get(kind, {}),
                }
            )

    def _emit(self, obj: dict) -> None:
        self._client._handle_line(json.dumps(obj))


def _done(text: str) -> dict:
    return {
        "type": "done",
        "reason": "stop",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stopReason": "stop",
        },
    }


@pytest.mark.asyncio
async def test_stream_yields_until_terminal():
    client = PiModelClient()
    script = [
        {"type": "start", "partial": {"role": "assistant", "content": []}},
        {"type": "text_delta", "contentIndex": 0, "delta": "Hi"},
        {"type": "text_delta", "contentIndex": 0, "delta": "!"},
        _done("Hi!"),
    ]
    client._transport = FakeModelTransport(client, stream_events=script)  # type: ignore[assignment]

    events = [
        ev
        async for ev in client.stream(
            provider="anthropic",
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi", "timestamp": 0}],
        )
    ]

    deltas = [e.delta for e in events if e.type == "text_delta"]
    assert "".join(deltas) == "Hi!"
    assert events[-1].is_terminal
    assert events[-1].type == "done"
    assert events[-1].final_message is not None
    assert events[-1].final_message.stopReason == "stop"

    # The stream command carried a context with our message and a stream id.
    sent = client._transport.written[0]  # type: ignore[union-attr]
    assert sent["type"] == "stream"
    assert sent["id"] == "stream_1"
    assert sent["context"]["messages"][0]["content"] == "hi"
    assert sent["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_stream_error_raises_pimodelerror():
    client = PiModelClient()
    client._transport = FakeModelTransport(client, stream_error="boom")  # type: ignore[assignment]

    with pytest.raises(PiModelError, match="boom"):
        async for _ in client.stream(
            provider="anthropic",
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi", "timestamp": 0}],
        ):
            pass


@pytest.mark.asyncio
async def test_error_event_is_in_band_terminal():
    """A model-produced ``error`` event is delivered, not raised."""
    client = PiModelClient()
    script = [
        {
            "type": "error",
            "reason": "error",
            "error": {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "rate limited"},
        }
    ]
    client._transport = FakeModelTransport(client, stream_events=script)  # type: ignore[assignment]

    events = [
        ev
        async for ev in client.stream(
            provider="anthropic",
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi", "timestamp": 0}],
        )
    ]
    assert len(events) == 1
    assert events[0].is_terminal
    assert events[0].final_message.errorMessage == "rate limited"


@pytest.mark.asyncio
async def test_complete_returns_final_message():
    client = PiModelClient()
    client._transport = FakeModelTransport(client, stream_events=[_done("done text")])  # type: ignore[assignment]

    msg = await client.complete(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi", "timestamp": 0}],
    )
    assert msg.stopReason == "stop"


@pytest.mark.asyncio
async def test_tool_definitions_and_reasoning_are_forwarded():
    client = PiModelClient()
    client._transport = FakeModelTransport(client, stream_events=[_done("ok")])  # type: ignore[assignment]
    tools = [{"name": "t", "description": "d", "parameters": {"type": "object"}}]

    async for _ in client.stream(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi", "timestamp": 0}],
        system_prompt="be brief",
        tools=tools,
        reasoning="high",
        maxTokens=64,
    ):
        pass

    sent = client._transport.written[0]  # type: ignore[union-attr]
    assert sent["context"]["systemPrompt"] == "be brief"
    assert sent["context"]["tools"] == tools
    assert sent["options"]["reasoning"] == "high"
    assert sent["options"]["maxTokens"] == 64


@pytest.mark.asyncio
async def test_closing_stream_early_sends_abort():
    """Closing the stream before its terminal event tells the shim to abort it.

    (A plain ``break`` defers async-generator cleanup to GC; ``aclose()`` runs the
    finally deterministically — which is what cancellation/early-exit ultimately does.)
    """
    client = PiModelClient()
    script = [
        {"type": "text_delta", "contentIndex": 0, "delta": "a"},
        {"type": "text_delta", "contentIndex": 0, "delta": "b"},
        _done("ab"),
    ]
    client._transport = FakeModelTransport(client, stream_events=script)  # type: ignore[assignment]

    agen = client.stream(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi", "timestamp": 0}],
    )
    first = await agen.__anext__()
    assert isinstance(first, StreamEvent) and first.delta == "a"
    await agen.aclose()  # consumer bails before the terminal event

    aborts = [w for w in client._transport.written if w["type"] == "abort"]  # type: ignore[union-attr]
    assert aborts and aborts[0]["id"] == "stream_1"
    # The stream id is no longer tracked after cleanup.
    assert client._streams == {}


@pytest.mark.asyncio
async def test_list_models_correlates_response():
    client = PiModelClient()
    client._transport = FakeModelTransport(  # type: ignore[assignment]
        client, responses={"list_models": {"models": [{"id": "m1"}, {"id": "m2"}]}}
    )
    models = await client.list_models("anthropic")
    assert [m["id"] for m in models] == ["m1", "m2"]
    # The provider filter is forwarded on the correlated command.
    assert client._transport.written[0]["provider"] == "anthropic"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_ping():
    client = PiModelClient()
    client._transport = FakeModelTransport(client, responses={"ping": {"ok": True}})  # type: ignore[assignment]
    assert await client.ping() is True
