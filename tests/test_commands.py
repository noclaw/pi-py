"""Round-trip tests for the expanded command surface (payload shape + data return)."""

from __future__ import annotations

import json

import pytest

from pi_py_sdk import PiAgent, PiCommandError


class EchoTransport:
    """Replies success to every command; optional canned ``data`` per command type.

    ``fail`` maps a command type to an error string to exercise the failure path.
    """

    def __init__(self, agent: PiAgent, *, data: dict | None = None, fail: dict | None = None):
        self._agent = agent
        self._data = data or {}
        self._fail = fail or {}
        self.written: list[dict] = []

    def stderr_text(self) -> str:
        return ""

    async def stop(self, timeout: float = 1.0) -> None:  # noqa: ARG002
        pass

    async def write_line(self, obj: dict) -> None:
        self.written.append(obj)
        if obj.get("type") == "extension_ui_response":
            return
        cmd = obj["type"]
        resp: dict = {"type": "response", "id": obj.get("id"), "command": cmd}
        if cmd in self._fail:
            resp.update(success=False, error=self._fail[cmd])
        else:
            resp["success"] = True
            if cmd in self._data:
                resp["data"] = self._data[cmd]
        self._agent._handle_line(json.dumps(resp))


def _last(transport: EchoTransport) -> dict:
    return transport.written[-1]


@pytest.mark.asyncio
async def test_set_steering_mode_payload():
    agent = PiAgent()
    agent._transport = EchoTransport(agent)  # type: ignore[assignment]
    await agent.set_steering_mode("all")
    sent = _last(agent._transport)  # type: ignore[arg-type]
    assert sent["type"] == "set_steering_mode"
    assert sent["mode"] == "all"
    assert sent["id"].startswith("req_")


@pytest.mark.asyncio
async def test_set_model_payload_and_data():
    agent = PiAgent()
    agent._transport = EchoTransport(  # type: ignore[assignment]
        agent, data={"set_model": {"provider": "anthropic", "id": "claude"}}
    )
    out = await agent.set_model("anthropic", "claude")
    sent = _last(agent._transport)  # type: ignore[arg-type]
    assert sent["provider"] == "anthropic" and sent["modelId"] == "claude"
    assert out == {"provider": "anthropic", "id": "claude"}


@pytest.mark.asyncio
async def test_bash_strips_none_and_returns_result():
    agent = PiAgent()
    agent._transport = EchoTransport(  # type: ignore[assignment]
        agent, data={"bash": {"output": "hi", "exitCode": 0}}
    )
    result = await agent.bash("echo hi")
    sent = _last(agent._transport)  # type: ignore[arg-type]
    assert sent["command"] == "echo hi"
    assert "excludeFromContext" not in sent  # None values are stripped before send
    assert result == {"output": "hi", "exitCode": 0}


@pytest.mark.asyncio
async def test_get_messages_unwraps_list():
    agent = PiAgent()
    agent._transport = EchoTransport(agent, data={"get_messages": {"messages": [1, 2, 3]}})  # type: ignore[assignment]
    assert await agent.get_messages() == [1, 2, 3]


@pytest.mark.asyncio
async def test_command_error_raises():
    agent = PiAgent()
    agent._transport = EchoTransport(agent, fail={"set_model": "Model not found: x/y"})  # type: ignore[assignment]
    with pytest.raises(PiCommandError) as excinfo:
        await agent.set_model("x", "y")
    assert excinfo.value.command == "set_model"
    assert "Model not found" in excinfo.value.error
