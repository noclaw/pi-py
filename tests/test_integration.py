"""Opt-in live tests against a real `pi` subprocess.

Run with:  pytest -m integration
Requires the `pi` binary on PATH. These exercises avoid LLM calls (get_state,
get_available_models, bash), so they don't need a provider key.
"""

from __future__ import annotations

import shutil

import pytest

from pi_py_sdk import (
    AgentEndEvent,
    PiAgent,
    PiAgentSync,
    PiConfig,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("pi") is None, reason="`pi` binary not on PATH"),
]


async def _started_agent() -> PiAgent:
    agent = PiAgent(config=PiConfig(no_session=True))
    await agent.start()
    return agent


@pytest.mark.asyncio
async def test_live_state_and_models():
    agent = await _started_agent()
    try:
        state = await agent.get_state()
        assert state.get("sessionId")
        models = await agent.get_available_models()
        assert len(models) > 0
        assert all("id" in m for m in models)
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_live_bash_execution():
    agent = await _started_agent()
    try:
        result = await agent.bash("echo integration-ok")
        assert result.get("exitCode") == 0
        assert "integration-ok" in (result.get("output") or "")
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_live_steering_mode_roundtrip():
    agent = await _started_agent()
    try:
        await agent.set_steering_mode("all")
        assert (await agent.get_state()).get("steeringMode") == "all"
    finally:
        await agent.stop()


def test_live_sync_facade():
    with PiAgentSync(config=PiConfig(no_session=True)) as agent:
        assert agent.get_state().get("sessionId")
        result = agent.bash("echo sync-ok")
        assert result.get("exitCode") == 0
        assert "sync-ok" in (result.get("output") or "")


@pytest.mark.asyncio
async def test_live_prompt_completes():
    """Requires a working model (local or keyed). Tolerant of retries; just checks the
    stream terminates on a final agent_end."""
    agent = await _started_agent()
    try:
        ended = False
        async for event in agent.prompt_stream("Reply with the single word: ok", timeout=120):
            if isinstance(event, AgentEndEvent) and not event.willRetry:
                ended = True
        assert ended
    finally:
        await agent.stop()
