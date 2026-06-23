"""Opt-in live tests for PiModelClient against the real pi-ai shim.

Run with:  pytest -m integration
Requires the `pi` binary and `node` on PATH. ping/list don't need credentials; the live
stream needs a provider key or Pi OAuth login and is gated behind ``PI_LIVE_LLM=1``.
"""

from __future__ import annotations

import os
import shutil

import pytest

from pi_py_sdk import PiModelClient, PiModelClientSync

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("pi") is None, reason="`pi` binary not on PATH"),
    pytest.mark.skipif(shutil.which("node") is None, reason="`node` not on PATH"),
]

_LIVE_LLM = os.environ.get("PI_LIVE_LLM") == "1"
_LIVE_MODEL = os.environ.get("PI_LIVE_MODEL", "claude-haiku-4-5")
_LIVE_PROVIDER = os.environ.get("PI_LIVE_PROVIDER", "anthropic")


@pytest.mark.asyncio
async def test_live_ping_and_list_models():
    async with PiModelClient() as client:
        assert await client.ping() is True
        providers = await client.list_providers()
        assert isinstance(providers, list) and providers
        models = await client.list_models(_LIVE_PROVIDER)
        assert isinstance(models, list)
        assert all("id" in m for m in models)


@pytest.mark.asyncio
@pytest.mark.skipif(not _LIVE_LLM, reason="set PI_LIVE_LLM=1 to run live model calls")
async def test_live_stream():
    async with PiModelClient() as client:
        text = ""
        terminal = None
        async for ev in client.stream(
            provider=_LIVE_PROVIDER,
            model=_LIVE_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: PONG", "timestamp": 0}],
            maxTokens=16,
        ):
            if ev.type == "text_delta" and ev.delta:
                text += ev.delta
            if ev.is_terminal:
                terminal = ev
        assert terminal is not None and terminal.type == "done"
        assert "PONG" in text


@pytest.mark.skipif(not _LIVE_LLM, reason="set PI_LIVE_LLM=1 to run live model calls")
def test_live_stream_sync():
    with PiModelClientSync() as client:
        text = "".join(
            ev.delta
            for ev in client.stream(
                provider=_LIVE_PROVIDER,
                model=_LIVE_MODEL,
                messages=[{"role": "user", "content": "Reply with exactly: PONG", "timestamp": 0}],
                maxTokens=16,
            )
            if ev.type == "text_delta" and ev.delta
        )
        assert "PONG" in text
