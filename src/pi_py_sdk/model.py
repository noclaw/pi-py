"""Low-level model-streaming client over the pi-ai shim.

Where :class:`~pi_py_sdk.client.PiAgent` drives the *full* Pi agent (``pi --mode rpc`` —
loop, tools, sessions, compaction), :class:`PiModelClient` exposes just the **raw model
layer**: stream an assistant response for a given context (system prompt + messages +
tools), and enumerate models/providers. It spawns the bundled ``_shim/stream.mjs``,
which calls ``@earendil-works/pi-ai``'s ``streamSimple``.

This is the seam a native-Python agent loop builds on: the loop owns turn structure and
tool execution, while pi-ai handles providers, auth (env keys + the coding agent's OAuth
login), transports, and local models.

    async with PiModelClient() as client:
        async for ev in client.stream(
            provider="anthropic",
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi", "timestamp": 0}],
        ):
            if ev.type == "text_delta":
                print(ev.delta, end="", flush=True)

Auth lives in pi-ai (see CLAUDE.md): the shim resolves credentials as
caller ``api_key`` > provider env var > ``~/.pi/agent/auth.json`` OAuth login.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator

from ._discovery import resolve_node, resolve_pi_ai_dir, shim_path
from .errors import (
    PiModelError,
    PiNotStartedError,
    PiProcessError,
    PiTimeoutError,
)
from .protocol import AssistantMessage, Response, StreamEvent, parse_stream_event
from .transport import Transport

_REQUEST_TIMEOUT = 30.0
#: Inactivity timeout: max seconds to wait for the *next* event of a live stream.
_DEFAULT_STREAM_TIMEOUT = 600.0


class PiModelClient:
    """Async handle to a running pi-ai model-streaming shim.

    Args:
        node: Path/name of the Node executable (default: ``PI_NODE`` env or ``node``).
        pi_ai_dir: The ``@earendil-works/pi-ai`` package directory (default: discovered
            from the ``pi`` install or the ``PI_AI_DIR`` env var).
        auth_path: Path to the coding agent's ``auth.json`` for OAuth login reuse
            (default: ``~/.pi/agent/auth.json``).
        cwd: Working directory for the subprocess.
        env: Extra environment variables layered on top of the current environment.
    """

    def __init__(
        self,
        *,
        node: str | None = None,
        pi_ai_dir: str | None = None,
        auth_path: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._node = node
        self._pi_ai_dir = pi_ai_dir
        self._auth_path = auth_path
        self._cwd = cwd
        self._env = env or {}
        self._transport: Transport | None = None
        self._pending: dict[str, asyncio.Future[Response]] = {}
        self._streams: dict[str, asyncio.Queue[Any]] = {}
        self._req_seq = 0

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self._transport is not None:
            raise PiProcessError("Client already started")
        node = resolve_node(self._node)
        pi_ai_dir = self._pi_ai_dir or resolve_pi_ai_dir()
        argv = [node, shim_path(), "--pi-ai-dir", pi_ai_dir]
        if self._auth_path:
            argv += ["--auth-path", self._auth_path]
        env = {**os.environ, **self._env}
        self._transport = Transport(argv, cwd=self._cwd, env=env, on_line=self._handle_line)
        await self._transport.start()

    async def stop(self) -> None:
        if self._transport is None:
            return
        await self._transport.stop()
        self._transport = None
        self._reject_pending(PiProcessError("Client stopped"))

    async def __aenter__(self) -> "PiModelClient":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------- commands

    async def list_models(self, provider: str | None = None) -> list[dict[str, Any]]:
        """List available models (optionally filtered to one provider)."""
        data = self._data(await self._send({"type": "list_models", "provider": provider}))
        return data.get("models", [])

    async def list_providers(self) -> list[str]:
        """List the providers pi-ai knows about."""
        data = self._data(await self._send({"type": "list_providers"}))
        return data.get("providers", [])

    async def ping(self) -> bool:
        """Round-trip a no-op to confirm the shim is alive."""
        data = self._data(await self._send({"type": "ping"}))
        return bool(data.get("ok"))

    # ---------------------------------------------------------------- streaming

    async def stream(
        self,
        *,
        model: str | dict[str, Any],
        provider: str | None = None,
        messages: list[Any],
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        reasoning: str | None = None,
        timeout: float = _DEFAULT_STREAM_TIMEOUT,
        **options: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream an assistant response, yielding each :class:`StreamEvent` to completion.

        Args:
            model: A model id (string, with ``provider``) or a full pi-ai ``Model`` dict
                (for custom/local models).
            provider: Provider id (required when ``model`` is a string id).
            messages: pi-ai wire messages (``{"role", "content", "timestamp"}``, plus
                ``toolResult``/assistant shapes). The caller's agent loop owns history.
            system_prompt: Optional system prompt.
            tools: Optional tool definitions (``{"name", "description", "parameters"}``).
            reasoning: Thinking level — one of minimal/low/medium/high/xhigh.
            timeout: Max seconds to wait for the next event before aborting the stream.
            **options: Passed through to pi-ai ``SimpleStreamOptions`` (e.g. ``maxTokens``,
                ``temperature``, ``apiKey``, ``cacheRetention``).

        Yields:
            :class:`StreamEvent` objects. The final one has ``type`` ``done`` or ``error``
            (see :attr:`StreamEvent.is_terminal` / :attr:`StreamEvent.final_message`).

        Raises:
            PiModelError: a shim-level failure (e.g. unknown model id).
            PiTimeoutError: no event arrived within ``timeout``.
        """
        if self._transport is None:
            raise PiNotStartedError("Client not started")

        stream_id = self._next_id("stream")
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._streams[stream_id] = queue

        context: dict[str, Any] = {"messages": messages}
        if system_prompt is not None:
            context["systemPrompt"] = system_prompt
        if tools is not None:
            context["tools"] = tools
        opts = dict(options)
        if reasoning is not None:
            opts["reasoning"] = reasoning

        body: dict[str, Any] = {"type": "stream", "id": stream_id, "context": context, "options": opts}
        body["model"] = model
        if not isinstance(model, dict):
            body["provider"] = provider

        terminated = False
        try:
            await self._transport.write_line(body)
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout)
                except asyncio.TimeoutError as exc:
                    raise PiTimeoutError(
                        f"Timed out after {timeout}s waiting for the next stream event"
                    ) from exc
                if isinstance(item, BaseException):
                    terminated = True
                    raise item
                yield item
                if item.is_terminal:
                    terminated = True
                    return
        finally:
            self._streams.pop(stream_id, None)
            # If the consumer broke early / cancelled, tell the shim to stop the stream.
            if not terminated and self._transport is not None:
                try:
                    await self._transport.write_line({"type": "abort", "id": stream_id})
                except PiProcessError:
                    pass

    async def complete(
        self,
        *,
        model: str | dict[str, Any],
        provider: str | None = None,
        messages: list[Any],
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        reasoning: str | None = None,
        timeout: float = _DEFAULT_STREAM_TIMEOUT,
        **options: Any,
    ) -> AssistantMessage:
        """Stream to completion and return the final assistant message.

        Returns the final message even on a model ``error`` event (inspect its
        ``stopReason`` / ``errorMessage``); raises :class:`PiModelError` only for
        shim-level failures.
        """
        final: AssistantMessage | None = None
        async for event in self.stream(
            model=model,
            provider=provider,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            reasoning=reasoning,
            timeout=timeout,
            **options,
        ):
            if event.is_terminal:
                final = event.final_message
        if final is None:
            raise PiModelError("Model stream ended without a final message")
        return final

    # -------------------------------------------------------------------- internal

    def _next_id(self, prefix: str) -> str:
        self._req_seq += 1
        return f"{prefix}_{self._req_seq}"

    async def _send(self, body: dict[str, Any]) -> Response:
        """Send an id-correlated command (list_*/ping) and await its response."""
        if self._transport is None:
            raise PiNotStartedError("Client not started")
        req_id = self._next_id("req")
        command = {k: v for k, v in body.items() if v is not None}
        command["id"] = req_id

        loop = asyncio.get_event_loop()
        future: asyncio.Future[Response] = loop.create_future()
        self._pending[req_id] = future
        try:
            await self._transport.write_line(command)
        except Exception:
            self._pending.pop(req_id, None)
            raise
        try:
            response = await asyncio.wait_for(future, _REQUEST_TIMEOUT)
        except asyncio.TimeoutError as exc:
            self._pending.pop(req_id, None)
            stderr = self._transport.stderr_text() if self._transport else None
            raise PiTimeoutError(
                f"Timed out after {_REQUEST_TIMEOUT}s waiting for response to {body.get('type')!r}"
                + (f"\nStderr:\n{stderr}" if stderr else "")
            ) from exc
        if not response.success:
            raise PiModelError(response.error or "unknown error")
        return response

    @staticmethod
    def _data(response: Response) -> dict[str, Any]:
        return response.data if isinstance(response.data, dict) else {}

    def _handle_line(self, line: str) -> None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return  # ignore non-JSON output
        if not isinstance(data, dict):
            return

        kind = data.get("type")
        if kind == "response":
            req_id = data.get("id")
            future = self._pending.pop(req_id, None) if req_id else None
            if future is not None and not future.done():
                future.set_result(Response.model_validate(data))
            return
        if kind == "stream_event":
            queue = self._streams.get(data.get("id"))
            if queue is not None:
                queue.put_nowait(parse_stream_event(data.get("event") or {}))
            return
        if kind == "stream_error":
            queue = self._streams.get(data.get("id"))
            if queue is not None:
                queue.put_nowait(PiModelError(data.get("error") or "model stream error"))
            return
        # Unknown line types are ignored (forward-compatible).

    def _reject_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
        for queue in self._streams.values():
            queue.put_nowait(error)
        self._streams.clear()
