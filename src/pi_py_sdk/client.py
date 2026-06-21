"""High-level async client for the Pi coding agent over RPC.

Phase 0 surface: lifecycle (`start`/`stop`/context manager), command sending with
id-correlated responses, event subscription, streaming a prompt to completion, and a
few core commands. The design mirrors Pi's reference ``RpcClient`` (rpc-client.ts) and
improves on its ``waitForIdle`` by honoring ``agent_end.willRetry``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable

from ._discovery import resolve_pi_command
from .config import PiConfig
from .errors import PiCommandError, PiNotStartedError, PiProcessError, PiTimeoutError
from .protocol import AgentEndEvent, Event, Response, parse_event
from .transport import Transport

EventListener = Callable[[Event], None]

_REQUEST_TIMEOUT = 30.0
_DEFAULT_PROMPT_TIMEOUT = 300.0


class PiAgent:
    """Async handle to a running Pi agent session.

    Example:
        async with PiAgent(model="anthropic/claude-sonnet-4-20250514") as agent:
            async for ev in agent.prompt_stream("hello"):
                ...
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        config: PiConfig | None = None,
    ) -> None:
        self._config = config or PiConfig(
            model=model, provider=provider, cwd=cwd, env=env or {}
        )
        self._transport: Transport | None = None
        self._listeners: list[EventListener] = []
        self._pending: dict[str, asyncio.Future[Response]] = {}
        self._req_seq = 0

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self._transport is not None:
            raise PiProcessError("Client already started")
        argv = resolve_pi_command(self._config.bin) + self._config.mode_args()
        self._transport = Transport(
            argv,
            cwd=self._config.cwd,
            env=self._config.build_env(),
            on_line=self._handle_line,
        )
        await self._transport.start()

    async def stop(self) -> None:
        if self._transport is None:
            return
        await self._transport.stop()
        self._transport = None
        self._reject_pending(PiProcessError("Client stopped"))

    async def __aenter__(self) -> "PiAgent":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # --------------------------------------------------------------- subscriptions

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        """Register an event listener; returns an unsubscribe callable."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    # ------------------------------------------------------------------- commands

    async def prompt(self, message: str, *, images: list[Any] | None = None) -> None:
        """Send a prompt. Returns once preflight succeeds — *not* when the run ends.

        Use :meth:`prompt_stream` or :meth:`wait_for_idle` to await completion.
        """
        await self._send({"type": "prompt", "message": message, "images": images})

    async def steer(self, message: str, *, images: list[Any] | None = None) -> None:
        await self._send({"type": "steer", "message": message, "images": images})

    async def follow_up(self, message: str, *, images: list[Any] | None = None) -> None:
        await self._send({"type": "follow_up", "message": message, "images": images})

    async def abort(self) -> None:
        await self._send({"type": "abort"})

    async def get_state(self) -> dict[str, Any]:
        return self._data(await self._send({"type": "get_state"}))

    async def get_last_assistant_text(self) -> str | None:
        data = self._data(await self._send({"type": "get_last_assistant_text"}))
        return data.get("text")

    # ---------------------------------------------------------------- streaming

    async def prompt_stream(
        self, message: str, *, images: list[Any] | None = None, timeout: float = _DEFAULT_PROMPT_TIMEOUT
    ) -> AsyncIterator[Event]:
        """Send a prompt and yield every event until the run completes.

        Completion = an ``agent_end`` event with ``willRetry`` false. An
        ``agent_end`` with ``willRetry`` true is yielded but does not end the stream
        (a retry follows).
        """
        queue: asyncio.Queue[Event] = asyncio.Queue()
        unsubscribe = self.subscribe(queue.put_nowait)
        try:
            await self.prompt(message, images=images)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout)
                except asyncio.TimeoutError as exc:
                    raise PiTimeoutError(
                        f"Timed out after {timeout}s waiting for prompt to complete"
                    ) from exc
                yield event
                if isinstance(event, AgentEndEvent) and not event.willRetry:
                    return
        finally:
            unsubscribe()

    async def wait_for_idle(self, timeout: float = _DEFAULT_PROMPT_TIMEOUT) -> None:
        """Wait until an ``agent_end`` with ``willRetry`` false is observed."""
        done: asyncio.Future[None] = asyncio.get_event_loop().create_future()

        def listener(event: Event) -> None:
            if isinstance(event, AgentEndEvent) and not event.willRetry and not done.done():
                done.set_result(None)

        unsubscribe = self.subscribe(listener)
        try:
            await asyncio.wait_for(done, timeout)
        except asyncio.TimeoutError as exc:
            raise PiTimeoutError(f"Timed out after {timeout}s waiting for idle") from exc
        finally:
            unsubscribe()

    async def prompt_and_wait(
        self, message: str, *, images: list[Any] | None = None, timeout: float = _DEFAULT_PROMPT_TIMEOUT
    ) -> list[Event]:
        """Send a prompt and collect all events until completion."""
        return [ev async for ev in self.prompt_stream(message, images=images, timeout=timeout)]

    # -------------------------------------------------------------------- internal

    async def _send(self, body: dict[str, Any]) -> Response:
        if self._transport is None:
            raise PiNotStartedError("Client not started")
        self._req_seq += 1
        req_id = f"req_{self._req_seq}"
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
            raise PiCommandError(response.command, response.error or "unknown error")
        return response

    @staticmethod
    def _data(response: Response) -> dict[str, Any]:
        return response.data if isinstance(response.data, dict) else {}

    def _handle_line(self, line: str) -> None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return  # ignore non-JSON output (mirrors the reference client)
        if not isinstance(data, dict):
            return

        # Correlated response?
        if data.get("type") == "response":
            req_id = data.get("id")
            future = self._pending.pop(req_id, None) if req_id else None
            if future is not None and not future.done():
                future.set_result(Response.model_validate(data))
                return
            # Uncorrelated/late response: drop it.
            return

        # Otherwise an event (the extension_ui_request sub-protocol arrives as an
        # event here too; a dedicated handler lands in Phase 2).
        event = parse_event(data)
        for listener in list(self._listeners):
            listener(event)

    def _reject_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
