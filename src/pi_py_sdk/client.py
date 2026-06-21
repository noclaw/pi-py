"""High-level async client for the Pi coding agent over RPC.

Phase 0 surface: lifecycle (`start`/`stop`/context manager), command sending with
id-correlated responses, event subscription, streaming a prompt to completion, and a
few core commands. The design mirrors Pi's reference ``RpcClient`` (rpc-client.ts) and
improves on its ``waitForIdle`` by honoring ``agent_end.willRetry``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, AsyncIterator, Awaitable, Callable, Union

from ._discovery import resolve_pi_command
from .config import PiConfig
from .errors import PiCommandError, PiNotStartedError, PiProcessError, PiTimeoutError
from .protocol import (
    DIALOG_METHODS,
    AgentEndEvent,
    Event,
    ExtensionUIRequest,
    Response,
    parse_event,
    parse_messages,
)
from .transport import Transport

EventListener = Callable[[Event], None]

#: A UI handler receives an ExtensionUIRequest and returns the reply value:
#:   * confirm  -> bool (True/False)
#:   * select/input/editor -> str (the chosen/entered value)
#:   * None     -> cancel the dialog
#: It may be sync or async.
UiResult = Union[str, bool, None]
UiHandler = Callable[[ExtensionUIRequest], Union[UiResult, Awaitable[UiResult]]]

_REQUEST_TIMEOUT = 30.0
_DEFAULT_PROMPT_TIMEOUT = 300.0


def _default_ui_handler(request: ExtensionUIRequest) -> UiResult:
    """Safe default: deny confirmations, cancel other dialogs (never blocks the agent)."""
    if request.method == "confirm":
        return False
    return None


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
        self._ui_listeners: list[Callable[[ExtensionUIRequest], None]] = []
        self._ui_handler: UiHandler = _default_ui_handler
        self._ui_tasks: set[asyncio.Task[None]] = set()
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
        for task in list(self._ui_tasks):
            task.cancel()
        self._ui_tasks.clear()
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

    def on_ui_request(self, handler: UiHandler) -> Callable[[], None]:
        """Install the handler that answers blocking extension dialogs (approvals,
        selects, inputs, editors). Replaces any previous handler.

        Returns a callable that restores the safe default (deny/cancel) handler.
        """
        self._ui_handler = handler

        def reset() -> None:
            self._ui_handler = _default_ui_handler

        return reset

    def observe_ui_requests(
        self, listener: Callable[[ExtensionUIRequest], None]
    ) -> Callable[[], None]:
        """Observe every UI request (including fire-and-forget notify/setStatus/...).

        Observers do not answer dialogs — that's the :meth:`on_ui_request` handler's job.
        """
        self._ui_listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._ui_listeners.remove(listener)
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

    # Model -----------------------------------------------------------------

    async def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        return self._data(
            await self._send({"type": "set_model", "provider": provider, "modelId": model_id})
        )

    async def cycle_model(self) -> Any:
        return (await self._send({"type": "cycle_model"})).data

    async def get_available_models(self) -> list[dict[str, Any]]:
        data = self._data(await self._send({"type": "get_available_models"}))
        return data.get("models", [])

    # Thinking --------------------------------------------------------------

    async def set_thinking_level(self, level: str) -> None:
        await self._send({"type": "set_thinking_level", "level": level})

    async def cycle_thinking_level(self) -> Any:
        return (await self._send({"type": "cycle_thinking_level"})).data

    # Queue modes -----------------------------------------------------------

    async def set_steering_mode(self, mode: str) -> None:
        """``mode`` is "all" or "one-at-a-time"."""
        await self._send({"type": "set_steering_mode", "mode": mode})

    async def set_follow_up_mode(self, mode: str) -> None:
        """``mode`` is "all" or "one-at-a-time"."""
        await self._send({"type": "set_follow_up_mode", "mode": mode})

    # Compaction ------------------------------------------------------------

    async def compact(self, custom_instructions: str | None = None) -> dict[str, Any]:
        return self._data(
            await self._send({"type": "compact", "customInstructions": custom_instructions})
        )

    async def set_auto_compaction(self, enabled: bool) -> None:
        await self._send({"type": "set_auto_compaction", "enabled": enabled})

    # Retry -----------------------------------------------------------------

    async def set_auto_retry(self, enabled: bool) -> None:
        await self._send({"type": "set_auto_retry", "enabled": enabled})

    async def abort_retry(self) -> None:
        await self._send({"type": "abort_retry"})

    # Bash ------------------------------------------------------------------

    async def bash(self, command: str, *, exclude_from_context: bool | None = None) -> dict[str, Any]:
        """Run a shell command. The result is stored as a BashExecutionMessage and only
        surfaced to the LLM on the *next* prompt (not sent immediately)."""
        return self._data(
            await self._send(
                {"type": "bash", "command": command, "excludeFromContext": exclude_from_context}
            )
        )

    async def abort_bash(self) -> None:
        await self._send({"type": "abort_bash"})

    # Session ---------------------------------------------------------------

    async def new_session(self, parent_session: str | None = None) -> dict[str, Any]:
        return self._data(
            await self._send({"type": "new_session", "parentSession": parent_session})
        )

    async def get_session_stats(self) -> dict[str, Any]:
        return self._data(await self._send({"type": "get_session_stats"}))

    async def export_html(self, output_path: str | None = None) -> dict[str, Any]:
        return self._data(await self._send({"type": "export_html", "outputPath": output_path}))

    async def switch_session(self, session_path: str) -> dict[str, Any]:
        return self._data(
            await self._send({"type": "switch_session", "sessionPath": session_path})
        )

    async def fork(self, entry_id: str) -> dict[str, Any]:
        return self._data(await self._send({"type": "fork", "entryId": entry_id}))

    async def clone(self) -> dict[str, Any]:
        return self._data(await self._send({"type": "clone"}))

    async def get_fork_messages(self) -> list[dict[str, Any]]:
        data = self._data(await self._send({"type": "get_fork_messages"}))
        return data.get("messages", [])

    async def set_session_name(self, name: str) -> None:
        await self._send({"type": "set_session_name", "name": name})

    # Messages / commands ---------------------------------------------------

    async def get_messages(self) -> list[Any]:
        data = self._data(await self._send({"type": "get_messages"}))
        return parse_messages(data.get("messages", []))

    async def get_commands(self) -> list[dict[str, Any]]:
        data = self._data(await self._send({"type": "get_commands"}))
        return data.get("commands", [])

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

        # Extension UI sub-protocol (dialogs / approvals / notifications).
        if data.get("type") == "extension_ui_request":
            self._dispatch_ui_request(ExtensionUIRequest.model_validate(data))
            return

        # Otherwise a session/agent event.
        event = parse_event(data)
        for listener in list(self._listeners):
            listener(event)

    def _dispatch_ui_request(self, request: ExtensionUIRequest) -> None:
        for listener in list(self._ui_listeners):
            listener(request)
        if request.method in DIALOG_METHODS:
            task = asyncio.create_task(self._answer_ui_request(request))
            self._ui_tasks.add(task)
            task.add_done_callback(self._ui_tasks.discard)

    async def _answer_ui_request(self, request: ExtensionUIRequest) -> None:
        try:
            result = self._ui_handler(request)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            result = None  # cancel on handler error rather than hang the agent
        await self._send_ui_response(request, result)

    async def _send_ui_response(self, request: ExtensionUIRequest, result: UiResult) -> None:
        if result is None:
            body: dict[str, Any] = {"type": "extension_ui_response", "id": request.id, "cancelled": True}
        elif request.method == "confirm":
            body = {"type": "extension_ui_response", "id": request.id, "confirmed": bool(result)}
        else:
            body = {"type": "extension_ui_response", "id": request.id, "value": str(result)}
        # Not an id-correlated command: write directly, no response is expected.
        if self._transport is not None:
            try:
                await self._transport.write_line(body)
            except PiProcessError:
                pass

    def _reject_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
