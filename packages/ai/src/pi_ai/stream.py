from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

from .types import AssistantMessage


class AssistantMessageEventStream:
    """Async-iterable stream of assistant message events.

    Events are plain dicts with a ``type`` key. Terminal event types are
    ``"done"`` and ``"error"``. Iterate with ``async for``, or skip iteration
    and call ``await stream.result()`` to get the final message directly.

    Must be used from within a running asyncio event loop.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()
        self._done = False
        self._final_message: Optional[AssistantMessage] = None
        self._result_ready = asyncio.Event()

    def push(self, event: dict[str, Any]) -> None:
        """Push an event onto the stream. Non-blocking."""
        if self._done:
            return
        if event.get("type") in ("done", "error"):
            self._done = True
            self._final_message = event.get("message") or event.get("error")
            self._result_ready.set()
        self._queue.put_nowait(event)

    def end(self, result: Optional[AssistantMessage] = None) -> None:
        """Close the stream. Call after pushing the terminal event."""
        self._done = True
        if result is not None and not self._result_ready.is_set():
            self._final_message = result
            self._result_ready.set()
        self._queue.put_nowait(None)  # sentinel

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._queue.get()
            if item is None:
                return  # sentinel: stream closed
            yield item
            if item.get("type") in ("done", "error"):
                return  # terminal event: stop iterating

    async def result(self) -> AssistantMessage:
        """Await the final AssistantMessage produced by the stream."""
        await self._result_ready.wait()
        assert self._final_message is not None, "Stream closed without a final message"
        return self._final_message
