"""Synchronous facade over :class:`PiAgent` for non-async callers.

Runs the async client on a dedicated background thread with its own event loop;
every method blocks until the underlying coroutine completes. Streaming is bridged to
a plain iterator.

    with PiAgentSync(model="anthropic/claude-sonnet-4-20250514") as agent:
        for event in agent.prompt_stream("hello"):
            ...
        print(agent.get_last_assistant_text())

Note: event listeners registered via ``subscribe``/``on_ui_request`` are invoked on the
background loop thread, not the caller's thread.
"""

from __future__ import annotations

import asyncio
import inspect
import queue
import threading
from typing import Any, Callable, Iterator

from .client import PiAgent
from .model import PiModelClient

_STREAM_DEFAULT_TIMEOUT = 300.0
_MODEL_STREAM_DEFAULT_TIMEOUT = 600.0


class PiAgentSync:
    """Blocking wrapper around :class:`PiAgent`.

    Unknown attributes are delegated to the wrapped agent: coroutine methods are run on
    the background loop and block until done; plain attributes/methods pass through.
    """

    def __init__(self, *, agent: PiAgent | None = None, **kwargs: Any) -> None:
        object.__setattr__(self, "_agent", agent if agent is not None else PiAgent(**kwargs))
        loop = asyncio.new_event_loop()
        object.__setattr__(self, "_loop", loop)
        thread = threading.Thread(target=loop.run_forever, name="pi-sync-loop", daemon=True)
        object.__setattr__(self, "_thread", thread)
        thread.start()

    def _submit(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._submit(self._agent.start())

    def stop(self) -> None:
        try:
            self._submit(self._agent.stop())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)

    def __enter__(self) -> "PiAgentSync":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- streaming --------------------------------------------------------

    def prompt_stream(
        self, message: str, *, images: list[Any] | None = None, timeout: float = _STREAM_DEFAULT_TIMEOUT
    ) -> Iterator[Any]:
        """Send a prompt and iterate its events synchronously until completion."""
        sentinel = object()
        bridge: "queue.Queue[Any]" = queue.Queue()

        async def pump() -> None:
            try:
                async for event in self._agent.prompt_stream(message, images=images, timeout=timeout):
                    bridge.put(event)
            except BaseException as exc:  # surface to the consumer thread
                bridge.put(exc)
            finally:
                bridge.put(sentinel)

        asyncio.run_coroutine_threadsafe(pump(), self._loop)

        def generator() -> Iterator[Any]:
            while True:
                item = bridge.get()
                if item is sentinel:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item

        return generator()

    def prompt_and_wait(
        self, message: str, *, images: list[Any] | None = None, timeout: float = _STREAM_DEFAULT_TIMEOUT
    ) -> list[Any]:
        return list(self.prompt_stream(message, images=images, timeout=timeout))

    # -- delegation -------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached when `name` isn't found normally. Guard internals to avoid
        # recursion during construction.
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._agent, name)
        if inspect.iscoroutinefunction(attr):

            def call(*args: Any, **kwargs: Any) -> Any:
                return self._submit(attr(*args, **kwargs))

            call.__name__ = name
            return call
        return attr

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        return self._agent.subscribe(listener)

    def on_ui_request(self, handler: Any) -> Callable[[], None]:
        return self._agent.on_ui_request(handler)


class PiModelClientSync:
    """Blocking wrapper around :class:`PiModelClient` (see :class:`PiAgentSync`).

    Coroutine methods (``list_models``, ``list_providers``, ``ping``, ``complete``) are
    delegated to the background loop and block until done; :meth:`stream` is bridged to a
    plain iterator.

        with PiModelClientSync() as client:
            for ev in client.stream(provider="anthropic", model="claude-sonnet-4-6",
                                     messages=[{"role": "user", "content": "hi", "timestamp": 0}]):
                if ev.type == "text_delta":
                    print(ev.delta, end="", flush=True)
    """

    def __init__(self, *, client: PiModelClient | None = None, **kwargs: Any) -> None:
        object.__setattr__(self, "_client", client if client is not None else PiModelClient(**kwargs))
        loop = asyncio.new_event_loop()
        object.__setattr__(self, "_loop", loop)
        thread = threading.Thread(target=loop.run_forever, name="pi-model-sync-loop", daemon=True)
        object.__setattr__(self, "_thread", thread)
        thread.start()

    def _submit(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._submit(self._client.start())

    def stop(self) -> None:
        try:
            self._submit(self._client.stop())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)

    def __enter__(self) -> "PiModelClientSync":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- streaming --------------------------------------------------------

    def stream(self, *, timeout: float = _MODEL_STREAM_DEFAULT_TIMEOUT, **kwargs: Any) -> Iterator[Any]:
        """Stream an assistant response synchronously (see :meth:`PiModelClient.stream`)."""
        sentinel = object()
        bridge: "queue.Queue[Any]" = queue.Queue()

        async def pump() -> None:
            try:
                async for event in self._client.stream(timeout=timeout, **kwargs):
                    bridge.put(event)
            except BaseException as exc:  # surface to the consumer thread
                bridge.put(exc)
            finally:
                bridge.put(sentinel)

        asyncio.run_coroutine_threadsafe(pump(), self._loop)

        def generator() -> Iterator[Any]:
            while True:
                item = bridge.get()
                if item is sentinel:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item

        return generator()

    # -- delegation -------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._client, name)
        if inspect.iscoroutinefunction(attr):

            def call(*args: Any, **kwargs: Any) -> Any:
                return self._submit(attr(*args, **kwargs))

            call.__name__ = name
            return call
        return attr
