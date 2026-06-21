"""Interactive REPL and one-shot runner for the Python coding agent.

Built entirely on :class:`pi_py_sdk.PiAgent` — the agent loop, tools, and model calls
all run inside Pi via the RPC bridge.

The REPL multiplexes a single stdin source across three consumers — idle prompts,
mid-turn steering, and approval dialogs — using one persistent reader thread feeding a
single async dispatch loop. A turn runs as a background task so the loop stays free to
route steering/abort input while the agent streams. SIGINT aborts the active turn.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
from typing import Any

from pi_py_sdk import ExtensionUIRequest, PiAgent, PiConfig, PiTimeoutError

from .render import Renderer

_HELP = """\
Commands (when idle):
  /help                  show this help
  /model | /models       show or list models
  /new                   start a fresh session
  /state                 model, message count, queue modes
  /compact               compact the conversation now
  /fork [n]              list forkable messages, or fork message n
  /clone                 clone the current branch into a new session
  /exit | /quit          leave

While the agent is responding, type to steer it:
  <text>                 steer (delivered after the current tool call)
  +<text>                queue a follow-up (delivered after the turn)
  /abort                 stop the current turn
Ctrl-C aborts the active turn; Ctrl-D (EOF) exits."""

_YES = {"y", "yes"}


def classify_turn_input(line: str) -> tuple[str, str]:
    """Classify a line typed during an active turn into (action, text)."""
    if line in ("/exit", "/quit"):
        return ("exit", "")
    if line in ("/abort", "/stop"):
        return ("abort", "")
    if line.startswith("+"):
        return ("follow_up", line[1:].strip())
    return ("steer", line)


def parse_approval(request: ExtensionUIRequest, line: str) -> Any:
    """Map a typed line to the value the SDK expects for a dialog request."""
    method = request.method
    if method == "confirm":
        return line.strip().lower() in _YES
    if method == "select":
        options = request.options or []
        stripped = line.strip()
        if stripped.isdigit() and int(stripped) < len(options):
            return options[int(stripped)]
        return None
    if method in ("input", "editor"):
        return line if line else None
    return None


class LineReader:
    """Reads stdin lines on a daemon thread into an asyncio queue.

    A daemon thread (not the loop executor) is used so a blocked ``readline`` never
    prevents interpreter exit. EOF enqueues a ``None`` sentinel.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while True:
            line = sys.stdin.readline()
            assert self._loop is not None
            if line == "":  # EOF
                self._loop.call_soon_threadsafe(self.queue.put_nowait, None)
                return
            self._loop.call_soon_threadsafe(self.queue.put_nowait, line.rstrip("\n"))

    async def get(self) -> str | None:
        return await self.queue.get()


class Repl:
    def __init__(self, agent: PiAgent, *, color: bool = True) -> None:
        self.agent = agent
        self.renderer = Renderer(color=color)
        self.color = color
        self.reader = LineReader()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._pending_approval: tuple[ExtensionUIRequest, asyncio.Future[Any]] | None = None
        self._fork_list: list[dict[str, Any]] = []
        self._exit = False

    # -- output helpers ---------------------------------------------------

    def _out(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def _idle_prompt(self) -> None:
        if not self._exit:
            self._out("\nYou: ")

    def _turn_active(self) -> bool:
        return self._turn_task is not None and not self._turn_task.done()

    # -- lifecycle --------------------------------------------------------

    async def run(self, initial: str | None = None) -> None:
        self._loop = asyncio.get_event_loop()
        self.reader.start()
        self.agent.on_ui_request(self._approve)
        self._install_sigint()
        self._out(f"pi-py · {await self._model_label()} · /help for commands\n")

        if initial:
            self._out(f"You: {initial}\n")
            self._start_turn(initial)
        else:
            self._idle_prompt()

        try:
            while not self._exit:
                line = await self.reader.get()
                if line is None:  # EOF
                    break
                await self._dispatch(line)
        finally:
            self._teardown()

    def _install_sigint(self) -> None:
        try:
            assert self._loop is not None
            self._loop.add_signal_handler(signal.SIGINT, self._on_sigint)
        except (NotImplementedError, RuntimeError):
            pass  # e.g. Windows / non-main thread — fall back to default behavior

    def _on_sigint(self) -> None:
        if self._turn_active():
            self._out("\n[aborting]\n")
            asyncio.ensure_future(self.agent.abort())
        else:
            self._exit = True
            self.reader.queue.put_nowait(None)  # unblock the dispatch loop

    def _teardown(self) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        if self._pending_approval and not self._pending_approval[1].done():
            self._pending_approval[1].set_result(None)

    # -- input dispatch ---------------------------------------------------

    async def _dispatch(self, line: str) -> None:
        # 1) An open approval dialog claims the next line.
        if self._pending_approval is not None:
            request, future = self._pending_approval
            if not future.done():
                future.set_result(parse_approval(request, line))
            return

        # 2) During a turn, input steers the agent.
        if self._turn_active():
            await self._route_steering(line)
            return

        # 3) Idle: blank, command, or a new prompt.
        stripped = line.strip()
        if not stripped:
            self._idle_prompt()
            return
        if stripped.startswith("/"):
            if await self._command(stripped):
                self._exit = True
                return
            self._idle_prompt()
            return
        self._start_turn(stripped)

    async def _route_steering(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        action, text = classify_turn_input(stripped)
        if action == "exit":
            await self.agent.abort()
            self._exit = True
        elif action == "abort":
            await self.agent.abort()
            self._out("[aborting]\n")
        elif action == "follow_up" and text:
            await self.agent.follow_up(text)
            self._out(f"[queued follow-up: {text}]\n")
        elif action == "steer":
            await self.agent.steer(text)
            self._out(f"[steered: {text}]\n")

    # -- turns ------------------------------------------------------------

    def _start_turn(self, message: str) -> None:
        self._turn_task = asyncio.ensure_future(self._run_turn(message))
        self._turn_task.add_done_callback(self._after_turn)

    async def _run_turn(self, message: str) -> None:
        try:
            async for event in self.agent.prompt_stream(message):
                self.renderer.handle(event)
        except asyncio.CancelledError:
            raise
        except PiTimeoutError as exc:
            self._out(f"\n[timeout] {exc}\n")
        except Exception as exc:  # keep the REPL alive on agent errors
            self._out(f"\n[error] {exc}\n")

    def _after_turn(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()  # retrieve to avoid "never retrieved" warnings
        self._idle_prompt()

    # -- approvals --------------------------------------------------------

    async def _approve(self, request: ExtensionUIRequest) -> Any:
        assert self._loop is not None
        future: asyncio.Future[Any] = self._loop.create_future()
        self._pending_approval = (request, future)
        self._print_approval_prompt(request)
        try:
            return await future
        finally:
            self._pending_approval = None

    def _print_approval_prompt(self, request: ExtensionUIRequest) -> None:
        self.renderer._break()  # ensure we start on a fresh line
        if request.method == "confirm":
            self._out(f"\n[approve] {request.title or 'Confirm'}? [y/N] ")
        elif request.method == "select":
            self._out(f"\n{request.title or 'Choose'}:\n")
            for i, option in enumerate(request.options or []):
                self._out(f"  {i}) {option}\n")
            self._out("[select] number: ")
        elif request.method in ("input", "editor"):
            self._out(f"\n[{request.method}] {request.title or 'Value'}: ")

    # -- commands ---------------------------------------------------------

    async def _model_label(self) -> str:
        state = await self.agent.get_state()
        model = state.get("model") or {}
        return f"{model.get('provider')}/{model.get('id')}"

    async def _command(self, line: str) -> bool:
        """Handle a slash command. Return True to exit the REPL."""
        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]
        if cmd in ("/exit", "/quit"):
            return True
        if cmd == "/help":
            self._out(_HELP + "\n")
        elif cmd == "/model":
            self._out(await self._model_label() + "\n")
        elif cmd == "/models":
            for m in await self.agent.get_available_models():
                self._out(f"  {m.get('provider')}/{m.get('id')}\n")
        elif cmd == "/new":
            await self.agent.new_session()
            self._out("[new session]\n")
        elif cmd == "/state":
            st = await self.agent.get_state()
            self._out(
                f"model={await self._model_label()} thinking={st.get('thinkingLevel')} "
                f"messages={st.get('messageCount')} steering={st.get('steeringMode')} "
                f"follow-up={st.get('followUpMode')}\n"
            )
        elif cmd == "/compact":
            self._out("[compacting…]\n")
            await self.agent.compact()
            self._out("[compacted]\n")
        elif cmd == "/clone":
            result = await self.agent.clone()
            self._out("[cloned]\n" if not result.get("cancelled") else "[clone cancelled]\n")
        elif cmd == "/fork":
            await self._fork(args)
        else:
            self._out(f"unknown command: {cmd} (try /help)\n")
        return False

    async def _fork(self, args: list[str]) -> None:
        if args and args[0].isdigit():
            index = int(args[0])
            if not self._fork_list:
                self._fork_list = await self.agent.get_fork_messages()
            if 0 <= index < len(self._fork_list):
                await self.agent.fork(self._fork_list[index]["entryId"])
                self._out(f"[forked at message {index}]\n")
            else:
                self._out(f"no message {index}; run /fork to list\n")
            return
        self._fork_list = await self.agent.get_fork_messages()
        if not self._fork_list:
            self._out("[no forkable messages]\n")
            return
        self._out("Forkable messages (use /fork <n>):\n")
        for i, msg in enumerate(self._fork_list):
            text = (msg.get("text") or "").replace("\n", " ")
            self._out(f"  {i}) {text[:80]}\n")


async def run_repl(config: PiConfig, *, color: bool = True, initial: str | None = None) -> None:
    async with PiAgent(config=config) as agent:
        await Repl(agent, color=color).run(initial=initial)


async def run_once(config: PiConfig, message: str, *, color: bool = True) -> None:
    renderer = Renderer(color=color)
    async with PiAgent(config=config) as agent:
        async for event in agent.prompt_stream(message):
            renderer.handle(event)
        print()
