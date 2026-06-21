"""Interactive REPL and one-shot runner for the Python coding agent.

Built entirely on :class:`pi_py_sdk.PiAgent` — the agent loop, tools, and model calls
all run inside Pi via the RPC bridge.
"""

from __future__ import annotations

import asyncio

from pi_py_sdk import ExtensionUIRequest, PiAgent, PiConfig, PiTimeoutError

from .render import Renderer

_HELP = """\
Commands:
  /help            show this help
  /model           show the current model
  /models          list available models
  /new             start a fresh session
  /state           show session state (model, counts, modes)
  /exit, /quit     leave
Ctrl-C during a response aborts that turn; Ctrl-D (EOF) exits."""


async def _ainput(prompt: str) -> str:
    """Read a line without blocking the event loop."""
    return await asyncio.get_event_loop().run_in_executor(None, input, prompt)


def _make_approver(color: bool):
    async def approve(req: ExtensionUIRequest):
        if req.method == "confirm":
            ans = (await _ainput(f"\n[approve] {req.title or 'Confirm'}? [y/N] ")).strip().lower()
            return ans in ("y", "yes")
        if req.method == "select":
            options = req.options or []
            print(f"\n{req.title or 'Choose'}:")
            for i, opt in enumerate(options):
                print(f"  {i}) {opt}")
            choice = (await _ainput("[select] number: ")).strip()
            return options[int(choice)] if choice.isdigit() and int(choice) < len(options) else None
        if req.method in ("input", "editor"):
            return (await _ainput(f"[{req.method}] {req.title or 'Value'}: ")) or None
        return None

    return approve


async def _stream_turn(agent: PiAgent, renderer: Renderer, message: str) -> None:
    try:
        async for event in agent.prompt_stream(message):
            renderer.handle(event)
    except KeyboardInterrupt:
        await agent.abort()
        print("\n[aborted]")
    except PiTimeoutError as exc:
        print(f"\n[timeout] {exc}")


async def _model_label(agent: PiAgent) -> str:
    state = await agent.get_state()
    model = state.get("model") or {}
    return f"{model.get('provider')}/{model.get('id')}"


async def _handle_command(agent: PiAgent, line: str) -> bool:
    """Return True if the REPL should exit."""
    cmd = line.split()[0].lower()
    if cmd in ("/exit", "/quit"):
        return True
    if cmd == "/help":
        print(_HELP)
    elif cmd == "/model":
        print(await _model_label(agent))
    elif cmd == "/models":
        for m in await agent.get_available_models():
            print(f"  {m.get('provider')}/{m.get('id')}")
    elif cmd == "/new":
        await agent.new_session()
        print("[new session]")
    elif cmd == "/state":
        st = await agent.get_state()
        print(
            f"model={await _model_label(agent)} thinking={st.get('thinkingLevel')} "
            f"messages={st.get('messageCount')} steering={st.get('steeringMode')} "
            f"followUp={st.get('followUpMode')}"
        )
    else:
        print(f"unknown command: {cmd} (try /help)")
    return False


async def run_repl(config: PiConfig, *, color: bool = True, initial: str | None = None) -> None:
    renderer = Renderer(color=color)
    async with PiAgent(config=config) as agent:
        agent.on_ui_request(_make_approver(color))
        print(f"pi-py · {await _model_label(agent)} · /help for commands")

        if initial:
            print(f"\nYou: {initial}")
            await _stream_turn(agent, renderer, initial)

        while True:
            try:
                line = (await _ainput("\nYou: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if await _handle_command(agent, line):
                    break
                continue
            await _stream_turn(agent, renderer, line)


async def run_once(config: PiConfig, message: str, *, color: bool = True) -> None:
    renderer = Renderer(color=color)
    async with PiAgent(config=config) as agent:
        agent.on_ui_request(_make_approver(color))
        async for event in agent.prompt_stream(message):
            renderer.handle(event)
        print()
