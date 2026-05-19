"""chat.py — interactive REPL with persistent sessions.

Conversations are saved to ~/.pi/chat-sessions/ and can be resumed
by their session ID. File and shell tools are enabled by default so
the agent can read files, run commands, etc.

Usage:
    uv run python examples/chat.py                        # new session
    uv run python examples/chat.py --session abc12345     # resume session
    uv run python examples/chat.py --no-tools             # text-only, no file access
    uv run python examples/chat.py --cwd /my/project      # set working directory

Commands during chat:
    /session    show current session ID
    /clear      start a new session (saves the old one)
    /quit       exit
"""
import argparse
import asyncio
import os
import sys

import pi_agent


SESSIONS_DIR = os.path.expanduser("~/.pi/chat-sessions")
COMMANDS = {"/quit", "/exit", "/q"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive agent chat.")
    p.add_argument("--session", "-s", metavar="ID", help="Resume a session by ID.")
    p.add_argument("--model", "-m", metavar="PROVIDER:ID", help="Model override.")
    p.add_argument("--cwd", metavar="PATH", help="Working directory (default: current dir).")
    p.add_argument("--no-tools", action="store_true", help="Disable file/shell tools.")
    return p.parse_args()


def _resolve_model(model_str: str | None):
    if not model_str:
        return None
    import pi_ai
    provider, model_id = model_str.split(":", 1)
    try:
        return pi_ai.get_model(provider, model_id)
    except Exception:
        m = pi_agent.find_custom_model(provider, model_id)
        if m is None:
            sys.exit(f"Model {model_str!r} not found.")
        return m


async def _new_harness(
    model, session_id: str | None, cwd: str | None, tools
) -> tuple[pi_agent.AgentHarness, str]:
    harness = await pi_agent.create_agent(
        model=model,
        cwd=cwd,
        sessions_dir=SESSIONS_DIR,
        session_id=session_id,
        tools=tools,
        system_prompt=(
            "You are a helpful personal assistant. "
            "When working with files, use the available tools rather than guessing."
        ),
    )
    meta = await harness._session.get_metadata()
    return harness, meta["id"]


async def _run_turn(harness: pi_agent.AgentHarness, text: str) -> bool:
    """Run one turn. Returns False if the turn errored."""
    printing = [False]

    def on_event(event: dict, *_) -> None:
        if event["type"] == "tool_execution_start":
            name = event.get("tool_name", "?")
            args = event.get("args") or {}
            hint = args.get("command") or args.get("path") or ""
            suffix = f": {str(hint)[:60]}" if hint else ""
            # Print on its own line, but only after a newline if we were streaming
            if printing[0]:
                print()
                printing[0] = False
            print(f"\033[36m  ⚙ {name}{suffix}\033[0m", flush=True)

        elif event["type"] == "message_update":
            ev = event.get("assistant_message_event") or {}
            if ev.get("type") == "text_delta":
                if not printing[0]:
                    print("\033[1mAssistant:\033[0m ", end="", flush=True)
                    printing[0] = True
                print(ev["delta"], end="", flush=True)

        elif event["type"] == "message_end":
            if printing[0]:
                print()
                printing[0] = False

    unsub = harness.subscribe(on_event)
    try:
        reply = await harness.prompt(text)
    except Exception as exc:
        print(f"\033[31mError: {exc}\033[0m", file=sys.stderr)
        return False
    finally:
        unsub()

    if reply.error_message:
        print(f"\033[31mError: {reply.error_message}\033[0m", file=sys.stderr)
        return False
    return True


async def main() -> None:
    args = _parse_args()
    model = _resolve_model(args.model)
    tools = None if args.no_tools else "all"
    cwd = os.path.abspath(args.cwd) if args.cwd else None

    harness, session_id = await _new_harness(model, args.session, cwd, tools)

    print(f"\033[90mSession: {session_id[:8]}  (resume with --session {session_id[:8]})\033[0m")
    print("\033[90mType /quit to exit, /session for session ID, /clear for a new session.\033[0m\n")

    while True:
        try:
            text = input("\033[32mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not text:
            continue
        if text in COMMANDS:
            print("Bye.")
            break
        if text == "/session":
            print(f"\033[90mSession: {session_id}\033[0m")
            continue
        if text == "/clear":
            harness, session_id = await _new_harness(model, None, cwd, tools)
            print(f"\033[90mNew session: {session_id[:8]}\033[0m\n")
            continue

        await _run_turn(harness, text)


if __name__ == "__main__":
    asyncio.run(main())
