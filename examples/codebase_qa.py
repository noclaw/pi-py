"""codebase_qa.py — answer questions about a codebase.

Loads CLAUDE.md / AGENTS.md from the project path into the system prompt
and enables the read/grep/find/ls tools so the agent can explore the code.

Usage:
    uv run python examples/codebase_qa.py /path/to/project "What does this codebase do?"
    uv run python examples/codebase_qa.py /path/to/project  # interactive mode
    uv run python examples/codebase_qa.py .                 # current directory
"""
import argparse
import asyncio
import sys

import pi_agent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q&A about a codebase.")
    p.add_argument("path", help="Path to the project root.")
    p.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode).")
    p.add_argument("--model", "-m", metavar="PROVIDER:ID", help="Model override.")
    p.add_argument("--session", "-s", metavar="ID", help="Resume session by ID.")
    p.add_argument("--sessions-dir", metavar="PATH", default=None,
                   help="Where to save sessions (default: in-memory).")
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


async def main() -> None:
    args = _parse_args()
    import os
    path = os.path.abspath(args.path)
    if not os.path.isdir(path):
        sys.exit(f"Not a directory: {path}")

    model = _resolve_model(args.model)

    # Load project context files (CLAUDE.md, AGENTS.md, etc.)
    context = pi_agent.load_context_files(path)
    context_note = f"\n\n{context}" if context else ""

    harness = await pi_agent.create_agent(
        model=model,
        cwd=path,
        session_id=args.session,
        sessions_dir=args.sessions_dir,
        # Disable write/edit/bash — read-only exploration
        tools=[
            pi_agent.create_read_tool(path),
            pi_agent.create_grep_tool(path),
            pi_agent.create_find_tool(path),
            pi_agent.create_ls_tool(path),
        ],
        system_prompt=(
            f"You are a helpful code reviewer. "
            f"The project is at {path}. "
            f"Use the available tools to read files and answer questions accurately. "
            f"When citing code, include the file path and relevant line numbers."
            f"{context_note}"
        ),
        context_files=[],  # already loaded above
    )

    meta = await harness._session.get_metadata()
    if context:
        print(f"\033[90mLoaded context from {path}\033[0m")
    print(f"\033[90mSession: {meta['id'][:8]}  cwd: {path}\033[0m\n")

    def on_event(event: dict, *_) -> None:
        if event["type"] == "tool_execution_start":
            name = event.get("tool_name", "?")
            args_ = event.get("args") or {}
            hint = args_.get("path") or args_.get("pattern") or ""
            suffix = f" {hint}" if hint else ""
            print(f"\033[36m  ⚙ {name}{suffix}\033[0m", flush=True)
        elif event["type"] == "message_update":
            ev = event.get("assistant_message_event") or {}
            if ev.get("type") == "text_delta":
                print(ev["delta"], end="", flush=True)
        elif event["type"] == "message_end":
            msg = event.get("message")
            if hasattr(msg, "stop_reason"):
                print()

    harness.subscribe(on_event)

    if args.question:
        # Single question mode
        try:
            reply = await harness.prompt(args.question)
        except Exception as exc:
            sys.exit(f"Error: {exc}")
        if reply.error_message:
            sys.exit(f"Error: {reply.error_message}")
    else:
        # Interactive mode
        print("Ask questions about the codebase. Type /quit to exit.\n")
        while True:
            try:
                question = input("\033[32mQuestion:\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not question or question in ("/quit", "/exit", "/q"):
                break
            try:
                await harness.prompt(question)
            except Exception as exc:
                print(f"\033[31mError: {exc}\033[0m", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
