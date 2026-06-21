"""`pi-py` command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys

from pi_py_sdk import PiConfig

from .app import run_once, run_repl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pi-py",
        description="A Python coding agent built on the Pi RPC bridge.",
    )
    parser.add_argument("prompt", nargs="*", help="optional prompt; omit for an interactive REPL")
    parser.add_argument("--model", help="model id, e.g. anthropic/claude-sonnet-4-20250514")
    parser.add_argument("--provider", help="provider override")
    parser.add_argument("--cwd", help="working directory for the agent")
    parser.add_argument("--session-dir", help="directory for session persistence")
    parser.add_argument("--no-session", action="store_true", help="disable session persistence")
    parser.add_argument("--print", action="store_true", help="one-shot: print the response and exit")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PiConfig(
        model=args.model,
        provider=args.provider,
        cwd=args.cwd,
        session_dir=args.session_dir,
        no_session=args.no_session,
    )
    color = sys.stdout.isatty() and not args.no_color
    prompt = " ".join(args.prompt).strip()

    try:
        if prompt and args.print:
            asyncio.run(run_once(config, prompt, color=color))
        else:
            asyncio.run(run_repl(config, color=color, initial=prompt or None))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
