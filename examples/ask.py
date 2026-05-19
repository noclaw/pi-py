"""ask.py — one-shot prompt with no session persistence.

The simplest pi-agent usage: ask a question, print the answer, exit.
Model and auth are resolved from ~/.pi/agent/ automatically.

Usage:
    uv run python examples/ask.py "What is the capital of France?"
    uv run python examples/ask.py --model anthropic:claude-haiku-4-5 "Explain closures"
    uv run python examples/ask.py --tools "Summarise the README in this directory"
"""
import argparse
import asyncio
import sys

import pi_agent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-shot agent prompt.")
    p.add_argument("prompt", help="The question or instruction.")
    p.add_argument(
        "--model", "-m", metavar="PROVIDER:ID",
        help="Model override, e.g. anthropic:claude-haiku-4-5",
    )
    p.add_argument(
        "--tools", action="store_true",
        help="Enable built-in file and shell tools (off by default).",
    )
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    model = None
    if args.model:
        import pi_ai
        provider, model_id = args.model.split(":", 1)
        try:
            model = pi_ai.get_model(provider, model_id)
        except Exception:
            model = pi_agent.find_custom_model(provider, model_id)
            if model is None:
                sys.exit(f"Model {args.model!r} not found. Run 'pi-py models list'.")

    harness = await pi_agent.create_agent(
        model=model,
        tools="all" if args.tools else None,
    )

    # Stream text deltas to stdout as they arrive
    def on_event(event: dict, *_) -> None:
        if event["type"] == "message_update":
            ev = event.get("assistant_message_event") or {}
            if ev.get("type") == "text_delta":
                print(ev["delta"], end="", flush=True)

    harness.subscribe(on_event)

    try:
        reply = await harness.prompt(args.prompt)
    except Exception as exc:
        sys.exit(f"\nError: {exc}")

    print()  # newline after streamed content

    if reply.error_message:
        sys.exit(f"Error: {reply.error_message}")

    # Brief cost/token summary on stderr so stdout stays clean
    u = reply.usage
    print(
        f"[{harness.get_model().id} · {u.input}in/{u.output}out"
        + (f" · ${u.cost.total:.6f}" if u.cost.total else "") + "]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    asyncio.run(main())
