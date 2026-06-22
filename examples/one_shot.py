"""Stream a single prompt's response to the terminal.

Prereqs:
  * `pi` on PATH (`npm i -g @earendil-works/pi-coding-agent`) — or Node/npx for the fallback.
  * A provider key in the environment, e.g. ANTHROPIC_API_KEY.

Usage:
  python examples/one_shot.py "List the Python files in this directory"
"""

from __future__ import annotations

import asyncio
import sys

from pi_py_sdk import (
    AgentEndEvent,
    AutoRetryStartEvent,
    MessageUpdateEvent,
    PiAgent,
    PiError,
    ToolExecutionStartEvent,
)


async def main(prompt: str) -> None:
    try:
        async with PiAgent(model="anthropic/claude-sonnet-4-6", cwd=".") as agent:
            got_text = False
            async for ev in agent.prompt_stream(prompt):
                if isinstance(ev, MessageUpdateEvent) and ev.assistantMessageEvent:
                    ame = ev.assistantMessageEvent
                    if ame.type in ("text_delta", "thinking_delta") and ame.delta:
                        print(ame.delta, end="", flush=True)
                        got_text = True
                elif isinstance(ev, ToolExecutionStartEvent):
                    print(f"\n[tool: {ev.toolName}]", flush=True)
                elif isinstance(ev, AutoRetryStartEvent):
                    print(f"\n[retry: {ev.errorMessage}]", file=sys.stderr, flush=True)
                elif isinstance(ev, AgentEndEvent):
                    # A run can finish with an error (bad model id, auth rejected)
                    # without raising — the message carries it on errorMessage.
                    for m in ev.messages:
                        err = m.get("errorMessage") if isinstance(m, dict) else None
                        if err:
                            print(f"\n[agent error: {err}]", file=sys.stderr)
            print()
            if not got_text:
                print(
                    "[no text produced — likely a bad model id or an auth error; "
                    "check the messages above and your provider key]",
                    file=sys.stderr,
                )
    except PiError as exc:
        # Preflight failures, subprocess crashes, and timeouts arrive here.
        # Timeouts/crashes include the pi subprocess stderr in the message.
        print(f"[PiError] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "Say hello in one short sentence."))
