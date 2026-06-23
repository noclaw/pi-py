"""Stream a raw model response via the low-level PiModelClient (no agent loop).

This uses pi-ai directly through the bundled Node shim — no tools, no sessions, just the
model. It's the building block a native-Python agent loop sits on top of.

Prereqs:
  * `pi` installed (`npm i -g @earendil-works/pi-coding-agent`) and Node on PATH.
  * Credentials: a provider env var (e.g. ANTHROPIC_API_KEY) OR an existing Pi OAuth
    login (`pi`, then `/login`) stored in ~/.pi/agent/auth.json.

Usage:
  python examples/model_stream.py "Say hello in one short sentence."
"""

from __future__ import annotations

import asyncio
import sys

from pi_py_sdk import PiError, PiModelClient


async def main(prompt: str) -> None:
    messages = [{"role": "user", "content": prompt, "timestamp": 0}]
    try:
        async with PiModelClient() as client:
            async for ev in client.stream(
                provider="anthropic",
                model="claude-sonnet-4-6",
                messages=messages,
                reasoning="low",
            ):
                if ev.type == "thinking_delta" and ev.delta:
                    print(ev.delta, end="", flush=True)
                elif ev.type == "text_delta" and ev.delta:
                    print(ev.delta, end="", flush=True)
                elif ev.type == "error":
                    msg = ev.error.errorMessage if ev.error else "unknown error"
                    print(f"\n[model error: {msg}]", file=sys.stderr)
            print()
    except PiError as exc:
        print(f"[PiError] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "Say hello in one short sentence."))
