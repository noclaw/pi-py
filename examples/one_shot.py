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

from pi_py_sdk import MessageUpdateEvent, PiAgent, ToolExecutionStartEvent


async def main(prompt: str) -> None:
    async with PiAgent(model="anthropic/claude-sonnet-4-20250514", cwd=".") as agent:
        async for ev in agent.prompt_stream(prompt):
            if isinstance(ev, MessageUpdateEvent) and ev.assistantMessageEvent:
                ame = ev.assistantMessageEvent
                if ame.type in ("text_delta", "thinking_delta") and ame.delta:
                    print(ame.delta, end="", flush=True)
            elif isinstance(ev, ToolExecutionStartEvent):
                print(f"\n[tool: {ev.toolName}]", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "Say hello in one short sentence."))
