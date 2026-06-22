"""Use the agent from synchronous (non-async) code via PiAgentSync.

Usage:
  python examples/sync_usage.py "List the Python files here"
"""

from __future__ import annotations

import sys

from pi_py_sdk import MessageUpdateEvent, PiAgentSync, message_text


def main(prompt: str) -> None:
    with PiAgentSync(model="anthropic/claude-sonnet-4-6", cwd=".") as agent:
        for event in agent.prompt_stream(prompt):
            if isinstance(event, MessageUpdateEvent) and event.assistantMessageEvent:
                ame = event.assistantMessageEvent
                if ame.type == "text_delta" and ame.delta:
                    print(ame.delta, end="", flush=True)
        print()

        # Typed message history is available too.
        for msg in agent.get_messages():
            text = message_text(msg).strip()
            if text:
                print(f"  [{getattr(msg, 'role', '?')}] {text[:60]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Say hello in one short sentence.")
