"""Drive the agent with an interactive approval handler.

Pi extensions request user decisions (allow this tool? pick an option? enter a value?)
via the extension-UI sub-protocol. ``on_ui_request`` installs the handler that answers
them; without one, the SDK safely denies confirmations and cancels other dialogs.

Usage:
  python examples/with_approvals.py "Refactor foo.py and run the tests"
"""

from __future__ import annotations

import asyncio
import sys

from pi_py_sdk import ExtensionUIRequest, MessageUpdateEvent, PiAgent


def approve(request: ExtensionUIRequest):
    """Console approval handler. Return value depends on the dialog method."""
    if request.method == "confirm":
        answer = input(f"\n[approve] {request.title or 'Confirm'} [y/N] ").strip().lower()
        return answer in ("y", "yes")
    if request.method == "select":
        opts = request.options or []
        for i, opt in enumerate(opts):
            print(f"  {i}) {opt}")
        choice = input(f"[select] {request.title or 'Pick one'} (number) ").strip()
        return opts[int(choice)] if choice.isdigit() and int(choice) < len(opts) else None
    if request.method in ("input", "editor"):
        return input(f"[{request.method}] {request.title or 'Enter value'}: ") or None
    return None


async def main(prompt: str) -> None:
    async with PiAgent(model="anthropic/claude-sonnet-4-20250514", cwd=".") as agent:
        agent.on_ui_request(approve)
        async for ev in agent.prompt_stream(prompt):
            if isinstance(ev, MessageUpdateEvent) and ev.assistantMessageEvent:
                ame = ev.assistantMessageEvent
                if ame.type == "text_delta" and ame.delta:
                    print(ame.delta, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "What would you like me to do?"))
