"""journal.py — append a dated entry to a markdown notes file.

The agent writes to NOTES_DIR/YYYY-MM-DD.md, creating the file if it doesn't
exist or appending a new entry with a timestamp. Useful for quick journaling,
meeting notes, or a daily log in an Obsidian vault.

Usage:
    python examples/journal.py ~/notes "Had a productive standup today"
    python examples/journal.py ~/notes          # reads entry from stdin
    python examples/journal.py ~/notes --file   # opens $EDITOR
    python examples/journal.py ~/notes --list   # show recent entries
"""
import argparse
import asyncio
import datetime
import os
import subprocess
import sys
import tempfile

import pi_agent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Append a daily journal entry.")
    p.add_argument("notes_dir", help="Directory where daily notes live.")
    p.add_argument("entry", nargs="?", help="Entry text (omit to read from stdin).")
    p.add_argument("--file", action="store_true",
                   help="Open $EDITOR to write a longer entry.")
    p.add_argument("--list", action="store_true",
                   help="Show the last 5 days of entries.")
    p.add_argument("--model", "-m", metavar="PROVIDER:ID", help="Model override.")
    return p.parse_args()


def _open_editor(initial: str = "") -> str:
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(initial)
        path = f.name
    try:
        subprocess.run([editor, path], check=True)
        return open(path).read().strip()
    finally:
        os.unlink(path)


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
    notes_dir = os.path.abspath(args.notes_dir)
    os.makedirs(notes_dir, exist_ok=True)

    today = datetime.date.today()
    note_file = os.path.join(notes_dir, f"{today}.md")

    # --list: show recent daily note files
    if args.list:
        files = sorted(
            (f for f in os.listdir(notes_dir) if f.endswith(".md")),
            reverse=True,
        )[:5]
        if not files:
            print("No journal entries found.")
        for fname in files:
            path = os.path.join(notes_dir, fname)
            lines = open(path).read().strip().splitlines()
            preview = lines[0][:80] if lines else "(empty)"
            print(f"\033[1m{fname[:-3]}\033[0m  {preview}")
        return

    # Collect the entry text
    if args.file:
        entry_text = _open_editor()
    elif args.entry:
        entry_text = args.entry
    elif not sys.stdin.isatty():
        entry_text = sys.stdin.read().strip()
    else:
        print("Entry text (Ctrl+D when done):")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        entry_text = "\n".join(lines).strip()

    if not entry_text:
        sys.exit("No entry text provided.")

    model = _resolve_model(args.model)

    harness = await pi_agent.create_agent(
        model=model,
        cwd=notes_dir,
        tools="all",
        system_prompt=(
            f"You are a personal journal assistant. "
            f"Today is {today.strftime('%A, %B %d, %Y')}. "
            f"Your job is to append a well-formatted entry to the daily note. "
            f"Use the write or edit tool to update {note_file}. "
            f"If the file exists, append after the last entry. "
            f"If it doesn't exist, create it with a # {today} header. "
            f"Format the entry with a ## HH:MM timestamp heading followed by the content. "
            f"Keep the tone and style consistent with existing entries if the file exists."
        ),
        context_files=[],
    )

    now = datetime.datetime.now().strftime("%H:%M")
    prompt = (
        f"Append this journal entry to {note_file}:\n\n"
        f"Time: {now}\n\n"
        f"{entry_text}"
    )

    def on_event(event: dict, *_) -> None:
        if event["type"] == "tool_execution_start":
            name = event.get("tool_name", "?")
            path_ = (event.get("args") or {}).get("path", "")
            print(f"\033[36m  ⚙ {name} {path_}\033[0m", flush=True)
        elif event["type"] == "message_end":
            msg = event.get("message")
            if hasattr(msg, "stop_reason"):
                text = " ".join(
                    b.text for b in (getattr(msg, "content", []) or [])
                    if getattr(b, "type", None) == "text" and b.text
                )
                if text:
                    print(text)

    harness.subscribe(on_event)

    try:
        reply = await harness.prompt(prompt)
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    if reply.error_message:
        sys.exit(f"Error: {reply.error_message}")

    print(f"\n\033[90mSaved to {note_file}\033[0m")


if __name__ == "__main__":
    asyncio.run(main())
