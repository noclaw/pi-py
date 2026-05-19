"""pi-py CLI — run agents, manage sessions, list models."""
from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import os
import sys
from typing import Any

import click


# ── Helpers ────────────────────────────────────────────────────────────────────

def _json_serialise(obj: Any) -> Any:
    """Custom JSON serialiser for AgentEvent fields."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)


def _emit_json(event: dict) -> None:
    sys.stdout.write(json.dumps(event, default=_json_serialise) + "\n")
    sys.stdout.flush()


def _summarise_tool_call(name: str, args: dict) -> str:
    """One-line description of a tool call for terminal display."""
    if name == "bash":
        cmd = str(args.get("command", ""))
        return ("$ " + cmd)[:72] + ("…" if len(cmd) > 70 else "")
    if name in ("read", "write", "find", "ls"):
        path = args.get("path") or "."
        return f"{name} {path}"
    if name == "edit":
        path = args.get("path", "?")
        edits = args.get("edits") or []
        n = len(edits) if isinstance(edits, list) else "?"
        return f"edit {path} ({n} edit{'s' if n != 1 else ''})"
    if name == "grep":
        pattern = args.get("pattern", "?")
        path = args.get("path") or "."
        return f"grep {pattern!r} in {path}"
    # Generic fallback
    parts = [f"{k}={v!r}" for k, v in list(args.items())[:2]]
    return f"{name}({', '.join(parts)})"


def _format_size(n: int) -> str:
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}K"


def _default_sessions_dir() -> str:
    return os.path.expanduser("~/.pi/sessions")


# ── Text streaming state ───────────────────────────────────────────────────────

class _TextState:
    """Track whether stdout is mid-line so tool annotations land cleanly."""

    def __init__(self) -> None:
        self._mid_line = False

    def write(self, text: str) -> None:
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()
        self._mid_line = not text.endswith("\n")

    def newline(self) -> None:
        if self._mid_line:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._mid_line = False


# ── Streaming event handler factory ───────────────────────────────────────────

def _make_handler(json_output: bool) -> tuple[Any, list]:
    """Return (subscribe_fn, results_list).

    results_list[0] is set to the final AssistantMessage once agent_end fires.
    """
    results: list = [None]

    if json_output:
        def handler(event: dict, *_) -> None:
            _emit_json(event)
            if event.get("type") == "agent_end":
                msgs = event.get("messages") or []
                for m in reversed(msgs):
                    if hasattr(m, "stop_reason"):
                        results[0] = m
                        break
        return handler, results

    state = _TextState()

    def handler(event: dict, *_) -> None:
        etype = event.get("type")

        if etype == "message_update":
            amsg_ev = event.get("assistant_message_event") or {}
            if amsg_ev.get("type") == "text_delta":
                state.write(amsg_ev["delta"])

        elif etype == "tool_execution_start":
            state.newline()
            name = event.get("tool_name", "?")
            args = event.get("args") or {}
            summary = _summarise_tool_call(name, args)
            click.echo(click.style(f"  ⚙ {summary}", fg="cyan"))

        elif etype == "tool_execution_end":
            if event.get("is_error"):
                result = event.get("result")
                content = getattr(result, "content", []) if result else []
                text = next((b.text for b in content if getattr(b, "text", None)), "error")
                click.echo(click.style(f"    ✗ {text[:100]}", fg="red"))
            elif event.get("tool_name") == "bash":
                result = event.get("result")
                content = getattr(result, "content", []) if result else []
                raw = next((b.text for b in content if getattr(b, "text", None)), "")
                if raw.strip():
                    lines = raw.strip().splitlines()
                    for line in lines[:6]:
                        click.echo(click.style(f"    {line}", fg="bright_black"))
                    if len(lines) > 6:
                        click.echo(click.style(f"    … ({len(lines)-6} more lines)", fg="bright_black"))

        elif etype == "message_end":
            msg = event.get("message")
            if hasattr(msg, "stop_reason"):
                state.newline()
                if msg.error_message:
                    click.echo(click.style(f"\nError: {msg.error_message}", fg="red"), err=True)

        elif etype == "agent_end":
            msgs = event.get("messages") or []
            for m in reversed(msgs):
                if hasattr(m, "stop_reason"):
                    results[0] = m
                    break

    return handler, results


def _print_run_summary(final_msg: Any, model_id: str) -> None:
    if final_msg is None:
        return
    usage = getattr(final_msg, "usage", None)
    if usage is None:
        return
    tokens_in = _format_size(usage.input)
    tokens_out = _format_size(usage.output)
    cost = usage.cost.total if usage.cost else 0
    cost_str = f"${cost:.6f}" if cost else ""
    parts = [model_id, f"{tokens_in} in / {tokens_out} out"]
    if cost_str:
        parts.append(cost_str)
    click.echo(click.style(f"\n[{' · '.join(parts)}]", fg="bright_black"), err=False)


# ── CLI entry point ────────────────────────────────────────────────────────────

@click.group()
@click.option(
    "--settings-dir",
    default=None,
    metavar="PATH",
    envvar="PI_SETTINGS_DIR",
    help="Settings directory (default: ~/.pi/agent).",
)
@click.pass_context
def main(ctx: click.Context, settings_dir: str | None) -> None:
    """pi-py — personal LLM agent CLI.

    Run 'pi-py prompt --help' for prompting options.
    """
    ctx.ensure_object(dict)
    ctx.obj["settings_dir"] = settings_dir


# ── prompt ─────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("prompt_text")
@click.option("--model", "-m", default=None, metavar="PROVIDER:MODEL",
              help="Model as 'provider:model-id', e.g. 'anthropic:claude-sonnet-4-6'.")
@click.option("--session", "-s", default=None, metavar="ID",
              help="Resume an existing session by ID.")
@click.option("--sessions-dir", default=None, metavar="PATH",
              help=f"Sessions root directory (default: ~/.pi/sessions).")
@click.option("--system", default=None, metavar="TEXT",
              help="System prompt (replaces auto-generated prompt).")
@click.option("--cwd", default=None, metavar="PATH",
              help="Working directory for file/shell tools (default: current dir).")
@click.option("--no-tools", is_flag=True, default=False,
              help="Disable built-in file and shell tools.")
@click.option("--json", "json_output", is_flag=True, default=False,
              help="Emit newline-delimited JSON events instead of text.")
@click.pass_context
def prompt(
    ctx: click.Context,
    prompt_text: str,
    model: str | None,
    session: str | None,
    sessions_dir: str | None,
    system: str | None,
    cwd: str | None,
    no_tools: bool,
    json_output: bool,
) -> None:
    """Run PROMPT_TEXT and stream the response."""
    asyncio.run(_run_prompt(
        prompt_text=prompt_text,
        model_str=model,
        session_id=session,
        sessions_dir=sessions_dir,
        system_prompt=system,
        cwd=cwd,
        tools="all" if not no_tools else None,
        json_output=json_output,
        settings_dir=ctx.obj.get("settings_dir"),
    ))


async def _run_prompt(
    prompt_text: str,
    model_str: str | None,
    session_id: str | None,
    sessions_dir: str | None,
    system_prompt: str | None,
    cwd: str | None,
    tools: Any,
    json_output: bool,
    settings_dir: str | None,
) -> None:
    import pi_agent

    try:
        resolved_model = None
        if model_str:
            import pi_ai
            parts = model_str.split(":", 1)
            if len(parts) != 2:
                raise click.UsageError("--model must be in 'provider:model-id' format.")
            # Try built-in catalog first, then custom models
            try:
                resolved_model = pi_ai.get_model(parts[0], parts[1])
            except Exception:
                resolved_model = pi_agent.find_custom_model(parts[0], parts[1], settings_dir)
                if resolved_model is None:
                    raise click.UsageError(
                        f"Model {model_str!r} not found. "
                        "Check 'pi-py models list' for available models."
                    )

        harness = await pi_agent.create_agent(
            model=resolved_model,
            cwd=cwd,
            session_id=session_id,
            sessions_dir=sessions_dir,
            system_prompt=system_prompt,
            tools=tools,
            settings_dir=settings_dir,
        )

    except ValueError as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    handler, results = _make_handler(json_output)
    harness.subscribe(handler)

    try:
        await harness.prompt(prompt_text)
    except Exception as exc:
        click.echo(click.style(f"\nError: {exc}", fg="red"), err=True)
        sys.exit(1)

    if not json_output:
        model_id = harness.get_model().id
        _print_run_summary(results[0], model_id)


# ── sessions ───────────────────────────────────────────────────────────────────

@main.group()
def sessions() -> None:
    """Manage saved sessions."""


@sessions.command("list")
@click.option("--sessions-dir", default=None, metavar="PATH",
              help=f"Sessions root directory (default: ~/.pi/sessions).")
@click.pass_context
def sessions_list(ctx: click.Context, sessions_dir: str | None) -> None:
    """List saved sessions, newest first."""
    asyncio.run(_list_sessions(sessions_dir))


async def _list_sessions(sessions_dir: str | None) -> None:
    import pi_agent

    sessions_root = os.path.expanduser(sessions_dir or _default_sessions_dir())
    if not os.path.exists(sessions_root):
        click.echo("No sessions found (sessions directory does not exist).")
        return

    from pi_agent.harness.env.python import PythonExecutionEnv
    from pi_agent.harness.session.jsonl_repo import JsonlSessionRepo

    env = PythonExecutionEnv(cwd=sessions_root)
    repo = JsonlSessionRepo(fs=env, sessions_root=sessions_root)

    try:
        all_sessions = await repo.list()
    except Exception as exc:
        click.echo(click.style(f"Error reading sessions: {exc}", fg="red"), err=True)
        return

    if not all_sessions:
        click.echo("No sessions found.")
        return

    # Header
    click.echo(click.style(
        f"{'ID':<10}  {'CREATED':<20}  {'CWD'}",
        bold=True,
    ))
    click.echo("─" * 70)

    for meta in all_sessions:
        sid = meta.get("id", "?")[:8]
        created_raw = meta.get("createdAt", "")
        try:
            dt = datetime.datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
            created = dt.astimezone().strftime("%Y-%m-%d %H:%M")
        except Exception:
            created = created_raw[:16]
        cwd_str = meta.get("cwd", "")
        # Shorten home dir
        cwd_str = cwd_str.replace(os.path.expanduser("~"), "~")
        click.echo(f"{sid:<10}  {created:<20}  {cwd_str}")

    click.echo(f"\n{len(all_sessions)} session(s)  •  {sessions_root}")


@sessions.command("show")
@click.argument("session_id")
@click.option("--sessions-dir", default=None, metavar="PATH",
              help="Sessions root directory (default: ~/.pi/sessions).")
@click.pass_context
def sessions_show(ctx: click.Context, session_id: str, sessions_dir: str | None) -> None:
    """Show the transcript for SESSION_ID."""
    asyncio.run(_show_session(session_id, sessions_dir))


async def _show_session(session_id: str, sessions_dir: str | None) -> None:
    sessions_root = os.path.expanduser(sessions_dir or _default_sessions_dir())

    from pi_agent.harness.env.python import PythonExecutionEnv
    from pi_agent.harness.session.jsonl_repo import JsonlSessionRepo

    env = PythonExecutionEnv(cwd=sessions_root)
    repo = JsonlSessionRepo(fs=env, sessions_root=sessions_root)

    all_sessions = await repo.list()
    metadata = next((s for s in all_sessions if s["id"].startswith(session_id)), None)
    if metadata is None:
        available = [s["id"][:8] for s in all_sessions[:5]]
        click.echo(click.style(f"Session {session_id!r} not found.", fg="red"), err=True)
        if available:
            click.echo(f"Available: {', '.join(available)}", err=True)
        sys.exit(1)

    session = await repo.open(metadata)
    meta = await session.get_metadata()
    created_raw = meta.get("createdAt", "")
    try:
        dt = datetime.datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        created = dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        created = created_raw

    click.echo(click.style(f"Session: {meta['id']}", bold=True))
    click.echo(f"Created: {created}")
    click.echo(f"CWD:     {meta.get('cwd', 'unknown')}")
    name = await session.get_session_name()
    if name:
        click.echo(f"Name:    {name}")
    click.echo()

    ctx = await session.build_context()
    messages = ctx.get("messages", [])

    if not messages:
        click.echo("(no messages)")
        return

    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "?")

        if role == "user":
            content = getattr(msg, "content", "") or ""
            if isinstance(content, list):
                text = " ".join(
                    b.text if hasattr(b, "text") else b.get("text", "")
                    for b in content
                    if (getattr(b, "type", None) or b.get("type")) == "text"
                )
            else:
                text = str(content)
            click.echo(click.style("user: ", fg="green", bold=True) + text.strip()[:200])

        elif role == "assistant":
            content = getattr(msg, "content", []) or []
            text_parts = []
            tool_parts = []
            for b in content:
                btype = getattr(b, "type", None)
                if btype == "text" and getattr(b, "text", ""):
                    text_parts.append(b.text)
                elif btype == "toolCall":
                    tool_parts.append(getattr(b, "name", "?"))
            text = " ".join(text_parts).strip()
            if text:
                click.echo(click.style("assistant: ", fg="blue", bold=True) + text[:200])
            if tool_parts:
                click.echo(click.style(
                    f"  [called: {', '.join(tool_parts)}]", fg="cyan"
                ))

        elif role == "toolResult":
            name = getattr(msg, "tool_name", "?")
            is_error = getattr(msg, "is_error", False)
            content = getattr(msg, "content", []) or []
            text = next(
                (b.text for b in content if getattr(b, "type", None) == "text" and b.text),
                ""
            )
            colour = "red" if is_error else "bright_black"
            click.echo(click.style(f"  [{name}]: {text[:120]}", fg=colour))

        elif role in ("compactionSummary", "branchSummary"):
            summary = getattr(msg, "summary", "")[:80]
            click.echo(click.style(f"  [summary: {summary}…]", fg="yellow"))


# ── models ─────────────────────────────────────────────────────────────────────

@main.group()
def models() -> None:
    """List available models."""


@models.command("list")
@click.pass_context
def models_list(ctx: click.Context) -> None:
    """List built-in and custom models."""
    settings_dir = ctx.obj.get("settings_dir")
    _print_models(settings_dir)


def _print_models(settings_dir: str | None) -> None:
    import pi_ai
    import pi_agent

    default_model = pi_agent.get_default_model(settings_dir=settings_dir)
    default_key = (default_model.provider, default_model.id) if default_model else None

    custom_models = {
        (p, m.id): m
        for p, m in pi_agent.load_custom_models(settings_dir)
    }

    def _ctx(n: int) -> str:
        if n <= 0:
            return "?"
        if n >= 1_000_000:
            return f"{n // 1_000_000}M"
        return f"{n // 1000}K"

    def _row(provider: str, model_id: str, name: str, ctx_win: int,
             reasoning: bool, tags: list[str]) -> None:
        tag_str = "  " + " ".join(click.style(f"[{t}]", fg="yellow") for t in tags) if tags else ""
        r = click.style("✓", fg="magenta") if reasoning else " "
        click.echo(
            f"  {provider:<12}  {model_id:<38}  {_ctx(ctx_win):>5}  {r}{tag_str}"
        )

    # Built-in models
    click.echo(click.style("\nBuilt-in models:", bold=True))
    click.echo(click.style(
        f"  {'PROVIDER':<12}  {'MODEL ID':<38}  {'CTX':>5}  REASONING",
        fg="bright_black",
    ))
    click.echo("  " + "─" * 66)

    for provider in pi_ai.get_providers():
        for model in pi_ai.get_models(provider):
            key = (provider, model.id)
            tags = []
            if key == default_key:
                tags.append("default")
            if key in custom_models:
                tags.append("overridden")
            _row(provider, model.id, model.name, model.context_window, model.reasoning, tags)

    # Custom models not already in built-in catalog
    built_in_keys = {
        (p, m.id)
        for p in pi_ai.get_providers()
        for m in pi_ai.get_models(p)
    }
    extra_pairs = [
        (pname, model)
        for pname, model in pi_agent.load_custom_models(settings_dir)
        if (pname, model.id) not in built_in_keys
    ]

    if extra_pairs:
        click.echo(click.style("\nCustom models (models.json):", bold=True))
        click.echo(click.style(
            f"  {'PROVIDER':<12}  {'MODEL ID':<38}  {'CTX':>5}  REASONING",
            fg="bright_black",
        ))
        click.echo("  " + "─" * 66)
        for pname, model in extra_pairs:
            key = (pname, model.id)
            tags = ["default"] if key == default_key else []
            _row(pname, model.id, model.name, model.context_window, model.reasoning, tags)

    click.echo()
    if default_model:
        click.echo(f"Default: {default_model.provider}:{default_model.id}")
    else:
        click.echo(click.style(
            "No default configured. Set defaultProvider + defaultModel in "
            "~/.pi/agent/settings.json or pass --model.",
            fg="yellow",
        ))
