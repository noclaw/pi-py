"""create_agent() — one-call factory for the common coding/automation agent pattern."""
from __future__ import annotations

import os
from typing import Any, Callable, Literal

from pi_ai.types import Model

from .harness.agent_harness import AgentHarness
from .harness.compaction.compaction import DEFAULT_COMPACTION_SETTINGS
from .harness.env.python import PythonExecutionEnv
from .harness.session.jsonl_repo import JsonlSessionRepo
from .harness.session.memory_repo import InMemorySessionRepo
from .harness.types import AgentHarnessResources
from .settings import get_default_model, load_auth, make_auth_provider
from .tools import build_system_prompt, create_tools, load_context_files
from .types import AgentTool, QueueMode, ThinkingLevel


async def create_agent(
    *,
    model: Model | None = None,
    cwd: str | None = None,
    session_dir: str | None = None,
    settings_dir: str | None = None,
    tools: list[AgentTool] | Literal["all"] | None = "all",
    context_files: list[str] | None = None,
    system_prompt: str | Callable[..., str] | None = None,
    auto_compact: bool = True,
    compact_reserve_tokens: int = DEFAULT_COMPACTION_SETTINGS.reserve_tokens,
    compact_keep_recent_tokens: int = DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens,
    get_api_key_and_headers: Callable[[Model], Any] | None = None,
    thinking_level: ThinkingLevel = "off",
    resources: AgentHarnessResources | None = None,
    active_tool_names: list[str] | None = None,
    steering_mode: QueueMode = "one-at-a-time",
    follow_up_mode: QueueMode = "one-at-a-time",
) -> AgentHarness:
    """Create a ready-to-use :class:`AgentHarness` with sensible defaults.

    This is the one-call entry point for the common pattern: working directory,
    built-in tools, context files, and a session — all wired together.

    Parameters
    ----------
    model:
        The LLM model to use.  When ``None``, the default model is resolved
        from ``settings_dir`` (``defaultProvider`` + ``defaultModel`` in
        settings.json + models.json).  Raises ``ValueError`` if ``model`` is
        ``None`` and no default can be resolved.
    cwd:
        Working directory for file tools and shell execution.
        Defaults to the current process working directory.
    settings_dir:
        Directory containing ``settings.json``, ``models.json``, and ``auth.json``.
        Defaults to ``~/.pi/agent``.  Pass ``None`` to use the default.
    session_dir:
        Directory for JSONL session persistence.
        ``None`` (default) uses an in-memory session that does not survive
        across process restarts.
    tools:
        ``"all"`` (default) creates all 7 built-in tools (read, bash, edit,
        write, grep, find, ls). Pass a list of :class:`AgentTool` objects to
        use custom tools, or ``None`` for no tools.
    context_files:
        File names to search for when walking up the directory tree.
        ``None`` (default) loads ``["CLAUDE.md", "AGENTS.md"]``.
        Pass ``[]`` to disable context loading.
    system_prompt:
        System prompt string or callable. When ``None`` (default), a prompt is
        generated automatically from the active tools and any loaded context.
    auto_compact:
        Automatically compact session history when the context window fills up.
        Defaults to ``True`` (unlike the lower-level :class:`AgentHarness` which
        defaults to ``False``).
    compact_reserve_tokens:
        Tokens reserved for the compaction summary and the next response.
    compact_keep_recent_tokens:
        Approximate tokens of recent history to keep after compaction.
    get_api_key_and_headers:
        Callable that returns ``{"apiKey": str, "headers": dict}`` for a given
        model.  When ``None`` and ``settings_dir`` is set (or defaults to
        ``~/.pi/agent``), credentials are loaded automatically from
        ``auth.json``.  Falls back to environment variables when no entry
        exists for the model's provider.
    thinking_level:
        Reasoning depth for models that support extended thinking.
    resources:
        Skill and prompt-template resources made available to the harness.
    active_tool_names:
        Subset of tool names to activate. ``None`` activates all provided tools.
    steering_mode:
        How queued steering messages are drained between turns.
    follow_up_mode:
        How queued follow-up messages are drained after the agent stops.

    Returns
    -------
    AgentHarness
        A configured harness ready to accept ``prompt()`` calls.

    Examples
    --------
    Minimal — resolves model from ``~/.pi/agent/settings.json``::

        import pi_agent

        harness = await pi_agent.create_agent(cwd="/my/project")
        reply = await harness.prompt("Explain the codebase structure.")

    Explicit model, persistent session, custom system prompt::

        import pi_ai, pi_agent

        harness = await pi_agent.create_agent(
            cwd="/my/project",
            model=pi_ai.get_model("openai", "gpt-4o"),
            session_dir="~/.pi/sessions",
            system_prompt="You are a security-focused code reviewer.",
            context_files=["SECURITY.md"],
        )
    """
    resolved_cwd = os.path.abspath(cwd or os.getcwd())
    env = PythonExecutionEnv(cwd=resolved_cwd)

    # ── Model ──────────────────────────────────────────────────────────────────
    if model is None:
        model = get_default_model(cwd=resolved_cwd, settings_dir=settings_dir)
        if model is None:
            raise ValueError(
                "No model provided and no default model found in settings. "
                "Either pass model= explicitly or configure defaultProvider + "
                "defaultModel in ~/.pi/agent/settings.json."
            )

    # ── Auth ───────────────────────────────────────────────────────────────────
    # When the caller doesn't supply get_api_key_and_headers, auto-load from
    # auth.json.  Returns None per-provider when no entry exists, which lets
    # pi_ai fall back to environment variables normally.
    if get_api_key_and_headers is None:
        get_api_key_and_headers = make_auth_provider(settings_dir)

    # ── Tools ──────────────────────────────────────────────────────────────────
    if tools == "all":
        agent_tools: list[AgentTool] = create_tools(env, resolved_cwd)
    elif tools is None:
        agent_tools = []
    else:
        agent_tools = list(tools)

    # ── Context files ──────────────────────────────────────────────────────────
    if context_files is None:
        context = load_context_files(resolved_cwd)          # uses defaults
    elif len(context_files) == 0:
        context = ""
    else:
        context = load_context_files(resolved_cwd, filenames=context_files)

    # ── System prompt ──────────────────────────────────────────────────────────
    if system_prompt is None:
        system_prompt = build_system_prompt(agent_tools or None, context)

    # ── Session ────────────────────────────────────────────────────────────────
    if session_dir is None:
        session = await InMemorySessionRepo().create()
    else:
        expanded = os.path.expanduser(session_dir)
        repo = JsonlSessionRepo(fs=env, sessions_root=expanded)
        session = await repo.create(cwd=resolved_cwd)

    return AgentHarness(
        env=env,
        session=session,
        model=model,
        tools=agent_tools,
        resources=resources,
        system_prompt=system_prompt,
        get_api_key_and_headers=get_api_key_and_headers,
        thinking_level=thinking_level,
        active_tool_names=active_tool_names,
        steering_mode=steering_mode,
        follow_up_mode=follow_up_mode,
        auto_compact=auto_compact,
        compact_reserve_tokens=compact_reserve_tokens,
        compact_keep_recent_tokens=compact_keep_recent_tokens,
    )
