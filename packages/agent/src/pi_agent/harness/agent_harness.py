"""AgentHarness — high-level orchestration built on Agent + Session."""
from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Callable

from pi_ai.types import AssistantMessage, ImageContent, Model, TextContent, UserMessage, Usage, UsageCost

from ..agent_loop import run_agent_loop
from ..types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    QueueMode,
    ThinkingLevel,
)
from .compaction.branch_summarization import collect_entries_for_branch_summary, generate_branch_summary
from .compaction.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    compact,
    estimate_context_tokens,
    prepare_compaction,
    should_compact,
)
from .session.session import build_session_context
from .messages import convert_to_llm
from .prompt_templates import format_prompt_template_invocation
from .session.session import Session
from .skills import format_skill_invocation
from .types import (
    AgentHarnessError,
    AgentHarnessEvent,
    AgentHarnessPhase,
    AgentHarnessResources,
    AgentHarnessStreamOptions,
    AgentHarnessStreamOptionsPatch,
    BranchSummaryError,
    CompactionError,
    CompactionSettings,
    ExecutionEnv,
    NavigateTreeResult,
    PromptTemplate,
    SessionError,
    Skill,
    ok,
    err,
    to_error,
)


_EMPTY_USAGE = Usage(
    input=0, output=0, cache_read=0, cache_write=0, total_tokens=0,
    cost=UsageCost(input=0, output=0, cache_read=0, cache_write=0, total=0),
)


def _create_user_message(text: str, images: list[ImageContent] | None = None) -> UserMessage:
    content: list[Any] = [TextContent(text=text)]
    if images:
        content.extend(images)
    return UserMessage(content=content, timestamp=int(time.time() * 1000))


def _create_failure_message(model: Model, error: Any, aborted: bool) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_EMPTY_USAGE,
        stop_reason="aborted" if aborted else "error",
        error_message=str(error) if error else "Unknown error",
        timestamp=int(time.time() * 1000),
    )


def _clone_stream_options(opts: AgentHarnessStreamOptions | None) -> AgentHarnessStreamOptions:
    if opts is None:
        return AgentHarnessStreamOptions()
    return AgentHarnessStreamOptions(
        transport=opts.transport,
        timeout_ms=opts.timeout_ms,
        max_retries=opts.max_retries,
        max_retry_delay_ms=opts.max_retry_delay_ms,
        headers=dict(opts.headers) if opts.headers else None,
        metadata=dict(opts.metadata) if opts.metadata else None,
        cache_retention=opts.cache_retention,
    )


def _merge_headers(*headers: dict | None) -> dict | None:
    merged: dict = {}
    has_any = False
    for h in headers:
        if h:
            merged.update(h)
            has_any = True
    return merged if has_any else None


def _apply_stream_options_patch(
    base: AgentHarnessStreamOptions,
    patch: AgentHarnessStreamOptionsPatch | None,
) -> AgentHarnessStreamOptions:
    result = _clone_stream_options(base)
    if patch is None:
        return result
    if patch.transport is not None:
        result.transport = patch.transport
    if patch.timeout_ms is not None:
        result.timeout_ms = patch.timeout_ms
    if patch.max_retries is not None:
        result.max_retries = patch.max_retries
    if patch.max_retry_delay_ms is not None:
        result.max_retry_delay_ms = patch.max_retry_delay_ms
    if patch.cache_retention is not None:
        result.cache_retention = patch.cache_retention
    if hasattr(patch, "headers"):
        if patch.headers is None:
            result.headers = None
        elif patch.headers is not None:
            headers = dict(result.headers or {})
            for k, v in patch.headers.items():
                if v is None:
                    headers.pop(k, None)
                else:
                    headers[k] = v
            result.headers = headers or None
    if hasattr(patch, "metadata"):
        if patch.metadata is None:
            result.metadata = None
        elif patch.metadata is not None:
            metadata = dict(result.metadata or {})
            for k, v in patch.metadata.items():
                if v is None:
                    metadata.pop(k, None)
                else:
                    metadata[k] = v
            result.metadata = metadata or None
    return result


def _normalize_harness_error(error: Any, fallback_code: str) -> AgentHarnessError:
    if isinstance(error, AgentHarnessError):
        return error
    cause = to_error(error)
    if isinstance(cause, SessionError):
        return AgentHarnessError("session", cause.args[0], cause)
    if isinstance(cause, CompactionError):
        return AgentHarnessError("compaction", cause.args[0], cause)
    if isinstance(cause, BranchSummaryError):
        return AgentHarnessError("branch_summary", cause.args[0], cause)
    return AgentHarnessError(fallback_code, str(cause), cause)


def _normalize_hook_error(error: Any) -> AgentHarnessError:
    return _normalize_harness_error(error, "hook")


class AgentHarness:
    """High-level orchestration: Session + Agent + Compaction + Hooks."""

    def __init__(
        self,
        *,
        env: ExecutionEnv,
        session: Session,
        model: Model,
        tools: list[AgentTool] | None = None,
        resources: AgentHarnessResources | None = None,
        system_prompt: str | Callable[..., str] | None = None,
        get_api_key_and_headers: Callable[[Model], Any] | None = None,
        stream_options: AgentHarnessStreamOptions | None = None,
        thinking_level: ThinkingLevel = "off",
        active_tool_names: list[str] | None = None,
        steering_mode: QueueMode = "one-at-a-time",
        follow_up_mode: QueueMode = "one-at-a-time",
        auto_compact: bool = False,
        compact_reserve_tokens: int = DEFAULT_COMPACTION_SETTINGS.reserve_tokens,
        compact_keep_recent_tokens: int = DEFAULT_COMPACTION_SETTINGS.keep_recent_tokens,
    ) -> None:
        self.env = env
        self._session = session
        self._model = model
        self._thinking_level: ThinkingLevel = thinking_level
        self._system_prompt = system_prompt
        self._stream_options = _clone_stream_options(stream_options)
        self._get_api_key_and_headers = get_api_key_and_headers
        self._resources: AgentHarnessResources = resources or AgentHarnessResources()
        self._tools: dict[str, AgentTool] = {t.name: t for t in (tools or [])}
        self._active_tool_names: list[str] = (
            active_tool_names if active_tool_names is not None
            else [t.name for t in (tools or [])]
        )
        self._steer_queue: list[UserMessage] = []
        self._follow_up_queue: list[UserMessage] = []
        self._next_turn_queue: list[AgentMessage] = []
        self._pending_session_writes: list[dict] = []
        self._steering_queue_mode: QueueMode = steering_mode
        self._follow_up_queue_mode: QueueMode = follow_up_mode
        self._phase: AgentHarnessPhase = "idle"
        self._run_abort: asyncio.Event | None = None
        self._run_promise: asyncio.Task | None = None
        # Event handlers: type -> list of handlers
        self._hook_handlers: dict[str, list[Callable]] = {}
        self._subscribers: list[Callable] = []
        # Auto-compaction
        self._auto_compact = auto_compact
        self._compact_reserve_tokens = compact_reserve_tokens
        self._compact_keep_recent_tokens = compact_keep_recent_tokens

    # ── Public session/model accessors ─────────────────────────────────────────

    def get_model(self) -> Model:
        return self._model

    def get_thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    def get_resources(self) -> AgentHarnessResources:
        return AgentHarnessResources(
            skills=list(self._resources.skills) if self._resources.skills else None,
            prompt_templates=list(self._resources.prompt_templates) if self._resources.prompt_templates else None,
        )

    def get_stream_options(self) -> AgentHarnessStreamOptions:
        return _clone_stream_options(self._stream_options)

    async def set_model(self, model: Model) -> None:
        try:
            prev = self._model
            if self._phase == "idle":
                await self._session.append_model_change(model.provider, model.id)
            else:
                self._pending_session_writes.append({"type": "model_change", "provider": model.provider, "modelId": model.id})
            self._model = model
            await self._emit_own({"type": "model_select", "model": model, "previousModel": prev, "source": "set"})
        except Exception as e:
            raise _normalize_harness_error(e, "session")

    async def set_thinking_level(self, level: ThinkingLevel) -> None:
        try:
            prev = self._thinking_level
            if self._phase == "idle":
                await self._session.append_thinking_level_change(level)
            else:
                self._pending_session_writes.append({"type": "thinking_level_change", "thinkingLevel": level})
            self._thinking_level = level
            await self._emit_own({"type": "thinking_level_select", "level": level, "previousLevel": prev})
        except Exception as e:
            raise _normalize_harness_error(e, "session")

    async def set_resources(self, resources: AgentHarnessResources) -> None:
        prev = self.get_resources()
        self._resources = AgentHarnessResources(
            skills=list(resources.skills) if resources.skills else None,
            prompt_templates=list(resources.prompt_templates) if resources.prompt_templates else None,
        )
        await self._emit_own({"type": "resources_update", "resources": self.get_resources(), "previousResources": prev})

    async def set_stream_options(self, opts: AgentHarnessStreamOptions) -> None:
        self._stream_options = _clone_stream_options(opts)

    async def set_tools(self, tools: list[AgentTool], active_tool_names: list[str] | None = None) -> None:
        next_tools = {t.name: t for t in tools}
        next_active = list(active_tool_names) if active_tool_names is not None else self._active_tool_names
        self._validate_tool_names(next_active, next_tools)
        self._tools = next_tools
        self._active_tool_names = next_active

    async def set_active_tools(self, tool_names: list[str]) -> None:
        self._validate_tool_names(tool_names)
        self._active_tool_names = list(tool_names)

    def _validate_tool_names(self, names: list[str], tools: dict | None = None) -> None:
        t = tools if tools is not None else self._tools
        missing = [n for n in names if n not in t]
        if missing:
            raise AgentHarnessError("invalid_argument", f"Unknown tool(s): {', '.join(missing)}")

    async def get_steering_mode(self) -> QueueMode:
        return self._steering_queue_mode

    async def set_steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue_mode = mode

    async def get_follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue_mode

    async def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue_mode = mode

    # ── Event subscription ─────────────────────────────────────────────────────

    def subscribe(self, listener: Callable) -> Callable:
        self._subscribers.append(listener)
        return lambda: self._subscribers.remove(listener)

    def on(self, type_: str, handler: Callable) -> Callable:
        self._hook_handlers.setdefault(type_, []).append(handler)
        return lambda: self._hook_handlers[type_].remove(handler)

    # ── Queue methods ──────────────────────────────────────────────────────────

    async def steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        if self._phase == "idle":
            raise AgentHarnessError("invalid_state", "Cannot steer while idle")
        self._steer_queue.append(_create_user_message(text, images))
        await self._emit_queue_update()

    async def follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        if self._phase == "idle":
            raise AgentHarnessError("invalid_state", "Cannot follow up while idle")
        self._follow_up_queue.append(_create_user_message(text, images))
        await self._emit_queue_update()

    async def next_turn(self, text: str, images: list[ImageContent] | None = None) -> None:
        self._next_turn_queue.append(_create_user_message(text, images))
        await self._emit_queue_update()

    async def append_message(self, message: AgentMessage) -> None:
        try:
            if self._phase == "idle":
                await self._session.append_message(message)
            else:
                msg_dict = message.model_dump() if hasattr(message, "model_dump") else message
                self._pending_session_writes.append({"type": "message", "message": msg_dict})
        except Exception as e:
            raise _normalize_harness_error(e, "session")

    # ── Main entry points ──────────────────────────────────────────────────────

    async def prompt(self, text: str, images: list[ImageContent] | None = None) -> AssistantMessage:
        if self._phase != "idle":
            raise AgentHarnessError("busy", "AgentHarness is busy")
        self._phase = "turn"
        done = self._start_run()
        try:
            turn_state = await self._create_turn_state()
            return await self._execute_turn(turn_state, text, images)
        except Exception as e:
            self._phase = "idle"
            raise _normalize_harness_error(e, "unknown")
        finally:
            done()

    async def skill(self, name: str, additional_instructions: str | None = None) -> AssistantMessage:
        if self._phase != "idle":
            raise AgentHarnessError("busy", "AgentHarness is busy")
        self._phase = "turn"
        done = self._start_run()
        try:
            turn_state = await self._create_turn_state()
            skills = turn_state["resources"].skills or []
            skill_obj = next((s for s in skills if s.name == name), None)
            if not skill_obj:
                raise AgentHarnessError("invalid_argument", f"Unknown skill: {name}")
            return await self._execute_turn(turn_state, format_skill_invocation(skill_obj, additional_instructions))
        except Exception as e:
            self._phase = "idle"
            raise _normalize_harness_error(e, "unknown")
        finally:
            done()

    async def prompt_from_template(self, name: str, args: list[str] | None = None) -> AssistantMessage:
        if self._phase != "idle":
            raise AgentHarnessError("busy", "AgentHarness is busy")
        self._phase = "turn"
        done = self._start_run()
        try:
            turn_state = await self._create_turn_state()
            templates = turn_state["resources"].prompt_templates or []
            tmpl = next((t for t in templates if t.name == name), None)
            if not tmpl:
                raise AgentHarnessError("invalid_argument", f"Unknown prompt template: {name}")
            return await self._execute_turn(turn_state, format_prompt_template_invocation(tmpl, args))
        except Exception as e:
            self._phase = "idle"
            raise _normalize_harness_error(e, "unknown")
        finally:
            done()

    async def compact(self, custom_instructions: str | None = None) -> dict:
        """Compact the session context. Requires the harness to be idle."""
        if self._phase != "idle":
            raise AgentHarnessError("busy", "compact() requires idle harness")
        self._phase = "compaction"
        try:
            return await self._do_compact(custom_instructions)
        except Exception as e:
            raise _normalize_harness_error(e, "compaction")
        finally:
            self._phase = "idle"

    async def _do_compact(self, custom_instructions: str | None = None) -> dict:
        """Core compaction logic — callable from any phase (no phase check)."""
        auth = await self._get_auth()
        api_key: str | None = auth.get("apiKey") or None  # None lets pi_ai read from env
        headers: dict | None = auth.get("headers")

        branch_entries = await self._session.get_branch()
        settings = CompactionSettings(
            enabled=True,
            reserve_tokens=self._compact_reserve_tokens,
            keep_recent_tokens=self._compact_keep_recent_tokens,
        )
        prep_result = prepare_compaction(branch_entries, settings)
        if not prep_result.ok:
            raise prep_result.error
        preparation = prep_result.value
        if not preparation:
            raise AgentHarnessError("compaction", "Nothing to compact")

        hook_result = await self._emit_hook({
            "type": "session_before_compact",
            "preparation": preparation,
            "branchEntries": branch_entries,
            "customInstructions": custom_instructions,
            "signal": asyncio.Event(),
        })
        if hook_result and hook_result.get("cancel"):
            raise AgentHarnessError("compaction", "Compaction cancelled")

        provided = hook_result.get("compaction") if hook_result else None
        if provided:
            compact_result_val = provided
        else:
            compact_r = await compact(
                preparation, self._model,
                api_key, headers,
                custom_instructions,
                signal=None,
                thinking_level=self._thinking_level,
            )
            if not compact_r.ok:
                raise compact_r.error
            compact_result_val = compact_r.value

        entry_id = await self._session.append_compaction(
            compact_result_val["summary"],
            compact_result_val["firstKeptEntryId"],
            compact_result_val["tokensBefore"],
            compact_result_val.get("details"),
            provided is not None,
        )
        entry = await self._session.get_entry(entry_id)
        if entry and entry.get("type") == "compaction":
            await self._emit_own({
                "type": "session_compact",
                "compactionEntry": entry,
                "fromHook": provided is not None,
            })
        return compact_result_val

    async def _maybe_auto_compact(self) -> bool:
        """Check token usage and compact if the context window is approaching its limit.

        Returns True if compaction ran, False if skipped or failed.
        Errors are swallowed — a failed auto-compact must never abort the agent loop.
        """
        if not self._auto_compact:
            return False
        context_window = self._model.context_window
        if context_window <= 0:
            return False  # Unknown context window — skip
        try:
            branch_entries = await self._session.get_branch()
            ctx = build_session_context(branch_entries)
            estimate = estimate_context_tokens(ctx["messages"])
            settings = CompactionSettings(
                enabled=True,
                reserve_tokens=self._compact_reserve_tokens,
                keep_recent_tokens=self._compact_keep_recent_tokens,
            )
            if not should_compact(estimate.tokens, context_window, settings):
                return False
            await self._do_compact()
            return True
        except Exception:
            return False

    async def navigate_tree(
        self,
        target_id: str,
        summarize: bool = False,
        custom_instructions: str | None = None,
        replace_instructions: bool = False,
        label: str | None = None,
    ) -> NavigateTreeResult:
        if self._phase != "idle":
            raise AgentHarnessError("busy", "navigateTree() requires idle harness")
        self._phase = "branch_summary"
        try:
            old_leaf_id = await self._session.get_leaf_id()
            if old_leaf_id == target_id:
                return NavigateTreeResult(cancelled=False)
            target_entry = await self._session.get_entry(target_id)
            if not target_entry:
                raise AgentHarnessError("invalid_argument", f"Entry {target_id} not found")

            collection = await collect_entries_for_branch_summary(self._session, old_leaf_id, target_id)
            entries = collection["entries"]
            common_ancestor_id = collection["commonAncestorId"]

            preparation = {
                "targetId": target_id, "oldLeafId": old_leaf_id,
                "commonAncestorId": common_ancestor_id,
                "entriesToSummarize": entries,
                "userWantsSummary": summarize,
                "customInstructions": custom_instructions,
                "replaceInstructions": replace_instructions,
                "label": label,
            }
            hook_result = await self._emit_hook({"type": "session_before_tree", "preparation": preparation, "signal": asyncio.Event()})
            if hook_result and hook_result.get("cancel"):
                return NavigateTreeResult(cancelled=True)

            summary_text: str | None = hook_result.get("summary", {}).get("summary") if hook_result else None
            summary_details: Any = hook_result.get("summary", {}).get("details") if hook_result else None

            if not summary_text and summarize and entries:
                auth = await self._get_auth()
                branch_r = await generate_branch_summary(
                    entries, self._model,
                    auth.get("apiKey", ""), auth.get("headers"),
                    custom_instructions=custom_instructions,
                    replace_instructions=replace_instructions,
                )
                if not branch_r.ok:
                    if branch_r.error.code == "aborted":
                        return NavigateTreeResult(cancelled=True)
                    raise AgentHarnessError("branch_summary", branch_r.error.args[0], branch_r.error)
                summary_text = branch_r.value["summary"]
                summary_details = {"readFiles": branch_r.value["readFiles"], "modifiedFiles": branch_r.value["modifiedFiles"]}

            editor_text: str | None = None
            t = target_entry.get("type")
            if t == "message" and (target_entry.get("message") or {}).get("role") == "user":
                new_leaf_id: str | None = target_entry.get("parentId")
                content = (target_entry.get("message") or {}).get("content", "")
                if isinstance(content, str):
                    editor_text = content
                elif isinstance(content, list):
                    editor_text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            elif t == "custom_message":
                new_leaf_id = target_entry.get("parentId")
                content = target_entry.get("content", "")
                if isinstance(content, str):
                    editor_text = content
                elif isinstance(content, list):
                    editor_text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            else:
                new_leaf_id = target_id

            summary_arg = (
                {"summary": summary_text, "details": summary_details, "fromHook": hook_result is not None and hook_result.get("summary") is not None}
                if summary_text else None
            )
            summary_entry_id = await self._session.move_to(new_leaf_id, summary_arg)
            summary_entry: dict | None = None
            if summary_entry_id:
                e = await self._session.get_entry(summary_entry_id)
                if e and e.get("type") == "branch_summary":
                    summary_entry = e

            await self._emit_own({
                "type": "session_tree",
                "newLeafId": await self._session.get_leaf_id(),
                "oldLeafId": old_leaf_id,
                "summaryEntry": summary_entry,
                "fromHook": hook_result is not None and hook_result.get("summary") is not None,
            })
            return NavigateTreeResult(cancelled=False, editor_text=editor_text, summary_entry=summary_entry)
        except Exception as e:
            raise _normalize_harness_error(e, "branch_summary")
        finally:
            self._phase = "idle"

    async def abort(self) -> dict:
        cleared_steer = list(self._steer_queue)
        cleared_follow_up = list(self._follow_up_queue)
        self._steer_queue = []
        self._follow_up_queue = []
        if self._run_abort:
            self._run_abort.set()
        errors: list[Exception] = []
        for op in [self._emit_queue_update, self.wait_for_idle,
                   lambda: self._emit_own({"type": "abort", "clearedSteer": cleared_steer, "clearedFollowUp": cleared_follow_up})]:
            try:
                await op()
            except Exception as e:
                errors.append(to_error(e))
        if errors:
            raise _normalize_harness_error(errors[0], "hook")
        return {"clearedSteer": cleared_steer, "clearedFollowUp": cleared_follow_up}

    async def wait_for_idle(self) -> None:
        future = self._run_promise
        if future is not None and not future.done():
            await future

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _start_run(self) -> Callable:
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._run_promise = future  # type: ignore[assignment]
        self._run_abort = asyncio.Event()

        def done() -> None:
            self._run_abort = None
            self._run_promise = None
            if not future.done():
                future.set_result(None)

        return done

    async def _get_auth(self) -> dict:
        if self._get_api_key_and_headers:
            result = self._get_api_key_and_headers(self._model)
            if asyncio.iscoroutine(result):
                result = await result
            return result or {}
        return {}

    async def _emit_own(self, event: dict, signal: asyncio.Event | None = None) -> None:
        for listener in self._subscribers:
            try:
                result = listener(event, signal)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                raise _normalize_hook_error(e)

    async def _emit_hook(self, event: dict) -> Any:
        handlers = self._hook_handlers.get(event["type"], [])
        last_result = None
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    last_result = result
            except Exception as e:
                raise _normalize_hook_error(e)
        return last_result

    async def _emit_queue_update(self) -> None:
        await self._emit_own({
            "type": "queue_update",
            "steer": list(self._steer_queue),
            "followUp": list(self._follow_up_queue),
            "nextTurn": list(self._next_turn_queue),
        })

    async def _flush_pending_writes(self) -> None:
        while self._pending_session_writes:
            write = self._pending_session_writes[0]
            t = write.get("type")
            if t == "message":
                await self._session.append_message(write["message"])
            elif t == "model_change":
                await self._session.append_model_change(write["provider"], write["modelId"])
            elif t == "thinking_level_change":
                await self._session.append_thinking_level_change(write["thinkingLevel"])
            elif t == "custom":
                await self._session.append_custom_entry(write["customType"], write.get("data"))
            elif t == "custom_message":
                await self._session.append_custom_message_entry(
                    write["customType"], write["content"], write["display"], write.get("details")
                )
            elif t == "label":
                await self._session.append_label(write["targetId"], write.get("label"))
            elif t == "session_info":
                await self._session.append_session_name(write.get("name") or "")
            elif t == "leaf":
                await self._session.get_storage().set_leaf_id(write.get("targetId"))
            self._pending_session_writes.pop(0)

    async def _create_turn_state(self) -> dict:
        context = await self._session.build_context()
        resources = self.get_resources()
        metadata = await self._session.get_metadata()
        tools = list(self._tools.values())
        active_tools = [self._tools[n] for n in self._active_tool_names if n in self._tools]

        if callable(self._system_prompt):
            system_prompt = self._system_prompt(
                env=self.env, session=self._session, model=self._model,
                thinking_level=self._thinking_level, active_tools=active_tools, resources=resources,
            )
            if asyncio.iscoroutine(system_prompt):
                system_prompt = await system_prompt
        elif isinstance(self._system_prompt, str):
            system_prompt = self._system_prompt
        else:
            system_prompt = "You are a helpful assistant."

        return {
            "messages": context["messages"],
            "resources": resources,
            "streamOptions": _clone_stream_options(self._stream_options),
            "sessionId": metadata["id"],
            "systemPrompt": system_prompt,
            "model": self._model,
            "thinkingLevel": self._thinking_level,
            "tools": tools,
            "activeTools": active_tools,
        }

    def _create_context(self, turn_state: dict, system_prompt: str | None = None) -> AgentContext:
        return AgentContext(
            system_prompt=system_prompt or turn_state["systemPrompt"],
            messages=list(turn_state["messages"]),
            tools=list(turn_state["activeTools"]),
        )

    def _create_loop_config(
        self,
        get_turn_state: Callable[[], dict],
        set_turn_state: Callable[[dict], None],
    ) -> AgentLoopConfig:
        ts = get_turn_state()

        async def get_steering():
            return await self._drain_queue(self._steer_queue, self._steering_queue_mode)

        async def get_follow_up():
            return await self._drain_queue(self._follow_up_queue, self._follow_up_queue_mode)

        async def transform_context(messages, signal=None):
            result = await self._emit_hook({"type": "context", "messages": list(messages)})
            return result["messages"] if result else messages

        async def before_tool_call(ctx, signal=None):
            result = await self._emit_hook({
                "type": "tool_call",
                "toolCallId": ctx.tool_call.id,
                "toolName": ctx.tool_call.name,
                "input": ctx.args,
            })
            if result:
                from ..types import BeforeToolCallResult
                return BeforeToolCallResult(block=result.get("block", False), reason=result.get("reason"))
            return None

        async def after_tool_call(ctx, signal=None):
            patch = await self._emit_hook({
                "type": "tool_result",
                "toolCallId": ctx.tool_call.id,
                "toolName": ctx.tool_call.name,
                "input": ctx.args,
                "content": ctx.result.content,
                "details": ctx.result.details,
                "isError": ctx.is_error,
            })
            if patch:
                from ..types import AfterToolCallResult
                return AfterToolCallResult(
                    content=patch.get("content"),
                    details=patch.get("details"),
                    is_error=patch.get("isError"),
                    terminate=patch.get("terminate"),
                )
            return None

        async def prepare_next_turn(ctx):
            await self._flush_pending_writes()
            await self._maybe_auto_compact()
            next_ts = await self._create_turn_state()
            set_turn_state(next_ts)
            from ..types import AgentLoopTurnUpdate
            return AgentLoopTurnUpdate(
                context=self._create_context(next_ts),
                model=next_ts["model"],
                thinking_level=next_ts["thinkingLevel"],
            )

        # Expose auth from get_api_key_and_headers to every streaming call,
        # not just compaction.  _get_auth() already takes self._model so we
        # ignore the provider argument here — the harness is single-model per run.
        get_api_key = None
        if self._get_api_key_and_headers:
            async def get_api_key(_provider: str) -> str | None:
                auth = await self._get_auth()
                return auth.get("apiKey") or None

        return AgentLoopConfig(
            model=ts["model"],
            reasoning=None if ts["thinkingLevel"] == "off" else ts["thinkingLevel"],
            convert_to_llm=convert_to_llm,
            transform_context=transform_context,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            prepare_next_turn=prepare_next_turn,
            get_steering_messages=get_steering,
            get_follow_up_messages=get_follow_up,
            get_api_key=get_api_key,
        )

    async def _drain_queue(self, queue: list, mode: QueueMode) -> list:
        if mode == "all":
            messages = queue[:]
            queue.clear()
        else:
            messages = queue[:1]
            del queue[:1]
        if not messages:
            return messages
        try:
            await self._emit_queue_update()
            return messages
        except Exception as e:
            queue[:0] = messages
            raise _normalize_hook_error(e)

    async def _handle_agent_event(self, event: AgentEvent, signal: asyncio.Event | None) -> None:
        etype = event.get("type")
        if etype == "message_end":
            await self._session.append_message(event["message"])
            await self._emit_any(event, signal)
            return
        if etype == "turn_end":
            event_error: Exception | None = None
            try:
                await self._emit_any(event, signal)
            except Exception as e:
                event_error = e
            had_pending = bool(self._pending_session_writes)
            await self._flush_pending_writes()
            if event_error:
                raise event_error
            await self._emit_own({"type": "save_point", "hadPendingMutations": had_pending})
            return
        if etype == "agent_end":
            await self._flush_pending_writes()
            self._phase = "idle"
            await self._emit_any(event, signal)
            await self._emit_own({"type": "settled", "nextTurnCount": len(self._next_turn_queue)}, signal)
            return
        await self._emit_any(event, signal)

    async def _emit_any(self, event: dict, signal: asyncio.Event | None = None) -> None:
        await self._emit_own(event, signal)

    async def _emit_run_failure(
        self, model: Model, error: Any, aborted: bool, signal: asyncio.Event | None
    ) -> list[AgentMessage]:
        msg = _create_failure_message(model, error, aborted)
        for evt in [
            {"type": "message_start", "message": msg},
            {"type": "message_end", "message": msg},
            {"type": "turn_end", "message": msg, "tool_results": []},
            {"type": "agent_end", "messages": [msg]},
        ]:
            await self._handle_agent_event(evt, signal)
        return [msg]

    async def _execute_turn(
        self,
        turn_state: dict,
        text: str,
        images: list[ImageContent] | None = None,
    ) -> AssistantMessage:
        active_ts_cell = [turn_state]
        messages: list[AgentMessage] = [_create_user_message(text, images)]

        if self._next_turn_queue:
            queued = self._next_turn_queue[:]
            self._next_turn_queue.clear()
            try:
                await self._emit_queue_update()
            except Exception as e:
                self._next_turn_queue[:0] = queued
                raise _normalize_hook_error(e)
            messages = [*queued, messages[0]]

        before_result = await self._emit_hook({
            "type": "before_agent_start",
            "prompt": text,
            "images": images,
            "systemPrompt": turn_state["systemPrompt"],
            "resources": turn_state["resources"],
        })
        if before_result and before_result.get("messages"):
            messages = [*messages, *before_result["messages"]]

        abort_signal = asyncio.Event()
        if self._run_abort:
            # Re-use the run's abort event
            abort_signal = self._run_abort

        get_ts = lambda: active_ts_cell[0]
        set_ts = lambda ts: active_ts_cell.__setitem__(0, ts)

        loop_config = self._create_loop_config(get_ts, set_ts)

        override_system_prompt = before_result.get("systemPrompt") if before_result else None
        context = self._create_context(turn_state, override_system_prompt)

        async def run_loop():
            try:
                return await run_agent_loop(
                    messages,
                    context,
                    loop_config,
                    lambda event: self._handle_agent_event(event, abort_signal),
                    abort_signal,
                )
            except Exception as e:
                try:
                    return await self._emit_run_failure(
                        active_ts_cell[0]["model"], e, abort_signal.is_set(), abort_signal
                    )
                except Exception as fail_err:
                    raise AgentHarnessError("unknown", f"Agent run failed and failure reporting failed: {fail_err}", to_error(fail_err))

        try:
            new_messages = await run_loop()
            for msg in reversed(new_messages):
                if isinstance(msg, AssistantMessage):
                    return msg
            raise AgentHarnessError("invalid_state", "AgentHarness prompt completed without an assistant message")
        finally:
            try:
                await self._flush_pending_writes()
            finally:
                self._run_abort = None
