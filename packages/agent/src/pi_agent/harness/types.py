"""Harness-level types: Result, error classes, ABC interfaces, session entry types, event types."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

if TYPE_CHECKING:
    from .session.session import Session

# ── Result type ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ok:
    ok: Literal[True] = True
    value: Any = None


@dataclass(frozen=True)
class Err:
    ok: Literal[False] = False
    error: Any = None


Result = Union[Ok, Err]


def ok(value: Any = None) -> Ok:
    return Ok(value=value)


def err(error: Any = None) -> Err:
    return Err(error=error)


def get_or_throw(result: Result) -> Any:
    if not result.ok:
        raise result.error
    return result.value


def get_or_undefined(result: Result) -> Any:
    return result.value if result.ok else None


def to_error(error: Any) -> Exception:
    if isinstance(error, Exception):
        return error
    if isinstance(error, str):
        return Exception(error)
    try:
        return Exception(str(error))
    except Exception:
        return Exception(repr(error))


# ── Error classes ──────────────────────────────────────────────────────────────

FileErrorCode = Literal[
    "aborted", "not_found", "permission_denied", "not_directory",
    "is_directory", "invalid", "not_supported", "unknown",
]

ExecutionErrorCode = Literal[
    "aborted", "timeout", "shell_unavailable", "spawn_error", "callback_error", "unknown",
]

CompactionErrorCode = Literal["aborted", "summarization_failed", "invalid_session", "unknown"]
BranchSummaryErrorCode = Literal["aborted", "summarization_failed", "invalid_session"]
SessionErrorCode = Literal[
    "not_found", "invalid_session", "invalid_entry", "invalid_fork_target", "storage", "unknown",
]
AgentHarnessErrorCode = Literal[
    "busy", "invalid_state", "invalid_argument", "session", "hook",
    "auth", "compaction", "branch_summary", "unknown",
]


class FileError(Exception):
    def __init__(self, code: str, message: str, path: str | None = None, cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.code = code
        self.path = path
        self.name = "FileError"


class ExecutionError(Exception):
    def __init__(self, code: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.code = code
        self.name = "ExecutionError"


class CompactionError(Exception):
    def __init__(self, code: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.code = code
        self.name = "CompactionError"


class BranchSummaryError(Exception):
    def __init__(self, code: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.code = code
        self.name = "BranchSummaryError"


class SessionError(Exception):
    def __init__(self, code: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.code = code
        self.name = "SessionError"


class AgentHarnessError(Exception):
    def __init__(self, code: str, message: str, cause: Exception | None = None) -> None:
        super().__init__(message, cause)
        self.code = code
        self.name = "AgentHarnessError"


# ── File info ──────────────────────────────────────────────────────────────────

FileKind = Literal["file", "directory", "symlink"]


@dataclass
class FileInfo:
    name: str
    path: str
    kind: FileKind
    size: int
    mtime_ms: float


# ── Execution env exec options ─────────────────────────────────────────────────

@dataclass
class ExecOptions:
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout: float | None = None          # seconds
    abort_signal: asyncio.Event | None = None
    on_stdout: Any | None = None          # Callable[[str], None]
    on_stderr: Any | None = None          # Callable[[str], None]


# ── ABC interfaces ─────────────────────────────────────────────────────────────

class FileSystem(ABC):
    """Filesystem capability. Methods must never raise — all errors are in Result.error."""

    @property
    @abstractmethod
    def cwd(self) -> str: ...

    @abstractmethod
    async def absolute_path(self, path: str, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def join_path(self, parts: list[str], abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def read_text_file(self, path: str, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def read_text_lines(self, path: str, max_lines: int | None = None, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def read_binary_file(self, path: str, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def write_file(self, path: str, content: str | bytes, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def append_file(self, path: str, content: str | bytes, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def file_info(self, path: str, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def list_dir(self, path: str, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def canonical_path(self, path: str, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def exists(self, path: str, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def create_dir(self, path: str, recursive: bool = True, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def remove(self, path: str, recursive: bool = False, force: bool = False, abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def create_temp_dir(self, prefix: str = "tmp-", abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def create_temp_file(self, prefix: str = "", suffix: str = "", abort_signal: asyncio.Event | None = None) -> Result: ...
    @abstractmethod
    async def cleanup(self) -> None: ...


class Shell(ABC):
    """Shell execution capability."""

    @abstractmethod
    async def exec(self, command: str, options: ExecOptions | None = None) -> Result: ...

    @abstractmethod
    async def cleanup(self) -> None: ...


class ExecutionEnv(FileSystem, Shell, ABC):
    """Combined filesystem + shell execution environment."""


# ── Session tree entry types (TypedDict-like dicts for JSON round-trip) ────────

# These are plain dicts in practice; the types below document the expected keys.

@dataclass
class SessionTreeEntryBase:
    type: str
    id: str
    parent_id: str | None
    timestamp: str  # ISO 8601


# ── Session storage interface ──────────────────────────────────────────────────

class SessionStorage(ABC):
    @abstractmethod
    async def get_metadata(self) -> dict: ...
    @abstractmethod
    async def get_leaf_id(self) -> str | None: ...
    @abstractmethod
    async def set_leaf_id(self, leaf_id: str | None) -> None: ...
    @abstractmethod
    async def create_entry_id(self) -> str: ...
    @abstractmethod
    async def append_entry(self, entry: dict) -> None: ...
    @abstractmethod
    async def get_entry(self, id: str) -> dict | None: ...
    @abstractmethod
    async def find_entries(self, type: str) -> list[dict]: ...
    @abstractmethod
    async def get_label(self, id: str) -> str | None: ...
    @abstractmethod
    async def get_path_to_root(self, leaf_id: str | None) -> list[dict]: ...
    @abstractmethod
    async def get_entries(self) -> list[dict]: ...


# ── Skill & prompt template ────────────────────────────────────────────────────

@dataclass
class Skill:
    name: str
    description: str
    content: str
    file_path: str
    disable_model_invocation: bool = False


@dataclass
class PromptTemplate:
    name: str
    content: str
    description: str = ""


@dataclass
class AgentHarnessResources:
    skills: list[Skill] | None = None
    prompt_templates: list[PromptTemplate] | None = None


# ── Stream options ─────────────────────────────────────────────────────────────

@dataclass
class AgentHarnessStreamOptions:
    transport: str | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    cache_retention: str | None = None


@dataclass
class AgentHarnessStreamOptionsPatch:
    transport: str | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    headers: dict[str, str | None] | None = None
    metadata: dict[str, Any | None] | None = None
    cache_retention: str | None = None


# ── Compaction data structures ─────────────────────────────────────────────────

@dataclass
class FileOperations:
    read: set[str]
    written: set[str]
    edited: set[str]


@dataclass
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000


@dataclass
class CompactionPreparation:
    first_kept_entry_id: str
    messages_to_summarize: list[Any]
    turn_prefix_messages: list[Any]
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None
    file_ops: FileOperations
    settings: CompactionSettings


@dataclass
class TreePreparation:
    target_id: str
    old_leaf_id: str | None
    common_ancestor_id: str | None
    entries_to_summarize: list[dict]
    user_wants_summary: bool
    custom_instructions: str | None = None
    replace_instructions: bool | None = None
    label: str | None = None


@dataclass
class BranchSummaryResult:
    summary: str
    read_files: list[str]
    modified_files: list[str]


@dataclass
class NavigateTreeResult:
    cancelled: bool
    editor_text: str | None = None
    summary_entry: dict | None = None


@dataclass
class AbortResult:
    cleared_steer: list[Any]
    cleared_follow_up: list[Any]


@dataclass
class CompactResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: Any = None


# ── AgentHarness event/result types ───────────────────────────────────────────

AgentHarnessPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]

# Hook event and result types used by AgentHarness.on() / AgentHarness.subscribe()
# All are plain dicts discriminated by "type" key.
AgentHarnessEvent = dict[str, Any]
