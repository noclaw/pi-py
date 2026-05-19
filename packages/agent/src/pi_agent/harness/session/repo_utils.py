"""Session repository utilities."""
from __future__ import annotations

import datetime

from ..types import FileError, Result, SessionError, SessionStorage
from .uuid import uuidv7


def create_session_id() -> str:
    return uuidv7()


def create_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def to_session(storage: SessionStorage) -> "Session":
    from .session import Session
    return Session(storage)


def get_fs_result_or_throw(result: Result, message: str) -> object:
    if not result.ok:
        code = "not_found" if result.error.code == "not_found" else "storage"
        raise SessionError(code, f"{message}: {result.error.args[0]}", result.error)
    return result.value


async def get_entries_to_fork(
    storage: SessionStorage,
    entry_id: str | None = None,
    position: str | None = None,
) -> list[dict]:
    if not entry_id:
        return await storage.get_entries()
    target = await storage.get_entry(entry_id)
    if not target:
        raise SessionError("invalid_fork_target", f"Entry {entry_id} not found")
    effective_position = position or "before"
    if effective_position == "at":
        effective_leaf_id: str | None = target["id"]
    else:
        msg = target.get("message", {})
        if target.get("type") != "message" or (isinstance(msg, dict) and msg.get("role") != "user"):
            raise SessionError("invalid_fork_target", f"Entry {entry_id} is not a user message")
        effective_leaf_id = target.get("parentId")
    return await storage.get_path_to_root(effective_leaf_id)
