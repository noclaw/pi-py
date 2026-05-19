"""JSONL-file-based session repository."""
from __future__ import annotations

import re

from ..types import FileSystem, SessionError, to_error
from .jsonl_storage import JsonlSessionStorage, load_jsonl_session_metadata
from .repo_utils import (
    create_session_id,
    create_timestamp,
    get_entries_to_fork,
    get_fs_result_or_throw,
    to_session,
)
from .session import Session


def _encode_cwd(cwd: str) -> str:
    stripped = re.sub(r"^[/\\]", "", cwd)
    return f"--{re.sub(r'[/\\:]', '-', stripped)}--"


class JsonlSessionRepo:
    def __init__(self, fs: FileSystem, sessions_root: str) -> None:
        self._fs = fs
        self._sessions_root_input = sessions_root
        self._sessions_root: str | None = None

    async def _get_sessions_root(self) -> str:
        if not self._sessions_root:
            result = await self._fs.absolute_path(self._sessions_root_input)
            self._sessions_root = get_fs_result_or_throw(result, f"Failed to resolve sessions root {self._sessions_root_input}")
        return self._sessions_root

    async def _get_session_dir(self, cwd: str) -> str:
        root = await self._get_sessions_root()
        result = await self._fs.join_path([root, _encode_cwd(cwd)])
        return get_fs_result_or_throw(result, f"Failed to resolve session directory for {cwd}")

    async def _create_session_file_path(self, cwd: str, session_id: str, timestamp: str) -> str:
        session_dir = await self._get_session_dir(cwd)
        safe_ts = re.sub(r"[:\.]", "-", timestamp)
        result = await self._fs.join_path([session_dir, f"{safe_ts}_{session_id}.jsonl"])
        return get_fs_result_or_throw(result, f"Failed to resolve session file path for {session_id}")

    async def create(
        self,
        cwd: str,
        id: str | None = None,
        parent_session_path: str | None = None,
    ) -> Session:
        session_id = id or create_session_id()
        created_at = create_timestamp()
        session_dir = await self._get_session_dir(cwd)
        mkdir_result = await self._fs.create_dir(session_dir, recursive=True)
        get_fs_result_or_throw(mkdir_result, f"Failed to create session directory {session_dir}")
        file_path = await self._create_session_file_path(cwd, session_id, created_at)
        storage = await JsonlSessionStorage.create(
            self._fs, file_path, cwd, session_id, parent_session_path
        )
        return to_session(storage)

    async def open(self, metadata: dict) -> Session:
        exists_result = await self._fs.exists(metadata["path"])
        exists = get_fs_result_or_throw(exists_result, f"Failed to check session {metadata['path']}")
        if not exists:
            raise SessionError("not_found", f"Session not found: {metadata['path']}")
        storage = await JsonlSessionStorage.open(self._fs, metadata["path"])
        return to_session(storage)

    async def list(self, cwd: str | None = None) -> list[dict]:
        if cwd:
            dirs = [await self._get_session_dir(cwd)]
        else:
            dirs = await self._list_session_dirs()
        sessions: list[dict] = []
        for dir_ in dirs:
            exists_result = await self._fs.exists(dir_)
            exists = get_fs_result_or_throw(exists_result, f"Failed to check session directory {dir_}")
            if not exists:
                continue
            files_result = await self._fs.list_dir(dir_)
            files = get_fs_result_or_throw(files_result, f"Failed to list sessions in {dir_}")
            for file in files:
                if file.kind == "directory" or not file.name.endswith(".jsonl"):
                    continue
                try:
                    sessions.append(await load_jsonl_session_metadata(self._fs, file.path))
                except SessionError as e:
                    if e.code != "invalid_session":
                        raise
        sessions.sort(key=lambda s: s.get("createdAt", ""), reverse=True)
        return sessions

    async def delete(self, metadata: dict) -> None:
        result = await self._fs.remove(metadata["path"], force=True)
        get_fs_result_or_throw(result, f"Failed to delete session {metadata['path']}")

    async def fork(
        self,
        source_metadata: dict,
        cwd: str,
        entry_id: str | None = None,
        position: str | None = None,
        id: str | None = None,
        parent_session_path: str | None = None,
    ) -> Session:
        source = await self.open(source_metadata)
        forked_entries = await get_entries_to_fork(source.get_storage(), entry_id, position)
        new_id = id or create_session_id()
        created_at = create_timestamp()
        session_dir = await self._get_session_dir(cwd)
        mkdir_result = await self._fs.create_dir(session_dir, recursive=True)
        get_fs_result_or_throw(mkdir_result, f"Failed to create session directory {session_dir}")
        file_path = await self._create_session_file_path(cwd, new_id, created_at)
        parent_path = parent_session_path or source_metadata.get("path")
        storage = await JsonlSessionStorage.create(self._fs, file_path, cwd, new_id, parent_path)
        for entry in forked_entries:
            await storage.append_entry(entry)
        return to_session(storage)

    async def _list_session_dirs(self) -> list[str]:
        root = await self._get_sessions_root()
        exists_result = await self._fs.exists(root)
        exists = get_fs_result_or_throw(exists_result, f"Failed to check sessions root {root}")
        if not exists:
            return []
        entries_result = await self._fs.list_dir(root)
        entries = get_fs_result_or_throw(entries_result, f"Failed to list sessions root {root}")
        return [e.path for e in entries if e.kind == "directory"]
