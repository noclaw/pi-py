"""In-memory session repository."""
from __future__ import annotations

from ..types import SessionError
from .memory_storage import InMemorySessionStorage
from .repo_utils import create_session_id, create_timestamp, get_entries_to_fork, to_session
from .session import Session


class InMemorySessionRepo:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def create(self, id: str | None = None) -> Session:
        session_id = id or create_session_id()
        metadata = {"id": session_id, "createdAt": create_timestamp()}
        storage = InMemorySessionStorage(metadata=metadata)
        session = to_session(storage)
        self._sessions[session_id] = session
        return session

    async def open(self, metadata: dict) -> Session:
        session = self._sessions.get(metadata["id"])
        if not session:
            raise SessionError("not_found", f"Session not found: {metadata['id']}")
        return session

    async def list(self) -> list[dict]:
        results = []
        for session in self._sessions.values():
            results.append(await session.get_metadata())
        return results

    async def delete(self, metadata: dict) -> None:
        self._sessions.pop(metadata["id"], None)

    async def fork(
        self,
        source_metadata: dict,
        entry_id: str | None = None,
        position: str | None = None,
        id: str | None = None,
    ) -> Session:
        source = await self.open(source_metadata)
        forked_entries = await get_entries_to_fork(source.get_storage(), entry_id, position)
        new_id = id or create_session_id()
        new_metadata = {"id": new_id, "createdAt": create_timestamp()}
        storage = InMemorySessionStorage(entries=forked_entries, metadata=new_metadata)
        session = to_session(storage)
        self._sessions[new_id] = session
        return session
