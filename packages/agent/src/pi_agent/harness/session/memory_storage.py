"""In-memory session storage."""
from __future__ import annotations

import datetime
from typing import Any

from ..types import SessionError, SessionStorage
from .uuid import uuidv7


def _update_label_cache(labels: dict[str, str], entry: dict) -> None:
    if entry.get("type") != "label":
        return
    label = (entry.get("label") or "").strip()
    target = entry.get("targetId")
    if not target:
        return
    if label:
        labels[target] = label
    else:
        labels.pop(target, None)


def _build_labels(entries: list[dict]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for entry in entries:
        _update_label_cache(labels, entry)
    return labels


def _generate_entry_id(by_id: dict[str, Any]) -> str:
    for _ in range(100):
        id_ = uuidv7()[:8]
        if id_ not in by_id:
            return id_
    return uuidv7()


def _leaf_id_after_entry(entry: dict) -> str | None:
    return entry.get("targetId") if entry.get("type") == "leaf" else entry.get("id")


class InMemorySessionStorage(SessionStorage):
    def __init__(
        self,
        entries: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._entries: list[dict] = list(entries) if entries else []
        self._by_id: dict[str, dict] = {e["id"]: e for e in self._entries}
        self._labels: dict[str, str] = _build_labels(self._entries)
        self._leaf_id: str | None = None
        for entry in self._entries:
            self._leaf_id = _leaf_id_after_entry(entry)
        if self._leaf_id is not None and self._leaf_id not in self._by_id:
            raise SessionError("invalid_session", f"Entry {self._leaf_id} not found")
        self._metadata: dict = metadata or {
            "id": uuidv7(),
            "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    async def get_metadata(self) -> dict:
        return self._metadata

    async def get_leaf_id(self) -> str | None:
        if self._leaf_id is not None and self._leaf_id not in self._by_id:
            raise SessionError("invalid_session", f"Entry {self._leaf_id} not found")
        return self._leaf_id

    async def set_leaf_id(self, leaf_id: str | None) -> None:
        if leaf_id is not None and leaf_id not in self._by_id:
            raise SessionError("not_found", f"Entry {leaf_id} not found")
        entry = {
            "type": "leaf",
            "id": _generate_entry_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "targetId": leaf_id,
        }
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry
        self._leaf_id = leaf_id

    async def create_entry_id(self) -> str:
        return _generate_entry_id(self._by_id)

    async def append_entry(self, entry: dict) -> None:
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry
        _update_label_cache(self._labels, entry)
        self._leaf_id = _leaf_id_after_entry(entry)

    async def get_entry(self, id: str) -> dict | None:
        return self._by_id.get(id)

    async def find_entries(self, type: str) -> list[dict]:
        return [e for e in self._entries if e.get("type") == type]

    async def get_label(self, id: str) -> str | None:
        return self._labels.get(id)

    async def get_path_to_root(self, leaf_id: str | None) -> list[dict]:
        if leaf_id is None:
            return []
        path: list[dict] = []
        current = self._by_id.get(leaf_id)
        if current is None:
            raise SessionError("not_found", f"Entry {leaf_id} not found")
        while current:
            path.insert(0, current)
            parent_id = current.get("parentId")
            if not parent_id:
                break
            parent = self._by_id.get(parent_id)
            if parent is None:
                raise SessionError("invalid_session", f"Entry {parent_id} not found")
            current = parent
        return path

    async def get_entries(self) -> list[dict]:
        return list(self._entries)
