"""JSONL-file-based session storage."""
from __future__ import annotations

import datetime
import json
from typing import Any

from ..types import FileSystem, SessionError, SessionStorage, to_error
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


def _invalid_session(file_path: str, message: str, cause: Exception | None = None) -> SessionError:
    return SessionError("invalid_session", f"Invalid JSONL session file {file_path}: {message}", cause)


def _invalid_entry(file_path: str, line_number: int, message: str, cause: Exception | None = None) -> SessionError:
    return SessionError(
        "invalid_entry",
        f"Invalid JSONL session file {file_path}: line {line_number} {message}",
        cause,
    )


def _parse_header_line(line: str, file_path: str) -> dict:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as e:
        raise _invalid_session(file_path, "first line is not a valid session header", e)
    if not isinstance(parsed, dict):
        raise _invalid_session(file_path, "first line is not a valid session header")
    if parsed.get("type") != "session":
        raise _invalid_session(file_path, "first line is not a valid session header")
    if parsed.get("version") != 3:
        raise _invalid_session(file_path, "unsupported session version")
    if not isinstance(parsed.get("id"), str) or not parsed["id"]:
        raise _invalid_session(file_path, "session header is missing id")
    if not isinstance(parsed.get("timestamp"), str) or not parsed["timestamp"]:
        raise _invalid_session(file_path, "session header is missing timestamp")
    if not isinstance(parsed.get("cwd"), str) or not parsed["cwd"]:
        raise _invalid_session(file_path, "session header is missing cwd")
    return parsed


def _parse_entry_line(line: str, file_path: str, line_number: int) -> dict:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as e:
        raise _invalid_entry(file_path, line_number, "is not valid JSON", e)
    if not isinstance(parsed, dict):
        raise _invalid_entry(file_path, line_number, "is not a valid session entry")
    if not isinstance(parsed.get("type"), str):
        raise _invalid_entry(file_path, line_number, "is missing entry type")
    if not isinstance(parsed.get("id"), str) or not parsed["id"]:
        raise _invalid_entry(file_path, line_number, "is missing entry id")
    if parsed.get("parentId") is not None and not isinstance(parsed["parentId"], str):
        raise _invalid_entry(file_path, line_number, "has invalid parentId")
    if not isinstance(parsed.get("timestamp"), str) or not parsed["timestamp"]:
        raise _invalid_entry(file_path, line_number, "is missing timestamp")
    return parsed


def _header_to_metadata(header: dict, path: str) -> dict:
    return {
        "id": header["id"],
        "createdAt": header["timestamp"],
        "cwd": header["cwd"],
        "path": path,
        "parentSessionPath": header.get("parentSession"),
    }


async def load_jsonl_session_metadata(fs: FileSystem, file_path: str) -> dict:
    result = await fs.read_text_lines(file_path, max_lines=1)
    if not result.ok:
        raise SessionError("storage", f"Failed to read session header {file_path}: {result.error.args[0]}", result.error)
    lines = result.value
    line = lines[0] if lines else ""
    if line.strip():
        return _header_to_metadata(_parse_header_line(line, file_path), file_path)
    raise _invalid_session(file_path, "missing session header")


class JsonlSessionStorage(SessionStorage):
    def __init__(
        self,
        fs: FileSystem,
        file_path: str,
        header: dict,
        entries: list[dict],
        leaf_id: str | None,
    ) -> None:
        self._fs = fs
        self._file_path = file_path
        self._metadata = _header_to_metadata(header, file_path)
        self._entries = list(entries)
        self._by_id: dict[str, dict] = {e["id"]: e for e in self._entries}
        self._labels = _build_labels(self._entries)
        self._leaf_id = leaf_id

    @classmethod
    async def open(cls, fs: FileSystem, file_path: str) -> "JsonlSessionStorage":
        result = await fs.read_text_file(file_path)
        if not result.ok:
            raise SessionError("storage", f"Failed to read session {file_path}: {result.error.args[0]}", result.error)
        lines = [l for l in result.value.split("\n") if l.strip()]
        if not lines:
            raise _invalid_session(file_path, "missing session header")
        header = _parse_header_line(lines[0], file_path)
        entries: list[dict] = []
        leaf_id: str | None = None
        for i, line in enumerate(lines[1:], start=2):
            entry = _parse_entry_line(line, file_path, i)
            entries.append(entry)
            leaf_id = _leaf_id_after_entry(entry)
        return cls(fs, file_path, header, entries, leaf_id)

    @classmethod
    async def create(
        cls,
        fs: FileSystem,
        file_path: str,
        cwd: str,
        session_id: str,
        parent_session_path: str | None = None,
    ) -> "JsonlSessionStorage":
        header = {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "cwd": cwd,
        }
        if parent_session_path:
            header["parentSession"] = parent_session_path
        write_result = await fs.write_file(file_path, json.dumps(header) + "\n")
        if not write_result.ok:
            raise SessionError("storage", f"Failed to create session {file_path}: {write_result.error.args[0]}", write_result.error)
        return cls(fs, file_path, header, [], None)

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
        result = await self._fs.append_file(self._file_path, json.dumps(entry) + "\n")
        if not result.ok:
            raise SessionError("storage", f"Failed to append session leaf {entry['id']}: {result.error.args[0]}", result.error)
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry
        self._leaf_id = leaf_id

    async def create_entry_id(self) -> str:
        return _generate_entry_id(self._by_id)

    async def append_entry(self, entry: dict) -> None:
        result = await self._fs.append_file(self._file_path, json.dumps(entry) + "\n")
        if not result.ok:
            raise SessionError("storage", f"Failed to append session entry {entry['id']}: {result.error.args[0]}", result.error)
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
