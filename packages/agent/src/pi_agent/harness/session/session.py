"""Session class and context builder."""
from __future__ import annotations

import datetime
from typing import Any

from ..messages import (
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from ..types import SessionError, SessionStorage


def _deserialize_message(msg: Any) -> Any:
    """Reconstruct a Pydantic Message model from a plain dict stored in the session."""
    if not isinstance(msg, dict):
        return msg
    role = msg.get("role")
    try:
        from pi_ai.types import AssistantMessage, ToolResultMessage, UserMessage
        if role == "user":
            return UserMessage.model_validate(msg)
        if role == "assistant":
            return AssistantMessage.model_validate(msg)
        if role == "toolResult":
            return ToolResultMessage.model_validate(msg)
    except Exception:
        pass
    return msg


def build_session_context(path_entries: list[dict]) -> dict:
    """Build SessionContext from a root-to-leaf path of session tree entries."""
    thinking_level = "off"
    model: dict | None = None
    compaction: dict | None = None

    for entry in path_entries:
        t = entry.get("type")
        if t == "thinking_level_change":
            thinking_level = entry.get("thinkingLevel", "off")
        elif t == "model_change":
            model = {"provider": entry["provider"], "modelId": entry["modelId"]}
        elif t == "message":
            msg = entry.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                model = {"provider": msg.get("provider"), "modelId": msg.get("model")}
        elif t == "compaction":
            compaction = entry

    messages: list[Any] = []

    def append_message(entry: dict) -> None:
        t = entry.get("type")
        if t == "message":
            messages.append(_deserialize_message(entry["message"]))
        elif t == "custom_message":
            messages.append(create_custom_message(
                entry.get("customType", ""),
                entry.get("content", ""),
                entry.get("display", True),
                entry.get("details"),
                entry.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            ))
        elif t == "branch_summary" and entry.get("summary"):
            messages.append(create_branch_summary_message(
                entry["summary"], entry.get("fromId", ""), entry["timestamp"]
            ))

    if compaction:
        messages.append(create_compaction_summary_message(
            compaction["summary"], compaction.get("tokensBefore", 0), compaction["timestamp"]
        ))
        comp_idx = next(
            (i for i, e in enumerate(path_entries) if e.get("type") == "compaction" and e.get("id") == compaction["id"]),
            -1,
        )
        found_first_kept = False
        for i, entry in enumerate(path_entries):
            if i >= comp_idx:
                break
            if entry.get("id") == compaction.get("firstKeptEntryId"):
                found_first_kept = True
            if found_first_kept:
                append_message(entry)
        for entry in path_entries[comp_idx + 1:]:
            append_message(entry)
    else:
        for entry in path_entries:
            append_message(entry)

    return {"messages": messages, "thinkingLevel": thinking_level, "model": model}


class Session:
    def __init__(self, storage: SessionStorage) -> None:
        self._storage = storage

    def get_storage(self) -> SessionStorage:
        return self._storage

    async def get_metadata(self) -> dict:
        return await self._storage.get_metadata()

    async def get_leaf_id(self) -> str | None:
        return await self._storage.get_leaf_id()

    async def get_entry(self, id: str) -> dict | None:
        return await self._storage.get_entry(id)

    async def get_entries(self) -> list[dict]:
        return await self._storage.get_entries()

    async def get_branch(self, from_id: str | None = None) -> list[dict]:
        leaf_id = from_id if from_id is not None else await self._storage.get_leaf_id()
        return await self._storage.get_path_to_root(leaf_id)

    async def build_context(self) -> dict:
        return build_session_context(await self.get_branch())

    async def get_label(self, id: str) -> str | None:
        return await self._storage.get_label(id)

    async def get_session_name(self) -> str | None:
        entries = await self._storage.find_entries("session_info")
        last = entries[-1] if entries else None
        name = (last.get("name") or "").strip() if last else None
        return name or None

    async def _append(self, entry: dict) -> str:
        await self._storage.append_entry(entry)
        return entry["id"]

    async def append_message(self, message: Any) -> str:
        msg_dict = message.model_dump() if hasattr(message, "model_dump") else message
        return await self._append({
            "type": "message",
            "id": await self._storage.create_entry_id(),
            "parentId": await self._storage.get_leaf_id(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "message": msg_dict,
        })

    async def append_thinking_level_change(self, thinking_level: str) -> str:
        return await self._append({
            "type": "thinking_level_change",
            "id": await self._storage.create_entry_id(),
            "parentId": await self._storage.get_leaf_id(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "thinkingLevel": thinking_level,
        })

    async def append_model_change(self, provider: str, model_id: str) -> str:
        return await self._append({
            "type": "model_change",
            "id": await self._storage.create_entry_id(),
            "parentId": await self._storage.get_leaf_id(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "provider": provider,
            "modelId": model_id,
        })

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        return await self._append({
            "type": "compaction",
            "id": await self._storage.create_entry_id(),
            "parentId": await self._storage.get_leaf_id(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
            "details": details,
            "fromHook": from_hook,
        })

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._append({
            "type": "custom",
            "id": await self._storage.create_entry_id(),
            "parentId": await self._storage.get_leaf_id(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "customType": custom_type,
            "data": data,
        })

    async def append_custom_message_entry(
        self,
        custom_type: str,
        content: Any,
        display: bool,
        details: Any = None,
    ) -> str:
        return await self._append({
            "type": "custom_message",
            "id": await self._storage.create_entry_id(),
            "parentId": await self._storage.get_leaf_id(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "customType": custom_type,
            "content": content,
            "display": display,
            "details": details,
        })

    async def append_label(self, target_id: str, label: str | None) -> str:
        if not await self._storage.get_entry(target_id):
            raise SessionError("not_found", f"Entry {target_id} not found")
        return await self._append({
            "type": "label",
            "id": await self._storage.create_entry_id(),
            "parentId": await self._storage.get_leaf_id(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "targetId": target_id,
            "label": label,
        })

    async def append_session_name(self, name: str) -> str:
        return await self._append({
            "type": "session_info",
            "id": await self._storage.create_entry_id(),
            "parentId": await self._storage.get_leaf_id(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "name": name.strip(),
        })

    async def move_to(
        self,
        entry_id: str | None,
        summary: dict | None = None,
    ) -> str | None:
        if entry_id is not None and not await self._storage.get_entry(entry_id):
            raise SessionError("not_found", f"Entry {entry_id} not found")
        await self._storage.set_leaf_id(entry_id)
        if not summary:
            return None
        return await self._append({
            "type": "branch_summary",
            "id": await self._storage.create_entry_id(),
            "parentId": entry_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "fromId": entry_id or "root",
            "summary": summary["summary"],
            "details": summary.get("details"),
            "fromHook": summary.get("fromHook", False),
        })
