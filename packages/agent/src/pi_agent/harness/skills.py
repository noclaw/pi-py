"""Skill loading from directories."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import ExecutionEnv, FileInfo, Skill


_MAX_NAME_LENGTH = 64
_MAX_DESC_LENGTH = 1024
_IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]


@dataclass
class SkillDiagnostic:
    type: str  # "warning"
    code: str  # SkillDiagnosticCode
    message: str
    path: str


# ── Path helpers ───────────────────────────────────────────────────────────────

def _join(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child.lstrip('/')}"


def _dirname(path: str) -> str:
    normalized = path.rstrip("/")
    idx = normalized.rfind("/")
    return "/" if idx <= 0 else normalized[:idx]


def _basename(path: str) -> str:
    normalized = path.rstrip("/")
    idx = normalized.rfind("/")
    return normalized if idx == -1 else normalized[idx + 1:]


def _relpath(root: str, path: str) -> str:
    root_n = root.rstrip("/")
    path_n = path.rstrip("/")
    if path_n == root_n:
        return ""
    prefix = f"{root_n}/"
    if path_n.startswith(prefix):
        return path_n[len(prefix):]
    return path_n.lstrip("/")


# ── Ignore matcher ─────────────────────────────────────────────────────────────

class _IgnoreMatcher:
    def __init__(self) -> None:
        self._patterns: list[str] = []
        self._spec: Any = None

    def add(self, patterns: list[str]) -> None:
        self._patterns.extend(patterns)
        self._spec = None  # invalidate cache

    def ignores(self, path: str) -> bool:
        if not self._patterns:
            return False
        if self._spec is None:
            try:
                from pathspec import PathSpec
                self._spec = PathSpec.from_lines("gitwildmatch", self._patterns)
            except ImportError:
                return False
        return self._spec.match_file(path)


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None
    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


# ── YAML frontmatter parser ────────────────────────────────────────────────────

def _parse_frontmatter(content: str) -> dict:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {"frontmatter": {}, "body": normalized}
    end_idx = normalized.find("\n---", 3)
    if end_idx == -1:
        return {"frontmatter": {}, "body": normalized}
    yaml_str = normalized[4:end_idx]
    body = normalized[end_idx + 4:].strip()
    try:
        import yaml
        frontmatter = yaml.safe_load(yaml_str) or {}
    except Exception:
        frontmatter = {}
    return {"frontmatter": frontmatter, "body": body}


# ── Validators ─────────────────────────────────────────────────────────────────

def _validate_name(name: str, parent_dir: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir}"')
    if len(name) > _MAX_NAME_LENGTH:
        errors.append(f"name exceeds {_MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    if not description or not description.strip():
        return ["description is required"]
    if len(description) > _MAX_DESC_LENGTH:
        return [f"description exceeds {_MAX_DESC_LENGTH} characters ({len(description)})"]
    return []


# ── Skill formatting ───────────────────────────────────────────────────────────

def format_skill_invocation(skill: Skill, additional_instructions: str | None = None) -> str:
    skill_dir = _dirname(skill.file_path)
    block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {skill_dir}.\n\n{skill.content}\n</skill>"
    )
    if additional_instructions:
        return f"{block}\n\n{additional_instructions}"
    return block


# ── Loading ────────────────────────────────────────────────────────────────────

async def _resolve_kind(
    env: ExecutionEnv,
    info: FileInfo,
    diagnostics: list[SkillDiagnostic],
) -> str | None:
    if info.kind in ("file", "directory"):
        return info.kind
    result = await env.canonical_path(info.path)
    if not result.ok:
        if result.error.code != "not_found":
            diagnostics.append(SkillDiagnostic(
                type="warning", code="file_info_failed",
                message=result.error.args[0], path=info.path,
            ))
        return None
    target = await env.file_info(result.value)
    if not target.ok:
        if target.error.code != "not_found":
            diagnostics.append(SkillDiagnostic(
                type="warning", code="file_info_failed",
                message=target.error.args[0], path=info.path,
            ))
        return None
    return target.value.kind if target.value.kind in ("file", "directory") else None


async def _add_ignore_rules(
    env: ExecutionEnv,
    ig: _IgnoreMatcher,
    dir_: str,
    root_dir: str,
    diagnostics: list[SkillDiagnostic],
) -> None:
    rel_dir = _relpath(root_dir, dir_)
    prefix = f"{rel_dir}/" if rel_dir else ""
    for filename in _IGNORE_FILE_NAMES:
        ignore_path = _join(dir_, filename)
        info = await env.file_info(ignore_path)
        if not info.ok:
            if info.error.code != "not_found":
                diagnostics.append(SkillDiagnostic(
                    type="warning", code="file_info_failed",
                    message=info.error.args[0], path=ignore_path,
                ))
            continue
        if info.value.kind != "file":
            continue
        content_result = await env.read_text_file(ignore_path)
        if not content_result.ok:
            diagnostics.append(SkillDiagnostic(
                type="warning", code="read_failed",
                message=content_result.error.args[0], path=ignore_path,
            ))
            continue
        patterns = [
            p for line in content_result.value.split("\n")
            if (p := _prefix_ignore_pattern(line, prefix)) is not None
        ]
        if patterns:
            ig.add(patterns)


async def _load_skill_from_file(
    env: ExecutionEnv,
    file_path: str,
) -> dict:
    diagnostics: list[SkillDiagnostic] = []
    result = await env.read_text_file(file_path)
    if not result.ok:
        diagnostics.append(SkillDiagnostic(
            type="warning", code="read_failed", message=result.error.args[0], path=file_path
        ))
        return {"skill": None, "diagnostics": diagnostics}

    try:
        parsed = _parse_frontmatter(result.value)
    except Exception as e:
        diagnostics.append(SkillDiagnostic(
            type="warning", code="parse_failed", message=str(e), path=file_path
        ))
        return {"skill": None, "diagnostics": diagnostics}

    fm = parsed["frontmatter"]
    body = parsed["body"]
    skill_dir = _dirname(file_path)
    parent_dir_name = _basename(skill_dir)
    description = fm.get("description") if isinstance(fm.get("description"), str) else None

    for error in _validate_description(description):
        diagnostics.append(SkillDiagnostic(
            type="warning", code="invalid_metadata", message=error, path=file_path
        ))

    frontmatter_name = fm.get("name") if isinstance(fm.get("name"), str) else None
    name = frontmatter_name or parent_dir_name
    for error in _validate_name(name, parent_dir_name):
        diagnostics.append(SkillDiagnostic(
            type="warning", code="invalid_metadata", message=error, path=file_path
        ))

    if not description or not description.strip():
        return {"skill": None, "diagnostics": diagnostics}

    skill = Skill(
        name=name,
        description=description,
        content=body,
        file_path=file_path,
        disable_model_invocation=fm.get("disable-model-invocation") is True,
    )
    return {"skill": skill, "diagnostics": diagnostics}


async def _load_from_dir(
    env: ExecutionEnv,
    dir_: str,
    include_root_files: bool,
    ig: _IgnoreMatcher,
    root_dir: str,
) -> dict:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []

    info_result = await env.file_info(dir_)
    if not info_result.ok:
        if info_result.error.code != "not_found":
            diagnostics.append(SkillDiagnostic(
                type="warning", code="file_info_failed", message=info_result.error.args[0], path=dir_
            ))
        return {"skills": skills, "diagnostics": diagnostics}

    kind = await _resolve_kind(env, info_result.value, diagnostics)
    if kind != "directory":
        return {"skills": skills, "diagnostics": diagnostics}

    await _add_ignore_rules(env, ig, dir_, root_dir, diagnostics)

    entries_result = await env.list_dir(dir_)
    if not entries_result.ok:
        diagnostics.append(SkillDiagnostic(
            type="warning", code="list_failed", message=entries_result.error.args[0], path=dir_
        ))
        return {"skills": skills, "diagnostics": diagnostics}

    entries = entries_result.value

    # Check for SKILL.md first
    for entry in entries:
        if entry.name != "SKILL.md":
            continue
        entry_kind = await _resolve_kind(env, entry, diagnostics)
        if entry_kind != "file":
            continue
        rel = _relpath(root_dir, entry.path)
        if ig.ignores(rel):
            continue
        result = await _load_skill_from_file(env, entry.path)
        if result["skill"]:
            skills.append(result["skill"])
        diagnostics.extend(result["diagnostics"])
        return {"skills": skills, "diagnostics": diagnostics}

    # Recurse into subdirs, load root .md files
    for entry in sorted(entries, key=lambda e: e.name):
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        entry_kind = await _resolve_kind(env, entry, diagnostics)
        if not entry_kind:
            continue
        rel = _relpath(root_dir, entry.path)
        ignore_path = f"{rel}/" if entry_kind == "directory" else rel
        if ig.ignores(ignore_path):
            continue
        if entry_kind == "directory":
            sub = await _load_from_dir(env, entry.path, False, ig, root_dir)
            skills.extend(sub["skills"])
            diagnostics.extend(sub["diagnostics"])
        elif entry_kind == "file" and include_root_files and entry.name.endswith(".md"):
            result = await _load_skill_from_file(env, entry.path)
            if result["skill"]:
                skills.append(result["skill"])
            diagnostics.extend(result["diagnostics"])

    return {"skills": skills, "diagnostics": diagnostics}


async def load_skills(
    env: ExecutionEnv,
    dirs: str | list[str],
) -> dict:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    dir_list = [dirs] if isinstance(dirs, str) else dirs
    for dir_ in dir_list:
        info_result = await env.file_info(dir_)
        if not info_result.ok:
            if info_result.error.code != "not_found":
                diagnostics.append(SkillDiagnostic(
                    type="warning", code="file_info_failed", message=info_result.error.args[0], path=dir_
                ))
            continue
        kind = await _resolve_kind(env, info_result.value, diagnostics)
        if kind != "directory":
            continue
        result = await _load_from_dir(env, info_result.value.path, True, _IgnoreMatcher(), info_result.value.path)
        skills.extend(result["skills"])
        diagnostics.extend(result["diagnostics"])
    return {"skills": skills, "diagnostics": diagnostics}
