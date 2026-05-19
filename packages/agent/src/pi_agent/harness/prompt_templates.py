"""Prompt template loading and invocation formatting."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .types import ExecutionEnv, FileInfo, PromptTemplate


@dataclass
class PromptTemplateDiagnostic:
    type: str  # "warning"
    code: str
    message: str
    path: str


def _basename(path: str) -> str:
    normalized = path.rstrip("/")
    idx = normalized.rfind("/")
    return normalized if idx == -1 else normalized[idx + 1:]


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


async def _resolve_kind(
    env: ExecutionEnv,
    info: FileInfo,
    diagnostics: list[PromptTemplateDiagnostic],
) -> str | None:
    if info.kind in ("file", "directory"):
        return info.kind
    result = await env.canonical_path(info.path)
    if not result.ok:
        if result.error.code != "not_found":
            diagnostics.append(PromptTemplateDiagnostic(
                type="warning", code="file_info_failed", message=result.error.args[0], path=info.path
            ))
        return None
    target = await env.file_info(result.value)
    if not target.ok:
        if target.error.code != "not_found":
            diagnostics.append(PromptTemplateDiagnostic(
                type="warning", code="file_info_failed", message=target.error.args[0], path=info.path
            ))
        return None
    return target.value.kind if target.value.kind in ("file", "directory") else None


async def _load_template_from_file(
    env: ExecutionEnv,
    file_path: str,
) -> dict:
    diagnostics: list[PromptTemplateDiagnostic] = []
    result = await env.read_text_file(file_path)
    if not result.ok:
        diagnostics.append(PromptTemplateDiagnostic(
            type="warning", code="read_failed", message=result.error.args[0], path=file_path
        ))
        return {"promptTemplate": None, "diagnostics": diagnostics}

    try:
        parsed = _parse_frontmatter(result.value)
    except Exception as e:
        diagnostics.append(PromptTemplateDiagnostic(
            type="warning", code="parse_failed", message=str(e), path=file_path
        ))
        return {"promptTemplate": None, "diagnostics": diagnostics}

    fm = parsed["frontmatter"]
    body = parsed["body"]
    first_line = next((l for l in body.split("\n") if l.strip()), "")
    description = fm.get("description") if isinstance(fm.get("description"), str) else ""
    if not description and first_line:
        description = first_line[:60] + ("..." if len(first_line) > 60 else "")

    name = re.sub(r"\.md$", "", _basename(file_path), flags=re.IGNORECASE)
    template = PromptTemplate(name=name, content=body, description=description)
    return {"promptTemplate": template, "diagnostics": diagnostics}


async def _load_from_dir(env: ExecutionEnv, dir_: str) -> dict:
    templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    entries_result = await env.list_dir(dir_)
    if not entries_result.ok:
        diagnostics.append(PromptTemplateDiagnostic(
            type="warning", code="list_failed", message=entries_result.error.args[0], path=dir_
        ))
        return {"promptTemplates": templates, "diagnostics": diagnostics}
    for entry in sorted(entries_result.value, key=lambda e: e.name):
        kind = await _resolve_kind(env, entry, diagnostics)
        if kind != "file" or not entry.name.endswith(".md"):
            continue
        result = await _load_template_from_file(env, entry.path)
        if result["promptTemplate"]:
            templates.append(result["promptTemplate"])
        diagnostics.extend(result["diagnostics"])
    return {"promptTemplates": templates, "diagnostics": diagnostics}


async def load_prompt_templates(
    env: ExecutionEnv,
    paths: str | list[str],
) -> dict:
    templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    path_list = [paths] if isinstance(paths, str) else paths
    for path in path_list:
        info_result = await env.file_info(path)
        if not info_result.ok:
            if info_result.error.code != "not_found":
                diagnostics.append(PromptTemplateDiagnostic(
                    type="warning", code="file_info_failed", message=info_result.error.args[0], path=path
                ))
            continue
        kind = await _resolve_kind(env, info_result.value, diagnostics)
        if kind == "directory":
            result = await _load_from_dir(env, info_result.value.path)
            templates.extend(result["promptTemplates"])
            diagnostics.extend(result["diagnostics"])
        elif kind == "file" and info_result.value.name.endswith(".md"):
            result = await _load_template_from_file(env, info_result.value.path)
            if result["promptTemplate"]:
                templates.append(result["promptTemplate"])
            diagnostics.extend(result["diagnostics"])
    return {"promptTemplates": templates, "diagnostics": diagnostics}


def parse_command_args(args_string: str) -> list[str]:
    args: list[str] = []
    current = ""
    in_quote: str | None = None
    for char in args_string:
        if in_quote:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char in (" ", "\t"):
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: list[str]) -> str:
    result = content
    result = re.sub(r"\$(\d+)", lambda m: args[int(m.group(1)) - 1] if int(m.group(1)) - 1 < len(args) else "", result)

    def replace_slice(m: re.Match) -> str:
        start = max(0, int(m.group(1)) - 1)
        length_str = m.group(2)
        if length_str:
            return " ".join(args[start:start + int(length_str)])
        return " ".join(args[start:])

    result = re.sub(r"\$\{@:(\d+)(?::(\d+))?\}", replace_slice, result)
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args).replace("$@", all_args)
    return result


def format_prompt_template_invocation(template: PromptTemplate, args: list[str] | None = None) -> str:
    return substitute_args(template.content, args or [])
