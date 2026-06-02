"""Harness-owned runner for external workspace environments.

This module configures a remote workspace tool backend and delegates planning,
tool dispatch, repair, prediction, and verification to the normal Harness
runtime. It intentionally avoids benchmark-runner adapter surfaces; callers
provide an environment object and Harness owns the execution.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, cast
from urllib.parse import unquote
from uuid import uuid4

import typer
from rich.console import Console

from harness.cli.common import _build_adapter
from harness.cli.config import HarnessConfig
from harness.cli.run_commands import run_once as _harness_run_once
from harness.cli.runtime_agent import build_agent as _runtime_build_agent
from harness.cli.runtime_helpers import build_critic as _build_critic
from harness.cli.runtime_helpers import resolve_runtime_strategy as _resolve_runtime_strategy
from harness.core import (
    ChainedVerifier,
    Done,
    ErrorEvent,
    Message,
    ModelSelectedEvent,
    TextDelta,
    Tool,
    ToolCall,
    ToolCallEvent,
    ToolRegistry,
    ToolResult,
    ToolResultEvent,
)
from harness.core.activity import ActivityEvent
from harness.core.schemas import VerificationResult
from harness.core.shell_feedback import shell_failure_hint
from harness.core.tools_verification import (
    _command_exits_before_trailing_command,
    _failure_branch_masks_exit_status,
    _output_reports_failure,
    _successful_shell_stderr_reports_failure,
    _verification_command_is_noop,
)
from harness.core.verification_judges import _parse_judge_response
from harness.storage.memory import InMemoryStorage
from harness.tools.web import FetchUrlTool, WebSearchTool

_GIT_STATUS_COMMAND = "git status --porcelain --untracked-files=all"
_NESTED_GIT_STATUS_COMMAND = (
    "tmp=$(mktemp) && "
    "trap 'rm -f \"$tmp\"' EXIT && "
    "find . -path './.git' -prune -o -type d -name .git -print > \"$tmp\" 2>/dev/null && "
    "while IFS= read -r gitdir; do "
    "repo=${gitdir%/.git}; "
    '[ "$repo" = "." ] && continue; '
    "repo=${repo#./}; "
    '[ -z "$repo" ] && continue; '
    'git -C "$repo" status --porcelain --untracked-files=all 2>/dev/null | '
    "while IFS= read -r line; do "
    '[ -z "$line" ] && continue; '
    "code=$(printf '%.2s' \"$line\"); "
    "path=${line#???}; "
    "case \"$path\" in *' -> '*) path=${path##* -> } ;; esac; "
    'printf \'%s %s/%s\\n\' "$code" "$repo" "$path"; '
    "done; "
    'done < "$tmp"'
)
_NESTED_GIT_LS_FILES_COMMAND = (
    "tmp=$(mktemp) && "
    "trap 'rm -f \"$tmp\"' EXIT && "
    "find . -path './.git' -prune -o -type d -name .git -print > \"$tmp\" 2>/dev/null && "
    "while IFS= read -r gitdir; do "
    "repo=${gitdir%/.git}; "
    '[ "$repo" = "." ] && continue; '
    "repo=${repo#./}; "
    '[ -z "$repo" ] && continue; '
    'git -C "$repo" ls-files 2>/dev/null | '
    "while IFS= read -r path; do "
    '[ -z "$path" ] && continue; '
    'printf \'%s/%s\\n\' "$repo" "$path"; '
    "done; "
    'done < "$tmp"'
)
_NESTED_GIT_ROOTS_COMMAND = (
    "tmp=$(mktemp) && "
    "trap 'rm -f \"$tmp\"' EXIT && "
    "find . -path './.git' -prune -o -type d -name .git -print > \"$tmp\" 2>/dev/null && "
    "while IFS= read -r gitdir; do "
    "repo=${gitdir%/.git}; "
    '[ "$repo" = "." ] && continue; '
    "repo=${repo#./}; "
    '[ -z "$repo" ] && continue; '
    "printf '%s\\n' \"$repo\"; "
    'done < "$tmp"'
)


def _nested_git_committed_status_command(required_base: str) -> str:
    quoted_base = shlex.quote(required_base)
    return (
        f"base={quoted_base}; "
        "tmp=$(mktemp) && "
        "trap 'rm -f \"$tmp\"' EXIT && "
        "find . -path './.git' -prune -o -type d -name .git -print > \"$tmp\" 2>/dev/null && "
        "while IFS= read -r gitdir; do "
        "repo=${gitdir%/.git}; "
        '[ "$repo" = "." ] && continue; '
        "repo=${repo#./}; "
        '[ -z "$repo" ] && continue; '
        'base_head=$(git -C "$repo" rev-parse "$base" 2>/dev/null || true); '
        '[ -z "$base_head" ] && continue; '
        'head=$(git -C "$repo" rev-parse HEAD 2>/dev/null || true); '
        '[ -z "$head" ] && continue; '
        '[ "$head" = "$base_head" ] && continue; '
        'git -C "$repo" merge-base --is-ancestor "$base_head" HEAD 2>/dev/null || continue; '
        'git -C "$repo" diff --name-status "$base_head" HEAD -- 2>/dev/null | '
        "while IFS=\"$(printf '\\t')\" read -r kind path extra; do "
        '[ -z "$kind" ] && continue; '
        'case "$kind" in '
        "A*) code='A ' ;; "
        "D*) code=' D' ;; "
        'R*|C*) code=\'R \'; [ -n "$extra" ] && path="$extra" ;; '
        "*) code=' M' ;; "
        "esac; "
        '[ -z "$path" ] && continue; '
        'printf \'%s %s/%s\\n\' "$code" "$repo" "$path"; '
        "done; "
        'done < "$tmp"'
    )


_GIT_WORKSPACE_FINGERPRINT_COMMAND = (
    "git diff --binary --no-ext-diff; "
    "git diff --cached --binary --no-ext-diff; "
    "git ls-files --others --exclude-standard | "
    "while IFS= read -r path; do "
    'case "$path" in '
    ".harness-home|.harness-home/*|*/.harness-home|*/.harness-home/*|"
    "*/__pycache__/*|__pycache__/*|*/.pytest_cache/*|.pytest_cache/*|"
    "*/.mypy_cache/*|.mypy_cache/*|*/.ruff_cache/*|.ruff_cache/*|"
    "*/.tox/*|.tox/*|*.pyc|*.pyo) continue ;; "
    "esac; "
    "printf 'untracked %s\\n' \"$path\"; "
    "if stat -f '%Lp' \"$path\" >/dev/null 2>&1; then "
    "printf 'mode '; stat -f '%Lp %N' \"$path\"; "
    "else printf 'mode '; stat -c '%a %n' \"$path\" 2>/dev/null || true; fi; "
    'git hash-object -- "$path" 2>/dev/null || true; '
    "done"
)
_ROOT_PROBE_COMMANDS = (
    "find",
    "ls",
    "tree",
    "du",
)
_SHELL_INSPECTION_COMMANDS = {
    "awk",
    "cat",
    "du",
    "egrep",
    "fd",
    "fgrep",
    "file",
    "find",
    "grep",
    "head",
    "less",
    "ls",
    "more",
    "pwd",
    "rg",
    "sed",
    "stat",
    "tail",
    "tree",
    "wc",
}
_GIT_INSPECTION_COMMANDS = {
    "branch",
    "diff",
    "grep",
    "log",
    "ls-files",
    "rev-parse",
    "show",
    "status",
}
_SETUP_COMMAND_WORDS = {
    "add",
    "bootstrap",
    "build",
    "create",
    "ensurepip",
    "install",
    "pull",
    "remove",
    "sync",
    "uninstall",
    "update",
    "upgrade",
}
_SETUP_COMMAND_EXECUTABLES = {
    "apt",
    "apt-get",
    "brew",
    "bundle",
    "cargo",
    "composer",
    "docker",
    "gem",
    "go",
    "npm",
    "npx",
    "pip",
    "pip3",
    "pipx",
    "pnpm",
    "poetry",
    "python",
    "python3",
    "uv",
    "yarn",
}
_TEST_DIR_NAMES = {
    "test",
    "tests",
    "testdata",
    "fixtures",
    "fixture",
    "t",
    "spec",
    "specs",
    "__test__",
    "__tests__",
    "__fixtures__",
}
_TEST_FIXTURE_DIR_NAMES = {"testdata", "fixtures", "fixture", "__fixtures__"}
_APPLY_PATCH_RECOVERY_GUIDANCE = (
    "The patch was not applied. Read the target file or range for exact context, "
    "then retry with a valid unified diff or Codex-style patch. If patch syntax is "
    "getting in the way, use edit_file with an exact old/new block, or write_file "
    "with overwrite=true when replacing the full file is intentional."
)
_EDIT_FILE_RECOVERY_GUIDANCE = "The edit was not applied."
_DEFAULT_VERIFIER_FAILURE_GUIDANCE = (
    "Configured default verifier evidence is required for completion."
)
_MIN_EXTERNAL_WORKSPACE_SOURCE_REPAIR_TURNS = 2
_MIN_EXTERNAL_WORKSPACE_VERIFICATION_REPAIR_TURNS = 8
_UNIFIED_DIFF_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<suffix>.*)$"
)


def _normalized_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    return tuple(
        value.strip() for value in values or () if isinstance(value, str) and value.strip()
    )


def external_workspace_repair_attempts(
    *, source_change_retries: int, verification_retries: int
) -> int:
    source_attempts = max(0, int(source_change_retries))
    verification_attempts = max(0, int(verification_retries))
    if source_attempts:
        source_attempts = max(
            source_attempts,
            _MIN_EXTERNAL_WORKSPACE_SOURCE_REPAIR_TURNS,
        )
    if verification_attempts:
        verification_attempts = max(
            verification_attempts,
            _MIN_EXTERNAL_WORKSPACE_VERIFICATION_REPAIR_TURNS,
        )
    return source_attempts + verification_attempts


def external_workspace_total_attempts(
    *, source_change_retries: int, verification_retries: int
) -> int:
    return 1 + external_workspace_repair_attempts(
        source_change_retries=source_change_retries,
        verification_retries=verification_retries,
    )


def _format_unified_count(start: str, count: int) -> str:
    if count == 1:
        return start
    return f"{start},{count}"


def _has_later_hunk_body_line(lines: list[str], start_index: int) -> bool:
    for line in lines[start_index:]:
        text = line.rstrip("\r\n")
        if _UNIFIED_DIFF_HUNK_RE.match(text):
            return False
        if text.startswith("diff --git ") or text.startswith(("--- ", "+++ ")):
            return False
        if not text:
            continue
        return text.startswith((" ", "-", "+", "\\"))
    return False


def _normalize_unified_diff_hunk_headers(patch: str) -> tuple[str, bool]:
    """Repair only unified-diff hunk line counts, leaving patch content untouched."""

    lines = patch.splitlines(keepends=True)
    changed = False
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _UNIFIED_DIFF_HUNK_RE.match(line.rstrip("\r\n"))
        if not match:
            index += 1
            continue

        body_index = index + 1
        old_count = 0
        new_count = 0
        while body_index < len(lines):
            body_line = lines[body_index]
            body_text = body_line.rstrip("\r\n")
            if _UNIFIED_DIFF_HUNK_RE.match(body_text):
                break
            if body_text.startswith("diff --git ") or body_text.startswith(("--- ", "+++ ")):
                break
            if body_text.startswith("\\"):
                body_index += 1
                continue
            if not body_text:
                if not _has_later_hunk_body_line(lines, body_index + 1):
                    break
                newline = "\n" if body_line.endswith("\n") else ""
                if body_line.endswith("\r\n"):
                    newline = "\r\n"
                lines[body_index] = f" {newline}"
                old_count += 1
                new_count += 1
                changed = True
                body_index += 1
                continue
            marker = body_text[0]
            if marker not in {" ", "-", "+"}:
                lines[body_index] = f" {body_line}"
                old_count += 1
                new_count += 1
                changed = True
                body_index += 1
                continue
            if marker in {" ", "-"}:
                old_count += 1
            if marker in {" ", "+"}:
                new_count += 1
            body_index += 1

        normalized_header = (
            f"@@ -{_format_unified_count(match.group('old_start'), old_count)} "
            f"+{_format_unified_count(match.group('new_start'), new_count)} @@"
            f"{match.group('suffix')}"
        )
        newline = "\n" if line.endswith("\n") else ""
        if line.endswith("\r\n"):
            newline = "\r\n"
        if lines[index] != f"{normalized_header}{newline}":
            lines[index] = f"{normalized_header}{newline}"
            changed = True
        index = body_index

    return "".join(lines), changed


def _looks_like_codex_apply_patch(patch: str) -> bool:
    text = patch.strip()
    return text.startswith("*** Begin Patch") and text.endswith("*** End Patch")


def _looks_like_unified_diff_patch(patch: str) -> bool:
    lines = [line for line in patch.splitlines() if line.strip()]
    if not lines:
        return False
    if any(line.startswith("diff --git ") for line in lines[:3]):
        return True
    for index, line in enumerate(lines[:-1]):
        if line.startswith("--- ") and lines[index + 1].startswith("+++ "):
            return True
    return False


def _strip_unified_diff_patch_envelope(patch: str) -> tuple[str, bool]:
    """Remove accidental Codex-style sentinels around a unified diff."""

    lines = patch.splitlines(keepends=True)
    changed = False
    while lines and lines[0].strip() == "*** Begin Patch":
        lines = lines[1:]
        changed = True
    while lines and lines[-1].strip() == "*** End Patch":
        lines = lines[:-1]
        changed = True
    candidate = "".join(lines)
    if changed and _looks_like_unified_diff_patch(candidate):
        return candidate, True
    return patch, False


@dataclass(frozen=True)
class _CodexPatchLine:
    old: str
    new: str


@dataclass(frozen=True)
class _CodexPatchOperation:
    kind: str
    path: str
    hunks: tuple[tuple[_CodexPatchLine, ...], ...] = ()
    lines: tuple[str, ...] | None = None
    move_to: str | None = None

    def paths(self) -> tuple[str, ...]:
        if self.move_to:
            return (self.path, self.move_to)
        return (self.path,)


def _normalize_codex_patch_path(raw_path: str) -> str:
    path = raw_path.strip()
    if not path:
        raise ValueError("patch path is empty")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"patch path escapes the workspace: {raw_path}")
    normalized = parsed.as_posix()
    if normalized in {"", "."}:
        raise ValueError(f"patch path is invalid: {raw_path}")
    return normalized


def _codex_patch_payload_line(raw_line: str) -> str:
    return raw_line + "\n"


def _parse_codex_hunk_line(raw_line: str) -> _CodexPatchLine | None:
    if raw_line.startswith("\\"):
        return None
    if raw_line.startswith("+"):
        return _CodexPatchLine(old="", new=_codex_patch_payload_line(raw_line[1:]))
    if raw_line.startswith("-"):
        return _CodexPatchLine(old=_codex_patch_payload_line(raw_line[1:]), new="")
    if raw_line.startswith(" "):
        payload = _codex_patch_payload_line(raw_line[1:])
        return _CodexPatchLine(old=payload, new=payload)
    payload = _codex_patch_payload_line(raw_line)
    return _CodexPatchLine(old=payload, new=payload)


def _parse_codex_apply_patch(patch: str) -> tuple[_CodexPatchOperation, ...]:
    lines = patch.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("Codex-style patch must start with *** Begin Patch")
    if lines[-1].strip() != "*** End Patch":
        raise ValueError("Codex-style patch must end with *** End Patch")

    operations: list[_CodexPatchOperation] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith("*** Add File: "):
            path = _normalize_codex_patch_path(line.removeprefix("*** Add File: "))
            index += 1
            added: list[str] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                add_line = lines[index]
                if not add_line.startswith("+"):
                    raise ValueError(f"add file line for {path} must start with '+'")
                added.append(_codex_patch_payload_line(add_line[1:]))
                index += 1
            operations.append(_CodexPatchOperation(kind="add", path=path, lines=tuple(added)))
            continue
        if line.startswith("*** Delete File: "):
            path = _normalize_codex_patch_path(line.removeprefix("*** Delete File: "))
            index += 1
            operations.append(_CodexPatchOperation(kind="delete", path=path))
            continue
        if line.startswith("*** Update File: "):
            path = _normalize_codex_patch_path(line.removeprefix("*** Update File: "))
            index += 1
            move_to: str | None = None
            if index < len(lines) - 1 and lines[index].startswith("*** Move to: "):
                move_to = _normalize_codex_patch_path(lines[index].removeprefix("*** Move to: "))
                index += 1
            hunks: list[tuple[_CodexPatchLine, ...]] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                if not lines[index].startswith("@@"):
                    raise ValueError(f"expected hunk header in update patch for {path}")
                index += 1
                hunk: list[_CodexPatchLine] = []
                while (
                    index < len(lines) - 1
                    and not lines[index].startswith("@@")
                    and not lines[index].startswith("*** ")
                ):
                    parsed = _parse_codex_hunk_line(lines[index])
                    if parsed is not None:
                        hunk.append(parsed)
                    index += 1
                if not hunk:
                    raise ValueError(f"empty update hunk for {path}")
                hunks.append(tuple(hunk))
            if not hunks and move_to is None:
                raise ValueError(f"update patch for {path} contains no hunks")
            operations.append(
                _CodexPatchOperation(
                    kind="update",
                    path=path,
                    hunks=tuple(hunks),
                    move_to=move_to,
                )
            )
            continue
        raise ValueError(f"unsupported Codex-style patch section: {line}")

    if not operations:
        raise ValueError("Codex-style patch contains no operations")
    return tuple(operations)


@dataclass(frozen=True)
class ExternalWorkspacePolicy:
    """Restrictions supplied by the caller for a specific external workspace.

    The harness runner stays general: benchmark-specific private paths, hidden
    verifier directories, and forbidden source repositories are configured here
    instead of being hardcoded into tool behavior.
    """

    forbidden_path_prefixes: tuple[str, ...] = ()
    forbidden_path_parts: tuple[str, ...] = ()
    forbidden_path_names: tuple[str, ...] = ()
    forbidden_absolute_paths: tuple[str, ...] = ()
    forbidden_text_fragments: tuple[str, ...] = ()
    forbidden_web_fragments: tuple[str, ...] = ()
    allowed_git_clone_fragments: tuple[str, ...] = ()
    required_git_base_commit: str = ""
    required_no_network_verify_image: str = ""
    allow_web_access: bool = True
    block_root_filesystem_probe: bool = True
    refusal_message: str = (
        "refused: external workspace policy blocks access to restricted artifacts"
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "forbidden_path_prefixes",
            _normalized_tuple(self.forbidden_path_prefixes),
        )
        object.__setattr__(
            self,
            "forbidden_path_parts",
            tuple(value.lower() for value in _normalized_tuple(self.forbidden_path_parts)),
        )
        object.__setattr__(
            self,
            "forbidden_path_names",
            tuple(value.lower() for value in _normalized_tuple(self.forbidden_path_names)),
        )
        object.__setattr__(
            self,
            "forbidden_absolute_paths",
            _normalized_tuple(self.forbidden_absolute_paths),
        )
        object.__setattr__(
            self,
            "forbidden_text_fragments",
            _normalized_tuple(self.forbidden_text_fragments),
        )
        object.__setattr__(
            self,
            "forbidden_web_fragments",
            _normalized_tuple(self.forbidden_web_fragments),
        )
        object.__setattr__(
            self,
            "allowed_git_clone_fragments",
            _normalized_tuple(self.allowed_git_clone_fragments),
        )
        object.__setattr__(self, "required_git_base_commit", self.required_git_base_commit.strip())
        object.__setattr__(
            self,
            "required_no_network_verify_image",
            self.required_no_network_verify_image.strip(),
        )

    def rejects_relative_path(self, path: str) -> bool:
        parts = [part.lower() for part in PurePosixPath(path.strip("/")).parts if part]
        name = parts[-1] if parts else ""
        for raw_prefix in self.forbidden_path_prefixes:
            prefix_parts = [
                part.lower() for part in PurePosixPath(raw_prefix.strip("/")).parts if part
            ]
            if prefix_parts and parts[: len(prefix_parts)] == prefix_parts:
                return True
        if any(part in self.forbidden_path_parts for part in parts):
            return True
        if name in self.forbidden_path_names:
            return True
        return self.references_forbidden_material(path)

    def references_forbidden_material(self, text: str) -> bool:
        return self._references_forbidden_fragments(
            text,
            fragments=self.forbidden_text_fragments,
            absolute_paths=self.forbidden_absolute_paths,
        )

    def references_forbidden_web_material(self, text: str) -> bool:
        return self.references_forbidden_material(text) or self._references_forbidden_fragments(
            text,
            fragments=self.forbidden_web_fragments,
            absolute_paths=(),
        )

    def references_allowed_git_clone_material(self, text: str) -> bool:
        return self._references_forbidden_fragments(
            text,
            fragments=self.allowed_git_clone_fragments,
            absolute_paths=(),
        )

    def _references_forbidden_fragments(
        self,
        text: str,
        *,
        fragments: tuple[str, ...],
        absolute_paths: tuple[str, ...],
    ) -> bool:
        compact_fragments = tuple(
            "".join(ch for ch in fragment.lower() if ch.isalnum() or ch in "/._-")
            for fragment in fragments
        )
        alnum_fragments = tuple(
            "".join(ch for ch in fragment.lower() if ch.isalnum()) for fragment in fragments
        )
        normalized_absolute_paths = tuple(path.lower().rstrip("/") for path in absolute_paths)
        for variant in _policy_text_variants(text):
            lowered = variant.lower()
            compact = _compact_policy_text(variant)
            alnum = _alnum_policy_text(variant)
            if (
                any(fragment.lower() in lowered for fragment in fragments)
                or any(fragment and fragment in compact for fragment in compact_fragments)
                or any(fragment and fragment in alnum for fragment in alnum_fragments)
                or any(
                    path and _contains_absolute_policy_path(lowered, path)
                    for path in normalized_absolute_paths
                )
            ):
                return True
        return False

    def redacts_web_output_line(self, line: str) -> bool:
        stripped = line.strip()
        return bool(stripped and self.references_forbidden_web_material(stripped))

    def redact_web_output(self, text: str | None) -> str:
        if not text:
            return ""
        kept: list[str] = []
        omitted_lines = 0
        omitted_blocks = 0
        lines = text.splitlines()
        index = 0
        result_block_re = re.compile(r"^\s*\d+\.\s+")
        while index < len(lines):
            line = lines[index]
            if result_block_re.match(line):
                block = [line]
                index += 1
                while index < len(lines) and not result_block_re.match(lines[index]):
                    block.append(lines[index])
                    index += 1
                if any(self.redacts_web_output_line(block_line) for block_line in block):
                    omitted_blocks += 1
                    continue
                kept.extend(block)
                continue
            if self.redacts_web_output_line(line):
                omitted_lines += 1
                index += 1
                continue
            kept.append(line)
            index += 1
        if omitted_blocks:
            kept.append(f"[{omitted_blocks} restricted artifact result(s) omitted]")
        if omitted_lines:
            kept.append(f"[{omitted_lines} restricted artifact line(s) omitted]")
        if not kept:
            return ""
        suffix = "\n" if text.endswith("\n") else ""
        return "\n".join(kept) + suffix

    def redacts_output_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if self.references_forbidden_material(stripped):
            return True
        candidates = [stripped]
        candidates.extend(
            token.strip("'\"`")
            for token in stripped.replace(":", " ").split()
            if token.strip("'\"`")
        )
        for candidate in candidates:
            while candidate.startswith("./"):
                candidate = candidate[2:]
            if (
                candidate
                and not PurePosixPath(candidate).is_absolute()
                and self.rejects_relative_path(candidate)
            ):
                return True
        return False

    def redact_output(self, text: str | None) -> str:
        if not text:
            return ""
        kept: list[str] = []
        omitted = 0
        for line in text.splitlines():
            if self.redacts_output_line(line):
                omitted += 1
                continue
            kept.append(line)
        if omitted:
            kept.append(f"[{omitted} restricted artifact line(s) omitted]")
        if not kept:
            return ""
        suffix = "\n" if text.endswith("\n") else ""
        return "\n".join(kept) + suffix


def _tool_error(call: ToolCall, name: str, message: str) -> ToolResult:
    return ToolResult(tool_call_id=call.id, name=name, content=message, is_error=True)


def _clean_relative_path(path: object) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return None
    candidate = PurePosixPath(path.strip())
    if candidate.is_absolute():
        return None
    if any(part in ("", ".", "..") for part in candidate.parts):
        return None
    return candidate.as_posix()


def _quote(path: str) -> str:
    return shlex.quote(path)


def _shell_words(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return []


def _strip_shell_heredoc_bodies(command: str) -> str:
    stripped_lines: list[str] = []
    pending_delimiters: list[tuple[str, bool]] = []
    heredoc_delimiter: tuple[str, bool] | None = None
    heredoc_re = re.compile(
        r"<<(?P<strip_tabs>-)?\s*(?P<delim>'[^']+'|\"[^\"]+\"|[A-Za-z0-9_./-]+)"
    )

    for line in command.splitlines():
        if heredoc_delimiter is not None:
            delimiter, strip_tabs = heredoc_delimiter
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate == delimiter:
                heredoc_delimiter = pending_delimiters.pop(0) if pending_delimiters else None
            continue
        stripped_lines.append(line)
        for match in heredoc_re.finditer(line):
            raw_delimiter = match.group("delim")
            delimiter = raw_delimiter[1:-1] if raw_delimiter[:1] in {"'", '"'} else raw_delimiter
            pending_delimiters.append((delimiter, bool(match.group("strip_tabs"))))
        if pending_delimiters:
            heredoc_delimiter = pending_delimiters.pop(0)
    return "\n".join(stripped_lines)


def _head_pipe_preview_exit_status(*, command: str, exit_code: int, stdout: str) -> bool:
    if exit_code not in {23, 56, 141}:
        return False
    if not stdout.strip():
        return False
    return bool(re.search(r"\|\s*head(?:\s|$)", command))


def _activity_event_command(event: ActivityEvent) -> str:
    arguments = event.data.get("arguments")
    if isinstance(arguments, dict):
        command = arguments.get("command") or arguments.get("cmd")
        if isinstance(command, str) and command.strip():
            return command
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict):
        command = metadata.get("command") or metadata.get("cmd")
        if isinstance(command, str) and command.strip():
            return command
    return ""


def _activity_event_content(event: ActivityEvent) -> str:
    return str(event.data.get("content_preview") or event.data.get("content") or "")


def _activity_event_reports_missing_command(event: ActivityEvent) -> bool:
    if event.kind != "tool_call.completed":
        return False
    if str(event.data.get("name") or "") not in {"shell", "verify_work"}:
        return False
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict) and metadata.get("exit_code") == 127:
        return True
    content = _activity_event_content(event).lower()
    return (
        "command not found" in content
        or "failed (exit 127)" in content
        or "exit_code: 127" in content
    )


def _activity_event_checks_docker_runtime(event: ActivityEvent) -> bool:
    if event.kind != "tool_call.completed":
        return False
    if str(event.data.get("name") or "") not in {"shell", "verify_work"}:
        return False
    command = _activity_event_command(event).lower()
    if not command:
        return False
    if re.search(r"\b(?:command\s+-v|which|type)\s+docker\b", command):
        return True
    return bool(
        re.search(
            r"\bdocker\s+(?:--version|version|info|run|build|pull|images?|compose)\b",
            command,
        )
    )


_NETWORK_SHELL_TOOLS = frozenset(
    {
        "curl",
        "wget",
        "git",
        "gh",
        "http",
        "https",
        "fetch",
        "python",
        "python3",
        "node",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "uv",
        "pip",
        "pip3",
    }
)


def _shell_command_may_access_network(command: str) -> bool:
    if re.search(r"\b(?:https?|ssh)://|git@[^:\s]+:", command, flags=re.IGNORECASE):
        return True
    parts = _shell_words(command)
    if not parts:
        return False
    while parts and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", parts[0]):
        parts = parts[1:]
    if not parts:
        return False
    executable = PurePosixPath(parts[0]).name.lower()
    return executable in _NETWORK_SHELL_TOOLS


def _shell_command_requests_external_network(command: str) -> bool:
    if re.search(r"\b(?:https?|ssh)://|git@[^:\s]+:", command, flags=re.IGNORECASE):
        return True
    parts = _shell_words(command)
    if not parts:
        return False
    while parts and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", parts[0]):
        parts = parts[1:]
    if not parts:
        return False
    executable = PurePosixPath(parts[0]).name.lower()
    if executable in {"curl", "wget", "http", "https", "gh"}:
        return True
    if executable == "git":
        network_subcommands = {"clone", "fetch", "pull", "ls-remote", "submodule"}
        return any(part.lower() in network_subcommands for part in parts[1:])
    return False


def _shell_command_is_allowed_git_clone(
    command: str,
    policy: ExternalWorkspacePolicy,
) -> bool:
    if not policy.allowed_git_clone_fragments:
        return False
    if not policy.references_allowed_git_clone_material(command):
        return False
    parts = _shell_words(command)
    if not parts:
        return False
    parts = _shell_words_after_env(parts)
    if not parts:
        return False
    executable = PurePosixPath(parts[0]).name.lower()
    if executable in {"bash", "sh", "zsh"}:
        for index, part in enumerate(parts[:-1]):
            if part in {"-c", "-lc"}:
                return _shell_command_is_allowed_git_clone(parts[index + 1], policy)
    if executable == "docker":
        inner_command = _docker_run_inner_command(parts)
        if inner_command:
            return _shell_command_is_allowed_git_clone(inner_command, policy)

    segments: list[list[str]] = [[]]
    for part in parts:
        if part in {"&&", ";", "||"}:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(part)

    saw_allowed_git_operation = False
    for segment in segments:
        while segment and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", segment[0]):
            segment = segment[1:]
        if not segment:
            continue
        executable = PurePosixPath(segment[0].strip("()")).name.lower()
        if executable in {
            "cd",
            "ls",
            "mkdir",
            "pwd",
            "test",
            "true",
            "python",
            "python3",
            "pip",
            "pip3",
            "uv",
            "pytest",
            "npm",
            "npx",
            "pnpm",
            "yarn",
        }:
            continue
        if executable == "rm":
            if not _safe_git_setup_rm_segment(segment, policy):
                return False
            continue
        if executable != "git":
            return False
        lowered = [part.strip("()").lower() for part in segment[1:]]
        if "clone" in lowered:
            if not policy.references_allowed_git_clone_material(" ".join(segment)):
                return False
            saw_allowed_git_operation = True
            continue
        if "ls-remote" in lowered:
            if not policy.references_allowed_git_clone_material(" ".join(segment)):
                return False
            saw_allowed_git_operation = True
            continue
        if "fetch" in lowered:
            if not policy.references_allowed_git_clone_material(" ".join(segment)):
                return False
            saw_allowed_git_operation = True
            continue
        if any(
            part in {"checkout", "switch", "submodule", "status", "rev-parse", "remote"}
            for part in lowered
        ):
            continue
        return False
    return saw_allowed_git_operation


def _docker_run_inner_command(words: list[str]) -> str:
    if len(words) < 3:
        return ""
    if words[1] != "run":
        return ""
    options_with_values = {
        "--add-host",
        "--cidfile",
        "--cpus",
        "--dns",
        "--entrypoint",
        "--env",
        "--env-file",
        "--expose",
        "--hostname",
        "--label",
        "--link",
        "--log-driver",
        "--log-opt",
        "--memory",
        "--mount",
        "--name",
        "--network",
        "--platform",
        "--publish",
        "--user",
        "--volume",
        "--volumes-from",
        "--workdir",
        "-e",
        "-h",
        "-l",
        "-m",
        "-p",
        "-u",
        "-v",
        "-w",
    }
    index = 2
    while index < len(words):
        word = words[index]
        if word == "--":
            index += 1
            break
        if word.startswith("--"):
            option = word.split("=", 1)[0]
            if "=" not in word and option in options_with_values:
                index += 2
            else:
                index += 1
            continue
        if word.startswith("-") and word != "-":
            if word in options_with_values:
                index += 2
            else:
                index += 1
            continue
        break
    if index >= len(words):
        return ""
    inner_words = words[index + 1 :]
    if not inner_words:
        return ""
    return " ".join(shlex.quote(word) for word in inner_words)


def _shell_command_uses_no_network_container(command: str, *, image: str) -> bool:
    required_image = image.strip()
    if not required_image:
        return False
    words = _shell_words(command)
    if not words:
        return False
    for index, word in enumerate(words):
        if PurePosixPath(word).name.lower() != "docker":
            continue
        if index + 1 >= len(words) or words[index + 1] != "run":
            continue
        segment = words[index:]
        saw_no_network = False
        saw_image = False
        waiting_for_network_value = False
        for part in segment[2:]:
            if waiting_for_network_value:
                if part == "none":
                    saw_no_network = True
                waiting_for_network_value = False
                continue
            if part in {"--network", "--net"}:
                waiting_for_network_value = True
                continue
            if part in {"--network=none", "--net=none"}:
                saw_no_network = True
                continue
            if part == required_image:
                saw_image = True
        if saw_no_network and saw_image:
            return True
    return False


def _safe_git_setup_rm_segment(
    segment: list[str],
    policy: ExternalWorkspacePolicy,
) -> bool:
    targets = [
        part.strip("()")
        for part in segment[1:]
        if part.strip("()") and not part.strip("()").startswith("-")
    ]
    if not targets:
        return False
    for target in targets:
        target_without_trailing_contents = target
        if target_without_trailing_contents.endswith("/*"):
            target_without_trailing_contents = target_without_trailing_contents[:-2]
        normalized = _normal_path(target)
        normalized_without_trailing_contents = _normal_path(target_without_trailing_contents)
        parts = PurePosixPath(normalized).parts
        concrete_parts = PurePosixPath(normalized_without_trailing_contents).parts
        has_only_trailing_contents_wildcard = (
            target.endswith("/*")
            and normalized_without_trailing_contents
            and normalized_without_trailing_contents not in {".", "./", "*"}
            and not target_without_trailing_contents.startswith("/")
            and ".." not in concrete_parts
            and not any("*" in part for part in concrete_parts)
        )
        if (
            not normalized
            or normalized in {".", "./", "*"}
            or target.startswith("/")
            or ".." in parts
            or (any("*" in part for part in parts) and not has_only_trailing_contents_wildcard)
        ) or policy.rejects_relative_path(normalized_without_trailing_contents):
            return False
    return True


def _compact_policy_text(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch in "/._-")


def _alnum_policy_text(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _policy_text_variants(text: str) -> tuple[str, ...]:
    variants = [text]
    collapsed = re.sub(r"(['\"])\s*\+\s*\1", "", text)
    if collapsed != text:
        variants.append(collapsed)
    decoded = text
    for _ in range(3):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        variants.append(next_decoded)
        collapsed_decoded = re.sub(r"(['\"])\s*\+\s*\1", "", next_decoded)
        if collapsed_decoded != next_decoded:
            variants.append(collapsed_decoded)
        decoded = next_decoded
    return tuple(dict.fromkeys(variants))


def _contains_absolute_policy_path(text: str, path: str) -> bool:
    normalized = path.lower().rstrip("/")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    boundary = r"(?<![A-Za-z0-9_.-])"
    suffix = r"(?=$|[^A-Za-z0-9_.-])"
    return re.search(f"{boundary}{re.escape(normalized)}{suffix}", text) is not None


def _required_git_base_scan_command(_required_base: str) -> str:
    quoted_base = shlex.quote(_required_base)
    return (
        f"base={quoted_base}; "
        "tmp=$(mktemp) && "
        "trap 'rm -f \"$tmp\"' EXIT && "
        "find . -path './.git' -prune -o -type d -name .git -print > \"$tmp\" 2>/dev/null && "
        "while IFS= read -r gitdir; do "
        "repo=${gitdir%/.git}; "
        '[ "$repo" = "." ] && continue; '
        'head=$(git -C "$repo" rev-parse HEAD 2>/dev/null || true); '
        '[ -z "$head" ] && continue; '
        'base_head=$(git -C "$repo" rev-parse "$base" 2>/dev/null || true); '
        "base_state=missing; "
        'if [ -n "$base_head" ]; then '
        'if [ "$head" = "$base_head" ]; then base_state=head; '
        'elif git -C "$repo" merge-base --is-ancestor "$base_head" HEAD 2>/dev/null; '
        "then base_state=descendant; else base_state=unrelated; fi; "
        "fi; "
        'status=$(git -C "$repo" status --porcelain --untracked-files=all 2>/dev/null || true); '
        'dirty=no; [ -n "$status" ] && dirty=yes; '
        'modified=no; [ "$dirty" = "yes" ] && modified=yes; '
        '[ "$base_state" = "descendant" ] && modified=yes; '
        "printf 'repo\\t%s\\thead\\t%s\\tmodified\\t%s\\tbase\\t%s\\tdirty\\t%s\\n' "
        '"$repo" "$head" "$modified" "$base_state" "$dirty"; '
        'done < "$tmp"'
    )


def _policy_with_forbidden_logs(
    policy: ExternalWorkspacePolicy | None,
    logs_relative_path: str | None,
) -> ExternalWorkspacePolicy:
    base = policy or ExternalWorkspacePolicy()
    if not logs_relative_path:
        return base
    normalized = PurePosixPath(logs_relative_path.strip("/")).as_posix()
    if not normalized or normalized == ".":
        return base
    return ExternalWorkspacePolicy(
        forbidden_path_prefixes=(*base.forbidden_path_prefixes, normalized),
        forbidden_path_parts=base.forbidden_path_parts,
        forbidden_path_names=base.forbidden_path_names,
        forbidden_absolute_paths=base.forbidden_absolute_paths,
        forbidden_text_fragments=(*base.forbidden_text_fragments, normalized),
        forbidden_web_fragments=base.forbidden_web_fragments,
        allowed_git_clone_fragments=base.allowed_git_clone_fragments,
        required_git_base_commit=base.required_git_base_commit,
        required_no_network_verify_image=base.required_no_network_verify_image,
        allow_web_access=base.allow_web_access,
        block_root_filesystem_probe=base.block_root_filesystem_probe,
        refusal_message=base.refusal_message,
    )


def _attempts_root_filesystem_probe(command: str) -> bool:
    words = _shell_words(command)
    for index, word in enumerate(words[:-1]):
        if word not in _ROOT_PROBE_COMMANDS:
            continue
        for candidate in words[index + 1 :]:
            if candidate.startswith("-"):
                continue
            return candidate == "/"
    return False


def _shell_word_references_parent_directory(word: str) -> bool:
    normalized = word
    if normalized in {"..", "./.."}:
        return True
    if "../" in normalized or normalized.startswith("./../"):
        return True
    return "/../" in normalized or normalized.endswith("/..")


def _shell_word_references_absolute_host_path(word: str) -> bool:
    normalized = word
    if "://" in normalized:
        return False
    if normalized.startswith("/"):
        return True
    return bool(
        re.search(
            r"(?<![A-Za-z0-9_:/.-])/"
            r"(?:Users|private|var|tmp|etc|app|home|opt|usr|bin|sbin|lib|Library|Volumes)"
            r"(?:/|$)",
            normalized,
        )
    )


def _command_references_parent_directory(command: str) -> bool:
    command = _strip_shell_heredoc_bodies(command)
    words = _shell_words(command)
    if not words:
        return False
    words = _shell_words_after_env(words)
    if not words:
        return False
    executable = PurePosixPath(words[0]).name.lower()
    if executable in {"bash", "sh", "zsh"}:
        for index, word in enumerate(words[:-1]):
            if word in {"-c", "-lc"}:
                return _command_references_parent_directory(words[index + 1])
    return any(_shell_word_references_parent_directory(word) for word in words[1:])


def _command_references_absolute_host_path(command: str) -> bool:
    command = _strip_shell_heredoc_bodies(command)
    words = _shell_words(command)
    if not words:
        return False
    words = _shell_words_after_env(words)
    if not words:
        return False
    executable = PurePosixPath(words[0]).name.lower()
    if executable in {"bash", "sh", "zsh"}:
        for index, word in enumerate(words[:-1]):
            if word in {"-c", "-lc"}:
                return _command_references_absolute_host_path(words[index + 1])
    segments = _shell_command_segments(words)
    if len(segments) > 1:
        return any(_command_words_reference_absolute_host_path(segment) for segment in segments)
    return _command_words_reference_absolute_host_path(words)


def _shell_command_segments(words: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for word in words:
        if word in {"&&", ";", "||"}:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(word)
    return [segment for segment in segments if segment]


def _command_words_reference_absolute_host_path(words: list[str]) -> bool:
    words = _shell_words_after_env(words)
    if not words:
        return False
    executable = PurePosixPath(words[0]).name.lower()
    if executable == "cd":
        return False
    if executable == "docker":
        return _docker_command_references_absolute_host_path(words)
    return any(_shell_word_references_absolute_host_path(word) for word in words)


def _docker_mount_spec_references_absolute_host_path(spec: str) -> bool:
    if not spec:
        return False
    if "," in spec and "=" in spec:
        for part in spec.split(","):
            key, sep, value = part.partition("=")
            if sep and key.strip().lower() in {"source", "src"}:
                return _shell_word_references_absolute_host_path(value.strip())
        return False
    source = spec.split(":", 1)[0]
    if source.startswith(("$", ".", "~")):
        return False
    return _shell_word_references_absolute_host_path(source)


def _docker_command_references_absolute_host_path(words: list[str]) -> bool:
    for index, word in enumerate(words[1:], start=1):
        if word in {"-v", "--volume"} and index + 1 < len(words):
            if _docker_mount_spec_references_absolute_host_path(words[index + 1]):
                return True
            continue
        if word.startswith("--volume="):
            if _docker_mount_spec_references_absolute_host_path(word.split("=", 1)[1]):
                return True
            continue
        if word.startswith("-v") and len(word) > 2:
            if _docker_mount_spec_references_absolute_host_path(word[2:]):
                return True
            continue
        if word == "--mount" and index + 1 < len(words):
            if _docker_mount_spec_references_absolute_host_path(words[index + 1]):
                return True
            continue
        if word.startswith("--mount="):
            if _docker_mount_spec_references_absolute_host_path(word.split("=", 1)[1]):
                return True
            continue
        if word in {"-f", "--file"} and index + 1 < len(words):
            if _shell_word_references_absolute_host_path(words[index + 1]):
                return True
            continue
        if word.startswith("--file="):
            if _shell_word_references_absolute_host_path(word.split("=", 1)[1]):
                return True
            continue
    return False


def _shell_words_after_env(words: list[str]) -> list[str]:
    index = 0
    if words and PurePosixPath(words[0]).name == "env":
        index = 1
    while index < len(words):
        word = words[index]
        if word.startswith("-"):
            index += 1
            continue
        if "=" in word and not word.startswith("="):
            name = word.split("=", 1)[0]
            if name.replace("_", "").isalnum():
                index += 1
                continue
        break
    return words[index:]


def _is_shell_inspection_command(command: str) -> bool:
    words = _shell_words_after_env(_shell_words(command))
    if not words:
        return True
    executable = PurePosixPath(words[0]).name.lower()
    if executable in {"bash", "sh", "zsh"}:
        for index, word in enumerate(words[:-1]):
            if word in {"-c", "-lc"}:
                return _is_shell_inspection_command(words[index + 1])
        return False
    if executable == "git":
        for word in words[1:]:
            if word.startswith("-"):
                continue
            return word.lower() in _GIT_INSPECTION_COMMANDS
        return True
    return executable in _SHELL_INSPECTION_COMMANDS


def _shell_command_is_environment_probe(command: str) -> bool:
    words = _shell_words_after_env(_shell_words(command))
    if not words:
        return False
    executable = PurePosixPath(words[0]).name.lower()
    if executable in {"bash", "sh", "zsh"}:
        for index, word in enumerate(words[:-1]):
            if word in {"-c", "-lc"}:
                return _shell_command_is_environment_probe(words[index + 1])
        return False
    segments = _shell_command_segments(words)
    if not segments:
        return False
    return all(_shell_segment_is_environment_probe(segment) for segment in segments)


def _shell_segment_is_environment_probe(words: list[str]) -> bool:
    words = _shell_words_after_env(words)
    if not words:
        return True
    executable = PurePosixPath(words[0]).name.lower()
    if executable in {"true", ":"}:
        return True
    if executable == "cd":
        return len(words) <= 2 and not (
            len(words) == 2 and _shell_word_references_absolute_host_path(words[1])
        )
    if executable == "command":
        return (
            len(words) >= 3
            and words[1] in {"-v", "-V"}
            and not any(word.startswith("-") for word in words[2:])
        )
    if executable in {"which", "type"}:
        return len(words) >= 2
    if executable == "docker":
        return _docker_segment_is_environment_probe(words)
    if executable == "git":
        for word in words[1:]:
            if word.startswith("-"):
                continue
            return word.lower() in _GIT_INSPECTION_COMMANDS
        return True
    if len(words) >= 2 and words[1] in {"--version", "-version", "-V", "-v", "version"}:
        return True
    return executable in _SHELL_INSPECTION_COMMANDS


def _docker_segment_is_environment_probe(words: list[str]) -> bool:
    if len(words) < 2:
        return False
    subcommand = words[1].lower()
    if subcommand in {"--version", "version", "info", "images"}:
        return True
    if subcommand == "context":
        return len(words) >= 3 and words[2].lower() in {"inspect", "ls", "show"}
    if subcommand == "image":
        return len(words) >= 3 and words[2].lower() in {"inspect", "ls"}
    if subcommand == "system":
        return len(words) >= 3 and words[2].lower() == "df"
    return False


def _shell_command_requests_setup(command: str) -> bool:
    words = _shell_words_after_env(_shell_words(command))
    if not words:
        return False
    executable = PurePosixPath(words[0]).name.lower()
    if executable in {"bash", "sh", "zsh"}:
        for index, word in enumerate(words[:-1]):
            if word in {"-c", "-lc"}:
                return _shell_command_requests_setup(words[index + 1])
        return False

    segments: list[list[str]] = [[]]
    for word in words:
        if word in {"&&", ";", "||"}:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(word)

    for segment in segments:
        segment = _shell_words_after_env(segment)
        if not segment:
            continue
        executable = PurePosixPath(segment[0]).name.lower()
        if executable == "cd":
            continue
        lowered = [part.strip("()").lower() for part in segment[1:]]
        if executable not in _SETUP_COMMAND_EXECUTABLES:
            continue
        if "ensurepip" in lowered:
            return True
        if executable in {"python", "python3"} and "-m" in lowered:
            module_indexes = [index + 1 for index, part in enumerate(lowered[:-1]) if part == "-m"]
            if any(lowered[index] in {"pip", "ensurepip"} for index in module_indexes):
                return any(word in _SETUP_COMMAND_WORDS for word in lowered)
        if any(word in _SETUP_COMMAND_WORDS for word in lowered):
            return True
    return False


def _porcelain_paths(status: str) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for raw_line in status.splitlines():
        if not raw_line.strip():
            continue
        code = raw_line[:2]
        path = raw_line[3:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[-1].strip()
        if not path:
            continue
        paths.append((code, path))
    return paths


def _workspace_source_change_status(
    status: str,
    *,
    tracked_paths: set[str] | None = None,
    baseline_untracked_paths: set[str] | None = None,
    ignored_paths: set[str] | None = None,
) -> tuple[bool, list[str], list[str]]:
    source_paths: list[str] = []
    scratch_paths: list[str] = []
    for code, path in _porcelain_paths(status):
        if code == "??" and path in (baseline_untracked_paths or set()):
            continue
        if _path_is_ignored(path, ignored_paths or set()):
            continue
        if _path_is_generated_artifact(path):
            continue
        if _path_counts_as_source_change(path, status_code=code, tracked_paths=tracked_paths):
            source_paths.append(path)
        elif _path_counts_as_test_change(path, status_code=code, tracked_paths=tracked_paths):
            continue
        else:
            scratch_paths.append(path)
    return bool(source_paths), source_paths, scratch_paths


def _path_is_test_only(path: str) -> bool:
    parts = [part for part in path.strip("/").split("/") if part]
    if not parts:
        return False
    first = parts[0].lower()
    name = parts[-1].lower()
    test_prefix_suffixes = {".py", ".go", ".rs", ".ts", ".tsx", ".js", ".jsx"}
    return (
        first in _TEST_DIR_NAMES
        or bool(_TEST_DIR_NAMES.intersection({part.lower() for part in parts}))
        or (name.startswith("test_") and PurePosixPath(name).suffix in test_prefix_suffixes)
        or name.endswith(("_test.py", "_test.go", "_test.rs", ".test.ts", ".test.js"))
        or ".spec." in name
    )


def _path_is_test_fixture(path: str) -> bool:
    parts = [part.lower() for part in path.strip("/").split("/") if part]
    return bool(_TEST_FIXTURE_DIR_NAMES.intersection(parts))


def _path_is_scratch_artifact(path: str) -> bool:
    name = PurePosixPath(path.strip("/")).name.lower()
    stem = PurePosixPath(name).stem
    documentation_suffixes = {".adoc", ".diff", ".md", ".patch", ".rst", ".txt"}
    placeholder_names = {".gitkeep", ".keep", ".placeholder"}
    scratch_prefixes = (
        "debug",
        "experiment",
        "probe",
        "repro",
        "scratch",
        "sandbox",
        "temp",
        "tmp",
    )
    return (
        name in placeholder_names
        or name.endswith("~")
        or name.endswith((".bak", ".backup", ".orig", ".tmp", ".temp"))
        or name.endswith(("_bak", "_backup", "_orig", "_tmp", "_temp"))
        or PurePosixPath(name).suffix.lower() in documentation_suffixes
        or any(stem == prefix or stem.startswith(f"{prefix}_") for prefix in scratch_prefixes)
    )


def _path_is_generated_artifact(path: str) -> bool:
    parts = [part.lower() for part in path.strip("/").split("/") if part]
    name = parts[-1] if parts else ""
    return (
        "__pycache__" in parts
        or ".harness-home" in parts
        or ".pytest_cache" in parts
        or ".mypy_cache" in parts
        or ".ruff_cache" in parts
        or ".tox" in parts
        or name.endswith((".pyc", ".pyo"))
    )


def _workspace_test_change_paths(
    status: str,
    *,
    tracked_paths: set[str] | None = None,
    baseline_untracked_paths: set[str] | None = None,
    ignored_paths: set[str] | None = None,
) -> list[str]:
    return [
        path
        for code, path in _porcelain_paths(status)
        if not (code == "??" and path in (baseline_untracked_paths or set()))
        and not _path_is_ignored(path, ignored_paths or set())
        and not _path_is_generated_artifact(path)
        and _path_counts_as_test_change(path, status_code=code, tracked_paths=tracked_paths)
    ]


def _merge_workspace_status(
    root_status: str,
    nested_status: str,
    *,
    nested_roots: set[str] | None = None,
) -> str:
    nested_repo_paths = {
        _normal_path(path).rstrip("/")
        for _code, path in _porcelain_paths(nested_status)
        if _normal_path(path).rstrip("/")
    }
    nested_repo_paths.update(
        _normal_path(path).rstrip("/")
        for path in nested_roots or set()
        if _normal_path(path).rstrip("/")
    )
    lines: list[str] = []
    for code, path in _porcelain_paths(root_status):
        normalized = _normal_path(path).rstrip("/")
        if any(
            nested_path == normalized
            or nested_path.startswith(f"{normalized}/")
            or normalized.startswith(f"{nested_path}/")
            for nested_path in nested_repo_paths
        ):
            continue
        lines.append(f"{code} {path}")
    for code, path in _porcelain_paths(nested_status):
        lines.append(f"{code} {path}")
    return "\n".join(lines)


def _workspace_relevant_status_entries(status: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (code, path)
        for code, path in _porcelain_paths(status)
        if not _path_is_generated_artifact(path)
    )


def _workspace_fingerprint_text(result: Any) -> str | None:
    return result.stdout if getattr(result, "return_code", 1) == 0 else None


def _fingerprint_has_untracked_test_path(fingerprint: str | None) -> bool:
    if not fingerprint:
        return False
    for line in fingerprint.splitlines():
        if not line.startswith("untracked "):
            continue
        path = line.removeprefix("untracked ").strip()
        if _path_counts_as_test_change(path, status_code="??"):
            return True
    return False


def _normal_path(path: str) -> str:
    normalized = path.strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return PurePosixPath(normalized).as_posix()


def _path_is_ignored(path: str, ignored_paths: set[str]) -> bool:
    normalized = _normal_path(path).rstrip("/")
    for ignored_path in ignored_paths:
        ignored = _normal_path(ignored_path).rstrip("/")
        if normalized == ignored or normalized.startswith(f"{ignored}/"):
            return True
    return False


_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(?P<delimiter>[A-Za-z_][A-Za-z0-9_]*)['\"]?")
_SHELL_COMMAND_RUNNERS = {"bash", "sh", "dash", "zsh", "ksh"}


def _command_without_heredoc_bodies(command: str) -> str:
    lines = command.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        match = _HEREDOC_RE.search(line)
        if not match:
            index += 1
            continue
        delimiter = match.group("delimiter")
        index += 1
        while index < len(lines) and lines[index].strip() != delimiter:
            index += 1
        if index < len(lines):
            index += 1
    return "\n".join(kept)


def _relative_command_path(path: str) -> str | None:
    candidate = path.strip().strip("'\"")
    if not candidate or candidate.startswith(("/", "~", "$")) or "://" in candidate:
        return None
    normalized = _normal_path(candidate)
    if not normalized:
        return None
    if any(part == ".." for part in PurePosixPath(normalized).parts):
        return None
    return normalized


def _join_command_prefix(prefix: str, path: str) -> str | None:
    relative = _relative_command_path(path)
    if relative is None:
        return None
    normalized_prefix = _normal_path(prefix) if prefix else ""
    if relative in {".", "./"}:
        return normalized_prefix or "."
    if not normalized_prefix or normalized_prefix == ".":
        return relative
    return _normal_path(f"{normalized_prefix.rstrip('/')}/{relative}")


def _shell_body_commands(words: list[str]) -> list[str]:
    bodies: list[str] = []
    for index, word in enumerate(words[:-1]):
        executable = PurePosixPath(word).name.lower()
        if executable not in _SHELL_COMMAND_RUNNERS:
            continue
        for flag_index in range(index + 1, len(words) - 1):
            flag = words[flag_index]
            if not flag.startswith("-"):
                continue
            if flag == "-c" or flag.endswith("c"):
                bodies.append(words[flag_index + 1])
                break
    return bodies


def _command_variants_with_prefix(command: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def visit(current_command: str, prefix: str) -> None:
        key = (current_command, prefix)
        if key in seen:
            return
        seen.add(key)
        variants.append(key)
        words = _shell_words(_command_without_heredoc_bodies(current_command))
        current_prefix = prefix
        for segment in _shell_command_segments(words):
            segment = _shell_words_after_env(segment)
            if not segment:
                continue
            executable = PurePosixPath(segment[0]).name.lower()
            if executable == "cd":
                if len(segment) >= 2:
                    joined = _join_command_prefix(current_prefix, segment[1])
                    if joined is not None:
                        current_prefix = "" if joined == "." else joined
                continue
            for body in _shell_body_commands(segment):
                visit(body, current_prefix)

    visit(command, "")
    return variants


def _command_segment_path_tokens(words: list[str], prefix: str) -> list[str]:
    paths: list[str] = []
    expect_command = True
    for word in words:
        cleaned = word.strip().strip("'\"")
        if not cleaned or cleaned.startswith("-"):
            continue
        if expect_command and "=" in cleaned and not cleaned.startswith("-"):
            continue
        if expect_command:
            expect_command = False
            continue
        if cleaned == "test":
            continue
        if "::" in cleaned:
            cleaned = cleaned.split("::", 1)[0]
        if cleaned in {".", "./", "./..."}:
            joined = _join_command_prefix(prefix, ".")
            paths.append(joined or ".")
            continue
        cleaned = cleaned.removesuffix("/...")
        if cleaned.startswith("./") or "/" in cleaned or _path_is_test_only(cleaned):
            joined = _join_command_prefix(prefix, cleaned)
            if joined is not None:
                paths.append(joined)
    return paths


def _command_path_tokens(command: str) -> list[str]:
    paths: list[str] = []
    for variant, prefix in _command_variants_with_prefix(command):
        words = _shell_words(_command_without_heredoc_bodies(variant))
        current_prefix = prefix
        for segment in _shell_command_segments(words):
            segment = _shell_words_after_env(segment)
            if not segment:
                continue
            executable = PurePosixPath(segment[0]).name.lower()
            if executable == "cd":
                if len(segment) >= 2:
                    joined = _join_command_prefix(current_prefix, segment[1])
                    if joined is not None:
                        current_prefix = "" if joined == "." else joined
                continue
            if executable == "docker":
                inner_command = _docker_run_inner_command(segment)
                if inner_command:
                    paths.extend(_command_path_tokens(inner_command))
                continue
            paths.extend(_command_segment_path_tokens(segment, current_prefix))
    return list(dict.fromkeys(paths))


def _is_broad_test_command(command: str) -> bool:
    return any(_test_segment_is_broad(segment) for segment, _raw in _command_test_segments(command))


def _test_segment_is_broad(segment: list[str]) -> bool:
    words = [PurePosixPath(word).name.lower() for word in segment]
    joined = " ".join(words)
    if any(
        pattern in joined
        for pattern in (
            "go test",
            "cargo test",
            "npm test",
            "pnpm test",
            "yarn test",
            "bun test",
            "node --test",
            "make test",
            "mix test",
            "python -m pytest",
            "python3 -m pytest",
            "uv run pytest",
        )
    ):
        return True
    test_runners = {
        "pytest",
        "tox",
        "nox",
        "vitest",
        "jest",
        "mocha",
        "rspec",
        "rebar3",
    }
    return any(word in test_runners for word in words)


def _command_test_segments(command: str) -> list[tuple[list[str], list[str]]]:
    segments: list[tuple[list[str], list[str]]] = []
    for variant, _prefix in _command_variants_with_prefix(command):
        words = _shell_words(_command_without_heredoc_bodies(variant))
        for raw_segment in _shell_command_segments(words):
            segment = _shell_words_after_env(raw_segment)
            if not segment:
                continue
            executable = PurePosixPath(segment[0]).name.lower()
            if executable == "docker":
                inner_command = _docker_run_inner_command(segment)
                if inner_command:
                    segments.extend(_command_test_segments(inner_command))
                continue
            if executable == "cd":
                continue
            if _test_segment_is_broad(segment):
                segments.append((segment, raw_segment))
    return segments


def _command_uses_test_selector(command: str) -> bool:
    selector_flags = {
        "-k",
        "-m",
        "--run",
        "-run",
        "-bench",
        "-t",
        "-g",
        "--grep",
        "--testnamepattern",
        "--test-name-pattern",
        "--filter",
        "--ignore",
        "--ignore-glob",
        "--deselect",
        "--exclude",
        "--exclude-from",
        "--collect-only",
        "--co",
        "--setup-only",
        "--setup-plan",
        "--fixtures",
        "--markers",
        "--version",
        "--help",
        "-h",
        "--dry-run",
        "--list",
        "--list-tests",
        "--listtests",
        "-list",
    }
    for segment, raw_segment in _command_test_segments(command):
        words = raw_segment if raw_segment else segment
        for index, word in enumerate(words):
            lowered = word.lower()
            if index == 0 and PurePosixPath(word).name.lower() == "env":
                continue
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word):
                value = word.split("=", 1)[1].strip("'\"")
                candidates = [value, *_shell_words(value)]
                if any(
                    candidate.lower() in selector_flags
                    or any(candidate.lower().startswith(f"{flag}=") for flag in selector_flags)
                    for candidate in candidates
                ):
                    return True
                continue
            lowered = lowered.strip("'\"")
            if "::" in lowered:
                node_path = lowered.split("::", 1)[0]
                if "/" in node_path or _path_is_test_only(node_path):
                    return True
                continue
            if lowered == "-m" and _is_module_invocation_flag(words, index):
                continue
            if lowered in selector_flags:
                return True
            if any(lowered.startswith(f"{flag}=") for flag in selector_flags):
                return True
            if "=" in lowered and not lowered.startswith("-"):
                value = lowered.split("=", 1)[1].strip("'\"")
                candidates = [value, *value.split()]
                if any(
                    candidate in selector_flags
                    or any(candidate.startswith(f"{flag}=") for flag in selector_flags)
                    for candidate in candidates
                ):
                    return True
    return False


def _is_module_invocation_flag(words: list[str], index: int) -> bool:
    if index <= 0 or index + 1 >= len(words):
        return False
    executable = PurePosixPath(words[index - 1]).name.lower()
    module = words[index + 1].lower()
    if module not in {"pytest", "unittest", "nose", "nose2"}:
        return False
    return executable == "python" or executable.startswith("python")


def _command_invokes_opaque_test_wrapper(command: str) -> bool:
    words = [
        PurePosixPath(word).name.lower()
        for word in _shell_words(_command_without_heredoc_bodies(command))
    ]
    joined = " ".join(words)
    return "make test" in joined


def _pytest_executable_failure_hint(command: str, stdout: str, stderr: str) -> str:
    words = _shell_words(_command_without_heredoc_bodies(command))
    if not words or PurePosixPath(words[0]).name.lower() != "pytest":
        return ""
    output = f"{stdout}\n{stderr}"
    if (
        "pytestconfigwarning" not in output.lower()
        and "unknown config option" not in output.lower()
        and "internalerror>" not in output.lower()
    ):
        return ""
    return (
        "[verification hint] The bare `pytest` executable failed during pytest startup. "
        "Inspect the project test setup and try an equivalent in-workspace command such "
        "as `python -m pytest ...` or a repository-provided test runner that executes "
        "the changed tests."
    )


def _command_local_script_paths(command: str | None) -> list[str]:
    if not command:
        return []
    words = _shell_words(_command_without_heredoc_bodies(command))
    paths: list[str] = []
    shell_runners = {"sh", "bash", "dash", "zsh", "ksh"}
    for index, word in enumerate(words):
        if not word or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word):
            continue
        name = PurePosixPath(word).name.lower()
        if name in shell_runners:
            for candidate in words[index + 1 :]:
                if candidate.startswith("-"):
                    continue
                paths.append(_normal_path(candidate))
                break
            continue
        if word.startswith("./") or "/" in word:
            suffix = PurePosixPath(word).suffix.lower()
            if suffix in {".sh", ".bash", ".zsh", ".py", ".js", ".mjs", ".cjs", ".ts"}:
                paths.append(_normal_path(word))
    return list(dict.fromkeys(path for path in paths if path))


def _command_runs_changed_test_script(command: str | None, test_paths: list[str]) -> bool:
    if not command or not test_paths:
        return False
    words = _shell_words(_command_without_heredoc_bodies(command))
    if not words:
        return False
    shell_runners = {"sh", "bash", "dash", "zsh", "ksh"}
    changed_candidates: set[str] = set()
    for path in test_paths:
        changed_candidates.update(_test_path_reference_candidates(path))
    for index, word in enumerate(words):
        if not word or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word):
            continue
        name = PurePosixPath(word).name.lower()
        if name == "docker":
            inner_command = _docker_run_inner_command(words[index:])
            if inner_command and _command_runs_changed_test_script(inner_command, test_paths):
                return True
            continue
        if name in shell_runners:
            for candidate in words[index + 1 :]:
                if candidate.startswith("-"):
                    continue
                normalized = _normal_path(candidate)
                if normalized in changed_candidates:
                    return True
                break
            continue
        if word.startswith("./"):
            normalized = _normal_path(word)
            if normalized in changed_candidates:
                return True
    return False


def _test_path_reference_candidates(path: str) -> set[str]:
    normalized = _normal_path(path)
    if not normalized:
        return set()
    pure = PurePosixPath(normalized)
    without_suffix = pure.with_suffix("").as_posix() if pure.suffix else normalized
    candidates = {
        normalized,
        without_suffix,
        pure.name,
        pure.stem,
    }
    if "/" in without_suffix:
        candidates.add(without_suffix.replace("/", "."))
    parts = pure.parts
    for index in range(1, len(parts)):
        suffix = PurePosixPath(*parts[index:]).as_posix()
        if _path_is_test_only(suffix):
            candidates.add(suffix)
            suffix_without_ext = (
                PurePosixPath(suffix).with_suffix("").as_posix()
                if PurePosixPath(suffix).suffix
                else suffix
            )
            candidates.add(suffix_without_ext)
    return {candidate for candidate in candidates if candidate and candidate != "."}


def _diff_references_test_paths(diff_text: str, test_paths: list[str]) -> bool:
    if not diff_text or not test_paths:
        return False
    for path in test_paths:
        if any(candidate in diff_text for candidate in _test_path_reference_candidates(path)):
            return True
    return False


def _verification_command_covers_test_changes(
    command: str | None,
    test_paths: list[str],
    *,
    untracked_test_paths: list[str] | None = None,
    runner_wires_untracked_tests: bool = False,
    runner_wires_changed_tests: bool = False,
) -> bool:
    if not command or not test_paths:
        return False
    if runner_wires_changed_tests:
        return True
    if _command_uses_test_selector(command):
        return False
    if _command_runs_changed_test_script(command, test_paths):
        return True
    changed = [_normal_path(path) for path in test_paths]
    if any(_path_is_test_fixture(path) for path in changed):
        return False
    untracked = {_normal_path(path) for path in untracked_test_paths or []}
    tracked_changed = [path for path in changed if path not in untracked]
    targets = _command_path_tokens(command)
    if not targets:
        if untracked and _command_invokes_opaque_test_wrapper(command):
            return bool(
                (tracked_changed or runner_wires_untracked_tests)
                and _is_broad_test_command(command)
            )
        return _is_broad_test_command(command)
    for target in targets:
        if target in {".", "./"}:
            return _is_broad_test_command(command)
        for path in changed:
            changed_candidates = _test_path_reference_candidates(path)
            if (
                path == target
                or path.startswith(f"{target.rstrip('/')}/")
                or target in changed_candidates
                or any(
                    candidate.startswith(f"{target.rstrip('/')}/")
                    for candidate in changed_candidates
                )
            ):
                return _is_broad_test_command(command)
    return False


def _diff_references_untracked_test_paths(
    diff_text: str,
    untracked_test_paths: list[str],
) -> bool:
    return _diff_references_test_paths(diff_text, untracked_test_paths)


def _root_source_suffixes(tracked_paths: set[str] | None) -> set[str]:
    suffixes: set[str] = set()
    for tracked_path in tracked_paths or set():
        if "/" in tracked_path.strip("/") or _path_is_test_only(tracked_path):
            continue
        suffix = PurePosixPath(tracked_path).suffix.lower()
        if suffix:
            suffixes.add(suffix)
    return suffixes


def _tracked_test_parents(tracked_paths: set[str] | None) -> set[str]:
    parents: set[str] = set()
    for tracked_path in tracked_paths or set():
        if not _path_is_test_only(tracked_path):
            continue
        parts = [part for part in tracked_path.strip("/").split("/") if part]
        for index, part in enumerate(parts[:-1]):
            if part.lower() in _TEST_DIR_NAMES:
                parents.add("/".join(parts[: index + 1]))
                break
        else:
            parent = PurePosixPath(tracked_path.strip("/")).parent.as_posix()
            parents.add("." if parent == "." else parent)
    return parents


def _path_counts_as_test_change(
    path: object,
    *,
    status_code: str | None = None,
    tracked_paths: set[str] | None = None,
) -> bool:
    if not isinstance(path, str) or not _path_is_test_only(path):
        return False
    if status_code is not None and any(code in status_code for code in ("D", "R")):
        return False
    if status_code != "??" or not tracked_paths:
        return True
    parents = _tracked_test_parents(tracked_paths)
    if not parents:
        return True
    normalized = _normal_path(path)
    parent = PurePosixPath(normalized).parent.as_posix()
    if parent == ".":
        return "." in parents
    return any(
        parent == test_parent or parent.startswith(f"{test_parent}/")
        for test_parent in parents
        if test_parent != "."
    )


def _path_counts_as_source_change(
    path: object,
    *,
    status_code: str | None = None,
    tracked_paths: set[str] | None = None,
) -> bool:
    if (
        not isinstance(path, str)
        or _path_counts_as_test_change(path, status_code=status_code, tracked_paths=tracked_paths)
        or _path_is_test_only(path)
        or _path_is_scratch_artifact(path)
    ):
        return False
    if status_code == "??" and path.rstrip().endswith("/"):
        return True
    normalized = path.strip("/")
    if "/" in normalized:
        return True
    if status_code is None:
        return False
    if status_code.strip("? ") != "":
        return True
    suffix = PurePosixPath(normalized).suffix.lower()
    return bool(suffix and suffix in _root_source_suffixes(tracked_paths))


def _tool_result_counts_as_source_change(result: ToolResult) -> bool:
    if result.is_error:
        return False
    if result.name == "apply_patch":
        return True
    if result.name in {"edit_file", "write_file"}:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        if _path_counts_as_source_change(metadata.get("path")):
            return True
        return (
            isinstance(metadata.get("path"), str)
            and metadata.get("overwrite") is True
            and not _path_is_test_only(metadata["path"])
        )
    return False


def _tool_result_changes_workspace(result: ToolResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if metadata.get("workspace_changed") is True:
        return True
    if result.is_error:
        return False
    if result.name == "apply_patch":
        return str(result.content).startswith("applied patch")
    if result.name in {"edit_file", "write_file"}:
        return "path" in (result.metadata if isinstance(result.metadata, dict) else {})
    if result.name == "shell":
        return metadata.get("workspace_changed") is True
    return False


class _RemoteToolBase:
    approval = "auto"
    effect_scope = "workspace_durable"
    phases = ("*",)

    def __init__(
        self,
        environment: Any,
        *,
        workdir: str,
        max_output_bytes: int = 64 * 1024,
        state: RemoteWorkspaceState | None = None,
        policy: ExternalWorkspacePolicy | None = None,
    ) -> None:
        self.environment = environment
        self.workdir = workdir
        self.max_output_bytes = max_output_bytes
        self.state = state or RemoteWorkspaceState()
        self.policy = policy or ExternalWorkspacePolicy()

    def _mark_workspace_changed(self) -> None:
        self.state.mutation_version += 1
        self.state.read_only_calls_since_change = 0

    def _mark_external_environment_changed(self) -> None:
        self.state.environment_version += 1
        self.state.read_only_calls_since_change = 0

    def _state_version(self) -> tuple[int, int]:
        return (self.state.mutation_version, self.state.environment_version)

    def _record_read_only_call(self) -> None:
        self.state.read_only_calls_since_change += 1

    async def _exec(self, command: str, *, timeout_sec: int = 120) -> Any:
        return await self.environment.exec(command, cwd=self.workdir, timeout_sec=timeout_sec)

    def _truncate(self, text: str | None) -> tuple[str, bool]:
        if not text:
            return "", False
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= self.max_output_bytes:
            return text, False
        return encoded[: self.max_output_bytes].decode("utf-8", errors="replace"), True


@dataclass
class RemoteWorkspaceState:
    mutation_version: int = 0
    environment_version: int = 0
    read_only_calls_since_change: int = 0


class RemoteReadFileTool(_RemoteToolBase):
    name = "read_file"
    description = "Read a UTF-8 text file from the external workspace."
    effect_scope = "read_only"
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to workspace."}},
        "required": ["path"],
    }

    async def __call__(self, call: ToolCall) -> ToolResult:
        path = _clean_relative_path(call.arguments.get("path"))
        if path is None:
            return _tool_error(call, self.name, "path must be a relative workspace path")
        if self.policy.rejects_relative_path(path):
            return _tool_error(
                call,
                self.name,
                self.policy.refusal_message,
            )
        self._record_read_only_call()
        q_path = _quote(path)
        command = (
            f"if [ ! -e {q_path} ]; then echo 'path does not exist' >&2; exit 2; fi; "
            f"if [ ! -f {q_path} ]; then echo 'not a regular file' >&2; exit 3; fi; "
            f"size=$(wc -c < {q_path}); "
            f'if [ "$size" -gt {self.max_output_bytes} ]; then '
            f'echo "file too large: $size bytes; use read_file_range" >&2; exit 4; fi; '
            f"cat {q_path}"
        )
        result = await self._exec(command, timeout_sec=30)
        if result.return_code != 0:
            stderr, _ = self._truncate(result.stderr)
            if result.return_code == 4 and not stderr:
                stderr = (
                    f"file too large to read in full: {path}; use read_file_range with "
                    "a focused start_line and line_count"
                )
            return _tool_error(
                call, self.name, stderr or f"read failed with exit {result.return_code}"
            )
        stdout, truncated = self._truncate(result.stdout)
        suffix = "\n[truncated]" if truncated else ""
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"{stdout}{suffix}",
            metadata={"path": path, "exit_code": result.return_code},
        )


class RemoteReadFileRangeTool(_RemoteToolBase):
    name = "read_file_range"
    description = "Read a line range from any UTF-8 text file in the external workspace."
    effect_scope = "read_only"
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to workspace."},
            "start_line": {
                "type": "integer",
                "description": "1-based line number to start reading from.",
            },
            "line_count": {
                "type": "integer",
                "description": "Number of lines to read, capped at 500.",
            },
        },
        "required": ["path", "start_line"],
    }

    async def __call__(self, call: ToolCall) -> ToolResult:
        path = _clean_relative_path(call.arguments.get("path"))
        if path is None:
            return _tool_error(call, self.name, "path must be a relative workspace path")
        if self.policy.rejects_relative_path(path):
            return _tool_error(
                call,
                self.name,
                self.policy.refusal_message,
            )
        self._record_read_only_call()
        try:
            start_line = max(1, int(call.arguments.get("start_line", 1)))
        except (TypeError, ValueError):
            return _tool_error(call, self.name, "start_line must be an integer")
        try:
            line_count = max(1, min(int(call.arguments.get("line_count", 200)), 500))
        except (TypeError, ValueError):
            line_count = 200
        end_line = start_line + line_count - 1
        q_path = _quote(path)
        command = (
            f"if [ ! -e {q_path} ]; then echo 'path does not exist' >&2; exit 2; fi; "
            f"if [ ! -f {q_path} ]; then echo 'not a regular file' >&2; exit 3; fi; "
            f"awk 'NR>={start_line} && NR<={end_line} "
            '{ printf "%d:%s\\n", NR, $0 }\' '
            f"{q_path}"
        )
        result = await self._exec(command, timeout_sec=30)
        if result.return_code != 0:
            stderr, _ = self._truncate(result.stderr)
            return _tool_error(
                call, self.name, stderr or f"read failed with exit {result.return_code}"
            )
        stdout, truncated = self._truncate(result.stdout)
        suffix = "\n[truncated]" if truncated else ""
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"{stdout}{suffix}",
            metadata={
                "path": path,
                "start_line": start_line,
                "line_count": line_count,
                "exit_code": result.return_code,
            },
        )


class RemoteListDirTool(_RemoteToolBase):
    name = "list_dir"
    description = "List entries in a external workspace directory."
    effect_scope = "read_only"
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory relative to workspace."}
        },
        "required": [],
    }

    async def __call__(self, call: ToolCall) -> ToolResult:
        raw_path = call.arguments.get("path", ".")
        path = "." if raw_path in (None, "", ".") else _clean_relative_path(raw_path)
        if path is None:
            return _tool_error(call, self.name, "path must be a relative workspace path")
        if self.policy.rejects_relative_path(path):
            return _tool_error(
                call,
                self.name,
                self.policy.refusal_message,
            )
        self._record_read_only_call()
        q_path = _quote(path)
        command = (
            f"if [ ! -d {q_path} ]; then echo 'not a directory' >&2; exit 2; fi; "
            f"find {q_path} -maxdepth 1 -mindepth 1 -exec basename {{}} \\; | sort"
        )
        result = await self._exec(command, timeout_sec=30)
        if result.return_code != 0:
            stderr, _ = self._truncate(result.stderr)
            return _tool_error(
                call, self.name, stderr or f"list failed with exit {result.return_code}"
            )
        entries = []
        for entry in (result.stdout or "").splitlines():
            entry_path = entry if path == "." else PurePosixPath(path, entry).as_posix()
            if not self.policy.rejects_relative_path(entry_path):
                entries.append(entry)
        visible_stdout = "\n".join(entries) + ("\n" if entries else "")
        stdout, truncated = self._truncate(visible_stdout)
        suffix = "\n[truncated]" if truncated else ""
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"{stdout}{suffix}",
            metadata={"path": path, "exit_code": result.return_code},
        )


class RemoteWriteFileTool(_RemoteToolBase):
    name = "write_file"
    description = (
        "Create a UTF-8 text file in the external workspace. For existing files, prefer "
        "edit_file or apply_patch; set overwrite=true only when replacing the full file is intentional."
    )
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Destination path relative to workspace."},
            "content": {"type": "string", "description": "Full file contents."},
            "overwrite": {
                "type": "boolean",
                "description": "Allow replacing an existing file. Defaults to false.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(
        self,
        environment: Any,
        *,
        workdir: str,
        max_output_bytes: int = 64 * 1024,
        state: RemoteWorkspaceState | None = None,
        policy: ExternalWorkspacePolicy | None = None,
    ) -> None:
        super().__init__(
            environment,
            workdir=workdir,
            max_output_bytes=max_output_bytes,
            state=state,
            policy=policy,
        )
        self._failed_argument_signatures: set[str] = set()
        self._failed_path_signatures: set[str] = set()

    async def __call__(self, call: ToolCall) -> ToolResult:
        argument_signature = json.dumps(call.arguments, sort_keys=True, default=str)
        if argument_signature in self._failed_argument_signatures:
            return _tool_error(
                call,
                self.name,
                "refused: these exact write_file arguments already failed at the "
                "current workspace state. write_file requires both path and content; "
                "path must be relative to the repository.",
            )
        raw_path = call.arguments.get("path")
        path_signature = json.dumps(raw_path, sort_keys=True, default=str)
        if path_signature in self._failed_path_signatures:
            return _tool_error(
                call,
                self.name,
                "refused: this write_file path was already rejected at the current "
                "workspace state. The path argument must be a relative workspace path.",
            )
        path = _clean_relative_path(raw_path)
        content = call.arguments.get("content")
        if path is None:
            self._failed_argument_signatures.add(argument_signature)
            self._failed_path_signatures.add(path_signature)
            return _tool_error(
                call,
                self.name,
                "path is required and must be a relative workspace path.",
            )
        if self.policy.rejects_relative_path(path):
            self._failed_argument_signatures.add(argument_signature)
            self._failed_path_signatures.add(path_signature)
            return _tool_error(
                call,
                self.name,
                self.policy.refusal_message,
            )
        if not isinstance(content, str):
            self._failed_argument_signatures.add(argument_signature)
            return _tool_error(call, self.name, "content must be a string")
        overwrite = call.arguments.get("overwrite") is True
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        q_path = _quote(path)
        q_parent = _quote(PurePosixPath(path).parent.as_posix())
        command = (
            "tmp=$(mktemp) && "
            f"base64 -d > \"$tmp\" <<'HARNESS_REMOTE_FILE'\n"
            f"{encoded}\n"
            "HARNESS_REMOTE_FILE\n"
            f'if [ -e {q_path} ] && cmp -s "$tmp" {q_path}; then '
            "rm -f \"$tmp\"; echo 'no-op: file already has requested content' >&2; exit 6; fi; "
            f"if [ -e {q_path} ] && [ {str(overwrite).lower()} != true ]; then "
            "rm -f \"$tmp\"; echo 'file exists and overwrite was not enabled' >&2; exit 5; fi; "
            f"mkdir -p {q_parent} && "
            f'cat "$tmp" > {q_path}; status=$?; rm -f "$tmp"; exit "$status"'
        )
        result = await self._exec(command, timeout_sec=30)
        if result.return_code != 0:
            stderr, _ = self._truncate(result.stderr)
            if not stderr and result.return_code == 5:
                stderr = "file exists and overwrite was not enabled"
            elif not stderr and result.return_code == 6:
                stderr = "no-op: file already has requested content"
            self._failed_argument_signatures.add(argument_signature)
            return _tool_error(
                call, self.name, stderr or f"write failed with exit {result.return_code}"
            )
        self._mark_workspace_changed()
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"wrote {path} ({len(content.encode('utf-8'))} bytes)",
            metadata={"path": path, "exit_code": result.return_code, "overwrite": overwrite},
        )


class RemoteEditFileTool(_RemoteToolBase):
    name = "edit_file"
    description = "Replace an exact string occurrence inside a UTF-8 workspace file."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Target path relative to workspace."},
            "old": {"type": "string", "description": "Exact text to replace once."},
            "new": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old", "new"],
    }

    def __init__(
        self,
        environment: Any,
        *,
        workdir: str,
        max_output_bytes: int = 64 * 1024,
        state: RemoteWorkspaceState | None = None,
        policy: ExternalWorkspacePolicy | None = None,
    ) -> None:
        super().__init__(
            environment,
            workdir=workdir,
            max_output_bytes=max_output_bytes,
            state=state,
            policy=policy,
        )
        self._failed_argument_versions: dict[str, int] = {}

    async def __call__(self, call: ToolCall) -> ToolResult:
        argument_signature = json.dumps(call.arguments, sort_keys=True, default=str)
        if self._failed_argument_versions.get(argument_signature) == self.state.mutation_version:
            return _tool_error(
                call,
                self.name,
                "refused: these exact edit_file arguments already failed at the "
                f"current workspace state. {_EDIT_FILE_RECOVERY_GUIDANCE}",
            )
        reader = RemoteReadFileTool(
            self.environment,
            workdir=self.workdir,
            max_output_bytes=2 * 1024 * 1024,
            state=self.state,
            policy=self.policy,
        )
        read_result = await reader(
            call.model_copy(update={"arguments": {"path": call.arguments.get("path")}})
        )
        if read_result.is_error:
            self._failed_argument_versions[argument_signature] = self.state.mutation_version
            return _tool_error(call, self.name, read_result.content)
        old = call.arguments.get("old")
        new = call.arguments.get("new")
        path = _clean_relative_path(call.arguments.get("path"))
        if path is None or not isinstance(old, str) or not isinstance(new, str):
            self._failed_argument_versions[argument_signature] = self.state.mutation_version
            return _tool_error(
                call,
                self.name,
                f"path, old, and new are required. {_EDIT_FILE_RECOVERY_GUIDANCE}",
            )
        count = read_result.content.count(old)
        if count != 1:
            self._failed_argument_versions[argument_signature] = self.state.mutation_version
            return _tool_error(
                call,
                self.name,
                f"old text must appear exactly once, found {count}. {_EDIT_FILE_RECOVERY_GUIDANCE}",
            )
        writer = RemoteWriteFileTool(
            self.environment,
            workdir=self.workdir,
            max_output_bytes=self.max_output_bytes,
            state=self.state,
            policy=self.policy,
        )
        write_result = await writer(
            call.model_copy(
                update={
                    "arguments": {
                        "path": path,
                        "content": read_result.content.replace(old, new, 1),
                        "overwrite": True,
                    }
                }
            )
        )
        if write_result.is_error:
            self._failed_argument_versions[argument_signature] = self.state.mutation_version
            return _tool_error(call, self.name, write_result.content)
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"edited {path}",
            metadata={"path": path, "overwrite": True},
        )


class RemoteApplyPatchTool(_RemoteToolBase):
    name = "apply_patch"
    description = "Apply a unified diff patch in the external workspace after git apply --check."
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "Unified diff patch with paths relative to the repository root.",
            }
        },
        "required": ["patch"],
    }

    def __init__(
        self,
        environment: Any,
        *,
        workdir: str,
        max_output_bytes: int = 64 * 1024,
        state: RemoteWorkspaceState | None = None,
        policy: ExternalWorkspacePolicy | None = None,
    ) -> None:
        super().__init__(
            environment,
            workdir=workdir,
            max_output_bytes=max_output_bytes,
            state=state,
            policy=policy,
        )
        self._failed_patch_versions: dict[str, int] = {}

    async def _run_git_apply(self, patch: str) -> Any:
        encoded = base64.b64encode(patch.encode("utf-8")).decode("ascii")
        command = (
            "tmp=$(mktemp) && "
            "base64 -d > \"$tmp\" <<'HARNESS_REMOTE_PATCH'\n"
            f"{encoded}\n"
            "HARNESS_REMOTE_PATCH\n"
            'git apply --check --whitespace=nowarn "$tmp" && '
            'git apply --whitespace=nowarn "$tmp"; '
            'status=$?; rm -f "$tmp"; exit "$status"'
        )
        return await self._exec(command, timeout_sec=30)

    async def _read_remote_text(self, path: str) -> str:
        result = await self._exec(f"[ -f {shlex.quote(path)} ] && base64 < {shlex.quote(path)}")
        if result.return_code != 0:
            raise ValueError(f"path does not exist or is not a file: {path}")
        try:
            return base64.b64decode(result.stdout.encode("ascii")).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"could not read UTF-8 text from {path}: {exc}") from exc

    async def _write_remote_text(self, path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        quoted_path = shlex.quote(path)
        command = (
            f'mkdir -p -- "$(dirname -- {quoted_path})" && '
            f"base64 -d > {quoted_path} <<'HARNESS_REMOTE_FILE'\n"
            f"{encoded}\n"
            "HARNESS_REMOTE_FILE"
        )
        result = await self._exec(command, timeout_sec=30)
        if result.return_code != 0:
            detail = result.stderr or result.stdout or f"exit {result.return_code}"
            raise ValueError(f"could not write {path}: {detail}")

    async def _delete_remote_path(self, path: str) -> None:
        result = await self._exec(f"[ -e {shlex.quote(path)} ] && rm -f -- {shlex.quote(path)}")
        if result.return_code != 0:
            detail = result.stderr or result.stdout or f"exit {result.return_code}"
            raise ValueError(f"could not delete {path}: {detail}")

    async def _move_remote_path(self, old_path: str, new_path: str) -> None:
        result = await self._exec(
            f'mkdir -p -- "$(dirname -- {shlex.quote(new_path)})" && '
            f"mv -- {shlex.quote(old_path)} {shlex.quote(new_path)}",
            timeout_sec=30,
        )
        if result.return_code != 0:
            detail = result.stderr or result.stdout or f"exit {result.return_code}"
            raise ValueError(f"could not move {old_path} to {new_path}: {detail}")

    async def _run_codex_apply_patch(self, patch: str) -> None:
        operations = _parse_codex_apply_patch(patch)
        for operation in operations:
            for path in operation.paths():
                if self.policy.rejects_relative_path(path):
                    raise ValueError(self.policy.refusal_message)

        for operation in operations:
            if operation.kind == "add":
                if operation.lines is None:
                    raise ValueError("add file operation has no content")
                await self._write_remote_text(operation.path, "".join(operation.lines))
                continue
            if operation.kind == "delete":
                await self._delete_remote_path(operation.path)
                continue
            if operation.kind != "update":
                raise ValueError(f"unsupported patch operation: {operation.kind}")

            content = await self._read_remote_text(operation.path)
            for hunk in operation.hunks:
                old_text = "".join(line.old for line in hunk)
                new_text = "".join(line.new for line in hunk)
                if not old_text:
                    raise ValueError(
                        f"update hunk for {operation.path} has no context/removal lines"
                    )
                occurrences = content.count(old_text)
                if occurrences == 0:
                    raise ValueError(f"update hunk did not match {operation.path}")
                if occurrences > 1:
                    raise ValueError(
                        f"update hunk matched {operation.path} {occurrences} times; "
                        "include more unique surrounding context from read_file_range "
                        "or use edit_file with an exact old block for the intended location"
                    )
                content = content.replace(old_text, new_text, 1)
            await self._write_remote_text(operation.path, content)
            if operation.move_to:
                await self._move_remote_path(operation.path, operation.move_to)

    async def __call__(self, call: ToolCall) -> ToolResult:
        patch = call.arguments.get("patch")
        patch_signature = json.dumps(patch, sort_keys=True, default=str)
        if self._failed_patch_versions.get(patch_signature) == self.state.mutation_version:
            return _tool_error(
                call,
                self.name,
                "refused: this exact apply_patch payload already failed at the "
                f"current workspace state. {_APPLY_PATCH_RECOVERY_GUIDANCE}",
            )
        if not isinstance(patch, str) or not patch.strip():
            self._failed_patch_versions[patch_signature] = self.state.mutation_version
            return _tool_error(
                call,
                self.name,
                f"patch must be a non-empty string. {_APPLY_PATCH_RECOVERY_GUIDANCE}",
            )
        if self.policy.references_forbidden_material(patch):
            self._failed_patch_versions[patch_signature] = self.state.mutation_version
            return _tool_error(
                call,
                self.name,
                self.policy.refusal_message,
            )
        patch, stripped_patch_envelope = _strip_unified_diff_patch_envelope(patch)
        if _looks_like_codex_apply_patch(patch):
            try:
                await self._run_codex_apply_patch(patch)
            except ValueError as exc:
                self._failed_patch_versions[patch_signature] = self.state.mutation_version
                return _tool_error(
                    call,
                    self.name,
                    f"{exc}\n{_APPLY_PATCH_RECOVERY_GUIDANCE}",
                )
            self._mark_workspace_changed()
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    f"applied patch ({len(patch.encode('utf-8'))} bytes)"
                    "\n[applied Codex-style patch format]"
                ),
                metadata={
                    "exit_code": 0,
                    "normalized_hunk_headers": False,
                    "patch_format": "codex",
                    **({"stripped_patch_envelope": True} if stripped_patch_envelope else {}),
                },
            )
        result = await self._run_git_apply(patch)
        applied_normalized_patch = False
        normalized_failure: Any | None = None
        normalized_patch, normalized_changed = _normalize_unified_diff_hunk_headers(patch)
        if result.return_code != 0 and normalized_changed:
            normalized_result = await self._run_git_apply(normalized_patch)
            if normalized_result.return_code == 0:
                result = normalized_result
                applied_normalized_patch = True
            else:
                normalized_failure = normalized_result
        stdout, stdout_truncated = self._truncate(result.stdout)
        stderr, stderr_truncated = self._truncate(result.stderr)
        if result.return_code != 0:
            content = stderr or stdout or f"patch failed with exit {result.return_code}"
            if normalized_failure is not None:
                normalized_stdout, _ = self._truncate(normalized_failure.stdout)
                normalized_stderr, _ = self._truncate(normalized_failure.stderr)
                normalized_content = (
                    normalized_stderr
                    or normalized_stdout
                    or f"patch failed with exit {normalized_failure.return_code}"
                )
                content += f"\nNormalized patch attempt also failed: {normalized_content}"
            content += f"\n{_APPLY_PATCH_RECOVERY_GUIDANCE}"
            self._failed_patch_versions[patch_signature] = self.state.mutation_version
            return _tool_error(call, self.name, content)
        content = f"applied patch ({len(patch.encode('utf-8'))} bytes)"
        if applied_normalized_patch:
            content += "\n[normalized unified diff hunk headers before applying]"
        if stdout or stderr:
            content += f"\nstdout:\n{stdout}\nstderr:\n{stderr}"
        if stdout_truncated or stderr_truncated:
            content += "\n[truncated]"
        self._mark_workspace_changed()
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=content,
            metadata={
                "exit_code": result.return_code,
                "normalized_hunk_headers": applied_normalized_patch,
                **({"stripped_patch_envelope": True} if stripped_patch_envelope else {}),
            },
        )


class RemoteShellTool(_RemoteToolBase):
    name = "shell"
    description = (
        "Run a shell command in the external workspace with a timeout. Use this for "
        "project setup, environment discovery, installs, builds, tests, and allowed "
        "network commands; external workspace policy will refuse forbidden access."
    )
    prediction_expected_status = "ok_or_error"
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {"type": "integer", "description": "Timeout in seconds, capped at 300."},
        },
        "required": ["command"],
    }

    def __init__(
        self,
        environment: Any,
        *,
        workdir: str,
        max_output_bytes: int = 64 * 1024,
        state: RemoteWorkspaceState | None = None,
        policy: ExternalWorkspacePolicy | None = None,
    ) -> None:
        super().__init__(
            environment,
            workdir=workdir,
            max_output_bytes=max_output_bytes,
            state=state,
            policy=policy,
        )
        self._failed_commands: dict[str, tuple[int, int]] = {}
        self._command_versions: dict[str, tuple[int, int]] = {}

    async def __call__(self, call: ToolCall) -> ToolResult:
        command = call.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return _tool_error(call, self.name, "command must be a non-empty string")
        state_version = self._state_version()
        if self._failed_commands.get(command) == state_version:
            return _tool_error(
                call,
                self.name,
                "refused: this exact shell command already failed at the current workspace state.",
            )
        if self._command_versions.get(command) == state_version:
            return _tool_error(
                call,
                self.name,
                "refused: this exact shell command already ran at the current workspace state.",
            )
        allowed_git_clone_command = _shell_command_is_allowed_git_clone(command, self.policy)
        if (
            (self.policy.references_forbidden_material(command) and not allowed_git_clone_command)
            or (
                self.policy.references_forbidden_web_material(command)
                and _shell_command_may_access_network(command)
                and not allowed_git_clone_command
            )
            or (
                not self.policy.allow_web_access
                and _shell_command_requests_external_network(command)
                and not allowed_git_clone_command
            )
        ):
            return _tool_error(
                call,
                self.name,
                self.policy.refusal_message,
            )
        if self.policy.block_root_filesystem_probe and _attempts_root_filesystem_probe(command):
            return _tool_error(
                call,
                self.name,
                "refused: external workspace agents may not inspect the container root filesystem",
            )
        if _command_references_parent_directory(command):
            return _tool_error(
                call,
                self.name,
                (
                    "refused: external workspace shell commands may not access parent "
                    "directories outside the repository. Use repository-relative paths "
                    "under the current workspace."
                ),
            )
        if _command_references_absolute_host_path(command):
            return _tool_error(
                call,
                self.name,
                (
                    "refused: external workspace shell commands may not access absolute "
                    "host filesystem paths. Use repository-relative paths under the "
                    "current workspace."
                ),
            )
        if _is_shell_inspection_command(command):
            self._record_read_only_call()
        timeout_arg = call.arguments.get("timeout", 120)
        try:
            timeout = max(1, min(int(timeout_arg), 300))
        except (TypeError, ValueError):
            timeout = 120
        before_fingerprint = await self._exec(_GIT_WORKSPACE_FINGERPRINT_COMMAND, timeout_sec=30)
        before_fingerprint_text = _workspace_fingerprint_text(before_fingerprint)
        execution_command = f"bash -lc {_quote('set -o pipefail; ' + command)}"
        result = await self._exec(execution_command, timeout_sec=timeout)
        after_fingerprint = await self._exec(_GIT_WORKSPACE_FINGERPRINT_COMMAND, timeout_sec=30)
        after_fingerprint_text = _workspace_fingerprint_text(after_fingerprint)
        workspace_changed = (
            before_fingerprint_text is not None
            and after_fingerprint_text is not None
            and after_fingerprint_text != before_fingerprint_text
        )
        if workspace_changed:
            self._mark_workspace_changed()
        stdout, stdout_truncated = self._truncate(self.policy.redact_output(result.stdout))
        stderr, stderr_truncated = self._truncate(self.policy.redact_output(result.stderr))
        raw_masked_failure_exit_status = (
            result.return_code == 0 and _failure_branch_masks_exit_status(command)
        )
        informational_environment_probe = (
            raw_masked_failure_exit_status
            and not workspace_changed
            and bool((stdout or stderr).strip())
            and _shell_command_is_environment_probe(command)
        )
        masked_failure_exit_status = (
            raw_masked_failure_exit_status and not informational_environment_probe
        )
        stderr_failure_exit_status = _successful_shell_stderr_reports_failure(
            exit_code=result.return_code,
            stderr=stderr,
        )
        stdout_failure_exit_status = (
            result.return_code == 0
            and _is_broad_test_command(command)
            and _output_reports_failure(stdout)
        )
        head_pipe_preview_exit_status = _head_pipe_preview_exit_status(
            command=command,
            exit_code=result.return_code,
            stdout=stdout,
        )
        content = f"exit_code: {result.return_code}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        if masked_failure_exit_status:
            content += (
                "\n\n[unreliable result] This shell command can hide a failed "
                "command behind a successful fallback. Re-run the check so the "
                "failing operation reports its real exit status."
            )
        elif informational_environment_probe:
            content += (
                "\n\n[environment probe] This command used a fallback branch, but it "
                "only inspected tool availability or runtime metadata and produced "
                "output without changing the workspace. Treat this as environment "
                "evidence; final verification must still use verify_work without "
                "masked failures."
            )
        if stderr_failure_exit_status:
            content += (
                "\n\n[unreliable result] The command exited 0, but stderr contains "
                "a shell/runtime failure. Re-run with a command form that preserves "
                "the failing operation's exit status."
            )
        if stdout_failure_exit_status:
            content += (
                "\n\n[unreliable result] The command exited 0, but stdout contains "
                "test failure output. Re-run with a command form that preserves the "
                "failing test command's exit status."
            )
        if head_pipe_preview_exit_status:
            content += (
                "\n\n[preview result] The command produced stdout and then exited with "
                "a pipe-close status from `head`. Treat the displayed output as valid "
                "preview evidence, and rerun without `| head` if an exact exit status "
                "is required."
            )
        if "destination path '.' already exists and is not an empty directory" in stderr.lower():
            content += (
                "\n\n[setup hint] The current workspace directory is already populated. "
                "Retry by cloning the project into a new subdirectory, then run later "
                "project commands from that subdirectory."
            )
        hint = shell_failure_hint(
            command,
            exit_code=result.return_code,
            stdout=stdout,
            stderr=stderr,
        )
        if hint:
            content += f"\n\n{hint}"
        if stdout_truncated or stderr_truncated:
            content += "\n[truncated]"
        is_error = (
            (result.return_code != 0 and not head_pipe_preview_exit_status)
            or masked_failure_exit_status
            or stderr_failure_exit_status
            or stdout_failure_exit_status
        )
        if not is_error and not workspace_changed and not _is_shell_inspection_command(command):
            self._mark_external_environment_changed()
        result_state_version = self._state_version()
        if is_error:
            self._failed_commands[command] = result_state_version
        self._command_versions[command] = result_state_version
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=content,
            is_error=is_error,
            metadata={
                "exit_code": result.return_code,
                "timeout": timeout,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint_changed": workspace_changed,
                "masked_failure_exit_status": masked_failure_exit_status,
                "informational_environment_probe": informational_environment_probe,
                "stderr_failure_exit_status": stderr_failure_exit_status,
                "stdout_failure_exit_status": stdout_failure_exit_status,
                "head_pipe_preview_exit_status": head_pipe_preview_exit_status,
                "pipefail": True,
            },
        )


class RemoteVerifyWorkTool(RemoteShellTool):
    name = "verify_work"
    description = (
        "Run a read-only in-repository verification command chosen for the current "
        "project. This can be the same read-only containerized command used during "
        "exploration when that is the available project runtime. The command passes "
        "only when its exit code and output support success."
    )
    effect_scope = "read_only"

    def __init__(
        self,
        environment: Any,
        *,
        workdir: str,
        max_output_bytes: int = 64 * 1024,
        state: RemoteWorkspaceState | None = None,
        policy: ExternalWorkspacePolicy | None = None,
        default_command: str | None = None,
        default_timeout_seconds: int | None = None,
    ) -> None:
        super().__init__(
            environment,
            workdir=workdir,
            max_output_bytes=max_output_bytes,
            state=state,
            policy=policy,
        )
        self.default_command = (default_command or "").strip()
        self.default_timeout_seconds = (
            max(1, int(default_timeout_seconds)) if default_timeout_seconds else None
        )
        self._passed_commands: dict[str, tuple[tuple[int, int], ToolResult]] = {}
        if self.default_command:
            self.__dict__["parameters_schema"] = {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Optional shell command. Omit it to run the configured "
                            "default verification command."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds, capped at 300.",
                    },
                },
                "required": [],
            }

    @property
    def has_default_command(self) -> bool:
        return bool(self.default_command)

    def _verification_error(
        self,
        call: ToolCall,
        message: str,
        *,
        command: str | None = None,
        reason: str | None = None,
    ) -> ToolResult:
        metadata: dict[str, object] = {}
        if command is not None:
            metadata["command"] = command
        if reason is not None:
            metadata["reason"] = reason
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=message,
            is_error=True,
            metadata=metadata,
        )

    @staticmethod
    def _default_verification_content(verdict: str, *, passed: bool) -> str:
        if passed:
            return (
                "PASSED\n\n"
                "Configured default verification passed. Detailed verifier output is "
                "withheld from the model because it may contain benchmark-private artifacts."
            )
        return (
            f"{verdict}\n\n"
            "Configured default verification failed. Detailed verifier output is withheld "
            "from the model because it may contain benchmark-private artifacts. "
            f"{_DEFAULT_VERIFIER_FAILURE_GUIDANCE}"
        )

    async def __call__(self, call: ToolCall) -> ToolResult:
        command = call.arguments.get("command")
        used_default_command = False
        if not isinstance(command, str) or not command.strip():
            command = self.default_command
            if not command:
                return self._verification_error(
                    call,
                    "command must be a non-empty string",
                    reason="missing_command",
                )
            used_default_command = True
        if not used_default_command and _verification_command_is_noop(command):
            return self._verification_error(
                call,
                (
                    "refused invalid verification command: the command does not "
                    "run a meaningful check. verify_work evidence must come from "
                    "a repository command that can fail when the work is wrong."
                ),
                command=command,
                reason="noop_verification_command",
            )
        if not used_default_command and _shell_command_requests_setup(command):
            return self._verification_error(
                call,
                (
                    "refused invalid verification command: setup or dependency "
                    "installation must use the shell tool first; run verify_work "
                    "only for read-only verification evidence after setup completes"
                ),
                command=command,
                reason="setup_command_not_verification",
            )
        state_version = self._state_version()
        if self._failed_commands.get(command) == state_version:
            if used_default_command:
                return self._verification_error(
                    call,
                    "refused: the configured default verification command already failed "
                    "at this workspace state. Detailed verifier output is withheld from "
                    "the model because it may contain benchmark-private artifacts. "
                    f"{_DEFAULT_VERIFIER_FAILURE_GUIDANCE}",
                    command=command,
                    reason="repeated_failed_default_command",
                )
            return self._verification_error(
                call,
                "refused: this exact verification command already failed at the "
                "current workspace state.",
                command=command,
                reason="repeated_failed_command",
            )
        if self._command_versions.get(command) == state_version:
            cached = self._passed_commands.get(command)
            allow_default_rerun = False
            if cached is not None and cached[0] == state_version:
                previous = cached[1]
                metadata: dict[str, Any] = (
                    dict(previous.metadata)
                    if isinstance(previous.metadata, dict)
                    else {"command": command}
                )
                if not used_default_command or metadata.get("used_default_command") is True:
                    metadata["cached_previous_success"] = True
                    return ToolResult(
                        tool_call_id=call.id,
                        name=self.name,
                        content=(
                            "PASSED (cached previous verification)\n\n"
                            "This exact verify_work command already passed at the current "
                            "workspace state. No files changed since that run, so this "
                            "is the same verification evidence."
                        ),
                        is_error=False,
                        metadata=metadata,
                    )
                allow_default_rerun = used_default_command
            if allow_default_rerun:
                pass
            else:
                return self._verification_error(
                    call,
                    "refused: this exact verification command already ran at the current "
                    "workspace state.",
                    command=command,
                    reason="repeated_command_without_workspace_change",
                )
        if _failure_branch_masks_exit_status(command):
            return self._verification_error(
                call,
                "refused invalid verification command: failed assertions must return a non-zero exit code",
                command=command,
                reason="masked_failure_exit_status",
            )
        if _command_exits_before_trailing_command(command):
            return self._verification_error(
                call,
                "refused invalid verification command: the command exits before a later check can run",
                command=command,
                reason="unreachable_verification_command",
            )
        if not used_default_command and self.policy.references_forbidden_material(command):
            return self._verification_error(
                call,
                self.policy.refusal_message,
                command=command,
                reason="external_workspace_policy",
            )
        if (
            not used_default_command
            and self.policy.block_root_filesystem_probe
            and _attempts_root_filesystem_probe(command)
        ):
            return self._verification_error(
                call,
                "refused: external workspace agents may not inspect the container root filesystem",
                command=command,
                reason="root_filesystem_probe",
            )
        if not used_default_command and _command_references_parent_directory(command):
            return self._verification_error(
                call,
                (
                    "refused: external workspace shell commands may not access parent "
                    "directories outside the repository. Use repository-relative paths "
                    "under the current workspace."
                ),
                command=command,
                reason="parent_directory_escape",
            )
        if not used_default_command and _command_references_absolute_host_path(command):
            return self._verification_error(
                call,
                (
                    "refused: external workspace verification commands may not access "
                    "absolute host filesystem paths. Use repository-relative paths under "
                    "the current workspace."
                ),
                command=command,
                reason="absolute_host_path_escape",
            )
        default_timeout = self.default_timeout_seconds if used_default_command else None
        timeout_arg = call.arguments.get("timeout", default_timeout or 120)
        timeout_cap = default_timeout or 300
        try:
            timeout = max(1, min(int(timeout_arg), timeout_cap))
        except (TypeError, ValueError):
            timeout = default_timeout or 120
        before_fingerprint = await self._exec(_GIT_WORKSPACE_FINGERPRINT_COMMAND, timeout_sec=30)
        before_fingerprint_text = _workspace_fingerprint_text(before_fingerprint)
        result = await self._exec(
            f"bash -lc {_quote('set -e -o pipefail; ' + command)}", timeout_sec=timeout
        )
        after_fingerprint = await self._exec(_GIT_WORKSPACE_FINGERPRINT_COMMAND, timeout_sec=30)
        after_fingerprint_text = _workspace_fingerprint_text(after_fingerprint)
        workspace_changed = (
            before_fingerprint_text is not None
            and after_fingerprint_text is not None
            and after_fingerprint_text != before_fingerprint_text
        )
        if workspace_changed:
            self._mark_workspace_changed()
        output_failure = result.return_code == 0 and _output_reports_failure(
            "\n".join(part for part in (result.stdout or "", result.stderr or "") if part)
        )
        passed = result.return_code == 0 and not output_failure and not workspace_changed
        rerun_required = (
            passed
            and not used_default_command
            and _fingerprint_has_untracked_test_path(after_fingerprint_text)
        )
        rerun_result: Any | None = None
        rerun_output_failure = False
        rerun_workspace_changed = False
        if rerun_required:
            rerun_result = await self._exec(
                f"bash -lc {_quote('set -e -o pipefail; ' + command)}", timeout_sec=timeout
            )
            assert rerun_result is not None
            rerun_after_fingerprint = await self._exec(
                _GIT_WORKSPACE_FINGERPRINT_COMMAND,
                timeout_sec=30,
            )
            rerun_after_fingerprint_text = _workspace_fingerprint_text(rerun_after_fingerprint)
            rerun_workspace_changed = (
                after_fingerprint_text is not None
                and rerun_after_fingerprint_text is not None
                and rerun_after_fingerprint_text != after_fingerprint_text
            )
            rerun_output_failure = rerun_result.return_code == 0 and _output_reports_failure(
                "\n".join(
                    part for part in (rerun_result.stdout or "", rerun_result.stderr or "") if part
                )
            )
            if rerun_result.return_code != 0 or rerun_output_failure or rerun_workspace_changed:
                passed = False
        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""
        if rerun_result is not None:
            stdout_text = (
                f"[first run]\n{stdout_text}\n[immediate rerun]\n{rerun_result.stdout or ''}"
            )
            stderr_text = (
                f"[first run]\n{stderr_text}\n[immediate rerun]\n{rerun_result.stderr or ''}"
            )
        stdout, stdout_truncated = self._truncate(self.policy.redact_output(stdout_text))
        stderr, stderr_truncated = self._truncate(self.policy.redact_output(stderr_text))
        workspace_changed = workspace_changed or rerun_workspace_changed
        if rerun_workspace_changed:
            self._mark_workspace_changed()
        verdict = (
            "PASSED"
            if passed
            else "FAILED (verification changed workspace)"
            if workspace_changed
            else "FAILED (output reports failure)"
            if output_failure or rerun_output_failure
            else f"FAILED (immediate rerun exit {rerun_result.return_code})"
            if rerun_result is not None and rerun_result.return_code != 0
            else f"FAILED (exit {result.return_code})"
        )
        content = f"{verdict}\n\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        pytest_hint = _pytest_executable_failure_hint(command, stdout, stderr)
        if not passed and pytest_hint:
            content += f"\n\n{pytest_hint}"
        setup_hint = shell_failure_hint(
            command,
            exit_code=result.return_code,
            stdout=stdout,
            stderr=stderr,
        )
        if not passed and setup_hint:
            content += f"\n\n{setup_hint}"
        if stdout_truncated or stderr_truncated:
            content += "\n[truncated]"
        if used_default_command:
            content = self._default_verification_content(verdict, passed=passed)
        result_state_version = self._state_version()
        if not passed:
            self._failed_commands[command] = result_state_version
        self._command_versions[command] = result_state_version
        tool_result = ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=content,
            is_error=not passed,
            metadata={
                "command": command,
                "exit_code": result.return_code,
                "timeout": timeout,
                "output_reports_failure": output_failure,
                "workspace_changed": workspace_changed,
                "workspace_fingerprint_changed": workspace_changed,
                "errexit": True,
                "pipefail": True,
                "used_default_command": used_default_command,
                "rerun_required": rerun_required,
                "rerun_exit_code": (rerun_result.return_code if rerun_result is not None else None),
                "rerun_output_reports_failure": rerun_output_failure,
                "rerun_workspace_changed": rerun_workspace_changed,
            },
        )
        if passed:
            self._passed_commands[command] = (result_state_version, tool_result)
        return tool_result


def _redact_web_metadata(
    metadata: dict[str, Any], policy: ExternalWorkspacePolicy
) -> dict[str, Any]:
    sanitized = dict(metadata)
    results = sanitized.get("results")
    if isinstance(results, list):
        kept: list[dict[str, Any]] = []
        omitted = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            text = "\n".join(str(value) for value in item.values())
            if policy.references_forbidden_web_material(text):
                omitted += 1
                continue
            kept.append(dict(item))
        sanitized["results"] = kept
        if omitted:
            sanitized["restricted_results_omitted"] = omitted
    return sanitized


def _web_policy_refusal_message(policy: ExternalWorkspacePolicy, text: str) -> str:
    if policy.references_allowed_git_clone_material(text):
        return (
            f"{policy.refusal_message}. Web lookup of this repository is blocked by "
            "policy; use shell git clone or git ls-remote for repository setup when "
            "public workspace metadata provides the repository URL."
        )
    return policy.refusal_message


class PolicyWebSearchTool:
    name = "web_search"
    description = (
        "Search the internet from the host environment. Use this to research missing "
        "tools, documentation, current APIs, or error messages when repository-visible "
        "evidence is insufficient. External workspace policy blocks benchmark-private "
        "artifacts and forbidden source repositories."
    )
    approval = "auto"
    effect_scope = "read_only"
    phases = ("*",)

    def __init__(
        self,
        *,
        policy: ExternalWorkspacePolicy | None = None,
        tool: WebSearchTool | None = None,
    ) -> None:
        self.policy = policy or ExternalWorkspacePolicy()
        self._tool = tool or WebSearchTool()
        self.parameters_schema = self._tool.parameters_schema

    async def __call__(self, call: ToolCall) -> ToolResult:
        query = call.arguments.get("query")
        if not self.policy.allow_web_access:
            return _tool_error(call, self.name, self.policy.refusal_message)
        if isinstance(query, str) and self.policy.references_forbidden_web_material(query):
            return _tool_error(call, self.name, _web_policy_refusal_message(self.policy, query))
        result = await self._tool(call)
        content = self.policy.redact_web_output(result.content)
        metadata = (
            _redact_web_metadata(result.metadata, self.policy)
            if isinstance(result.metadata, dict)
            else result.metadata
        )
        return result.model_copy(update={"content": content, "metadata": metadata})


class PolicyFetchUrlTool:
    name = "fetch_url"
    description = (
        "Fetch an http(s) URL from the host environment. Use this for documentation "
        "or public references when repository-visible evidence is insufficient. "
        "External workspace policy blocks benchmark-private artifacts and forbidden "
        "source repositories."
    )
    approval = "auto"
    effect_scope = "read_only"
    phases = ("*",)

    def __init__(
        self,
        *,
        policy: ExternalWorkspacePolicy | None = None,
        tool: FetchUrlTool | None = None,
    ) -> None:
        self.policy = policy or ExternalWorkspacePolicy()
        self._tool = tool or FetchUrlTool()
        self.parameters_schema = self._tool.parameters_schema

    async def __call__(self, call: ToolCall) -> ToolResult:
        url = call.arguments.get("url")
        if not self.policy.allow_web_access:
            return _tool_error(call, self.name, self.policy.refusal_message)
        if isinstance(url, str) and self.policy.references_forbidden_web_material(url):
            return _tool_error(call, self.name, _web_policy_refusal_message(self.policy, url))
        result = await self._tool(call)
        content = self.policy.redact_web_output(result.content)
        return result.model_copy(update={"content": content})


def build_remote_tool_registry(
    environment: Any,
    *,
    workdir: str,
    policy: ExternalWorkspacePolicy | None = None,
    default_verify_command: str | None = None,
    default_verify_timeout_seconds: int | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    state = RemoteWorkspaceState()
    tool_policy = policy or ExternalWorkspacePolicy()
    for tool in (
        RemoteApplyPatchTool(environment, workdir=workdir, state=state, policy=tool_policy),
        RemoteReadFileTool(environment, workdir=workdir, state=state, policy=tool_policy),
        RemoteReadFileRangeTool(environment, workdir=workdir, state=state, policy=tool_policy),
        RemoteListDirTool(environment, workdir=workdir, state=state, policy=tool_policy),
        RemoteWriteFileTool(environment, workdir=workdir, state=state, policy=tool_policy),
        RemoteEditFileTool(environment, workdir=workdir, state=state, policy=tool_policy),
        RemoteVerifyWorkTool(
            environment,
            workdir=workdir,
            state=state,
            policy=tool_policy,
            default_command=default_verify_command,
            default_timeout_seconds=default_verify_timeout_seconds,
        ),
        RemoteShellTool(environment, workdir=workdir, state=state, policy=tool_policy),
        PolicyWebSearchTool(policy=tool_policy),
        PolicyFetchUrlTool(policy=tool_policy),
    ):
        registry.register(cast(Tool, tool))
    return registry


def _dict_metadata(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _activity_tool_changes_workspace(event: ActivityEvent) -> bool:
    if event.kind != "tool_call.completed":
        return False
    name = str(event.data.get("name") or "")
    metadata = _dict_metadata(event.data.get("metadata"))
    if metadata.get("workspace_changed") is True:
        return True
    if event.data.get("is_error") is True:
        return False
    if name == "apply_patch":
        return True
    if name in {"write_file", "edit_file"}:
        return "path" in metadata
    if name == "shell":
        return metadata.get("workspace_changed") is True
    return False


def _activity_tool_is_passing_verify(event: ActivityEvent) -> bool:
    if event.kind != "tool_call.completed" or event.data.get("is_error") is True:
        return False
    name = str(event.data.get("name") or "")
    metadata = _dict_metadata(event.data.get("metadata"))
    if metadata.get("workspace_changed") is True:
        return False
    if name == "verify_work":
        return True
    if name != "shell":
        return False
    if metadata.get("exit_code") not in (None, 0):
        return False
    if metadata.get("masked_failure_exit_status") is True:
        return False
    if metadata.get("stderr_failure_exit_status") is True:
        return False
    if metadata.get("stdout_failure_exit_status") is True:
        return False
    command = _activity_event_command(event)
    if not command:
        return False
    return not (_verification_command_is_noop(command) or _shell_command_requests_setup(command))


def _activity_tool_is_verification_attempt(
    event: ActivityEvent,
    *,
    test_paths: list[str],
    untracked_test_paths: list[str],
) -> bool:
    if event.kind != "tool_call.completed":
        return False
    name = str(event.data.get("name") or "")
    if name == "verify_work":
        return True
    if name != "shell":
        return False
    command = _activity_event_command(event)
    if not command:
        return False
    if _verification_command_is_noop(command) or _shell_command_requests_setup(command):
        return False
    if not test_paths:
        return _is_broad_test_command(command)
    return _verification_command_covers_test_changes(
        command,
        test_paths,
        untracked_test_paths=untracked_test_paths,
    )


def _latest_assistant_message_content(session: Any) -> str:
    messages = getattr(session, "messages", None)
    if not isinstance(messages, list | tuple):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if role == "assistant" and isinstance(content, str) and content.strip():
            return content
    return ""


def _assistant_defers_environment_setup_to_user(session: Any) -> bool:
    text = " ".join(_latest_assistant_message_content(session).lower().split())
    if not text:
        return False
    blocked_terms = (
        "cannot complete",
        "can't complete",
        "unable to complete",
        "cannot proceed",
        "can't proceed",
        "unable to proceed",
        "blocked",
        "not available",
        "missing",
    )
    setup_terms = (
        "tool",
        "dependency",
        "dependencies",
        "package",
        "binary",
        "generator",
        "compiler",
        "runtime",
        "environment",
        "credential",
        "api key",
        "network",
        "docker",
        "git",
    )
    user_supply_terms = (
        "if you can",
        "if you want",
        "provide",
        "install",
        "enable",
        "allow",
        "tell me",
        "give me",
        "supply",
        "configure",
        "set up",
    )
    return (
        any(term in text for term in blocked_terms)
        and any(term in text for term in setup_terms)
        and any(term in text for term in user_supply_terms)
    )


def _assistant_defers_source_setup_to_user(session: Any) -> bool:
    text = " ".join(_latest_assistant_message_content(session).lower().split())
    if not text:
        return False
    blocked_terms = (
        "cannot",
        "can't",
        "unable",
        "blocked",
        "missing",
        "not present",
        "not available",
        "nothing to patch",
        "no code",
        "no source",
        "need",
    )
    source_terms = (
        "source code",
        "source tree",
        "repository contents",
        "repository files",
        "project files",
        "codebase",
        "code is located",
        "repo is located",
        "repository is located",
        "checkout",
        "clone",
    )
    user_supply_terms = (
        "provide",
        "upload",
        "add",
        "checkout",
        "check out",
        "clone",
        "tell me",
        "where",
        "allow me",
        "ask me",
    )
    return (
        any(term in text for term in blocked_terms)
        and any(term in text for term in source_terms)
        and any(term in text for term in user_supply_terms)
    )


def _assistant_waits_for_permission_to_continue(session: Any) -> bool:
    text = " ".join(_latest_assistant_message_content(session).lower().split())
    if not text:
        return False
    permission_terms = (
        "unless you object",
        "if you want me to continue",
        "if you want me to keep going",
        "if you want me to proceed",
        "reply with",
        "confirm",
        "confirmation",
    )
    continuation_terms = (
        "i will implement",
        "i'll implement",
        "i will start",
        "i'll start",
        "start implementing",
        "keep going",
        "continue",
        "proceed",
    )
    return any(term in text for term in permission_terms) and any(
        term in text for term in continuation_terms
    )


def _assistant_defers_setup_to_user(session: Any) -> bool:
    return _assistant_defers_environment_setup_to_user(
        session
    ) or _assistant_defers_source_setup_to_user(session)


def _autonomous_setup_repair_reason(*, latest_verify_error: str | None = None) -> str:
    reason = (
        "The final response defers repository, source, environment, or dependency "
        "setup to the user. Continue autonomously with the available tools: inspect "
        "workspace files, project setup instructions, and public metadata. If public "
        "metadata names a repository URL, base commit, container image, or setup "
        "commands, use that evidence to prepare a patchable checkout inside the "
        "workspace. Run available setup, clone, install, build, or equivalent "
        "commands yourself. When metadata names a Dockerfile or container image, "
        "check Docker availability with shell tools and try the declared container "
        "path before concluding the toolchain is unavailable. Use web_search for "
        "public documentation when repository-visible evidence is insufficient; "
        "choose a different implementation path if one setup path cannot work; then "
        "verify the result instead of asking the user to provide source files or "
        "tooling."
    )
    if latest_verify_error is None:
        return reason
    return f"{reason} Latest verify_work error: {latest_verify_error or 'none observed'}."


def _autonomous_permission_repair_reason(*, latest_verify_error: str | None = None) -> str:
    reason = (
        "The final response asks the user for permission or confirmation to continue "
        "work that the harness can perform with available tools. Continue "
        "autonomously from the current evidence: inspect, edit, run setup or "
        "verification commands as needed, and only finish after the requested "
        "behavior is implemented and verified. Do not ask the user to approve the "
        "next implementation, debugging, setup, or verification step unless the "
        "available tools cannot make meaningful progress."
    )
    if latest_verify_error is None:
        return reason
    return f"{reason} Latest verify_work error: {latest_verify_error or 'none observed'}."


def _verification_failure_replan_hint() -> str:
    return (
        " Treat the failing public verification as evidence that at least one current "
        "assumption is false. Before the next mutation, inspect the failing output and "
        "the project path exercised by the command, update the expected outcome, and "
        "switch implementation path if the evidence contradicts the current approach. "
        "Check whether the edited files are actually consumed by the verifier, "
        "including generated or derived artifacts, build outputs, entrypoints, and "
        "nested checkouts."
    )


@dataclass
class ExternalWorkspaceVerificationSnapshot:
    source_change_passed: bool = False
    source_change_paths: list[str] | None = None
    scratch_paths: list[str] | None = None
    test_change_paths: list[str] | None = None
    verification_passed_after_source_change: bool = False
    latest_verification_error: str | None = None
    latest_verification_command: str | None = None
    latest_verification_used_default_command: bool = False
    source_change_error: str | None = None
    verification_error: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source_change_passed": self.source_change_passed,
            "source_change_paths": list(self.source_change_paths or []),
            "scratch_paths": list(self.scratch_paths or []),
            "test_change_paths": list(self.test_change_paths or []),
            "source_change_error": self.source_change_error,
            "verification_passed_after_source_change": self.verification_passed_after_source_change,
            "latest_verification_error": self.latest_verification_error,
            "latest_verification_command": self.latest_verification_command,
            "latest_verification_used_default_command": (
                self.latest_verification_used_default_command
            ),
            "verification_error": self.verification_error,
        }


class ExternalWorkspaceVerifier:
    """Verifier gate for external workspace runs executed through Harness.

    The normal Harness runtime owns planning, tool use, prediction, and repair.
    This verifier only states external-run completion rules: real source changes,
    in-repository regression coverage, and a passing verify_work after the final
    workspace mutation.
    """

    name = "external_workspace"

    def __init__(
        self,
        environment: Any,
        *,
        workdir: str,
        baseline_untracked_paths: set[str] | None = None,
        ignored_workspace_paths: set[str] | None = None,
        fail_without_source_change: bool = True,
        fail_without_verification: bool = True,
        require_regression_test_change: bool = True,
        require_default_verify_command: bool = False,
        policy: ExternalWorkspacePolicy | None = None,
    ) -> None:
        self.environment = environment
        self.workdir = workdir
        self.policy = policy or ExternalWorkspacePolicy()
        self.baseline_untracked_paths = set(baseline_untracked_paths or set())
        self.ignored_workspace_paths = set(ignored_workspace_paths or set())
        self.fail_without_source_change = fail_without_source_change
        self.fail_without_verification = fail_without_verification
        self.require_regression_test_change = require_regression_test_change
        self.require_default_verify_command = require_default_verify_command
        self.latest = ExternalWorkspaceVerificationSnapshot()

    async def _workspace_status(self) -> tuple[str, set[str]]:
        status_result = await self.environment.exec(
            _GIT_STATUS_COMMAND,
            cwd=self.workdir,
            timeout_sec=30,
        )
        tracked_result = await self.environment.exec(
            "git ls-files",
            cwd=self.workdir,
            timeout_sec=30,
        )
        nested_status_result = await self.environment.exec(
            _NESTED_GIT_STATUS_COMMAND,
            cwd=self.workdir,
            timeout_sec=30,
        )
        nested_committed_status = ""
        if self.policy.required_git_base_commit:
            nested_committed_result = await self.environment.exec(
                _nested_git_committed_status_command(self.policy.required_git_base_commit),
                cwd=self.workdir,
                timeout_sec=30,
            )
            if nested_committed_result.return_code == 0:
                nested_committed_status = nested_committed_result.stdout or ""
        nested_tracked_result = await self.environment.exec(
            _NESTED_GIT_LS_FILES_COMMAND,
            cwd=self.workdir,
            timeout_sec=30,
        )
        nested_roots_result = await self.environment.exec(
            _NESTED_GIT_ROOTS_COMMAND,
            cwd=self.workdir,
            timeout_sec=30,
        )
        status_text = _merge_workspace_status(
            status_result.stdout or "",
            "\n".join(
                part
                for part in (nested_committed_status, nested_status_result.stdout or "")
                if part.strip()
            ),
            nested_roots=set((nested_roots_result.stdout or "").splitlines())
            if nested_roots_result.return_code == 0
            else set[str](),
        )
        tracked_paths: set[str] = set((tracked_result.stdout or "").splitlines())
        tracked_paths.update((nested_tracked_result.stdout or "").splitlines())
        return status_text, tracked_paths

    async def _workspace_declares_container_runtime(self) -> bool:
        result = await self.environment.exec(
            (
                "if [ -f environment/Dockerfile ] || [ -f Dockerfile ] || "
                "[ -f docker-compose.yml ] || [ -f docker-compose.yaml ] || "
                "[ -f compose.yml ] || [ -f compose.yaml ]; then "
                "printf '__container_runtime_declared__\\n'; exit 0; fi; "
                "for file in task.toml environment.toml harness.toml; do "
                'if [ -f "$file" ] && grep -Eiq '
                "'docker[_-]?image|container[_-]?image|prebuilt[_-]?image|dockerfile' "
                "\"$file\"; then printf '__container_runtime_declared__\\n'; exit 0; fi; "
                "done; exit 1"
            ),
            cwd=self.workdir,
            timeout_sec=30,
        )
        return "__container_runtime_declared__" in (result.stdout or "")

    async def _missing_tool_declared_runtime_reason(
        self,
        tool_events: list[ActivityEvent],
    ) -> str:
        if not any(_activity_event_reports_missing_command(event) for event in tool_events):
            return ""
        if any(_activity_event_checks_docker_runtime(event) for event in tool_events):
            return ""
        if not await self._workspace_declares_container_runtime():
            return ""
        return (
            "Verification failed because a required command was missing, and project "
            "metadata declares a Dockerfile or container image, but the run did not "
            "check Docker availability. Continue autonomously: run `command -v docker` "
            "or `docker info`, then use the declared container path, an available "
            "install path, or another project-evidenced runtime before asking the user "
            "for toolchain support."
        )

    async def _required_git_base_commit_reason(self, source_paths: list[str]) -> str:
        required_base = self.policy.required_git_base_commit
        if not required_base or not source_paths:
            return ""
        result = await self.environment.exec(
            _required_git_base_scan_command(required_base),
            cwd=self.workdir,
            timeout_sec=30,
        )
        if result.return_code != 0:
            return ""
        modified_repos: list[tuple[str, str]] = []
        matching_modified_repos: list[str] = []
        for line in (result.stdout or "").splitlines():
            parts = line.split("\t")
            if len(parts) < 6 or parts[0] != "repo" or parts[2] != "head" or parts[4] != "modified":
                continue
            repo, head, modified = parts[1], parts[3], parts[5]
            base_state = "head" if head == required_base else ""
            dirty = "yes"
            if len(parts) >= 8 and parts[6] == "base":
                base_state = parts[7]
            if len(parts) >= 10 and parts[8] == "dirty":
                dirty = parts[9]
            if modified != "yes":
                continue
            modified_repos.append((repo, head))
            if (
                head == required_base
                or base_state == "head"
                or (base_state == "descendant" and dirty != "yes")
            ):
                matching_modified_repos.append(repo)
        if matching_modified_repos:
            return ""
        if modified_repos:
            repo_summary = ", ".join(f"{repo}@{head[:12]}" for repo, head in modified_repos)
            return (
                "Public workspace metadata requires target source changes in a nested "
                f"git checkout at base commit {required_base}, but the modified nested "
                f"git checkout(s) are at different HEADs: {repo_summary}. "
                "Continue autonomously: preserve useful edits if needed, checkout the "
                "required base commit in the target repository, reapply the fix there, "
                "then run verification again."
            )
        return (
            "Public workspace metadata requires target source changes in a nested "
            f"git checkout at base commit {required_base}, but no modified nested git "
            "checkout at that base commit was found. Continue autonomously: clone or "
            "checkout the public repository in a subdirectory, checkout the required "
            "base commit, apply the fix inside that checkout, then run verification "
            "again. Do not stage or git add the embedded checkout in the parent "
            "workspace; parent-level gitlink or untracked-directory changes do not "
            "count as the task fix."
        )

    async def _no_source_change_nested_repo_hint(self) -> str:
        required_base = self.policy.required_git_base_commit
        if not required_base:
            return ""
        result = await self.environment.exec(
            _required_git_base_scan_command(required_base),
            cwd=self.workdir,
            timeout_sec=30,
        )
        if result.return_code != 0:
            return ""
        clean_target_repos: list[str] = []
        clean_wrong_repos: list[str] = []
        for line in (result.stdout or "").splitlines():
            parts = line.split("\t")
            if (
                len(parts) < 10
                or parts[0] != "repo"
                or parts[2] != "head"
                or parts[4] != "modified"
                or parts[6] != "base"
                or parts[8] != "dirty"
            ):
                continue
            repo = parts[1]
            base_state = parts[7]
            dirty = parts[9]
            modified = parts[5]
            if modified == "yes" or dirty == "yes":
                continue
            if base_state == "head":
                clean_target_repos.append(repo)
            elif base_state in {"missing", "unrelated"}:
                clean_wrong_repos.append(repo)
        if clean_target_repos:
            repo_summary = ", ".join(clean_target_repos[:3])
            return (
                " A nested target git checkout already exists at the required base "
                f"commit but is clean: {repo_summary}. Apply the implementation "
                "inside that checkout, leave the wrapper task metadata intact, then "
                "run verification from the workspace."
            )
        if clean_wrong_repos:
            repo_summary = ", ".join(clean_wrong_repos[:3])
            return (
                " Nested git checkout(s) exist but are not at the required base "
                f"commit {required_base}: {repo_summary}. Checkout the required "
                "base in the target repository before applying the fix."
            )
        return ""

    async def _runner_wires_untracked_tests(
        self,
        *,
        source_paths: list[str],
        untracked_test_paths: list[str],
        command: str | None,
        used_default_command: bool,
    ) -> bool:
        if (
            not source_paths
            or not untracked_test_paths
            or not command
            or not used_default_command
            or not _command_invokes_opaque_test_wrapper(command)
            or not _is_broad_test_command(command)
        ):
            return False
        quoted_paths = " ".join(shlex.quote(path) for path in source_paths)
        if not quoted_paths:
            return False
        result = await self.environment.exec(
            f"git diff --no-ext-diff -- {quoted_paths}",
            cwd=self.workdir,
            timeout_sec=30,
        )
        if result.return_code != 0:
            return False
        return _diff_references_untracked_test_paths(
            result.stdout or "",
            untracked_test_paths,
        )

    async def _runner_wires_changed_tests(
        self,
        *,
        source_paths: list[str],
        test_paths: list[str],
        command: str | None,
    ) -> bool:
        if not source_paths or not test_paths or not command:
            return False
        runner_paths = set(_command_local_script_paths(command))
        if not runner_paths:
            return False
        source_path_set = {_normal_path(path) for path in source_paths}
        changed_runner_paths = [
            path for path in sorted(runner_paths) if path and path in source_path_set
        ]
        if not changed_runner_paths:
            return False
        quoted_paths = " ".join(shlex.quote(path) for path in changed_runner_paths)
        result = await self.environment.exec(
            f"git diff --no-ext-diff -- {quoted_paths}",
            cwd=self.workdir,
            timeout_sec=30,
        )
        if result.return_code != 0:
            return False
        return _diff_references_test_paths(result.stdout or "", test_paths)

    async def verify(
        self,
        *,
        session: Any,
        activity: list[ActivityEvent],
    ) -> VerificationResult:
        status_text, tracked_paths = await self._workspace_status()
        source_passed, source_paths, scratch_paths = _workspace_source_change_status(
            status_text,
            tracked_paths=tracked_paths,
            baseline_untracked_paths=self.baseline_untracked_paths,
            ignored_paths=self.ignored_workspace_paths,
        )
        test_paths = _workspace_test_change_paths(
            status_text,
            tracked_paths=tracked_paths,
            baseline_untracked_paths=self.baseline_untracked_paths,
            ignored_paths=self.ignored_workspace_paths,
        )
        untracked_test_paths = [
            path
            for code, path in _porcelain_paths(status_text)
            if code == "??" and path in test_paths
        ]

        snapshot = ExternalWorkspaceVerificationSnapshot(
            source_change_passed=source_passed,
            source_change_paths=source_paths,
            scratch_paths=scratch_paths,
            test_change_paths=test_paths,
        )
        self.latest = snapshot

        if self.fail_without_source_change and not source_passed:
            if test_paths:
                snapshot.source_change_error = (
                    "Harness did not make an implementation/source change; only "
                    "in-repository test or fixture changes were detected. Keep or add "
                    "objective-covering regression tests, then implement the source "
                    "behavior they exercise before finishing."
                )
            else:
                snapshot.source_change_error = (
                    "Harness did not make an implementation/source change; refusing to submit "
                    "scratch-only or inspection-only external workspace work."
                )
            if _assistant_defers_setup_to_user(session):
                snapshot.source_change_error = _autonomous_setup_repair_reason()
                return VerificationResult(
                    can_finish=False,
                    reason=snapshot.source_change_error,
                    confidence=1.0,
                    verifier_name=self.name,
                )
            if _assistant_waits_for_permission_to_continue(session):
                snapshot.source_change_error = _autonomous_permission_repair_reason()
                return VerificationResult(
                    can_finish=False,
                    reason=snapshot.source_change_error,
                    confidence=1.0,
                    verifier_name=self.name,
                )
            recovery_hint = (
                " If the source tree is not visible in the current directory, continue "
                "autonomously with the public workspace evidence: inspect task metadata, "
                "environment files, project setup files, and available shell tools to "
                "discover or create a patchable repository inside the workspace. If the "
                "current workspace contains only task metadata, keep those files intact "
                "and clone or create the patchable project in a subdirectory such as "
                "`repo/` instead of overwriting `.`; do not treat the metadata wrapper "
                "`.git` checkout as the target source repository. If "
                "public metadata or environment files show setup commands, a source "
                "repository URL, a base commit, a container image, or dependency install "
                "steps, run the appropriate available command yourself before asking "
                "the user. Do not ask the user to provide repository files, tools, or "
                "setup unless those public setup paths have been attempted and failed."
            )
            nested_repo_hint = await self._no_source_change_nested_repo_hint()
            return VerificationResult(
                can_finish=False,
                reason=(
                    f"{snapshot.source_change_error} "
                    f"Scratch/non-source paths: {', '.join(scratch_paths) or 'none'}."
                    f"{nested_repo_hint}"
                    f"{recovery_hint}"
                ),
                confidence=1.0,
                verifier_name=self.name,
            )

        required_base_reason = await self._required_git_base_commit_reason(source_paths)
        if required_base_reason:
            snapshot.verification_error = required_base_reason
            snapshot.latest_verification_error = required_base_reason
            return VerificationResult(
                can_finish=False,
                reason=required_base_reason,
                confidence=1.0,
                verifier_name=self.name,
            )

        if self.require_regression_test_change and source_passed and not test_paths:
            snapshot.latest_verification_error = (
                "Implementation files changed but no in-repository regression test changes "
                "were detected. Add or update a focused project test in the existing test "
                "structure, then run verify_work after the final code/test change."
            )
            snapshot.verification_error = snapshot.latest_verification_error
            return VerificationResult(
                can_finish=False,
                reason=snapshot.latest_verification_error,
                confidence=0.95,
                verifier_name=self.name,
            )

        if source_passed and scratch_paths:
            snapshot.latest_verification_error = (
                "Workspace still contains scratch/non-source artifacts after the "
                "implementation change. Remove temporary debug files or convert them "
                "into intentional source/test artifacts before finishing."
            )
            snapshot.verification_error = snapshot.latest_verification_error
            return VerificationResult(
                can_finish=False,
                reason=(
                    f"{snapshot.latest_verification_error} "
                    f"Scratch/non-source paths: {', '.join(scratch_paths)}."
                ),
                confidence=0.95,
                verifier_name=self.name,
            )

        tool_events = [event for event in activity if event.kind == "tool_call.completed"]
        last_workspace_change_index = -1
        for index, event in enumerate(tool_events):
            if _activity_tool_changes_workspace(event):
                last_workspace_change_index = index

        latest_verify_after_change: ActivityEvent | None = None
        latest_verify_error: str | None = None
        for index, event in enumerate(tool_events):
            if not _activity_tool_is_verification_attempt(
                event,
                test_paths=test_paths,
                untracked_test_paths=untracked_test_paths,
            ):
                continue
            metadata = _dict_metadata(event.data.get("metadata"))
            command = metadata.get("command")
            if not isinstance(command, str):
                command = _activity_event_command(event)
            if isinstance(command, str):
                snapshot.latest_verification_command = command
            snapshot.latest_verification_used_default_command = (
                metadata.get("used_default_command") is True
            )
            if index < last_workspace_change_index:
                continue
            latest_verify_error = None
            if event.data.get("is_error") is True:
                preview = str(event.data.get("content_preview") or "")
                latest_verify_error = (
                    preview or f"{event.data.get('name') or 'verification'} failed"
                )
            elif metadata.get("workspace_changed") is True:
                preview = str(event.data.get("content_preview") or "")
                latest_verify_error = (
                    preview or f"{event.data.get('name') or 'verification'} changed workspace"
                )
            latest_verify_after_change = event

        if latest_verify_after_change is not None:
            metadata = _dict_metadata(latest_verify_after_change.data.get("metadata"))
            command = metadata.get("command")
            if not isinstance(command, str):
                command = _activity_event_command(latest_verify_after_change)
            if isinstance(command, str):
                snapshot.latest_verification_command = command
            snapshot.latest_verification_used_default_command = (
                metadata.get("used_default_command") is True
            )
            if not _activity_tool_is_passing_verify(latest_verify_after_change):
                preview = str(latest_verify_after_change.data.get("content_preview") or "")
                if latest_verify_after_change.data.get("is_error") is True:
                    latest_verify_error = preview or "verify_work failed"
                elif metadata.get("workspace_changed") is True:
                    latest_verify_error = preview or "verify_work changed workspace"
                snapshot.latest_verification_error = latest_verify_error
                declared_runtime_reason = await self._missing_tool_declared_runtime_reason(
                    tool_events
                )
                if declared_runtime_reason:
                    snapshot.verification_error = declared_runtime_reason
                    return VerificationResult(
                        can_finish=False,
                        reason=declared_runtime_reason,
                        confidence=1.0,
                        verifier_name=self.name,
                        evidence_event_ids=[latest_verify_after_change.id],
                    )
                if _assistant_defers_setup_to_user(session):
                    snapshot.verification_error = _autonomous_setup_repair_reason(
                        latest_verify_error=latest_verify_error or None
                    )
                    return VerificationResult(
                        can_finish=False,
                        reason=snapshot.verification_error,
                        confidence=1.0,
                        verifier_name=self.name,
                        evidence_event_ids=[latest_verify_after_change.id],
                    )
                if _assistant_waits_for_permission_to_continue(session):
                    snapshot.verification_error = _autonomous_permission_repair_reason(
                        latest_verify_error=latest_verify_error or None
                    )
                    return VerificationResult(
                        can_finish=False,
                        reason=snapshot.verification_error,
                        confidence=1.0,
                        verifier_name=self.name,
                        evidence_event_ids=[latest_verify_after_change.id],
                    )
                snapshot.verification_error = (
                    "The latest verify_work after the final workspace mutation did not pass; "
                    "Harness did not produce a later passing in-repository verification "
                    "command, so refusing to submit stale or contradicted verification evidence."
                )
                return VerificationResult(
                    can_finish=False,
                    reason=(
                        f"{snapshot.verification_error} "
                        f"Latest verify_work error: {latest_verify_error or 'none observed'}."
                        f"{_verification_failure_replan_hint()}"
                    ),
                    confidence=1.0,
                    verifier_name=self.name,
                    evidence_event_ids=[latest_verify_after_change.id],
                )
            if (
                self.require_default_verify_command
                and not snapshot.latest_verification_used_default_command
            ):
                snapshot.latest_verification_error = (
                    "Passing verify_work did not use the configured default verifier. "
                    "External benchmark runs require harness-owned verifier evidence."
                )
                snapshot.verification_error = snapshot.latest_verification_error
                return VerificationResult(
                    can_finish=False,
                    reason=snapshot.latest_verification_error,
                    confidence=1.0,
                    verifier_name=self.name,
                    evidence_event_ids=[latest_verify_after_change.id],
                )
            if self.policy.required_no_network_verify_image and not (
                isinstance(snapshot.latest_verification_command, str)
                and _shell_command_uses_no_network_container(
                    snapshot.latest_verification_command,
                    image=self.policy.required_no_network_verify_image,
                )
            ):
                snapshot.latest_verification_error = (
                    "Passing verify_work did not reproduce the public check inside the "
                    "declared no-network task image. Public task metadata declares "
                    f"`{self.policy.required_no_network_verify_image}` with network "
                    "disabled for isolated grading; run the relevant project check in "
                    "that Docker image with `--network none`, using the current target "
                    "checkout, then call verify_work again. Do not use hidden tests or "
                    "reference solutions."
                )
                snapshot.verification_error = snapshot.latest_verification_error
                return VerificationResult(
                    can_finish=False,
                    reason=snapshot.latest_verification_error,
                    confidence=1.0,
                    verifier_name=self.name,
                    evidence_event_ids=[latest_verify_after_change.id],
                )
            runner_wires_untracked_tests = await self._runner_wires_untracked_tests(
                source_paths=source_paths,
                untracked_test_paths=untracked_test_paths,
                command=snapshot.latest_verification_command,
                used_default_command=snapshot.latest_verification_used_default_command,
            )
            runner_wires_changed_tests = await self._runner_wires_changed_tests(
                source_paths=source_paths,
                test_paths=test_paths,
                command=snapshot.latest_verification_command,
            )
            must_cover_changed_tests = bool(
                test_paths and not snapshot.latest_verification_used_default_command
            )
            if must_cover_changed_tests and not _verification_command_covers_test_changes(
                snapshot.latest_verification_command,
                test_paths,
                untracked_test_paths=untracked_test_paths,
                runner_wires_untracked_tests=runner_wires_untracked_tests,
                runner_wires_changed_tests=runner_wires_changed_tests,
            ):
                default_clause = (
                    " External benchmark runs must finish with the configured default "
                    "verify_work when one exists."
                    if self.require_default_verify_command
                    else ""
                )
                snapshot.latest_verification_error = (
                    "Passing verify_work did not cover the changed regression tests. Wire "
                    "changed tests into the project test runner, run the changed test "
                    "file with its real test framework, or run a broad project test "
                    "command that actually includes them after the final code/test "
                    f"change.{default_clause}"
                )
                snapshot.verification_error = snapshot.latest_verification_error
                return VerificationResult(
                    can_finish=False,
                    reason=snapshot.latest_verification_error,
                    confidence=0.95,
                    verifier_name=self.name,
                    evidence_event_ids=[latest_verify_after_change.id],
                )
            snapshot.verification_passed_after_source_change = True
            return VerificationResult(
                can_finish=True,
                reason=(
                    "External workspace gate passed: source files changed, regression tests "
                    "changed, and verify_work passed after the final workspace mutation."
                ),
                confidence=0.95,
                verifier_name=self.name,
                evidence_event_ids=[latest_verify_after_change.id],
            )

        if self.fail_without_verification and source_passed:
            snapshot.latest_verification_error = latest_verify_error
            declared_runtime_reason = await self._missing_tool_declared_runtime_reason(tool_events)
            if declared_runtime_reason:
                snapshot.verification_error = declared_runtime_reason
                return VerificationResult(
                    can_finish=False,
                    reason=declared_runtime_reason,
                    confidence=1.0,
                    verifier_name=self.name,
                )
            if _assistant_defers_setup_to_user(session):
                snapshot.verification_error = _autonomous_setup_repair_reason(
                    latest_verify_error=latest_verify_error or None
                )
                return VerificationResult(
                    can_finish=False,
                    reason=snapshot.verification_error,
                    confidence=1.0,
                    verifier_name=self.name,
                )
            if _assistant_waits_for_permission_to_continue(session):
                snapshot.verification_error = _autonomous_permission_repair_reason(
                    latest_verify_error=latest_verify_error or None
                )
                return VerificationResult(
                    can_finish=False,
                    reason=snapshot.verification_error,
                    confidence=1.0,
                    verifier_name=self.name,
                )
            snapshot.verification_error = (
                "Harness made an implementation/source change but did not produce a later "
                "passing in-repository verification command; refusing to submit unverified "
                "external workspace work."
            )
            return VerificationResult(
                can_finish=False,
                reason=(
                    f"{snapshot.verification_error} "
                    f"Latest verify_work error: {latest_verify_error or 'none observed'}."
                    f"{_verification_failure_replan_hint()}"
                ),
                confidence=1.0,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=True,
            reason="External workspace source-change gate did not require verification.",
            confidence=0.5,
            verifier_name=self.name,
        )


_COVERAGE_REVIEW_SYSTEM_PROMPT = (
    "You are an adversarial reviewer for an autonomous coding agent. You see the "
    "original task, the changed source/test files, the verification command, and "
    "the agent's final answer. Decide whether the changed in-repository tests are "
    "strong enough to catch obvious failures of the requested behavior.\n\n"
    "Fail if the tests only cover a narrow happy path while important behavior from "
    "the task is untested, if edge cases naturally implied by the task are missing, "
    "if the tests encode expectations that contradict the task, or if the final "
    "answer claims broader behavior than the tests exercise. Pass if the tests "
    "cover the main observable requirements well enough for an agent to rely on "
    "them. Do not require exhaustive testing, and do not fail on expectations "
    "that are merely arguable or unspecified by the task. Stay inside the user's "
    "stated behavior: do not invent adjacent protocol, security, encoding, "
    "performance, concurrency, or platform semantics unless the task, changed "
    "source, or final answer explicitly claims them. For example, preserving a "
    "URL scheme/host does not by itself require query-string or fragment "
    "semantics; sorting text does not by itself require locale/Unicode collation; "
    "and handling dates does not by itself require timezone calendars unless "
    "those behaviors are requested. If your only objection is that an adjacent "
    "domain feature 'should not break when present' but that feature is not "
    "requested or claimed, return can_finish=true. For boundary words such as "
    "leading, trailing, first, last, root, prefix, or suffix, compare tests "
    "against the boundary of the whole relevant input/result. Do not let tests "
    "redefine an interior separator or later segment as a leading/trailing "
    "property unless the task explicitly says any segment can do that.\n\n"
    "Before approving, actively try to name one simple counterexample that satisfies "
    "the task but could pass the changed tests. Treat words such as normalize, "
    "sanitize, safe, valid, parse, canonical, escape, trim, collapse, preserve, "
    "or ignore as signals that representative invalid, boundary, and mixed-input "
    "cases matter. If the changed source appears to mishandle an obvious "
    "counterexample and the tests would not catch it, fail and state that concrete "
    "counterexample. Do not require exhaustive variants. If tests already cover a "
    "representative equivalence class and the changed source clearly handles a "
    "new variant from that same class, pass instead of asking for another near-"
    "duplicate case. Fail only for a distinct stated behavior, a test expectation "
    "that conflicts with the stated behavior, or an obvious source bug that the "
    "changed tests would miss.\n\n"
    "Return only JSON on one line: "
    '{"can_finish": true|false, "reason": "<short actionable reason>", '
    '"confidence": 0.0..1.0}'
)


class ExternalWorkspaceCoverageVerifier:
    """Semantic regression-coverage review for external workspace tasks.

    This verifier does not receive hidden benchmark tests or reference solutions.
    It reviews only agent-visible artifacts after the structural gate has already
    found source changes, changed project tests, and a passing verify_work.
    """

    name = "external_workspace_coverage"

    def __init__(
        self,
        *,
        environment: Any,
        workdir: str,
        instruction: str,
        adapter: Any,
        model: str,
        structural_verifier: ExternalWorkspaceVerifier,
        max_retries: int = 2,
        block_confidence: float | None = None,
    ) -> None:
        self.environment = environment
        self.workdir = workdir
        self.instruction = instruction
        self.adapter = adapter
        self.model = model
        self.structural_verifier = structural_verifier
        self.max_retries = max_retries
        self.block_confidence = (
            _external_workspace_coverage_block_confidence()
            if block_confidence is None
            else max(0.0, min(1.0, block_confidence))
        )

    async def verify(
        self,
        *,
        session: Any,
        activity: list[ActivityEvent],
    ) -> VerificationResult:
        snapshot = self.structural_verifier.latest
        if not snapshot.verification_passed_after_source_change:
            return VerificationResult(
                can_finish=True,
                reason="structural verifier has not accepted the workspace yet",
                confidence=0.5,
                verifier_name=self.name,
            )

        source_content = await self._read_changed_paths(snapshot.source_change_paths or [])
        test_content = await self._read_changed_paths(snapshot.test_change_paths or [])
        final_answer = ""
        for message in reversed(getattr(session, "messages", []) or []):
            if getattr(message, "role", "") == "assistant" and getattr(message, "content", ""):
                final_answer = str(message.content)
                break
        prompt = (
            f"ORIGINAL TASK:\n{self.instruction}\n\n"
            f"CHANGED SOURCE FILES:\n{source_content or '(not readable)'}\n\n"
            f"CHANGED TEST FILES:\n{test_content or '(not readable)'}\n\n"
            f"VERIFY COMMAND:\n{snapshot.latest_verification_command or '(unknown)'}\n\n"
            f"AGENT FINAL ANSWER:\n{final_answer or '(empty)'}\n"
        )
        messages = [
            Message(role="system", content=_COVERAGE_REVIEW_SYSTEM_PROMPT),
            Message(role="user", content=prompt[:24_000]),
        ]

        last_reason = "coverage reviewer failed after retries"
        for attempt in range(self.max_retries):
            if attempt:
                await asyncio.sleep(2**attempt)
            accumulated: list[str] = []
            final_content: str | None = None
            try:
                async for event in self.adapter.stream(model=self.model, messages=messages):
                    if isinstance(event, TextDelta):
                        accumulated.append(event.text)
                    elif isinstance(event, Done):
                        final_content = (
                            event.final_message.content
                            if event.final_message and event.final_message.content
                            else "".join(accumulated)
                        )
                        break
            except Exception as exc:
                last_reason = f"coverage reviewer call failed: {exc!s}"
                continue
            parsed = _parse_judge_response(final_content or "")
            if parsed is None:
                last_reason = (
                    f"coverage reviewer returned non-JSON: {(final_content or '')[:200]!r}"
                )
                continue
            can_finish, reason, confidence = parsed
            confidence_value = confidence if confidence is not None else 0.0
            if can_finish:
                return VerificationResult(
                    can_finish=True,
                    reason=f"coverage review passed: {reason}",
                    confidence=confidence_value,
                    verifier_name=self.name,
                )
            if confidence_value < self.block_confidence:
                return VerificationResult(
                    can_finish=True,
                    reason=(
                        "coverage review advisory below blocking confidence "
                        f"({confidence_value:.2f} < {self.block_confidence:.2f}): {reason}"
                    ),
                    confidence=confidence_value,
                    verifier_name=self.name,
                )
            return VerificationResult(
                can_finish=False,
                reason=f"Regression coverage is too weak: {reason}",
                confidence=confidence,
                verifier_name=self.name,
            )
        return VerificationResult(
            can_finish=False,
            reason=last_reason,
            confidence=0.0,
            verifier_name=self.name,
        )

    async def _read_changed_paths(self, paths: list[str]) -> str:
        if not paths:
            return ""
        selected = [path for path in paths if path][:12]
        quoted = " ".join(shlex.quote(path) for path in selected)
        result = await self.environment.exec(
            (
                "for path in "
                f"{quoted}; do "
                '[ -f "$path" ] || continue; '
                "printf '\\n--- %s ---\\n' \"$path\"; "
                "sed -n '1,240p' \"$path\"; "
                "done"
            ),
            cwd=self.workdir,
            timeout_sec=30,
        )
        return (result.stdout or "")[:16_000] if result.return_code == 0 else ""


async def _noop_task_attachment(
    _storage: object,
    _task_ref: str | None,
    _session_id: str | None,
) -> tuple[None, None]:
    return None, None


def _memory_storage(*, db: Any, in_memory: bool, cwd: Any = None) -> InMemoryStorage:
    return InMemoryStorage()


async def _noop_defense_ledger(*_args: Any, **_kwargs: Any) -> None:
    return None


def _noop_search_fn() -> None:
    return None


def _workspace_relative_path(path: str, *, root: str) -> str | None:
    try:
        relative = Path(path).resolve(strict=False).relative_to(Path(root).resolve(strict=False))
    except (OSError, ValueError):
        return None
    relative_text = relative.as_posix()
    if not relative_text or relative_text == ".":
        return None
    return relative_text


def _positive_seconds_env_value(value: float | int | None, *, name: str) -> str | None:
    if value is None:
        return None
    seconds = float(value)
    if seconds <= 0:
        raise ValueError(f"{name} must be positive")
    return f"{seconds:g}"


def _normalized_model_name(value: str) -> str:
    return value.strip().removeprefix("openrouter/").strip()


def _split_model_names(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace("\n", ",")
    return [model for item in normalized.split(",") if (model := _normalized_model_name(item))]


def _unique_model_names(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        model = _normalized_model_name(value)
        if not model or model in seen:
            continue
        seen.add(model)
        unique.append(model)
    return unique


_DEFAULT_EXCLUDED_MODEL_FALLBACKS = ("qwen/qwen3-coder",)


def _excluded_model_fallback_names() -> set[str]:
    source = os.environ.get("HARNESS_EXTERNAL_WORKSPACE_EXCLUDED_MODEL_FALLBACKS")
    if source is None:
        source = os.environ.get("HARNESS_OPENROUTER_EXCLUDED_MODEL_FALLBACKS")
    if source is None:
        source = ",".join(_DEFAULT_EXCLUDED_MODEL_FALLBACKS)
    return {model.lower() for model in _split_model_names(source)}


def _external_workspace_model_candidates(model: str) -> list[str]:
    primary = _normalized_model_name(model)
    fallback_source = os.environ.get("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACKS")
    if fallback_source is None:
        fallback_source = os.environ.get("HARNESS_OPENROUTER_MODEL_FALLBACKS")
    excluded_fallbacks = _excluded_model_fallback_names()
    fallbacks = [
        fallback
        for fallback in _split_model_names(fallback_source)
        if fallback.lower() not in excluded_fallbacks
    ]
    limit_source = os.environ.get("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACK_LIMIT")
    if limit_source is None:
        limit_source = os.environ.get("HARNESS_OPENROUTER_MODEL_FALLBACK_LIMIT")
    try:
        fallback_limit = int(limit_source) if limit_source and limit_source.strip() else 2
    except ValueError:
        fallback_limit = 2
    fallback_limit = max(0, fallback_limit)
    return _unique_model_names([primary, *fallbacks[:fallback_limit]])


_EXTERNAL_WORKSPACE_RETRYABLE_RUNTIME_KINDS = frozenset(
    {"internal", "model_unavailable", "network", "rate_limit", "timeout"}
)


def _external_workspace_coverage_review_enabled() -> bool:
    raw = os.environ.get("HARNESS_EXTERNAL_WORKSPACE_COVERAGE_REVIEW", "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "disabled"}:
        return False
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _external_workspace_coverage_block_confidence() -> float:
    raw = os.environ.get("HARNESS_EXTERNAL_WORKSPACE_COVERAGE_BLOCK_CONFIDENCE", "0.65")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.65


def _should_retry_external_workspace_with_fallback(
    *,
    run_error: str | None,
    latest_runtime_error_kind: str | None,
    latest_runtime_error_recoverable: bool,
    snapshot: ExternalWorkspaceVerificationSnapshot,
    attempt_index: int,
    model_candidates: list[str],
) -> bool:
    if run_error is None or attempt_index + 1 >= len(model_candidates):
        return False
    if not latest_runtime_error_recoverable:
        return False
    if latest_runtime_error_kind not in _EXTERNAL_WORKSPACE_RETRYABLE_RUNTIME_KINDS:
        return False
    if _snapshot_has_no_workspace_mutation(snapshot):
        return True
    return bool(snapshot.source_change_paths)


def _snapshot_has_no_workspace_mutation(snapshot: ExternalWorkspaceVerificationSnapshot) -> bool:
    return not (
        snapshot.source_change_paths or snapshot.test_change_paths or snapshot.scratch_paths
    )


def _snapshot_has_external_workspace_state(
    snapshot: ExternalWorkspaceVerificationSnapshot,
) -> bool:
    return bool(
        snapshot.source_change_passed
        or snapshot.source_change_paths
        or snapshot.test_change_paths
        or snapshot.scratch_paths
        or snapshot.verification_passed_after_source_change
        or snapshot.latest_verification_error
        or snapshot.latest_verification_command
        or snapshot.source_change_error
        or snapshot.verification_error
    )


async def run_harness_on_external_environment(
    *,
    instruction: str,
    environment: Any,
    context: Any,
    logs_dir: str,
    model_name: str | None = None,
    max_steps: int = 30,
    max_output_tokens: int | None = 4096,
    source_change_retries: int = 1,
    verification_retries: int = 1,
    pass_timeout_seconds: float = 180.0,
    fail_without_source_change: bool = True,
    fail_without_verification: bool = True,
    require_regression_test_change: bool = True,
    policy: ExternalWorkspacePolicy | None = None,
    default_verify_command: str | None = None,
    default_verify_timeout_seconds: int | None = None,
    model_stream_idle_timeout_seconds: float | int | None = None,
    model_turn_timeout_seconds: float | int | None = None,
) -> None:
    workdir_result = await environment.exec("pwd")
    workdir = (workdir_result.stdout or "").strip() or "/app"
    model = (model_name or "google/gemma-4-26b-a4b-it").removeprefix("openrouter/")
    log_path = os.path.join(str(logs_dir), "harness.txt")
    events_path = os.path.join(str(logs_dir), "harness-events.jsonl")
    os.makedirs(str(logs_dir), exist_ok=True)

    baseline_status_result = await environment.exec(
        _GIT_STATUS_COMMAND,
        cwd=workdir,
        timeout_sec=30,
    )
    baseline_untracked_paths = {
        path for code, path in _porcelain_paths(baseline_status_result.stdout or "") if code == "??"
    }
    ignored_workspace_paths = set()
    logs_relative_path = _workspace_relative_path(str(logs_dir), root=workdir)
    if logs_relative_path is not None:
        ignored_workspace_paths.add(logs_relative_path)
    policy = _policy_with_forbidden_logs(policy, logs_relative_path)
    prompt = instruction
    model_candidates = _external_workspace_model_candidates(model)
    total_event_count = 0
    final_text = ""
    run_error: str | None = None
    latest_runtime_error_kind: str | None = None
    latest_runtime_error: str | None = None
    latest_runtime_error_recoverable = False
    effective_model: str | None = None
    model_selection: dict[str, Any] | None = None
    snapshot = ExternalWorkspaceVerificationSnapshot()
    source_change_error: str | None = None
    verification_error: str | None = None
    attempt_model = model
    runtime_attempts: list[dict[str, Any]] = []
    repair_attempts = external_workspace_repair_attempts(
        source_change_retries=source_change_retries,
        verification_retries=verification_retries,
    )
    total_attempt_budget = external_workspace_total_attempts(
        source_change_retries=source_change_retries,
        verification_retries=verification_retries,
    )

    log = await asyncio.to_thread(open, log_path, "w", encoding="utf-8", buffering=1)
    event_log = await asyncio.to_thread(open, events_path, "w", encoding="utf-8", buffering=1)
    try:
        for attempt_index, attempt_model in enumerate(model_candidates):
            if attempt_index:
                retry_scope = (
                    "current workspace"
                    if snapshot.source_change_paths
                    else "workspace before mutation"
                )
                log.write(
                    "\n[harness:model] retrying external workspace run with fallback "
                    f"model {attempt_model} after a runtime failure; continuing from "
                    f"{retry_scope}.\n"
                )
                event_log.write(
                    json.dumps(
                        {
                            "event": "ExternalWorkspaceModelRetry",
                            "attempt": attempt_index,
                            "model": attempt_model,
                            "requested_model": model,
                            "source_change_paths": list(snapshot.source_change_paths or []),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

            registry = build_remote_tool_registry(
                environment,
                workdir=workdir,
                policy=policy,
                default_verify_command=default_verify_command,
                default_verify_timeout_seconds=default_verify_timeout_seconds,
            )
            verifier = ExternalWorkspaceVerifier(
                environment,
                workdir=workdir,
                baseline_untracked_paths=baseline_untracked_paths,
                ignored_workspace_paths=ignored_workspace_paths,
                fail_without_source_change=fail_without_source_change,
                fail_without_verification=fail_without_verification,
                require_regression_test_change=require_regression_test_change,
                require_default_verify_command=bool(default_verify_command),
                policy=policy,
            )
            cfg = HarnessConfig(default_provider="openrouter", default_model=attempt_model)

            def _build_remote_tools(
                _cwd: Any,
                *,
                config: Any = None,
                include: set[str] | None = None,
                extras: dict[str, Any] | None = None,
                _registry: ToolRegistry = registry,
            ) -> ToolRegistry:
                if include is None:
                    return _registry
                filtered = ToolRegistry()
                for name in sorted(include):
                    if _registry.has(name):
                        filtered.register(_registry.get(name))
                return filtered

            def _build_remote_agent(**kwargs: Any) -> Any:
                kwargs.pop("build_tools", None)
                return _runtime_build_agent(
                    **kwargs,
                    build_adapter=_build_adapter,
                    build_tools=_build_remote_tools,
                    build_search_fn=_noop_search_fn,
                    console=Console(file=io.StringIO()),
                    auxiliary_tools_enabled=False,
                    memory_tools_enabled=False,
                    project_context_enabled=False,
                )

            def _build_external_workspace_verifier(
                *_args: Any,
                _verifier: ExternalWorkspaceVerifier = verifier,
                _cfg: HarnessConfig = cfg,
                _attempt_model: str = attempt_model,
                **_kwargs: Any,
            ) -> Any:
                if not _external_workspace_coverage_review_enabled():
                    return ChainedVerifier(_verifier, fail_fast=True)
                coverage_verifier = ExternalWorkspaceCoverageVerifier(
                    environment=environment,
                    workdir=workdir,
                    instruction=instruction,
                    adapter=_build_adapter("openrouter", base_url=None, config=_cfg),
                    model=_attempt_model,
                    structural_verifier=_verifier,
                )
                return ChainedVerifier(_verifier, coverage_verifier, fail_fast=True)

            event_count = 0
            final_text = ""
            run_error = None
            latest_runtime_error_kind = None
            latest_runtime_error = None
            latest_runtime_error_recoverable = False
            effective_model = None
            model_selection = None
            session_id = f"sess_harness_external_{uuid4().hex[:12]}"

            def _render(event: Any) -> None:
                nonlocal event_count
                nonlocal final_text
                nonlocal run_error
                nonlocal latest_runtime_error_kind
                nonlocal latest_runtime_error
                nonlocal latest_runtime_error_recoverable
                nonlocal effective_model
                nonlocal model_selection
                event_count += 1
                try:
                    event_log.write(event.model_dump_json() + "\n")
                except Exception:
                    event_log.write(json.dumps({"event": event.__class__.__name__}) + "\n")
                event_log.flush()
                if isinstance(event, TextDelta):
                    log.write(event.text)
                    final_text += event.text
                elif isinstance(event, ToolCallEvent):
                    args = json.dumps(event.call.arguments, sort_keys=True)
                    log.write(f"\n→ {event.call.name}({args})\n")
                elif isinstance(event, ToolResultEvent):
                    log.write(f"✓ {event.result.name}: {event.result.content}\n")
                elif isinstance(event, ErrorEvent):
                    latest_runtime_error_kind = event.kind
                    latest_runtime_error = event.error
                    latest_runtime_error_recoverable = event.recoverable
                    run_error = f"{event.kind}: {event.error}"
                    log.write(f"\nError ({event.kind}): {event.error}\n")
                elif isinstance(event, ModelSelectedEvent):
                    effective_model = event.model
                    model_selection = {
                        "provider": event.provider,
                        "requested_model": event.requested_model,
                        "model": event.model,
                        "fallback": event.fallback,
                        "attempt": event.attempt,
                    }
                    if event.fallback:
                        log.write(
                            "\n[harness:model] "
                            f"{event.provider} selected fallback model {event.model} "
                            f"for requested model {event.requested_model}.\n"
                        )
                elif isinstance(event, Done):
                    if event.final_message and event.final_message.content:
                        final_text = event.final_message.content
                    log.write("\n[DONE]\n")
                log.flush()

            env_updates: dict[str, str] = {}
            idle_timeout = _positive_seconds_env_value(
                model_stream_idle_timeout_seconds,
                name="model_stream_idle_timeout_seconds",
            )
            turn_timeout = _positive_seconds_env_value(
                model_turn_timeout_seconds,
                name="model_turn_timeout_seconds",
            )
            if idle_timeout is not None:
                env_updates["HARNESS_MODEL_STREAM_IDLE_TIMEOUT"] = idle_timeout
            if turn_timeout is not None:
                env_updates["HARNESS_MODEL_TURN_TIMEOUT"] = turn_timeout
            previous_env = {key: os.environ.get(key) for key in env_updates}
            try:
                os.environ.update(env_updates)
                final = await asyncio.wait_for(
                    _harness_run_once(
                        prompt=prompt,
                        model=attempt_model,
                        chain=["openrouter"],
                        base_url=None,
                        cwd=Path(workdir),
                        max_steps=max_steps,
                        max_output_tokens=max_output_tokens,
                        session_id=session_id,
                        task_ref=None,
                        db=None,
                        in_memory=True,
                        yes=True,
                        inbox=False,
                        verify="external-workspace",
                        verify_command=None,
                        critic=None,
                        require_tools=True,
                        goal=True,
                        max_context_tokens=None,
                        predict=True,
                        auto_compact=False,
                        max_repair=repair_attempts,
                        profile="minimal",
                        domain="coding",
                        phases=None,
                        loop_detect=True,
                        contracts=False,
                        tips=False,
                        include_workspace_context=False,
                        silent=False,
                        config=cfg,
                        build_storage=_memory_storage,
                        resolve_task_attachment=_noop_task_attachment,
                        resolve_runtime_strategy=_resolve_runtime_strategy,
                        build_verifier=_build_external_workspace_verifier,
                        build_critic=_build_critic,
                        build_adapter=_build_adapter,
                        build_tools=_build_remote_tools,
                        build_agent=_build_remote_agent,
                        print_defense_ledger=_noop_defense_ledger,
                        render=_render,
                        default_system_prompt=(
                            "You are a coding agent operating through tools in a repository. "
                            "Use the available tools to inspect, edit, and verify real work."
                        ),
                        console=Console(file=log),
                    ),
                    timeout=max(1.0, pass_timeout_seconds) * max(1, total_attempt_budget),
                )
                if final:
                    final_text = final
            except TimeoutError:
                run_error = f"harness run timed out after {pass_timeout_seconds:.1f}s"
                log.write(f"\nError (timeout): {run_error}\n")
            except typer.Exit as exc:
                run_error = f"harness run exited with {exc.exit_code}"
                log.write(f"\nError (exit): {run_error}\n")
            finally:
                for key, previous in previous_env.items():
                    if previous is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous

            if run_error is not None and not _snapshot_has_external_workspace_state(
                verifier.latest
            ):
                await verifier.verify(session=None, activity=[])

            snapshot = verifier.latest
            source_change_error = snapshot.source_change_error
            verification_error = snapshot.verification_error
            total_event_count += event_count
            runtime_attempts.append(
                {
                    "attempt": attempt_index,
                    "model": attempt_model,
                    "effective_model": effective_model or attempt_model,
                    "run_error": run_error,
                    "latest_runtime_error_kind": latest_runtime_error_kind,
                    "latest_runtime_error": latest_runtime_error,
                    "latest_runtime_error_recoverable": latest_runtime_error_recoverable,
                    "source_change_passed": snapshot.source_change_passed,
                    "source_change_paths": list(snapshot.source_change_paths or []),
                    "test_change_paths": list(snapshot.test_change_paths or []),
                    "scratch_paths": list(snapshot.scratch_paths or []),
                    "verification_passed_after_source_change": (
                        snapshot.verification_passed_after_source_change
                    ),
                }
            )
            if _should_retry_external_workspace_with_fallback(
                run_error=run_error,
                latest_runtime_error_kind=latest_runtime_error_kind,
                latest_runtime_error_recoverable=latest_runtime_error_recoverable,
                snapshot=snapshot,
                attempt_index=attempt_index,
                model_candidates=model_candidates,
            ):
                continue
            break
    finally:
        log.close()
        event_log.close()

    context.n_agent_steps = total_event_count
    context.metadata = {
        "agent": "harness",
        "runtime": "harness.run_once",
        "model": effective_model or attempt_model,
        "requested_model": model,
        "effective_model": effective_model or attempt_model,
        "attempt_model": attempt_model,
        "model_fallback_used": (
            attempt_model != model or bool(model_selection and model_selection.get("fallback"))
        ),
        "model_selection": model_selection,
        "model_attempts": runtime_attempts,
        "workdir": workdir,
        "final_text_preview": final_text[:500],
        "run_error": run_error,
        "latest_runtime_error_kind": latest_runtime_error_kind,
        "latest_runtime_error": latest_runtime_error,
        "latest_runtime_error_recoverable": latest_runtime_error_recoverable,
        **snapshot.to_metadata(),
        "source_change_retries_used": None,
        "verification_retries_used": None,
        "repair_attempt_budget": repair_attempts,
        "total_attempt_budget": max(1, total_attempt_budget),
        "pass_timeout_seconds": pass_timeout_seconds,
        "require_regression_test_change": require_regression_test_change,
        "default_verify_timeout_seconds": default_verify_timeout_seconds,
        "model_stream_idle_timeout_seconds": model_stream_idle_timeout_seconds,
        "model_turn_timeout_seconds": model_turn_timeout_seconds,
    }
    if source_change_error is not None:
        if latest_runtime_error is not None:
            runtime_prefix = latest_runtime_error_kind or "runtime_error"
            raise RuntimeError(f"{runtime_prefix}: {latest_runtime_error}; {source_change_error}")
        raise RuntimeError(source_change_error)
    if verification_error is not None:
        raise RuntimeError(verification_error)
    if run_error is not None:
        raise RuntimeError(run_error)


__all__ = [
    "ExternalWorkspacePolicy",
    "ExternalWorkspaceVerificationSnapshot",
    "ExternalWorkspaceVerifier",
    "PolicyFetchUrlTool",
    "PolicyWebSearchTool",
    "RemoteApplyPatchTool",
    "RemoteEditFileTool",
    "RemoteListDirTool",
    "RemoteReadFileRangeTool",
    "RemoteReadFileTool",
    "RemoteShellTool",
    "RemoteVerifyWorkTool",
    "RemoteWorkspaceState",
    "RemoteWriteFileTool",
    "_workspace_source_change_status",
    "build_remote_tool_registry",
    "external_workspace_repair_attempts",
    "external_workspace_total_attempts",
    "run_harness_on_external_environment",
]
