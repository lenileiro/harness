"""Filesystem tools for Harness agents.

All tools are scoped to a working directory passed at construction time.
Paths that resolve outside the cwd are refused as a security backstop
independent of the approval policy. The set:

- `ReadFileTool`  — read UTF-8 text (auto approval)
- `WriteFileTool` — create or overwrite a UTF-8 text file (prompt approval)
- `EditFileTool`  — exact-string replacement inside a UTF-8 text file (prompt)
- `ListDirTool`   — list directory entries (auto)
- `GlobTool`      — glob-match files under cwd (auto)
"""

from __future__ import annotations

import asyncio
import fnmatch
from pathlib import Path
from typing import Any

from harness.core import ApprovalDecision, ToolCall, ToolResult

__version__ = "0.0.0"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _error(call: ToolCall, name: str, message: str) -> ToolResult:
    return ToolResult(tool_call_id=call.id, name=name, content=message, is_error=True)


def _resolve_in_cwd(cwd: Path, path_arg: str) -> Path | None:
    """Resolve `path_arg` relative to `cwd`; return None if it escapes."""
    candidate = Path(path_arg)
    target = (candidate if candidate.is_absolute() else cwd / candidate).resolve()
    try:
        target.relative_to(cwd)
    except ValueError:
        return None
    return target


_READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to a UTF-8 text file, relative to cwd.",
        }
    },
    "required": ["path"],
}

_WRITE_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Destination path, relative to cwd."},
        "content": {"type": "string", "description": "Full file contents to write (UTF-8)."},
    },
    "required": ["path", "content"],
}

_EDIT_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Target file, relative to cwd."},
        "old": {
            "type": "string",
            "description": "Exact substring to replace. Must appear exactly once.",
        },
        "new": {"type": "string", "description": "Replacement string."},
    },
    "required": ["path", "old", "new"],
}

_LIST_DIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Directory path relative to cwd. Defaults to cwd itself.",
        }
    },
    "required": [],
}

_GLOB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Glob pattern (e.g. '**/*.py'). Matched against paths relative to cwd.",
        }
    },
    "required": ["pattern"],
}


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------


class ReadFileTool:
    """Read the contents of a UTF-8 text file under the session's cwd."""

    name = "read_file"
    description = (
        "Read the UTF-8 text contents of a file. The `path` argument is resolved "
        "relative to the session's working directory and must remain inside it."
    )
    approval: ApprovalDecision = "auto"
    # Read-only — safe in any phase, including any new ones a future planner
    # might define.
    phases: tuple[str, ...] = ("*",)

    def __init__(self, *, cwd: Path | str, max_bytes: int = 1024 * 1024) -> None:
        self.cwd = Path(cwd).resolve()
        self.max_bytes = max_bytes
        self.parameters_schema: dict[str, Any] = _READ_FILE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        path_arg = call.arguments.get("path")
        if not isinstance(path_arg, str) or not path_arg:
            return _error(call, self.name, "missing or invalid `path` argument")

        target = _resolve_in_cwd(self.cwd, path_arg)
        if target is None:
            return _error(
                call,
                self.name,
                f"path {path_arg!r} resolves outside the session working directory",
            )

        if not target.exists():
            return _error(call, self.name, f"path does not exist: {path_arg}")
        if not target.is_file():
            return _error(call, self.name, f"not a regular file: {path_arg}")

        size = target.stat().st_size
        if size > self.max_bytes:
            return _error(
                call,
                self.name,
                f"file too large: {size} bytes exceeds limit of {self.max_bytes}",
            )

        try:
            content = await asyncio.to_thread(target.read_text, encoding="utf-8")
        except UnicodeDecodeError:
            return _error(call, self.name, f"file is not UTF-8 text: {path_arg}")
        except OSError as exc:
            return _error(call, self.name, f"could not read {path_arg}: {exc}")

        return ToolResult(tool_call_id=call.id, name=self.name, content=content)


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------


class WriteFileTool:
    """Create or overwrite a UTF-8 text file under the session's cwd.

    Parent directories are created if missing. Default approval is `prompt`
    because writes are not idempotent and can clobber user work.
    """

    name = "write_file"
    description = (
        "Create or overwrite a UTF-8 text file. The `path` is resolved relative "
        "to cwd and must remain inside it. Parent directories are auto-created."
    )
    approval: ApprovalDecision = "prompt"
    # Mutates state — restricted to the `act` phase. Callers who want it
    # available elsewhere should construct an Agent with no `current_phase`.
    phases: tuple[str, ...] = ("act",)

    def __init__(self, *, cwd: Path | str, max_bytes: int = 1024 * 1024) -> None:
        self.cwd = Path(cwd).resolve()
        self.max_bytes = max_bytes
        self.parameters_schema: dict[str, Any] = _WRITE_FILE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        path_arg = call.arguments.get("path")
        content = call.arguments.get("content")
        if not isinstance(path_arg, str) or not path_arg:
            return _error(call, self.name, "missing or invalid `path` argument")
        if not isinstance(content, str):
            return _error(call, self.name, "missing or non-string `content` argument")
        if len(content.encode("utf-8")) > self.max_bytes:
            return _error(
                call,
                self.name,
                f"content exceeds {self.max_bytes}-byte limit",
            )

        target = _resolve_in_cwd(self.cwd, path_arg)
        if target is None:
            return _error(
                call,
                self.name,
                f"path {path_arg!r} resolves outside the session working directory",
            )

        try:
            await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
        except OSError as exc:
            return _error(call, self.name, f"could not write {path_arg}: {exc}")

        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"wrote {len(content.encode('utf-8'))} bytes to {path_arg}",
        )


# ---------------------------------------------------------------------------
# EditFileTool
# ---------------------------------------------------------------------------


class EditFileTool:
    """Exact-string replacement inside an existing UTF-8 text file.

    Requires that `old` appears exactly once in the file. Refusing on
    ambiguous matches is intentional — the model must disambiguate by
    including more surrounding context.
    """

    name = "edit_file"
    description = (
        "Replace one occurrence of `old` with `new` inside a UTF-8 text file. "
        "Fails if `old` is absent or appears more than once — disambiguate by "
        "passing a longer `old` snippet."
    )
    approval: ApprovalDecision = "prompt"
    phases: tuple[str, ...] = ("act",)

    def __init__(self, *, cwd: Path | str, max_bytes: int = 1024 * 1024) -> None:
        self.cwd = Path(cwd).resolve()
        self.max_bytes = max_bytes
        self.parameters_schema: dict[str, Any] = _EDIT_FILE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        path_arg = call.arguments.get("path")
        old = call.arguments.get("old")
        new = call.arguments.get("new")
        if not isinstance(path_arg, str) or not path_arg:
            return _error(call, self.name, "missing or invalid `path` argument")
        if not isinstance(old, str) or not old:
            return _error(call, self.name, "missing or empty `old` argument")
        if not isinstance(new, str):
            return _error(call, self.name, "missing or non-string `new` argument")

        target = _resolve_in_cwd(self.cwd, path_arg)
        if target is None:
            return _error(
                call,
                self.name,
                f"path {path_arg!r} resolves outside the session working directory",
            )
        if not target.exists() or not target.is_file():
            return _error(call, self.name, f"not a regular file: {path_arg}")
        if target.stat().st_size > self.max_bytes:
            return _error(call, self.name, f"file too large: {path_arg}")

        try:
            original = await asyncio.to_thread(target.read_text, encoding="utf-8")
        except UnicodeDecodeError:
            return _error(call, self.name, f"file is not UTF-8 text: {path_arg}")

        occurrences = original.count(old)
        if occurrences == 0:
            return _error(
                call,
                self.name,
                f"`old` not found in {path_arg}; nothing to replace",
            )
        if occurrences > 1:
            return _error(
                call,
                self.name,
                f"`old` matches {occurrences} times in {path_arg}; pass more context to disambiguate",
            )

        updated = original.replace(old, new, 1)
        try:
            await asyncio.to_thread(target.write_text, updated, encoding="utf-8")
        except OSError as exc:
            return _error(call, self.name, f"could not write {path_arg}: {exc}")

        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"replaced 1 occurrence in {path_arg}",
        )


# ---------------------------------------------------------------------------
# ListDirTool
# ---------------------------------------------------------------------------


class ListDirTool:
    """List immediate children of a directory under cwd. Sorted; dirs marked with /."""

    name = "list_dir"
    description = (
        "List directory entries. Returns one entry per line. Directories are "
        "suffixed with '/'. Path is relative to cwd; defaults to cwd itself."
    )
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(self, *, cwd: Path | str) -> None:
        self.cwd = Path(cwd).resolve()
        self.parameters_schema: dict[str, Any] = _LIST_DIR_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        path_arg = call.arguments.get("path", ".")
        if not isinstance(path_arg, str):
            return _error(call, self.name, "invalid `path` argument")

        target = _resolve_in_cwd(self.cwd, path_arg or ".")
        if target is None:
            return _error(
                call,
                self.name,
                f"path {path_arg!r} resolves outside the session working directory",
            )
        if not target.exists():
            return _error(call, self.name, f"path does not exist: {path_arg}")
        if not target.is_dir():
            return _error(call, self.name, f"not a directory: {path_arg}")

        try:
            entries = await asyncio.to_thread(lambda: sorted(target.iterdir()))
        except OSError as exc:
            return _error(call, self.name, f"could not list {path_arg}: {exc}")

        lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries]
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="\n".join(lines) if lines else "(empty)",
        )


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------


class GlobTool:
    """Match files under cwd against a glob pattern.

    Uses `Path.glob` semantics (e.g. `**/*.py`). Returns one path per line,
    each relative to cwd. The result is hard-capped at `max_results` to keep
    transcripts manageable.
    """

    name = "glob"
    description = (
        "List files under cwd that match a glob pattern (e.g. '**/*.py'). "
        "Returns paths relative to cwd, one per line, sorted."
    )
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(self, *, cwd: Path | str, max_results: int = 500) -> None:
        self.cwd = Path(cwd).resolve()
        self.max_results = max_results
        self.parameters_schema: dict[str, Any] = _GLOB_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        pattern = call.arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return _error(call, self.name, "missing or empty `pattern` argument")

        # Defence-in-depth: refuse patterns that try to escape via `..`.
        if any(part == ".." for part in Path(pattern).parts):
            return _error(call, self.name, "`pattern` may not contain '..'")
        if Path(pattern).is_absolute():
            return _error(call, self.name, "`pattern` must be relative")

        def _do_glob() -> list[str]:
            results: list[str] = []
            for path in self.cwd.glob(pattern):
                # Filter strictly to inside cwd via fnmatch on the resolved path.
                try:
                    rel = path.resolve().relative_to(self.cwd)
                except ValueError:
                    continue
                if fnmatch.fnmatch(str(rel), pattern) or path.is_dir() or path.is_file():
                    results.append(str(rel))
                if len(results) >= self.max_results:
                    break
            return sorted(results)

        try:
            matches = await asyncio.to_thread(_do_glob)
        except OSError as exc:
            return _error(call, self.name, f"glob failed: {exc}")

        if not matches:
            return ToolResult(tool_call_id=call.id, name=self.name, content="(no matches)")

        suffix = f"\n… ({self.max_results} max reached)" if len(matches) >= self.max_results else ""
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="\n".join(matches) + suffix,
        )


__all__ = [
    "EditFileTool",
    "GlobTool",
    "ListDirTool",
    "ReadFileTool",
    "WriteFileTool",
    "__version__",
]
