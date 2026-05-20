"""Filesystem tools for Harness agents.

All tools are scoped to a working directory passed at construction time. Paths
that resolve outside the cwd are refused as a security backstop independent
of the approval policy.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from harness.core import ApprovalDecision, ToolCall, ToolResult

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

__version__ = "0.0.0"


def _error(call: ToolCall, name: str, message: str) -> ToolResult:
    return ToolResult(tool_call_id=call.id, name=name, content=message, is_error=True)


class ReadFileTool:
    """Read the contents of a UTF-8 text file under the session's cwd.

    The agent receives the file's text as the tool result. Binary files,
    files outside the cwd, and files exceeding `max_bytes` are refused with
    `is_error=True`.
    """

    name = "read_file"
    description = (
        "Read the UTF-8 text contents of a file. The `path` argument is "
        "resolved relative to the session's working directory and must "
        "remain inside it."
    )
    approval: ApprovalDecision = "auto"

    def __init__(
        self,
        *,
        cwd: Path | str,
        max_bytes: int = 1024 * 1024,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.max_bytes = max_bytes
        # parameters_schema is per-instance to satisfy the Tool Protocol's
        # instance-attribute declaration. The actual dict is shared.
        self.parameters_schema: dict[str, Any] = _READ_FILE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        path_arg = call.arguments.get("path")
        if not isinstance(path_arg, str) or not path_arg:
            return _error(call, self.name, "missing or invalid `path` argument")

        candidate = Path(path_arg)
        target = (candidate if candidate.is_absolute() else self.cwd / candidate).resolve()

        try:
            target.relative_to(self.cwd)
        except ValueError:
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


__all__ = ["ReadFileTool", "__version__"]
