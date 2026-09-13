"""Keep shared Harness state out of remote filesystem tool access.

This guards the standard path-based filesystem tools only. ``shell`` and
``verify_work`` execute arbitrary host commands and bypass path guards; remote
builders must remove them until an isolated execution backend exists. Custom
host-execution tools need the same treatment. Local scheduled jobs retain their
ordinary filesystem and approval behavior via ``local_only=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult
from harness.core.tools import Tool

if TYPE_CHECKING:
    from harness.core.runtime import Agent

HOST_EXECUTION_TOOL_NAMES = frozenset({"shell", "verify_work"})
_FILESYSTEM_TOOL_NAMES = frozenset({"read_file", "write_file", "edit_file", "list_dir", "glob"})


def _private(path: Path) -> bool:
    # Case-folding also protects case-insensitive filesystems.
    return any(part.casefold() == ".harness" for part in path.parts)


class _GatewayFilesystemTool:
    def __init__(self, tool: Tool, *, cwd: Path) -> None:
        self._tool = tool
        self._cwd = cwd.resolve()
        self.name = tool.name
        self.description = tool.description
        self.approval: ApprovalDecision = tool.approval
        self.parameters_schema = tool.parameters_schema
        self.effect_scope = getattr(tool, "effect_scope", None)
        self.phases = getattr(tool, "phases", ("*",))

    def _public_path(self, raw: str, *, base: Path | None = None) -> Path | None:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (base or self._cwd) / candidate
        try:
            lexical = candidate.relative_to(self._cwd)
            resolved = candidate.resolve()
            relative = resolved.relative_to(self._cwd)
        except (OSError, RuntimeError, ValueError):
            return None
        if _private(lexical) or _private(relative):
            return None
        return resolved

    def _denied(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="Path is outside the public project workspace. Harness private state is unavailable through remote filesystem tools.",
            is_error=True,
        )

    async def __call__(self, call: ToolCall) -> ToolResult:
        listing_base = self._cwd
        forwarded = call
        if self.name == "glob":
            pattern = call.arguments.get("pattern")
            if isinstance(pattern, str) and _private(Path(pattern)):
                return self._denied(call)
        else:
            raw = call.arguments.get("path", "." if self.name == "list_dir" else None)
            if not isinstance(raw, str):
                return self._denied(call)
            target = self._public_path(raw or ".")
            if target is None:
                return self._denied(call)
            listing_base = target
            # Forward the checked canonical target, not an unchecked symlink alias.
            forwarded = call.model_copy(
                update={"arguments": {**call.arguments, "path": str(target)}}
            )
        result = await self._tool(forwarded)
        if result.is_error or self.name not in {"list_dir", "glob"}:
            return result

        visible: list[str] = []
        for line in result.content.splitlines():
            target = self._public_path(line.rstrip("/"), base=listing_base)
            if target is None or not target.exists():
                continue
            visible.append(line)
        if self.name == "list_dir":
            return result.model_copy(
                update={
                    "content": "\n".join(visible) or "(empty)",
                    "metadata": {"path": str(listing_base), "entries": len(visible)},
                }
            )
        return result.model_copy(
            update={
                "content": "\n".join(visible) or "(no matches)",
                "metadata": {
                    "pattern": call.arguments.get("pattern"),
                    "matches": len(visible),
                    "capped": len(visible) >= getattr(self._tool, "max_results", 500),
                },
            }
        )


def install_gateway_tool_boundary(agent: Agent, cwd: Path, *, local_only: bool = False) -> None:
    """Wrap standard filesystem tools; leave custom tools and local jobs unchanged."""
    if local_only:
        return
    for tool in agent.tools.all():
        if tool.name in _FILESYSTEM_TOOL_NAMES and not isinstance(tool, _GatewayFilesystemTool):
            agent.tools.unregister(tool.name)
            agent.tools.register(_GatewayFilesystemTool(tool, cwd=cwd))


__all__ = ["HOST_EXECUTION_TOOL_NAMES", "install_gateway_tool_boundary"]
