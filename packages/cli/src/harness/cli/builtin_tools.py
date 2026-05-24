from __future__ import annotations

from pathlib import Path

from harness.core.extensions import ToolProvider
from harness.core.tool_entry import ToolBuildContext, ToolSpec
from harness.core.tools import ToolRegistry
from harness.tools.fs import EditFileTool, GlobTool, ListDirTool, ReadFileTool, WriteFileTool
from harness.tools.shell import ShellTool
from harness.tools.web import FetchUrlTool, TavilySearchTool


class BuiltinToolProvider(ToolProvider):
    """Declarative source of the standard CLI toolset."""

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="read_file",
                category="filesystem",
                factory=lambda ctx: ReadFileTool(cwd=ctx.cwd),
            ),
            ToolSpec(
                name="write_file",
                category="filesystem",
                factory=lambda ctx: WriteFileTool(cwd=ctx.cwd),
            ),
            ToolSpec(
                name="edit_file",
                category="filesystem",
                factory=lambda ctx: EditFileTool(cwd=ctx.cwd),
            ),
            ToolSpec(
                name="list_dir",
                category="filesystem",
                factory=lambda ctx: ListDirTool(cwd=ctx.cwd),
            ),
            ToolSpec(
                name="glob",
                category="filesystem",
                factory=lambda ctx: GlobTool(cwd=ctx.cwd),
            ),
            ToolSpec(
                name="shell",
                category="execution",
                factory=lambda ctx: ShellTool(cwd=ctx.cwd),
            ),
            ToolSpec(
                name="web_search",
                category="web",
                factory=lambda ctx: TavilySearchTool(),
            ),
            ToolSpec(
                name="fetch_url",
                category="web",
                factory=lambda ctx: FetchUrlTool(),
            ),
        ]

    def build_registry(
        self,
        *,
        cwd: Path,
        include: set[str] | None = None,
    ) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register_provider(self)
        built = registry.materialize_specs(cwd=cwd)
        if include is None:
            return built

        filtered = ToolRegistry()
        for name in sorted(include):
            if built.has(name):
                filtered.register(built.get(name))
        return filtered

    def build_context(self, *, cwd: Path) -> ToolBuildContext:
        return ToolBuildContext(cwd=cwd)


__all__ = ["BuiltinToolProvider"]
