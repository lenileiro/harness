"""Declarative tool registration metadata.

The runtime still works with concrete ``Tool`` instances, but extension points
need a way to describe how tools are built before a run starts. ``ToolSpec``
captures that factory-level metadata without forcing the runtime to know about
CLI wiring details.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.core.tools import Tool


@dataclass(frozen=True, slots=True)
class ToolBuildContext:
    """Context passed to ``ToolSpec.factory`` when materializing a tool."""

    cwd: Path
    extras: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.extras.get(key, default)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Declarative registration for a tool factory."""

    name: str
    factory: Callable[[ToolBuildContext], Tool]
    category: str = "general"
    availability_check: Callable[[ToolBuildContext], bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def available(self, ctx: ToolBuildContext) -> bool:
        if self.availability_check is None:
            return True
        return bool(self.availability_check(ctx))

    def build(self, ctx: ToolBuildContext) -> Tool:
        return self.factory(ctx)


__all__ = ["ToolBuildContext", "ToolSpec"]
