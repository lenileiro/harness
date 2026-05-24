from __future__ import annotations

from pathlib import Path

from harness.core.tool_entry import ToolBuildContext, ToolSpec
from harness.core.tools import ToolRegistry

from .conftest import MockTool


def test_register_spec_tracks_generation_and_names(tmp_path: Path) -> None:
    registry = ToolRegistry()
    assert registry.generation == 0

    spec = ToolSpec(
        name="echo",
        category="test",
        factory=lambda ctx: MockTool(name="echo"),
    )
    registry.register_spec(spec)

    assert registry.generation == 1
    assert registry.has_spec("echo") is True
    assert registry.spec_names() == ["echo"]
    assert registry.get_spec("echo") == spec


def test_materialize_specs_builds_only_available_tools(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_spec(
        ToolSpec(
            name="enabled",
            category="test",
            factory=lambda ctx: MockTool(name="enabled"),
            availability_check=lambda ctx: True,
        )
    )
    registry.register_spec(
        ToolSpec(
            name="disabled",
            category="test",
            factory=lambda ctx: MockTool(name="disabled"),
            availability_check=lambda ctx: False,
        )
    )

    built = registry.materialize_specs(cwd=tmp_path)

    assert built.names() == ["enabled"]
    assert built.has("enabled") is True
    assert built.has("disabled") is False


def test_from_specs_materializes_registry(tmp_path: Path) -> None:
    specs = [
        ToolSpec(
            name="alpha",
            category="test",
            factory=lambda ctx: MockTool(name="alpha"),
        ),
        ToolSpec(
            name="bravo",
            category="test",
            factory=lambda ctx: MockTool(name="bravo"),
        ),
    ]

    built = ToolRegistry.from_specs(specs, cwd=tmp_path)

    assert built.names() == ["alpha", "bravo"]


def test_register_provider_registers_all_specs(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_provider(_Provider())

    built = registry.materialize_specs(cwd=tmp_path)

    assert registry.spec_names() == ["alpha", "bravo"]
    assert built.names() == ["alpha", "bravo"]


def test_build_context_extras_are_available_to_factories(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    spec = ToolSpec(
        name="echo",
        category="test",
        factory=lambda ctx: _recording_tool(ctx, seen),
    )
    built = ToolRegistry.from_specs([spec], cwd=tmp_path, extras={"mode": "debug"})

    assert built.has("echo") is True
    assert seen == {"cwd": str(tmp_path), "mode": "debug"}


def _recording_tool(ctx: ToolBuildContext, seen: dict[str, str]) -> MockTool:
    seen["cwd"] = str(ctx.cwd)
    seen["mode"] = str(ctx.get("mode"))
    return MockTool(name="echo")


class _Provider:
    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name="alpha", category="test", factory=lambda ctx: MockTool(name="alpha")),
            ToolSpec(name="bravo", category="test", factory=lambda ctx: MockTool(name="bravo")),
        ]
