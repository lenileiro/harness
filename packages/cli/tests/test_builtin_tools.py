from __future__ import annotations

from pathlib import Path

from harness.cli.builtin_tools import BuiltinToolProvider


def test_builtin_tool_provider_builds_expected_tool_names(tmp_path: Path) -> None:
    registry = BuiltinToolProvider().build_registry(cwd=tmp_path)

    assert registry.names() == [
        "edit_file",
        "fetch_url",
        "glob",
        "list_dir",
        "read_file",
        "shell",
        "web_search",
        "write_file",
    ]


def test_builtin_tool_provider_uses_cwd_for_workspace_tools(tmp_path: Path) -> None:
    registry = BuiltinToolProvider().build_registry(cwd=tmp_path)
    read_tool = registry.get("read_file")
    shell_tool = registry.get("shell")

    assert Path(read_tool.cwd) == tmp_path  # type: ignore[attr-defined]
    assert Path(shell_tool.cwd) == tmp_path  # type: ignore[attr-defined]


def test_builtin_tool_provider_can_filter_tool_subset(tmp_path: Path) -> None:
    registry = BuiltinToolProvider().build_registry(
        cwd=tmp_path,
        include={"read_file", "list_dir", "shell"},
    )

    assert registry.names() == ["list_dir", "read_file", "shell"]
