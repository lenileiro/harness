"""Tests for write/edit/list/glob fs tools.

ReadFileTool is covered separately in test_read_file.py.
"""

from __future__ import annotations

# pyright: reportOptionalSubscript=false
from pathlib import Path

import pytest

from harness.core import ToolCall
from harness.tools.fs import (
    SYNTAX_CHECKERS,
    EditFileTool,
    GlobTool,
    ListDirTool,
    WriteFileTool,
)


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


# ---------------------------------------------------------------------------
# WriteFileTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWriteFile:
    async def test_creates_file(self, tmp_path: Path) -> None:
        tool = WriteFileTool(cwd=tmp_path)
        result = await tool(_call("write_file", path="new.txt", content="hi"))
        assert result.is_error is False
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hi"

    async def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        tool = WriteFileTool(cwd=tmp_path)
        result = await tool(_call("write_file", path="a/b/c.txt", content="deep"))
        assert result.is_error is False
        assert (tmp_path / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "deep"

    async def test_overwrites_existing(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("old", encoding="utf-8")
        tool = WriteFileTool(cwd=tmp_path)
        result = await tool(_call("write_file", path="f.txt", content="new"))
        assert result.is_error is False
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "new"

    async def test_refuses_path_outside_cwd(self, tmp_path: Path) -> None:
        tool = WriteFileTool(cwd=tmp_path)
        result = await tool(_call("write_file", path="../escape.txt", content="x"))
        assert result.is_error is True

    async def test_refuses_oversized_content(self, tmp_path: Path) -> None:
        tool = WriteFileTool(cwd=tmp_path, max_bytes=10)
        result = await tool(_call("write_file", path="big.txt", content="x" * 100))
        assert result.is_error is True
        assert "limit" in result.content

    async def test_default_approval_is_prompt(self) -> None:
        tool = WriteFileTool(cwd=Path.cwd())
        assert tool.approval == "prompt"

    async def test_does_not_special_case_python_syntax_by_default(self, tmp_path: Path) -> None:
        assert ".py" not in SYNTAX_CHECKERS
        tool = WriteFileTool(cwd=tmp_path)
        result = await tool(_call("write_file", path="src.py", content="def broken(:\n"))

        assert result.is_error is False
        assert (tmp_path / "src.py").read_text(encoding="utf-8") == "def broken(:\n"


# ---------------------------------------------------------------------------
# EditFileTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEditFile:
    async def test_replaces_single_occurrence(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("foo bar baz", encoding="utf-8")
        tool = EditFileTool(cwd=tmp_path)
        result = await tool(_call("edit_file", path="f.txt", old="bar", new="QUX"))
        assert result.is_error is False
        assert (tmp_path / "f.txt").read_text() == "foo QUX baz"

    async def test_missing_old_is_error(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("nothing here", encoding="utf-8")
        tool = EditFileTool(cwd=tmp_path)
        result = await tool(_call("edit_file", path="f.txt", old="missing", new="x"))
        assert result.is_error is True
        assert "not found" in result.content

    async def test_ambiguous_match_is_error(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("foo foo foo", encoding="utf-8")
        tool = EditFileTool(cwd=tmp_path)
        result = await tool(_call("edit_file", path="f.txt", old="foo", new="X"))
        assert result.is_error is True
        assert "3 times" in result.content

    async def test_refuses_path_outside_cwd(self, tmp_path: Path) -> None:
        tool = EditFileTool(cwd=tmp_path)
        result = await tool(_call("edit_file", path="../x.txt", old="a", new="b"))
        assert result.is_error is True

    async def test_refuses_missing_file(self, tmp_path: Path) -> None:
        tool = EditFileTool(cwd=tmp_path)
        result = await tool(_call("edit_file", path="ghost.txt", old="a", new="b"))
        assert result.is_error is True
        assert "regular file" in result.content

    async def test_does_not_special_case_python_syntax_by_default(self, tmp_path: Path) -> None:
        assert ".py" not in SYNTAX_CHECKERS
        (tmp_path / "src.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        tool = EditFileTool(cwd=tmp_path)

        result = await tool(_call("edit_file", path="src.py", old="def ok():", new="def broken(:"))

        assert result.is_error is False
        assert (tmp_path / "src.py").read_text(encoding="utf-8").startswith("def broken(:")


# ---------------------------------------------------------------------------
# ListDirTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestListDir:
    async def test_lists_root_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("", encoding="utf-8")
        (tmp_path / "b").mkdir()
        tool = ListDirTool(cwd=tmp_path)
        result = await tool(_call("list_dir"))
        assert result.is_error is False
        # Directories suffixed with /
        assert "a.txt" in result.content
        assert "b/" in result.content

    async def test_hides_workspace_noise_by_default(self, tmp_path: Path) -> None:
        (tmp_path / ".harness").mkdir()
        (tmp_path / ".venv").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "src").mkdir()
        tool = ListDirTool(cwd=tmp_path)
        result = await tool(_call("list_dir"))
        assert result.is_error is False
        assert result.content.strip() == "src/"
        assert result.metadata["ignored_entries"] == 3

    async def test_lists_subdir(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "x.txt").write_text("", encoding="utf-8")
        tool = ListDirTool(cwd=tmp_path)
        result = await tool(_call("list_dir", path="sub"))
        assert result.content.strip() == "x.txt"

    async def test_empty_dir_returns_empty_marker(self, tmp_path: Path) -> None:
        tool = ListDirTool(cwd=tmp_path)
        result = await tool(_call("list_dir"))
        assert result.content == "(empty)"

    async def test_refuses_path_outside_cwd(self, tmp_path: Path) -> None:
        tool = ListDirTool(cwd=tmp_path)
        result = await tool(_call("list_dir", path=".."))
        assert result.is_error is True

    async def test_refuses_file_target(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("", encoding="utf-8")
        tool = ListDirTool(cwd=tmp_path)
        result = await tool(_call("list_dir", path="f.txt"))
        assert result.is_error is True
        assert "not a directory" in result.content


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGlob:
    async def test_basic_match(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        (tmp_path / "b.py").write_text("", encoding="utf-8")
        (tmp_path / "c.txt").write_text("", encoding="utf-8")
        tool = GlobTool(cwd=tmp_path)
        result = await tool(_call("glob", pattern="*.py"))
        names = result.content.split("\n")
        assert names == ["a.py", "b.py"]

    async def test_recursive_match(self, tmp_path: Path) -> None:
        (tmp_path / "x").mkdir()
        (tmp_path / "x" / "y.py").write_text("", encoding="utf-8")
        (tmp_path / "z.py").write_text("", encoding="utf-8")
        tool = GlobTool(cwd=tmp_path)
        result = await tool(_call("glob", pattern="**/*.py"))
        names = set(result.content.split("\n"))
        assert "z.py" in names
        assert "x/y.py" in names

    async def test_fnmatch_fallback_for_non_path_component_double_star(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "httpx").mkdir()
        (tmp_path / "httpx" / "_models.py").write_text("", encoding="utf-8")
        (tmp_path / "httpx" / "_client.py").write_text("", encoding="utf-8")
        tool = GlobTool(cwd=tmp_path)

        result = await tool(_call("glob", pattern="httpx/**.py"))

        assert result.is_error is False
        names = set(result.content.split("\n"))
        assert "httpx/_models.py" in names
        assert "httpx/_client.py" in names

    async def test_recursive_match_hides_workspace_noise(self, tmp_path: Path) -> None:
        (tmp_path / ".venv" / "lib").mkdir(parents=True)
        (tmp_path / ".venv" / "lib" / "ignored.py").write_text("", encoding="utf-8")
        (tmp_path / ".harness").mkdir()
        (tmp_path / ".harness" / "ignored.py").write_text("", encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
        tool = GlobTool(cwd=tmp_path)
        result = await tool(_call("glob", pattern="**/*.py"))
        assert result.is_error is False
        assert result.content.strip() == "src/app.py"

    async def test_no_match_returns_marker(self, tmp_path: Path) -> None:
        tool = GlobTool(cwd=tmp_path)
        result = await tool(_call("glob", pattern="*.nope"))
        assert result.content == "(no matches)"

    async def test_refuses_absolute_pattern(self, tmp_path: Path) -> None:
        tool = GlobTool(cwd=tmp_path)
        result = await tool(_call("glob", pattern="/etc/*"))
        assert result.is_error is True

    async def test_refuses_traversal_pattern(self, tmp_path: Path) -> None:
        tool = GlobTool(cwd=tmp_path)
        result = await tool(_call("glob", pattern="../*"))
        assert result.is_error is True

    async def test_respects_max_results(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f"f{i:02d}.txt").write_text("", encoding="utf-8")
        tool = GlobTool(cwd=tmp_path, max_results=5)
        result = await tool(_call("glob", pattern="*.txt"))
        assert "max reached" in result.content
