"""Tests for content_before / content_after diff metadata in write_file and edit_file."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import ToolCall
from harness.tools.fs import EditFileTool, WriteFileTool


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


# ---------------------------------------------------------------------------
# WriteFileTool diff metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWriteFileDiffMetadata:
    async def test_new_file_content_before_is_none(self, tmp_path: Path) -> None:
        tool = WriteFileTool(cwd=tmp_path)
        result = await tool(_call("write_file", path="new.txt", content="hello"))
        assert result.metadata is not None
        assert result.metadata["content_before"] is None
        assert result.metadata["content_after"] == "hello"
        assert result.metadata["created"] is True

    async def test_overwrite_captures_old_content(self, tmp_path: Path) -> None:
        (tmp_path / "existing.txt").write_text("original content", encoding="utf-8")
        tool = WriteFileTool(cwd=tmp_path)
        result = await tool(_call("write_file", path="existing.txt", content="new content"))
        assert result.metadata is not None
        assert result.metadata["content_before"] == "original content"
        assert result.metadata["content_after"] == "new content"
        assert result.metadata["created"] is False

    async def test_large_before_content_is_truncated(self, tmp_path: Path) -> None:
        big = "x" * (9 * 1024)
        (tmp_path / "big.txt").write_text(big, encoding="utf-8")
        tool = WriteFileTool(cwd=tmp_path)
        result = await tool(_call("write_file", path="big.txt", content="small"))
        assert result.metadata is not None
        before = result.metadata["content_before"]
        assert before is not None
        assert before.endswith("…[truncated]")
        assert len(before) < len(big)

    async def test_large_after_content_is_truncated(self, tmp_path: Path) -> None:
        tool = WriteFileTool(cwd=tmp_path)
        big = "y" * (9 * 1024)
        result = await tool(_call("write_file", path="out.txt", content=big))
        assert result.metadata is not None
        after = result.metadata["content_after"]
        assert after is not None
        assert after.endswith("…[truncated]")


# ---------------------------------------------------------------------------
# EditFileTool diff metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEditFileDiffMetadata:
    async def test_edit_captures_before_and_after(self, tmp_path: Path) -> None:
        (tmp_path / "src.py").write_text('def greet():\n    return "hello"\n', encoding="utf-8")
        tool = EditFileTool(cwd=tmp_path)
        result = await tool(_call("edit_file", path="src.py", old='"hello"', new='"hello, world"'))
        assert result.metadata is not None
        assert '"hello"' in result.metadata["content_before"]
        assert '"hello, world"' in result.metadata["content_after"]
        assert '"hello"' not in result.metadata["content_after"]

    async def test_large_file_both_sides_truncated(self, tmp_path: Path) -> None:
        big = "a" * (4 * 1024) + "MARKER" + "a" * (4 * 1024)
        (tmp_path / "big.py").write_text(big, encoding="utf-8")
        tool = EditFileTool(cwd=tmp_path)
        result = await tool(_call("edit_file", path="big.py", old="MARKER", new="REPLACED"))
        assert result.metadata is not None
        assert result.metadata["content_before"].endswith("…[truncated]")
        assert result.metadata["content_after"].endswith("…[truncated]")
