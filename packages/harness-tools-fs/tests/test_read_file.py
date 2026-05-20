"""Tests for fs.ReadFileTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import ToolCall
from harness.tools.fs import ReadFileTool


def _call(path: str) -> ToolCall:
    return ToolCall(id="c1", name="read_file", arguments={"path": path})


@pytest.mark.asyncio
class TestReadFile:
    async def test_reads_relative_path(self, tmp_path: Path) -> None:
        (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
        tool = ReadFileTool(cwd=tmp_path)
        result = await tool(_call("hello.txt"))
        assert result.is_error is False
        assert result.content == "hello world"
        assert result.tool_call_id == "c1"

    async def test_reads_nested_path(self, tmp_path: Path) -> None:
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        (sub / "deep.txt").write_text("deep", encoding="utf-8")
        tool = ReadFileTool(cwd=tmp_path)
        result = await tool(_call("a/b/deep.txt"))
        assert result.content == "deep"

    async def test_absolute_path_inside_cwd_works(self, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("ok", encoding="utf-8")
        tool = ReadFileTool(cwd=tmp_path)
        result = await tool(_call(str(tmp_path / "f.txt")))
        assert result.content == "ok"

    async def test_path_outside_cwd_refused(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        try:
            tool = ReadFileTool(cwd=tmp_path)
            result = await tool(_call(str(outside)))
            assert result.is_error is True
            assert "outside" in result.content
        finally:
            outside.unlink(missing_ok=True)

    async def test_traversal_refused(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("nope", encoding="utf-8")
        try:
            tool = ReadFileTool(cwd=tmp_path)
            result = await tool(_call("../secret.txt"))
            assert result.is_error is True
        finally:
            outside.unlink(missing_ok=True)

    async def test_missing_path_argument_is_error(self, tmp_path: Path) -> None:
        tool = ReadFileTool(cwd=tmp_path)
        result = await tool(ToolCall(id="c1", name="read_file", arguments={}))
        assert result.is_error is True
        assert "missing" in result.content

    async def test_nonexistent_file_is_error(self, tmp_path: Path) -> None:
        tool = ReadFileTool(cwd=tmp_path)
        result = await tool(_call("ghost.txt"))
        assert result.is_error is True
        assert "does not exist" in result.content

    async def test_directory_target_is_error(self, tmp_path: Path) -> None:
        (tmp_path / "dir").mkdir()
        tool = ReadFileTool(cwd=tmp_path)
        result = await tool(_call("dir"))
        assert result.is_error is True
        assert "not a regular file" in result.content

    async def test_file_size_limit_enforced(self, tmp_path: Path) -> None:
        (tmp_path / "big.txt").write_text("x" * 2048, encoding="utf-8")
        tool = ReadFileTool(cwd=tmp_path, max_bytes=1024)
        result = await tool(_call("big.txt"))
        assert result.is_error is True
        assert "too large" in result.content

    async def test_non_utf8_binary_refused(self, tmp_path: Path) -> None:
        (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00\x01\x02")
        tool = ReadFileTool(cwd=tmp_path)
        result = await tool(_call("bin.dat"))
        assert result.is_error is True
        assert "UTF-8" in result.content


class TestToolMetadata:
    def test_protocol_attributes(self) -> None:
        tool = ReadFileTool(cwd=Path.cwd())
        assert tool.name == "read_file"
        assert tool.approval == "auto"
        assert "path" in tool.parameters_schema["properties"]
        assert "path" in tool.parameters_schema["required"]
