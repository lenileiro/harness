"""Tests for WebSearchTool. Mocks DDGS to avoid network I/O."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from harness.core import ToolCall
from harness.tools.web import WebSearchTool


def _call(query: str, **extra: object) -> ToolCall:
    return ToolCall(id="c1", name="web_search", arguments={"query": query, **extra})


_SAMPLE_RESULTS = [
    {
        "title": "Python 3.12 Release Notes",
        "body": "New features in Python 3.12.",
        "href": "https://docs.python.org/3.12/whatsnew/3.12.html",
    },
    {
        "title": "Real Python — Python 3.12",
        "body": "A practical guide to what changed.",
        "href": "https://realpython.com/python312",
    },
]


def _patch_ddgs(results: list[dict] | None = None, *, raise_exc: Exception | None = None):
    """Context manager that patches DDGS().text() with deterministic output."""
    mock_ddgs = MagicMock()
    if raise_exc is not None:
        mock_ddgs.return_value.text.side_effect = raise_exc
    else:
        mock_ddgs.return_value.text.return_value = results or _SAMPLE_RESULTS
    return patch(
        "harness.tools.web.WebSearchTool._search",
        side_effect=lambda q, n: (
            (_ for _ in ()).throw(raise_exc)
            if raise_exc
            else (results if results is not None else _SAMPLE_RESULTS)[:n]
        ),
    )


@pytest.mark.asyncio
class TestWebSearchTool:
    async def test_returns_results_with_title_and_url(self) -> None:
        tool = WebSearchTool()
        with patch.object(WebSearchTool, "_search", return_value=_SAMPLE_RESULTS):
            result = await tool(_call("Python 3.12"))
        assert result.is_error is False
        assert "Python 3.12 Release Notes" in result.content
        assert "https://docs.python.org" in result.content
        assert "New features" in result.content

    async def test_result_count_in_metadata(self) -> None:
        tool = WebSearchTool()
        with patch.object(WebSearchTool, "_search", return_value=_SAMPLE_RESULTS):
            result = await tool(_call("Python 3.12"))
        assert result.metadata is not None
        assert result.metadata["result_count"] == 2
        assert result.metadata["query"] == "Python 3.12"

    async def test_empty_query_returns_error(self) -> None:
        tool = WebSearchTool()
        result = await tool(_call("   "))
        assert result.is_error is True
        assert "query" in result.content

    async def test_missing_query_returns_error(self) -> None:
        tool = WebSearchTool()
        call = ToolCall(id="c1", name="web_search", arguments={})
        result = await tool(call)
        assert result.is_error is True

    async def test_no_results_returns_non_error_message(self) -> None:
        tool = WebSearchTool()
        with patch.object(WebSearchTool, "_search", return_value=[]):
            result = await tool(_call("xyzzy_no_results"))
        assert result.is_error is False
        assert "no results" in result.content

    async def test_max_results_capped_at_20(self) -> None:
        tool = WebSearchTool()
        captured: list[int] = []

        def fake_search(q: str, n: int) -> list[dict]:
            captured.append(n)
            return _SAMPLE_RESULTS

        with patch.object(WebSearchTool, "_search", side_effect=fake_search):
            await tool(_call("query", max_results=999))
        assert captured[0] == 20

    async def test_search_exception_returns_error(self) -> None:
        tool = WebSearchTool()
        with patch.object(WebSearchTool, "_search", side_effect=RuntimeError("rate limited")):
            result = await tool(_call("query"))
        assert result.is_error is True
        assert "rate limited" in result.content

    async def test_approval_is_auto(self) -> None:
        assert WebSearchTool().approval == "auto"

    async def test_name_is_web_search(self) -> None:
        assert WebSearchTool().name == "web_search"

    async def test_numbered_results_format(self) -> None:
        tool = WebSearchTool()
        with patch.object(WebSearchTool, "_search", return_value=_SAMPLE_RESULTS):
            result = await tool(_call("Python 3.12"))
        assert "1." in result.content
        assert "2." in result.content
