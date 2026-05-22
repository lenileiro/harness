"""Tests for TavilySearchTool. Mocks _search to avoid network I/O."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from harness.core import ToolCall
from harness.tools.web import TavilySearchTool


def _call(query: str, **extra: object) -> ToolCall:
    return ToolCall(id="c1", name="web_search", arguments={"query": query, **extra})


_SAMPLE_RESULTS = [
    {
        "title": "Python 3.12 Release Notes",
        "content": "New features in Python 3.12 including improved error messages.",
        "url": "https://docs.python.org/3.12/whatsnew/3.12.html",
    },
    {
        "title": "Real Python — Python 3.12 Guide",
        "content": "A practical guide to what changed in 3.12.",
        "url": "https://realpython.com/python312",
    },
]


@pytest.mark.asyncio
class TestTavilySearchTool:
    async def test_returns_results_with_title_snippet_url(self) -> None:
        tool = TavilySearchTool(api_key="test-key")
        with patch.object(TavilySearchTool, "_search", return_value=_SAMPLE_RESULTS):
            result = await tool(_call("Python 3.12"))
        assert result.is_error is False
        assert "Python 3.12 Release Notes" in result.content
        assert "https://docs.python.org" in result.content
        assert "New features" in result.content

    async def test_metadata_has_backend_tavily(self) -> None:
        tool = TavilySearchTool(api_key="test-key")
        with patch.object(TavilySearchTool, "_search", return_value=_SAMPLE_RESULTS):
            result = await tool(_call("Python 3.12"))
        assert result.metadata is not None
        assert result.metadata["backend"] == "tavily"
        assert result.metadata["result_count"] == 2
        assert result.metadata["query"] == "Python 3.12"

    async def test_empty_query_returns_error(self) -> None:
        tool = TavilySearchTool(api_key="test-key")
        result = await tool(_call("   "))
        assert result.is_error is True
        assert "query" in result.content

    async def test_missing_query_returns_error(self) -> None:
        tool = TavilySearchTool(api_key="test-key")
        call = ToolCall(id="c1", name="web_search", arguments={})
        result = await tool(call)
        assert result.is_error is True

    async def test_missing_api_key_returns_helpful_error(self) -> None:
        tool = TavilySearchTool()
        import os

        env_backup = os.environ.pop("TAVILY_API_KEY", None)
        try:
            result = await tool(_call("test"))
        finally:
            if env_backup is not None:
                os.environ["TAVILY_API_KEY"] = env_backup
        assert result.is_error is True
        assert "TAVILY_API_KEY" in result.content

    async def test_no_results_returns_non_error(self) -> None:
        tool = TavilySearchTool(api_key="test-key")
        with patch.object(TavilySearchTool, "_search", return_value=[]):
            result = await tool(_call("xyzzy_no_results"))
        assert result.is_error is False
        assert "no results" in result.content

    async def test_max_results_capped_at_20(self) -> None:
        tool = TavilySearchTool(api_key="test-key")
        captured: list[int] = []

        async def fake_search(self_: object, q: str, key: str, n: int) -> list[dict]:
            captured.append(n)
            return []

        with patch.object(TavilySearchTool, "_search", fake_search):
            await tool(_call("query", max_results=999))
        assert captured[0] == 20

    async def test_search_exception_returns_error(self) -> None:
        tool = TavilySearchTool(api_key="test-key")
        with patch.object(TavilySearchTool, "_search", side_effect=RuntimeError("rate limited")):
            result = await tool(_call("query"))
        assert result.is_error is True
        assert "rate limited" in result.content

    async def test_approval_is_auto(self) -> None:
        assert TavilySearchTool().approval == "auto"

    async def test_name_is_web_search(self) -> None:
        assert TavilySearchTool().name == "web_search"

    async def test_numbered_results_format(self) -> None:
        tool = TavilySearchTool(api_key="test-key")
        with patch.object(TavilySearchTool, "_search", return_value=_SAMPLE_RESULTS):
            result = await tool(_call("Python 3.12"))
        assert "1." in result.content
        assert "2." in result.content

    async def test_api_key_from_env(self) -> None:
        import os

        os.environ["TAVILY_API_KEY"] = "env-test-key"
        try:
            tool = TavilySearchTool()
            assert tool._resolve_key() == "env-test-key"
        finally:
            del os.environ["TAVILY_API_KEY"]

    async def test_api_key_constructor_overrides_env(self) -> None:
        import os

        os.environ["TAVILY_API_KEY"] = "env-key"
        try:
            tool = TavilySearchTool(api_key="explicit-key")
            assert tool._resolve_key() == "explicit-key"
        finally:
            del os.environ["TAVILY_API_KEY"]
