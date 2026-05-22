"""Web tools for Harness agents: HTTP fetch, DuckDuckGo search, and SearXNG.

Tools:
- ``fetch_url(url, timeout?)`` — GET-only HTTP fetch, capped + allow-listed.
- ``WebSearchTool`` — DuckDuckGo search via the ``ddgs`` library; no API key.
- ``SearXNGSearchTool`` — Search via a self-hosted SearXNG instance; open source,
  no API key, aggregates 70+ engines. Requires a running SearXNG server.

fetch_url defences:
- Only ``http://`` and ``https://`` schemes are accepted.
- Response body is capped at ``max_bytes``.
- Content-Type must match the allow-list.
- Configurable timeout, hard-capped by ``max_timeout``.
- Approval default is ``prompt``.

SearXNG quick-start (Docker, one command)::

    docker run -d --name searxng -p 8080:8080 \\
      -e SEARXNG_SECRET=$(openssl rand -hex 32) \\
      searxng/searxng:latest

Then enable JSON format in settings.yml (inside the container)::

    search:
      formats:
        - html
        - json

Point ``SearXNGSearchTool(base_url="http://localhost:8080")`` at it.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlparse

import httpx

from harness.core import ApprovalDecision, ToolCall, ToolResult

__version__ = "0.0.0"


_FETCH_URL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "Absolute http:// or https:// URL to GET.",
        },
        "timeout": {
            "type": "integer",
            "description": "Request timeout in seconds (capped by the tool's max).",
        },
    },
    "required": ["url"],
}


DEFAULT_ALLOWED_MIME_PREFIXES: tuple[str, ...] = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-yaml",
)


def _mime_allowed(content_type: str, allowed: tuple[str, ...]) -> bool:
    primary = content_type.split(";", 1)[0].strip().lower()
    return any(primary.startswith(prefix) for prefix in allowed)


def _error(call: ToolCall, name: str, message: str) -> ToolResult:
    return ToolResult(tool_call_id=call.id, name=name, content=message, is_error=True)


class FetchUrlTool:
    """GET a URL and return the body. Caps + allow-list applied."""

    name = "fetch_url"
    description = (
        "GET an http(s) URL and return the response body as text. Refuses "
        "non-http(s) schemes, non-allowlisted MIME types, oversized bodies, "
        "and non-2xx responses."
    )
    approval: ApprovalDecision = "prompt"
    # GET is observational from harness' perspective — safe across phases.
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        max_bytes: int = 1024 * 1024,
        default_timeout: float = 15.0,
        max_timeout: float = 60.0,
        allowed_mime_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_MIME_PREFIXES,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.allowed_mime_prefixes = allowed_mime_prefixes
        self._injected_client = client
        self.parameters_schema: dict[str, Any] = _FETCH_URL_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        url = call.arguments.get("url")
        if not isinstance(url, str) or not url:
            return _error(call, self.name, "missing or empty `url` argument")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return _error(
                call, self.name, f"unsupported scheme {parsed.scheme!r}; use http or https"
            )
        if not parsed.netloc:
            return _error(call, self.name, "URL is missing a host")

        timeout_arg = call.arguments.get("timeout", self.default_timeout)
        try:
            timeout = float(timeout_arg)
        except (TypeError, ValueError):
            timeout = self.default_timeout
        timeout = max(0.1, min(timeout, self.max_timeout))

        owns_client = self._injected_client is None
        client = self._injected_client or httpx.AsyncClient(timeout=timeout, follow_redirects=True)

        try:
            try:
                response = await client.get(url, timeout=timeout, follow_redirects=True)
            except httpx.ConnectError as exc:
                return _error(call, self.name, f"connection error: {exc}")
            except httpx.TimeoutException:
                return _error(call, self.name, f"request timed out after {timeout}s")
            except httpx.HTTPError as exc:
                return _error(call, self.name, f"http error: {exc}")
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 400:
            preview = response.text[:200] if response.text else ""
            return _error(
                call,
                self.name,
                f"HTTP {response.status_code}: {preview}",
            )

        content_type = response.headers.get("content-type", "")
        if not _mime_allowed(content_type, self.allowed_mime_prefixes):
            return _error(
                call,
                self.name,
                f"content-type {content_type!r} is not in the allow-list",
            )

        # Re-check size after the fact (Content-Length may be missing or wrong).
        body = response.content
        if len(body) > self.max_bytes:
            return _error(
                call,
                self.name,
                f"response body too large: {len(body)} bytes exceeds {self.max_bytes}",
            )

        text = body.decode(response.encoding or "utf-8", errors="replace")
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"status: {response.status_code}\ncontent-type: {content_type}\n\n{text}",
            metadata={
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "bytes": len(body),
            },
        )


_WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default 5, max 20).",
        },
    },
    "required": ["query"],
}


class WebSearchTool:
    """Search the web via DuckDuckGo. No API key required."""

    name = "web_search"
    description = (
        "Search the internet using DuckDuckGo. Returns titles, snippets, and URLs "
        "for the most relevant results. Use this to research topics, find current "
        "information, or look up documentation."
    )
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(self, *, max_results: int = 5) -> None:
        self._default_max_results = min(max_results, 20)
        self.parameters_schema: dict[str, Any] = _WEB_SEARCH_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        query = call.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="missing or empty `query` argument",
                is_error=True,
            )

        max_results_arg = call.arguments.get("max_results", self._default_max_results)
        try:
            max_results = max(1, min(int(max_results_arg), 20))
        except (TypeError, ValueError):
            max_results = self._default_max_results

        try:
            results = await asyncio.to_thread(self._search, query.strip(), max_results)
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"search failed: {exc}",
                is_error=True,
            )

        if not results:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"no results found for: {query}",
            )

        lines = [f"Results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            href = r.get("href", "").strip()
            lines.append(f"{i}. {title}")
            if body:
                lines.append(f"   {body}")
            if href:
                lines.append(f"   URL: {href}")
            lines.append("")

        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="\n".join(lines).rstrip(),
            metadata={"query": query, "result_count": len(results)},
        )

    @staticmethod
    def _search(query: str, max_results: int) -> list[dict[str, str]]:
        from duckduckgo_search import DDGS

        return list(DDGS().text(query, max_results=max_results))


_SEARXNG_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default 5, max 20).",
        },
        "engines": {
            "type": "string",
            "description": "Comma-separated list of search engines to use (e.g. 'google,bing,duckduckgo'). Leave empty to use SearXNG defaults.",
        },
    },
    "required": ["query"],
}


class SearXNGSearchTool:
    """Search the web via a self-hosted SearXNG instance. No API key required.

    Start SearXNG with Docker::

        docker run -d --name searxng -p 8080:8080 \\
          -e SEARXNG_SECRET=$(openssl rand -hex 32) \\
          searxng/searxng:latest

    Then enable JSON format in the container's settings.yml::

        search:
          formats: [html, json]
    """

    name = "web_search"
    description = (
        "Search the internet via SearXNG (open-source, self-hosted, no API key). "
        "Returns titles, snippets, and URLs for the most relevant results. "
        "Use this to research topics, find current information, or look up documentation."
    )
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        default_max_results: int = 5,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_max_results = min(default_max_results, 20)
        self._timeout = timeout
        self._injected_client = client
        self.parameters_schema: dict[str, Any] = _SEARXNG_SEARCH_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        query = call.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="missing or empty `query` argument",
                is_error=True,
            )

        max_results_arg = call.arguments.get("max_results", self._default_max_results)
        try:
            max_results = max(1, min(int(max_results_arg), 20))
        except (TypeError, ValueError):
            max_results = self._default_max_results

        engines = call.arguments.get("engines", "")

        params: dict[str, str] = {
            "q": query.strip(),
            "format": "json",
            "pageno": "1",
        }
        if engines and isinstance(engines, str):
            params["engines"] = engines

        owns_client = self._injected_client is None
        client = self._injected_client or httpx.AsyncClient(
            timeout=self._timeout, follow_redirects=True
        )
        try:
            try:
                response = await client.get(
                    f"{self._base_url}/search", params=params, timeout=self._timeout
                )
            except httpx.ConnectError as exc:
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content=f"could not connect to SearXNG at {self._base_url}: {exc}",
                    is_error=True,
                )
            except httpx.TimeoutException:
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content=f"SearXNG request timed out after {self._timeout}s",
                    is_error=True,
                )
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code == 403:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    "SearXNG returned 403 Forbidden. "
                    "Enable JSON format in settings.yml: search.formats: [html, json]"
                ),
                is_error=True,
            )
        if response.status_code >= 400:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"SearXNG returned HTTP {response.status_code}",
                is_error=True,
            )

        try:
            data = response.json()
        except Exception:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="SearXNG returned non-JSON response; is JSON format enabled?",
                is_error=True,
            )

        results = data.get("results", [])[:max_results]

        if not results:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"no results found for: {query}",
            )

        lines = [f"Results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()
            snippet = r.get("content", "").strip()
            url = r.get("url", "").strip()
            engine = r.get("engine", "")
            lines.append(f"{i}. {title}" + (f" [{engine}]" if engine else ""))
            if snippet:
                lines.append(f"   {snippet}")
            if url:
                lines.append(f"   URL: {url}")
            lines.append("")

        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="\n".join(lines).rstrip(),
            metadata={
                "query": query,
                "result_count": len(results),
                "backend": "searxng",
                "base_url": self._base_url,
            },
        )


_PLAYWRIGHT_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query.",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default 5, max 20).",
        },
    },
    "required": ["query"],
}

# XPath expressions for DuckDuckGo HTML endpoint result parsing.
_DDG_HTML_RESULT_XPATH = '//div[contains(@class, "web-result")]'
_DDG_HTML_TITLE_XPATH = './/a[contains(@class, "result__a")]'
_DDG_HTML_SNIPPET_XPATH = './/*[contains(@class, "result__snippet")]'
_DDG_HTML_URL_XPATH = './/*[contains(@class, "result__url")]'


class PlaywrightSearchTool:
    """Search the web via DuckDuckGo's HTML endpoint.

    Uses ``httpx`` + ``lxml`` to query DuckDuckGo's plain-HTML search
    (``html.duckduckgo.com``), bypassing bot detection that blocks headless
    browsers on all major search engines. No API key, no browser install,
    no rate-limit concerns for normal usage.

    Playwright (``playwright install chromium``) is available on the same
    agent for rendering specific pages via ``fetch_url`` once you have URLs
    from this tool.
    """

    name = "web_search"
    description = (
        "Search the internet via DuckDuckGo HTML. Returns titles, snippets, "
        "and URLs for the most relevant results. No API key required. "
        "Use for research, current events, documentation lookup."
    )
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        default_max_results: int = 5,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._default_max_results = min(default_max_results, 20)
        self._timeout = timeout
        self._injected_client = client
        self.parameters_schema: dict[str, Any] = _PLAYWRIGHT_SEARCH_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        query = call.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="missing or empty `query` argument",
                is_error=True,
            )

        max_results_arg = call.arguments.get("max_results", self._default_max_results)
        try:
            max_results = max(1, min(int(max_results_arg), 20))
        except (TypeError, ValueError):
            max_results = self._default_max_results

        try:
            results = await self._search(query.strip(), max_results)
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"search failed: {exc}",
                is_error=True,
            )

        if not results:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"no results found for: {query}",
            )

        lines = [f"Results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            if r.get("url"):
                lines.append(f"   URL: {r['url']}")
            lines.append("")

        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="\n".join(lines).rstrip(),
            metadata={
                "query": query,
                "result_count": len(results),
                "backend": "playwright-chromium",
            },
        )

    async def _search(self, query: str, max_results: int) -> list[dict[str, str]]:
        from lxml import html as lxml_html

        owns_client = self._injected_client is None
        client = self._injected_client or httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        try:
            response = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                timeout=self._timeout,
            )
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 400:
            raise RuntimeError(f"DDG returned HTTP {response.status_code}")

        tree = lxml_html.fromstring(response.content)
        result_nodes = tree.xpath(_DDG_HTML_RESULT_XPATH)

        parsed: list[dict[str, str]] = []
        for node in result_nodes[:max_results]:
            title_nodes = node.xpath(_DDG_HTML_TITLE_XPATH)
            title = title_nodes[0].text_content().strip() if title_nodes else ""

            snippet_nodes = node.xpath(_DDG_HTML_SNIPPET_XPATH)
            snippet = snippet_nodes[0].text_content().strip() if snippet_nodes else ""

            url_nodes = node.xpath(_DDG_HTML_URL_XPATH)
            display_url = url_nodes[0].text_content().strip() if url_nodes else ""

            if title or display_url:
                parsed.append({"title": title, "snippet": snippet, "url": display_url})

        return parsed


__all__ = [
    "DEFAULT_ALLOWED_MIME_PREFIXES",
    "FetchUrlTool",
    "PlaywrightSearchTool",
    "SearXNGSearchTool",
    "WebSearchTool",
    "__version__",
]
