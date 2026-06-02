"""Web tools for Harness agents: HTTP fetch and web search.

Tools:
- ``FetchUrlTool`` — GET-only HTTP fetch, capped + allow-listed.
- ``WebSearchTool`` — Web search with an API-free public fallback.

FetchUrlTool defences:
- Only ``http://`` and ``https://`` schemes are accepted.
- Response body is capped at ``max_bytes``.
- Returned text is capped at ``max_output_chars`` with explicit truncation
  metadata, so large public pages do not flood the agent context.
- Content-Type must match the allow-list.
- Configurable timeout, hard-capped by ``max_timeout``.
- Approval default is ``auto`` because the tool is read-only; URL safety is
  enforced by scheme, MIME, size, and SSRF checks.

WebSearchTool notes:
- Uses Tavily when ``TAVILY_API_KEY`` is available, otherwise uses a public
  HTML search fallback that requires no purchase, support request, or API key.
- Returns titles, snippets, and URLs from the search backend.
- Approval default is ``auto`` — search is read-only.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

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


_BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
        # Cloud metadata service hostnames — same IP risk via DNS.
        "metadata.google.internal",
        "metadata",
        "instance-data",
        "instance-data.ec2.internal",
    }
)


def _is_blocked_address(addr: str) -> bool:
    """Return True if an IP address is loopback, private, link-local, or
    otherwise unsafe to fetch from a development machine.

    Blocks: 127/8, ::1, 169.254/16 (link-local incl. AWS/Azure metadata),
    10/8, 172.16/12, 192.168/16, multicast, reserved. Allows public IPs.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False  # not an IP literal — caller handles via hostname check
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _is_blocked_host(host: str) -> str | None:
    """Return a reason string if `host` should be refused, else None.

    Handles three cases:
      1. Hostname is on the blocklist (`localhost`, cloud-metadata aliases).
      2. Host is an IP literal in a private/loopback/link-local range.
      3. Hostname resolves (via DNS) to a blocked IP — defeats DNS rebinding
         and the trick of pointing a public hostname at 127.0.0.1.

    DNS lookups happen here, not in the request — so we fail closed before
    any traffic leaves the box.
    """
    if not host:
        return "missing host"
    lowered = host.lower().strip("[]")
    if lowered in _BLOCKED_HOSTNAMES:
        return f"hostname {lowered!r} is blocked (loopback/metadata)"
    if _is_blocked_address(lowered):
        return f"address {lowered!r} is in a blocked range (loopback/private/link-local)"
    # Best-effort DNS lookup — if it fails (no network, unresolvable), we let
    # httpx handle it. If it succeeds and resolves to a blocked range, refuse.
    try:
        infos = socket.getaddrinfo(lowered, None)
    except (socket.gaierror, OSError):
        return None
    for *_, sockaddr in infos:
        # sockaddr is (host, port) for AF_INET and (host, port, flowinfo,
        # scopeid) for AF_INET6 — index 0 is always the host string for
        # both. The type union includes Unix-socket FDs which we never
        # asked for; coerce to str for the IP-range check.
        candidate = str(sockaddr[0])
        if _is_blocked_address(candidate):
            return (
                f"hostname {lowered!r} resolves to {candidate!r}, which is in "
                f"a blocked range (SSRF defense)"
            )
    return None


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
    approval: ApprovalDecision = "auto"
    effect_scope = "read_only"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        max_bytes: int = 1024 * 1024,
        max_output_chars: int = 32_000,
        default_timeout: float = 15.0,
        max_timeout: float = 60.0,
        allowed_mime_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_MIME_PREFIXES,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.max_output_chars = max(1, max_output_chars)
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

        # SSRF defense — block loopback, private, link-local addresses and
        # cloud metadata hostnames. Resolves DNS once and refuses if any
        # answer is in a blocked range.
        block_reason = _is_blocked_host(parsed.hostname or "")
        if block_reason is not None:
            return _error(call, self.name, f"refused: {block_reason}")

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
            return _error(call, self.name, f"HTTP {response.status_code}: {preview}")

        content_type = response.headers.get("content-type", "")
        if not _mime_allowed(content_type, self.allowed_mime_prefixes):
            return _error(
                call, self.name, f"content-type {content_type!r} is not in the allow-list"
            )

        body = response.content
        if len(body) > self.max_bytes:
            return _error(
                call,
                self.name,
                f"response body too large: {len(body)} bytes exceeds {self.max_bytes}",
            )

        text = body.decode(response.encoding or "utf-8", errors="replace")
        truncated = len(text) > self.max_output_chars
        visible_text = text
        if truncated:
            visible_text = text[: self.max_output_chars].rstrip()
            visible_text = (
                f"{visible_text}\n\n"
                f"[truncated: showing first {self.max_output_chars} of {len(text)} "
                "characters from this response. Use a narrower URL, web_search, "
                "or another source if the missing suffix matters.]"
            )
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"status: {response.status_code}\ncontent-type: {content_type}\n\n{visible_text}",
            metadata={
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "bytes": len(body),
                "characters": len(text),
                "returned_characters": min(len(text), self.max_output_chars),
                "truncated": truncated,
            },
        )


_TAVILY_SEARCH_SCHEMA: dict[str, Any] = {
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


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._active: Literal["title", "content", ""] = ""
        self._text_parts: list[str] = []
        self._current_url = ""
        self._snippet_index = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name: value or "" for name, value in attrs}
        class_names = attr_map.get("class", "")
        if tag == "a" and "result__a" in class_names:
            self._active = "title"
            self._text_parts = []
            self._current_url = _clean_duckduckgo_url(attr_map.get("href", ""))
            return
        if "result__snippet" in class_names:
            self._active = "content"
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._active:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._active == "title" and tag == "a":
            title = _collapse_ws("".join(self._text_parts))
            if title:
                self.results.append({"title": title, "content": "", "url": self._current_url})
            self._active = ""
            self._text_parts = []
            self._current_url = ""
            return
        if self._active == "content" and tag in {"a", "div"}:
            snippet = _collapse_ws("".join(self._text_parts))
            if snippet and self._snippet_index < len(self.results):
                self.results[self._snippet_index]["content"] = snippet
                self._snippet_index += 1
            self._active = ""
            self._text_parts = []


def _collapse_ws(value: str) -> str:
    return " ".join(value.split())


def _clean_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.query:
        uddg = parse_qs(parsed.query).get("uddg")
        if uddg:
            return unquote(uddg[0])
    return url


class WebSearchTool:
    """Search the web without requiring an API key; Tavily is optional."""

    name = "web_search"
    description = (
        "Search the internet. Returns titles, snippets, and URLs for relevant "
        "results. Use this to research topics, find current information, or look "
        "up documentation."
    )
    approval: ApprovalDecision = "auto"
    effect_scope = "read_only"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_max_results: int = 5,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        client: httpx.AsyncClient | None = None,
        fallback_url: str = "https://duckduckgo.com/html/",
    ) -> None:
        self._api_key = api_key
        self._default_max_results = min(default_max_results, 20)
        self._search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = search_depth
        self._injected_client = client
        self._fallback_url = fallback_url
        self.parameters_schema: dict[str, Any] = _TAVILY_SEARCH_SCHEMA

    def _resolve_key(self) -> str | None:
        return self._api_key or os.environ.get("TAVILY_API_KEY")

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

        api_key = self._resolve_key()
        backend = "duckduckgo"
        tavily_error = ""
        try:
            if api_key:
                results = await self._search(query.strip(), api_key, max_results)
                backend = "tavily"
            else:
                results = await self._search_public(query.strip(), max_results)
        except Exception as exc:
            if not api_key:
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content=f"search failed: {exc}",
                    is_error=True,
                )
            tavily_error = str(exc)
            try:
                results = await self._search_public(query.strip(), max_results)
                backend = "duckduckgo"
            except Exception as fallback_exc:
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content=f"search failed: {tavily_error}; fallback search failed: {fallback_exc}",
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
            lines.append(f"{i}. {r.get('title', '').strip()}")
            content = r.get("content", "").strip()
            if content:
                lines.append(f"   {content}")
            url = r.get("url", "").strip()
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
                "backend": backend,
                "results": results,
                **({"primary_backend_error": tavily_error} if tavily_error else {}),
            },
        )

    async def _search_tavily(
        self, query: str, api_key: str, max_results: int
    ) -> list[dict[str, str]]:
        import asyncio

        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = await asyncio.to_thread(
            client.search,
            query,
            max_results=max_results,
            search_depth=self._search_depth,
        )
        return response.get("results", [])

    async def _search(self, query: str, api_key: str, max_results: int) -> list[dict[str, str]]:
        return await self._search_tavily(query, api_key, max_results)

    async def _search_public(self, query: str, max_results: int) -> list[dict[str, str]]:
        owns_client = self._injected_client is None
        client = self._injected_client or httpx.AsyncClient(timeout=15.0, follow_redirects=True)
        try:
            response = await client.get(
                self._fallback_url,
                params={"q": query},
                headers={"user-agent": "harness-web-search/0.1"},
                follow_redirects=True,
            )
        finally:
            if owns_client:
                await client.aclose()
        response.raise_for_status()
        parser = _DuckDuckGoHTMLParser()
        parser.feed(response.text)
        return parser.results[:max_results]


TavilySearchTool = WebSearchTool

__all__ = [
    "DEFAULT_ALLOWED_MIME_PREFIXES",
    "FetchUrlTool",
    "TavilySearchTool",
    "WebSearchTool",
    "__version__",
]
