"""Web tools for Harness agents: HTTP fetch and Tavily search.

Tools:
- ``FetchUrlTool`` — GET-only HTTP fetch, capped + allow-listed.
- ``TavilySearchTool`` — Web search via the Tavily API.

FetchUrlTool defences:
- Only ``http://`` and ``https://`` schemes are accepted.
- Response body is capped at ``max_bytes``.
- Content-Type must match the allow-list.
- Configurable timeout, hard-capped by ``max_timeout``.
- Approval default is ``prompt``.

TavilySearchTool notes:
- Requires ``TAVILY_API_KEY`` environment variable (or pass ``api_key`` directly).
- Returns titles, snippets, and URLs from Tavily's search index.
- Approval default is ``auto`` — search is read-only.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any, Literal
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
    approval: ApprovalDecision = "prompt"
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


class TavilySearchTool:
    """Search the web via Tavily. Requires TAVILY_API_KEY environment variable."""

    name = "web_search"
    description = (
        "Search the internet using Tavily. Returns titles, snippets, and URLs "
        "for the most relevant results. Use this to research topics, find current "
        "information, or look up documentation."
    )
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_max_results: int = 5,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
    ) -> None:
        self._api_key = api_key
        self._default_max_results = min(default_max_results, 20)
        self._search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = search_depth
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

        api_key = self._resolve_key()
        if not api_key:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    "TAVILY_API_KEY is not set. "
                    "Export it before running: export TAVILY_API_KEY=<your-key>"
                ),
                is_error=True,
            )

        max_results_arg = call.arguments.get("max_results", self._default_max_results)
        try:
            max_results = max(1, min(int(max_results_arg), 20))
        except (TypeError, ValueError):
            max_results = self._default_max_results

        try:
            results = await self._search(query.strip(), api_key, max_results)
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
                "backend": "tavily",
            },
        )

    async def _search(self, query: str, api_key: str, max_results: int) -> list[dict[str, str]]:
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


__all__ = [
    "DEFAULT_ALLOWED_MIME_PREFIXES",
    "FetchUrlTool",
    "TavilySearchTool",
    "__version__",
]
