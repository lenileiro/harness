"""HTTP fetch tool for Harness agents.

Single tool: `fetch_url(url, timeout?)`. GET-only by design — the agent has
no business issuing POST/PUT/DELETE through a generic tool.

Defences:
- Only `http://` and `https://` schemes are accepted.
- Response body is capped at `max_bytes`.
- Content-Type must match the allow-list (default: text/*, application/json,
  application/xml, application/javascript).
- Configurable timeout, hard-capped by `max_timeout`.
- Approval default is `prompt` — fetching arbitrary URLs is a network egress
  capability worth confirming.
"""

from __future__ import annotations

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
        )


__all__ = ["DEFAULT_ALLOWED_MIME_PREFIXES", "FetchUrlTool", "__version__"]
