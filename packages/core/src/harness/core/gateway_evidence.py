"""Gateway evidence rendering helpers.

The gateway command owns transport/session plumbing. Turning runtime activity
into a safe fallback reply is shared behavior, so it lives in core.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from harness.core.activity import ActivityEvent

_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"\bOPENROUTER_API_KEY=([^\s]+)"),
    re.compile(r"\bTAVILY_API_KEY=([^\s]+)"),
    re.compile(r"sk-or-v1" + r"-[A-Za-z0-9_-]+"),
    re.compile(r"tvly-[A-Za-z0-9_-]+"),
)


def _compact_text(text: str, *, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _redact_sensitive_text(text: str) -> str:
    redacted = str(text or "")
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(
                lambda match: match.group(0).split("=", 1)[0] + "=<redacted>",
                redacted,
            )
        else:
            redacted = pattern.sub(
                lambda match: match.group(0).split("-", 2)[0] + "-<redacted>",
                redacted,
            )
    return redacted


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def successful_tool_evidence_reply(events: Iterable[ActivityEvent]) -> str | None:
    """Return a user-facing fallback answer from the latest successful evidence."""

    for event in reversed(list(events)):
        data: dict[str, Any] = event.data or {}
        if data.get("is_error"):
            continue
        name = str(data.get("name") or "")
        metadata = data.get("metadata")
        if name == "web_search" and isinstance(metadata, dict):
            results = metadata.get("results")
            if not isinstance(results, list) or not results:
                continue
            first = next((item for item in results if isinstance(item, dict)), None)
            if not first:
                continue
            title = str(first.get("title") or "Search result").strip()
            url = str(first.get("url") or "").strip()
            content = _compact_text(str(first.get("content") or ""), limit=700)
            source = f"{title}: {url}" if url else title
            if content:
                return f"I found current tool evidence from {source}\n\n{content}"
            return f"I found current tool evidence from {source}"
        if name == "fetch_url":
            arguments = _dict_or_empty(data.get("arguments"))
            url = str(arguments.get("url") or "").strip()
            preview = _compact_text(str(data.get("content_preview") or ""), limit=700)
            if url and preview:
                return f"I found current tool evidence from {url}\n\n{preview}"
            if preview:
                return f"I found current tool evidence:\n\n{preview}"
        if name == "shell":
            arguments = _dict_or_empty(data.get("arguments"))
            command = _compact_text(
                _redact_sensitive_text(str(arguments.get("command") or "")),
                limit=180,
            )
            preview = _compact_text(
                _redact_sensitive_text(str(data.get("content_preview") or "")),
                limit=700,
            )
            if command and preview:
                return f"I verified this with local shell evidence from `{command}`:\n\n{preview}"
            if preview:
                return f"I verified this with local shell evidence:\n\n{preview}"
        if name == "verify_work":
            arguments = _dict_or_empty(data.get("arguments"))
            command = _compact_text(
                _redact_sensitive_text(str(arguments.get("command") or "")),
                limit=180,
            )
            preview = _compact_text(
                _redact_sensitive_text(str(data.get("content_preview") or "")),
                limit=700,
            )
            if command and preview:
                return f"I verified this with `verify_work` using `{command}`:\n\n{preview}"
            if preview:
                return f"I verified this with `verify_work`:\n\n{preview}"
        if name in {"write_file", "edit_file"}:
            arguments = _dict_or_empty(data.get("arguments"))
            path = str(arguments.get("path") or "").strip()
            preview = _compact_text(str(data.get("content_preview") or ""), limit=700)
            if path and preview:
                return f"The core runtime changed `{path}`:\n\n{preview}"
            if preview:
                return f"The core runtime changed the workspace:\n\n{preview}"
    return None


__all__ = ["successful_tool_evidence_reply"]
