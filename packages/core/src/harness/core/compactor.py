"""LLM-based context compaction for long-running sessions.

When a session's message history grows past a threshold, the compactor
summarizes the oldest messages via a separate LLM call and replaces them
with a synthetic summary message. This preserves coherence while freeing
context space — strictly better than silent truncation.

The cut-point algorithm walks backward from the most recent messages,
accumulating tokens until a ``keep_recent_tokens`` budget is exhausted.
Everything before the cut goes to the summarizer; everything after is
kept verbatim.  The summarizer's output is injected as a ``system`` role
message at the head of the kept history.
"""

from __future__ import annotations

import json

from harness.core.adapter import Adapter
from harness.core.budget import _atomic_blocks, count_tokens
from harness.core.schemas import Message

_COMPACT_THRESHOLD = 0.80
"""Compact when token count exceeds this fraction of max_tokens."""

_KEEP_RECENT_TOKENS = 20_000
"""Tokens of recent history always preserved verbatim."""

_SUMMARIZE_SYSTEM = (
    "You are a context summarization assistant. "
    "Produce a concise summary of the conversation excerpt below, covering:\n"
    "- What was accomplished\n"
    "- Key files read or modified (name them explicitly)\n"
    "- Decisions made and their rationale\n"
    "- Current state and what remains to do\n\n"
    "Write 3-8 bullet points. Be specific — name files, commands, and results. "
    "Do NOT continue the conversation. Output ONLY the summary."
)


class ContextCompactor:
    """Summarize-and-replace old turns when context approaches the budget.

    Args:
        adapter: LLM adapter used for the summarization call.
        model: Model identifier for summarization (may differ from the main model).
        max_tokens: Context ceiling in tokens. Compact fires at 80% of this.
        keep_recent_tokens: Token budget for the verbatim recent tail.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        model: str,
        max_tokens: int = 64_000,
        keep_recent_tokens: int = _KEEP_RECENT_TOKENS,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._max_tokens = max_tokens
        self._keep_recent = keep_recent_tokens

    def should_compact(self, messages: list[Message]) -> bool:
        """Return True if the message list is large enough to warrant compaction."""
        return count_tokens(messages, self._model) > int(self._max_tokens * _COMPACT_THRESHOLD)

    async def compact(self, messages: list[Message]) -> list[Message]:
        """Summarize old messages and return a shorter list.

        Returns the original list unchanged if there is nothing to summarize
        (e.g. history is already minimal).
        """
        blocks = _atomic_blocks(messages)

        # Walk backward, accumulating the recent tail up to keep_recent_tokens.
        recent_blocks: list[list[Message]] = []
        recent_tokens = 0
        for block in reversed(blocks):
            block_tokens = count_tokens(block, self._model)
            if recent_tokens + block_tokens > self._keep_recent and recent_blocks:
                break
            recent_blocks.insert(0, block)
            recent_tokens += block_tokens

        cut = len(blocks) - len(recent_blocks)
        old_blocks = blocks[:cut]

        if not old_blocks:
            return messages  # nothing to compact

        old_messages = [m for block in old_blocks for m in block]
        summary = await self._summarize(old_messages)

        kept = [m for block in recent_blocks for m in block]
        summary_msg = Message(role="system", content=f"[Compacted context summary]\n{summary}")
        return [summary_msg, *kept]

    async def _summarize(self, messages: list[Message]) -> str:
        """One-shot LLM call to summarize a list of messages."""
        from harness.core.events import Done

        conversation = _serialize(messages)
        call_messages = [
            Message(role="system", content=_SUMMARIZE_SYSTEM),
            Message(role="user", content=f"Conversation excerpt:\n\n{conversation}"),
        ]

        text: list[str] = []
        try:
            async for event in self._adapter.stream(
                model=self._model, messages=call_messages, tools=None
            ):
                if isinstance(event, Done):
                    if event.final_message and event.final_message.content:
                        text.append(event.final_message.content)
                    break
        except Exception:
            return "(summary unavailable — compaction LLM call failed)"

        return "".join(text).strip() or "(empty summary)"


def _serialize(messages: list[Message]) -> str:
    """Convert a message list to a readable text form for the summarizer."""
    parts: list[str] = []
    for m in messages:
        if m.role == "system":
            parts.append(f"[System]: {m.content}")
        elif m.role == "user":
            parts.append(f"[User]: {m.content}")
        elif m.role == "assistant":
            if m.content:
                parts.append(f"[Assistant]: {m.content}")
            if m.tool_calls:
                for tc in m.tool_calls:
                    parts.append(f"[Tool call]: {tc.name}({json.dumps(tc.arguments)})")
        elif m.role == "tool":
            # Cap long tool results so the summarization prompt stays manageable.
            content = (m.content or "")[:800]
            parts.append(f"[Tool result ({m.name})]: {content}")
    return "\n".join(parts)


__all__ = ["ContextCompactor"]
