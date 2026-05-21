"""Context budget governance.

A long-running session's message history grows unbounded. When it grows
past the model's context window, the adapter call fails. `ContextBudget`
+ `prune()` give us a token-aware sliding window — we keep the first N
messages (typically the system prompt), the last N (most recent context),
and drop from the middle until we're under budget.

Critical invariant: assistant messages with `tool_calls` and the
`role=tool` messages that follow them are one atomic block. The OpenAI
wire format requires that every `tool_call_id` referenced by an assistant
turn has a matching `tool` message — and vice versa. `_atomic_blocks`
groups them so the pruner never produces an orphan.

Token counting uses `tiktoken` for OpenAI-family models. For unknown
models (Ollama, Gemini, etc.) we fall back to `cl100k_base`, which is
approximate but stable enough for budget decisions.
"""

from __future__ import annotations

import functools
import json
from typing import Literal

import tiktoken
from pydantic import BaseModel, ConfigDict

from harness.core.schemas import Message

PruneStrategy = Literal["sliding_window"]


class ContextBudget(BaseModel):
    """Sliding-window pruning policy for a session's message history.

    `keep_first_n` typically protects the system prompt. `keep_last_n`
    protects the most recent turns (which the agent usually needs to
    continue coherently). The pruner drops middle blocks until the total
    token count is at or below `max_tokens`.

    If even keeping just the first+last blocks exceeds the budget, the
    pruner returns the kept blocks anyway — overshooting beats producing
    an unsendable message list.
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: int = 64_000
    keep_first_n: int = 1
    """Blocks (not individual messages) at the head to always keep."""
    keep_last_n: int = 8
    """Blocks at the tail to always keep."""
    strategy: PruneStrategy = "sliding_window"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _encoder_for(model: str) -> tiktoken.Encoding:
    """Resolve a model id to a tiktoken encoder, falling back to cl100k_base."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


_PER_MESSAGE_OVERHEAD = 3
"""Approximate per-message overhead in OpenAI's chat-completions tokenization."""


def count_tokens(messages: list[Message], model: str) -> int:
    """Approximate the total token count for sending `messages` to `model`.

    Counts `content`, serialized tool_call arguments, and a small per-message
    overhead. Approximate for non-OpenAI models (Ollama etc.) — see module
    docstring.
    """
    enc = _encoder_for(model)
    total = 0
    for msg in messages:
        total += _PER_MESSAGE_OVERHEAD
        if msg.content:
            total += len(enc.encode(msg.content))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total += len(enc.encode(tc.name))
                total += len(enc.encode(json.dumps(tc.arguments)))
                total += len(enc.encode(tc.id))
        if msg.tool_call_id:
            total += len(enc.encode(msg.tool_call_id))
        if msg.name:
            total += len(enc.encode(msg.name))
    return total


# ---------------------------------------------------------------------------
# Atomic-block grouping
# ---------------------------------------------------------------------------


def _atomic_blocks(messages: list[Message]) -> list[list[Message]]:
    """Group messages into prunable blocks.

    Each block is a contiguous list of messages that must be kept or
    dropped together. Rules:

      - `system` → its own block.
      - `user` → its own block.
      - `assistant` with NO `tool_calls` → its own block.
      - `assistant` with `tool_calls` AND its immediately-following `tool`
        messages whose `tool_call_id` matches one of the call ids → a
        single block.
      - An orphan `tool` message (one that doesn't match any preceding
        unconsumed tool_call) gets its own single-message block. The
        runtime shouldn't produce this, but the pruner doesn't crash if it
        encounters one.
    """
    blocks: list[list[Message]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "assistant" and m.tool_calls:
            block = [m]
            call_ids = {tc.id for tc in m.tool_calls}
            j = i + 1
            # Consume any directly-following tool messages whose id is in
            # this assistant's call set.
            while (
                j < len(messages)
                and messages[j].role == "tool"
                and messages[j].tool_call_id in call_ids
            ):
                block.append(messages[j])
                call_ids.discard(messages[j].tool_call_id or "")
                j += 1
            blocks.append(block)
            i = j
        else:
            blocks.append([m])
            i += 1
    return blocks


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def prune(messages: list[Message], *, budget: ContextBudget, model: str) -> list[Message]:
    """Return a possibly-shorter message list that fits within the budget.

    Drops from the middle, preserving `keep_first_n` blocks at the head and
    `keep_last_n` blocks at the tail. Never splits an
    assistant-with-tool-calls block. Returns a NEW list — does not mutate
    the input.
    """
    if budget.strategy != "sliding_window":  # pragma: no cover — future-proof
        raise NotImplementedError(f"unknown strategy: {budget.strategy!r}")

    current = count_tokens(messages, model)
    if current <= budget.max_tokens:
        return list(messages)

    blocks = _atomic_blocks(messages)
    head = blocks[: budget.keep_first_n]
    tail_start = max(len(blocks) - budget.keep_last_n, budget.keep_first_n)
    tail = blocks[tail_start:]
    middle = blocks[len(head) : tail_start]

    # Keep dropping middle blocks (newest first — we keep the oldest middle
    # blocks as context for as long as possible) until we fit or run out.
    kept_middle = list(middle)
    while kept_middle:
        kept = _flatten(head + kept_middle + tail)
        if count_tokens(kept, model) <= budget.max_tokens:
            return kept
        # Drop the oldest middle block first (it's least recent — we prefer
        # keeping more recent context).
        kept_middle.pop(0)

    # All middle gone; check if head+tail alone fit. If they still overshoot,
    # return them anyway — we'd rather send a slightly-too-big list than an
    # invalid one missing tool/assistant pairs.
    return _flatten(head + tail)


def _flatten(blocks: list[list[Message]]) -> list[Message]:
    return [m for block in blocks for m in block]


__all__ = [
    "ContextBudget",
    "PruneStrategy",
    "count_tokens",
    "prune",
]
