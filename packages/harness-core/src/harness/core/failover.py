"""Failover policy.

Classifies errors raised by adapters and decides whether the runtime should
retry against the next provider in `chain`. Stateless — the runtime tracks
attempt counts.
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.core.errors import (
    ApprovalDeniedError,
    CancelledError,
    ConfigurationError,
    HarnessError,
    InternalError,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TimeoutError,
)

ErrorKind = Literal[
    "network",
    "rate_limit",
    "timeout",
    "model_unavailable",
    "internal",
    "configuration",
    "approval_denied",
    "cancelled",
    "unknown",
]


def classify(exc: BaseException) -> ErrorKind:
    """Map an exception instance to a stable error-kind tag.

    Used by FailoverPolicy.should_retry and by the runtime when building
    ErrorEvents.
    """
    if isinstance(exc, NetworkError):
        return "network"
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ModelUnavailableError):
        return "model_unavailable"
    if isinstance(exc, InternalError):
        return "internal"
    if isinstance(exc, ConfigurationError):
        return "configuration"
    if isinstance(exc, ApprovalDeniedError):
        return "approval_denied"
    if isinstance(exc, CancelledError):
        return "cancelled"
    if isinstance(exc, HarnessError):
        return "unknown"
    return "unknown"


_DEFAULT_RETRY_ON: tuple[ErrorKind, ...] = (
    "network",
    "rate_limit",
    "timeout",
    "model_unavailable",
    "internal",
)


class FailoverPolicy(BaseModel):
    """Ordered fallback chain across providers with bounded retries.

    The runtime walks `chain` in order. On each attempt it asks the policy
    whether the error is retryable; if so, it moves to the next provider and
    waits `backoff(attempt)` seconds.
    """

    model_config = ConfigDict(extra="forbid")

    chain: list[str] = Field(min_length=1)
    """Ordered provider names. The first is the primary; the rest are fallbacks."""

    retry_on: tuple[ErrorKind, ...] = _DEFAULT_RETRY_ON
    max_attempts: int = 3
    backoff_base: float = 0.5
    """Initial backoff in seconds. Doubled per attempt, capped by `backoff_max`."""
    backoff_max: float = 10.0
    backoff_jitter: float = 0.2
    """Fraction of computed backoff to randomize (0.0 disables jitter)."""

    def should_retry(self, exc: BaseException, *, attempt: int) -> bool:
        """True if `attempt+1 < max_attempts` AND the error kind is in `retry_on`."""
        if attempt + 1 >= self.max_attempts:
            return False
        return classify(exc) in self.retry_on

    def next_provider(self, *, attempt: int) -> str:
        """Provider to use on a given attempt index (0-based). Wraps via modulo."""
        return self.chain[attempt % len(self.chain)]

    def backoff(self, *, attempt: int) -> float:
        """Backoff in seconds for the *next* attempt after `attempt` failed."""
        base = min(self.backoff_base * (2**attempt), self.backoff_max)
        if self.backoff_jitter <= 0:
            return base
        jitter = base * self.backoff_jitter * (random.random() * 2 - 1)
        return max(0.0, base + jitter)


__all__ = ["ErrorKind", "FailoverPolicy", "classify"]
