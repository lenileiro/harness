"""Error hierarchy for Harness.

The FailoverPolicy classifies errors by these types to decide whether to retry
against the next adapter in the chain or fail terminally.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all Harness errors."""


# --- Retryable / failover-eligible ------------------------------------------


class NetworkError(HarnessError):
    """Transport-level failure (connection refused, DNS, TLS, socket reset)."""


class TimeoutError(HarnessError):
    """A bounded operation exceeded its deadline."""


class RateLimitError(HarnessError):
    """Provider rate-limited us. Carries an optional retry-after hint in seconds."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ModelUnavailableError(HarnessError):
    """The requested model isn't installed / accessible on this provider."""


class InternalError(HarnessError):
    """Provider 5xx or malformed response — generally retryable, occasionally not."""


# --- Terminal ---------------------------------------------------------------


class ConfigurationError(HarnessError):
    """The runtime is misconfigured (missing API key, unknown provider, etc.)."""


class ApprovalDeniedError(HarnessError):
    """The user (or policy) denied a tool call.

    Not retryable: the agent observes this as a tool result and proceeds.
    """


class ToolError(HarnessError):
    """A tool raised during execution. Surfaced to the agent as a tool_result with is_error=True."""


class CancelledError(HarnessError):
    """The session or run was cancelled."""


__all__ = [
    "ApprovalDeniedError",
    "CancelledError",
    "ConfigurationError",
    "HarnessError",
    "InternalError",
    "ModelUnavailableError",
    "NetworkError",
    "RateLimitError",
    "TimeoutError",
    "ToolError",
]
