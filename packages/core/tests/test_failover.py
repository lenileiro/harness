from __future__ import annotations

import pytest

from harness.core import (
    ApprovalDeniedError,
    CancelledError,
    ConfigurationError,
    FailoverPolicy,
    InternalError,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TimeoutError,
    classify,
)


class TestClassify:
    @pytest.mark.parametrize(
        ("exc", "kind"),
        [
            (NetworkError("x"), "network"),
            (RateLimitError("x"), "rate_limit"),
            (TimeoutError("x"), "timeout"),
            (ModelUnavailableError("x"), "model_unavailable"),
            (InternalError("x"), "internal"),
            (ConfigurationError("x"), "configuration"),
            (ApprovalDeniedError("x"), "approval_denied"),
            (CancelledError("x"), "cancelled"),
            (RuntimeError("other"), "unknown"),
        ],
    )
    def test_classifies(self, exc: BaseException, kind: str) -> None:
        assert classify(exc) == kind


class TestFailoverPolicy:
    def test_minimum_chain_length(self) -> None:
        with pytest.raises(ValueError):
            FailoverPolicy(chain=[])

    def test_next_provider_cycles(self) -> None:
        policy = FailoverPolicy(chain=["a", "b", "c"], max_attempts=5)
        assert policy.next_provider(attempt=0) == "a"
        assert policy.next_provider(attempt=1) == "b"
        assert policy.next_provider(attempt=2) == "c"
        assert policy.next_provider(attempt=3) == "a"  # wraps

    def test_should_retry_obeys_max_attempts(self) -> None:
        policy = FailoverPolicy(chain=["a"], max_attempts=2)
        assert policy.should_retry(NetworkError("x"), attempt=0) is True
        assert policy.should_retry(NetworkError("x"), attempt=1) is False

    def test_should_retry_only_for_listed_kinds(self) -> None:
        policy = FailoverPolicy(chain=["a"], max_attempts=5, retry_on=("network",))
        assert policy.should_retry(NetworkError("x"), attempt=0) is True
        assert policy.should_retry(RateLimitError("x"), attempt=0) is False
        assert policy.should_retry(ConfigurationError("x"), attempt=0) is False

    def test_backoff_doubles_until_capped(self) -> None:
        policy = FailoverPolicy(
            chain=["a"],
            backoff_base=1.0,
            backoff_max=8.0,
            backoff_jitter=0.0,
        )
        # 1, 2, 4, 8, 8, 8 ...
        assert policy.backoff(attempt=0) == 1.0
        assert policy.backoff(attempt=1) == 2.0
        assert policy.backoff(attempt=2) == 4.0
        assert policy.backoff(attempt=3) == 8.0
        assert policy.backoff(attempt=4) == 8.0

    def test_backoff_jitter_within_bounds(self) -> None:
        policy = FailoverPolicy(
            chain=["a"],
            backoff_base=1.0,
            backoff_max=10.0,
            backoff_jitter=0.2,
        )
        # With jitter 0.2 and base 1.0, result is in [0.8, 1.2].
        for _ in range(20):
            value = policy.backoff(attempt=0)
            assert 0.8 - 1e-9 <= value <= 1.2 + 1e-9

    def test_default_retry_excludes_terminal_errors(self) -> None:
        policy = FailoverPolicy(chain=["a"], max_attempts=5)
        assert policy.should_retry(ApprovalDeniedError("x"), attempt=0) is False
        assert policy.should_retry(ConfigurationError("x"), attempt=0) is False
        assert policy.should_retry(CancelledError("x"), attempt=0) is False
