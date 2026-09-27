"""Shared fixtures for the eval test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _deterministic_git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give git an identity so tests do not depend on the developer's config.

    Several eval tests shell out to `git commit`. On a machine with no
    `user.name`/`user.email` configured -- a fresh CI runner, a container --
    git refuses to commit and the test fails for a reason that has nothing to
    do with what it is checking. These variables take precedence over
    `git config`, so they also keep the author of test commits out of the
    developer's real identity.
    """

    for variable, value in (
        ("GIT_AUTHOR_NAME", "Harness Eval"),
        ("GIT_AUTHOR_EMAIL", "eval@harness.invalid"),
        ("GIT_COMMITTER_NAME", "Harness Eval"),
        ("GIT_COMMITTER_EMAIL", "eval@harness.invalid"),
    ):
        monkeypatch.setenv(variable, value)
