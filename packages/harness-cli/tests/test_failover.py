"""End-to-end test that `--failover` actually engages the second provider.

Strategy: monkeypatch BOTH OllamaAdapter and OpenRouterAdapter to FakeAdapters
the test controls. The primary (ollama) is configured to raise a retryable
NetworkError; the secondary (openrouter) is configured to return a Done.
The CLI run should yield the secondary's output.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import (
    Capabilities,
    Done,
    Event,
    Message,
    NetworkError,
    TextDelta,
)


class _FakeAdapter:
    """Per-class queue + per-class error sentinel; one per provider."""

    next_script: ClassVar[list[list[Event]]] = []
    error: ClassVar[BaseException | None] = None
    calls: ClassVar[int] = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        cls = type(self)
        cls.calls += 1
        err = cls.error
        if err is not None:
            raise err
        if not cls.next_script:
            raise RuntimeError(f"{cls.__name__}: no scripts left")
        for ev in cls.next_script.pop(0):
            yield ev

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        pass


class FakeOllama(_FakeAdapter):
    name = "ollama"
    next_script: ClassVar[list[list[Event]]] = []
    error: ClassVar[BaseException | None] = None
    calls: ClassVar[int] = 0


class FakeOpenRouter(_FakeAdapter):
    name = "openrouter"
    next_script: ClassVar[list[list[Event]]] = []
    error: ClassVar[BaseException | None] = None
    calls: ClassVar[int] = 0


@pytest.fixture
def patch_both_adapters(monkeypatch: pytest.MonkeyPatch):
    """Replace both adapter classes with their FakeXxx counterparts."""
    monkeypatch.setattr(cli_main, "OllamaAdapter", FakeOllama)
    monkeypatch.setattr(cli_main, "OpenRouterAdapter", FakeOpenRouter)
    # Avoid OpenRouterAdapter's ConfigurationError on missing env key.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Reset class state.
    FakeOllama.next_script = []
    FakeOllama.error = None
    FakeOllama.calls = 0
    FakeOpenRouter.next_script = []
    FakeOpenRouter.error = None
    FakeOpenRouter.calls = 0
    yield


def _run(cli_args: list[str]) -> Result:
    return CliRunner().invoke(cli_main.app, cli_args)


class TestFailover:
    def test_primary_error_falls_through_to_secondary(
        self, patch_both_adapters, tmp_path: Path
    ) -> None:
        FakeOllama.error = NetworkError("primary unreachable")
        FakeOpenRouter.next_script = [
            [
                TextDelta(text="from openrouter"),
                Done(final_message=Message(role="assistant", content="from openrouter")),
            ]
        ]
        result = _run(
            [
                "run",
                "hello",
                "--failover",
                "ollama,openrouter",
                "--cwd",
                str(tmp_path),
                "--in-memory",
                "--yes",
            ]
        )
        assert result.exit_code == 0, result.stdout
        assert "from openrouter" in result.stdout
        assert FakeOllama.calls == 1
        assert FakeOpenRouter.calls == 1

    def test_no_failover_keeps_single_provider_chain(
        self, patch_both_adapters, tmp_path: Path
    ) -> None:
        FakeOllama.next_script = [
            [
                Done(final_message=Message(role="assistant", content="just ollama")),
            ]
        ]
        result = _run(
            [
                "run",
                "hello",
                "--cwd",
                str(tmp_path),
                "--in-memory",
                "--yes",
            ]
        )
        assert result.exit_code == 0
        assert FakeOllama.calls == 1
        assert FakeOpenRouter.calls == 0

    def test_secondary_also_failing_propagates_error(
        self, patch_both_adapters, tmp_path: Path
    ) -> None:
        FakeOllama.error = NetworkError("primary down")
        FakeOpenRouter.error = NetworkError("secondary down too")
        result = _run(
            [
                "run",
                "hello",
                "--failover",
                "ollama,openrouter",
                "--cwd",
                str(tmp_path),
                "--in-memory",
                "--yes",
            ]
        )
        # Both providers attempted; final error surfaces.
        assert FakeOllama.calls == 1
        assert FakeOpenRouter.calls == 1
        assert "network" in result.stdout.lower() or "error" in result.stdout.lower()
