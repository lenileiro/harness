from __future__ import annotations

from typing import ClassVar

from harness.cli import common
from harness.cli.config import HarnessConfig


class _OpenRouterAdapterProbe:
    captured: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).captured = kwargs


class _CodexAdapterProbe:
    captured: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).captured = kwargs


class _ClaudeAdapterProbe:
    captured: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).captured = kwargs


class _FakeLoop:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False
        self.created_task: object | None = None

    def run_until_complete(self, awaitable: object) -> str | None:
        self.calls += 1
        if self.calls == 1:
            return "ok"
        if self.calls == 2:
            raise RuntimeError("aclose(): asynchronous generator is already running")
        return None

    def shutdown_asyncgens(self) -> object:
        return object()

    def shutdown_default_executor(self) -> object:
        return object()

    def close(self) -> None:
        self.closed = True

    def create_task(self, awaitable: object) -> object:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        self.created_task = awaitable
        return awaitable


class _FakeAwaitable:
    def __await__(self):
        if False:
            yield None
        return "ok"


def test_run_async_suppresses_running_asyncgen_shutdown_error(monkeypatch) -> None:
    fake_loop = _FakeLoop()
    monkeypatch.setattr("harness.cli.common.asyncio.new_event_loop", lambda: fake_loop)
    monkeypatch.setattr("harness.cli.common.asyncio.set_event_loop", lambda _loop: None)
    monkeypatch.setattr("harness.cli.common.asyncio.all_tasks", lambda _loop: set())

    result = common._run_async(_FakeAwaitable())

    assert result == "ok"
    assert fake_loop.closed is True


def test_common_openrouter_adapter_respects_configured_timeout(monkeypatch) -> None:
    monkeypatch.setattr(common, "OpenRouterAdapter", _OpenRouterAdapterProbe)
    cfg = HarnessConfig(provider_settings={"openrouter": {"timeout": 17.5}})

    common._build_adapter("openrouter", base_url=None, config=cfg)

    assert _OpenRouterAdapterProbe.captured["timeout"] == 17.5


def test_common_codex_adapter_uses_model_timeout_env_overrides(monkeypatch) -> None:
    monkeypatch.setattr(common, "CodexAdapter", _CodexAdapterProbe)
    monkeypatch.setenv("HARNESS_MODEL_TURN_TIMEOUT", "1200")
    monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "240")
    cfg = HarnessConfig(provider_settings={"codex": {"timeout": 600.0, "idle_timeout": 120.0}})

    common._build_adapter("codex", base_url=None, config=cfg)

    assert _CodexAdapterProbe.captured["timeout"] == 1200.0
    assert _CodexAdapterProbe.captured["idle_timeout"] == 240.0


def test_common_claude_adapter_reads_configured_settings(monkeypatch) -> None:
    monkeypatch.setattr(common, "ClaudeAdapter", _ClaudeAdapterProbe)
    monkeypatch.delenv("HARNESS_MODEL_TURN_TIMEOUT", raising=False)
    monkeypatch.delenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", raising=False)
    cfg = HarnessConfig(
        provider_settings={
            "claude": {
                "timeout": 900.0,
                "idle_timeout": 90.0,
                "effort": "high",
                "max_budget_usd": 0.5,
                "cwd": "/tmp/workspace",
            }
        }
    )

    common._build_adapter("claude", base_url=None, config=cfg)

    captured = _ClaudeAdapterProbe.captured
    assert captured["timeout"] == 900.0
    assert captured["idle_timeout"] == 90.0
    assert captured["effort"] == "high"
    assert captured["max_budget_usd"] == 0.5
    assert captured["cwd"] == "/tmp/workspace"


def test_common_claude_adapter_defaults(monkeypatch) -> None:
    monkeypatch.setattr(common, "ClaudeAdapter", _ClaudeAdapterProbe)
    monkeypatch.delenv("HARNESS_MODEL_TURN_TIMEOUT", raising=False)
    monkeypatch.delenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", raising=False)

    common._build_adapter("claude", base_url=None, config=HarnessConfig())

    captured = _ClaudeAdapterProbe.captured
    assert captured["effort"] == "medium"
    assert captured["max_budget_usd"] is None


def test_claude_is_a_known_provider() -> None:
    assert "claude" in common.KNOWN_PROVIDERS
