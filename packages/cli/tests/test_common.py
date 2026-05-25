from __future__ import annotations

from harness.cli import common


class _FakeLoop:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

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


class _FakeAwaitable:
    def __await__(self):
        if False:
            yield None
        return "ok"


def test_run_async_suppresses_running_asyncgen_shutdown_error(monkeypatch) -> None:
    fake_loop = _FakeLoop()
    monkeypatch.setattr("harness.cli.common.asyncio.new_event_loop", lambda: fake_loop)
    monkeypatch.setattr("harness.cli.common.asyncio.set_event_loop", lambda _loop: None)

    result = common._run_async(_FakeAwaitable())

    assert result == "ok"
    assert fake_loop.closed is True
