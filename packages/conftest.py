from __future__ import annotations

import asyncio
import contextlib
import warnings
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _isolated_harness_config(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the developer's real config out of the suite.

    `default_config_path()` resolves `$HARNESS_CONFIG`, then
    `$XDG_CONFIG_HOME/harness/config.toml`, then `~/.config/harness/config.toml`.
    Without this, anyone who has actually configured Harness runs a different
    suite than CI does: a `[default] provider` alone is enough to send tests down
    a real provider path and fail dozens of them.

    The path deliberately does not exist, so `load_config()` returns defaults.
    """

    isolated = tmp_path_factory.mktemp("harness-config") / "config.toml"
    monkeypatch.setenv("HARNESS_CONFIG", str(isolated))


@pytest.fixture(autouse=True)
def _close_stray_event_loop() -> Generator[None, None, None]:
    yield

    with contextlib.suppress(RuntimeError):
        # Python 3.12 warns when this fallback path is consulted without a
        # current loop. The fixture only needs best-effort cleanup of a stray
        # loop left behind by a prior test, so suppress that probe warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_running() or loop.is_closed():
            return
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
