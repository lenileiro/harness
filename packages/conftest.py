from __future__ import annotations

import asyncio
import contextlib
import warnings
from collections.abc import Generator

import pytest


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
