from __future__ import annotations

import subprocess
import sys


def test_configured_structlog_writes_to_stderr_not_stdout() -> None:
    script = """
from harness.core.telemetry import configure_logging, get_logger
configure_logging(level='INFO')
get_logger('probe').warning('probe.message')
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout == ""
    assert "probe.message" in result.stderr
