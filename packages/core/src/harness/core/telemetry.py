"""Telemetry hooks.

structlog is the only hard dep. OTel is a soft dep: if the user installs
`opentelemetry-api` and sets `HARNESS_OTEL=1`, the `span` context manager
emits real spans; otherwise it's a no-op decorator that still logs start/end
through structlog.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

_configured = False


def configure_logging(*, level: str | int = "INFO", json: bool = False) -> None:
    """Configure structlog. Safe to call multiple times — the second call is a no-op.

    `json=True` emits one JSON object per line (good for log aggregation).
    `json=False` emits human-readable colored output (default; better for CLI).
    """
    global _configured
    if _configured:
        return

    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    # Route stdlib logging through structlog so libraries we depend on are visible.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )

    # Quiet noisy third-party loggers unless caller asked for DEBUG.
    # httpx logs every request at INFO ("HTTP Request: POST ..."), which is
    # noise for normal CLI runs but useful when --verbose.
    if level > logging.DEBUG:
        for noisy in ("httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger. Calls `configure_logging()` if needed."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()


def _otel_enabled() -> bool:
    return os.environ.get("HARNESS_OTEL", "").lower() in {"1", "true", "yes"}


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    """A logical span. Logs start/end at DEBUG; emits an OTel span if enabled.

    OTel is imported lazily inside the `if` branch so it stays a soft dep.
    """
    logger = get_logger("harness.span").bind(span=name, **attrs)
    logger.debug("span.start")

    if _otel_enabled():
        try:
            from opentelemetry import trace  # type: ignore[import-not-found]

            tracer = trace.get_tracer("harness")
            with tracer.start_as_current_span(name, attributes=attrs):
                yield
                logger.debug("span.end")
                return
        except ImportError:
            # OTel asked for but not installed — fall through to no-op.
            pass

    try:
        yield
    finally:
        logger.debug("span.end")


__all__ = ["configure_logging", "get_logger", "span"]
