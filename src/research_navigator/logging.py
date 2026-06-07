"""Structured logging setup built on ``structlog``.

Call :func:`configure_logging` once at process start (CLI entrypoints do this),
then obtain loggers via :func:`get_logger`. No ``print`` statements should appear
outside CLI user-facing output.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import TextIO, cast

import structlog


class _LazyStderr:
    """A write proxy that always targets the *current* ``sys.stderr``.

    Resolving the stream per-write (rather than capturing it once) keeps logging
    correct when the stream is swapped at runtime — e.g. by test harnesses that
    capture output, or by code that reassigns ``sys.stderr``.
    """

    def write(self, message: str) -> int:
        return sys.stderr.write(message)

    def flush(self) -> None:
        with contextlib.suppress(ValueError, OSError):  # stream may be closed mid-teardown
            sys.stderr.flush()


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure ``structlog`` and the stdlib logging bridge.

    Args:
        level: Root log level, e.g. ``"INFO"`` or ``"DEBUG"``.
        json_logs: If ``True``, render logs as JSON (useful in production /
            log aggregation); otherwise render human-readable console output.
    """
    logging.basicConfig(format="%(message)s", level=level.upper(), stream=sys.stderr)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=cast("TextIO", _LazyStderr())),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, optionally named."""
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
