"""Structured logging for feaweld.

Provides a unified logging setup for all feaweld modules with support for
console output, JSON-line format (for containers), and optional systemd
journal integration.
"""

from __future__ import annotations

import json
import logging
import sys


class _JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(
    level: int | str = logging.INFO,
    *,
    use_journal: bool = False,
    json_format: bool = False,
    stream: object | None = None,
) -> None:
    """Configure the ``feaweld`` logger hierarchy.

    Args:
        level: Logging level (int or string like ``"DEBUG"``).
        use_journal: If *True*, add a ``systemd.journal.JournalHandler``
            (requires the ``systemd-python`` package).
        json_format: If *True*, emit records as single-line JSON.
        stream: Output stream for the console handler (default *stderr*).
    """
    root = logging.getLogger("feaweld")
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls.
    root.handlers.clear()

    if use_journal:
        try:
            from systemd.journal import JournalHandler  # type: ignore[import-untyped]
            jh = JournalHandler(SYSLOG_IDENTIFIER="feaweld")
            jh.setLevel(level)
            root.addHandler(jh)
            return  # journal handler is sufficient on its own
        except ImportError:
            pass  # fall through to console handler

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    if json_format:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``feaweld`` namespace.

    Usage::

        from feaweld.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("stage complete")
    """
    if not name.startswith("feaweld"):
        name = f"feaweld.{name}"
    return logging.getLogger(name)
