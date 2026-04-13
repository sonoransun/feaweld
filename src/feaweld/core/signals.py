"""Graceful shutdown support for long-running analyses.

Provides a context manager that installs SIGTERM / SIGINT handlers and
exposes a threading event that pipeline stages can poll between steps.
"""

from __future__ import annotations

import signal
import threading
from types import FrameType
from typing import Any


class GracefulShutdown:
    """Context manager for cooperative shutdown on SIGTERM / SIGINT.

    Usage::

        shutdown = GracefulShutdown()
        with shutdown:
            for stage in stages:
                if shutdown.requested:
                    break
                stage.run()
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._original_handlers: dict[int, Any] = {}

    @property
    def requested(self) -> bool:
        """Return *True* if a termination signal has been received."""
        return self._event.is_set()

    def request(self) -> None:
        """Manually request shutdown (useful from other threads)."""
        self._event.set()

    def _handler(self, signum: int, frame: FrameType | None) -> None:
        self._event.set()

    def __enter__(self) -> GracefulShutdown:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handler)
            except (OSError, ValueError):
                # Cannot set signal handler outside main thread.
                pass
        return self

    def __exit__(self, *exc: object) -> None:
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass
        self._original_handlers.clear()
