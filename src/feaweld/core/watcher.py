"""File-system watching for hot-reload of case files and material databases.

Uses the ``watchdog`` library (inotify on Linux) to detect changes and
trigger re-validation or cache invalidation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from feaweld.core.logging import get_logger

logger = get_logger(__name__)


def _require_watchdog() -> None:
    try:
        import watchdog  # noqa: F401
    except ImportError:
        raise ImportError(
            "watchdog is required for file watching. "
            "Install it with: pip install feaweld[daemon]"
        ) from None


class CaseFileWatcher:
    """Watch a directory for YAML case file changes.

    On modification or creation of a ``.yaml`` / ``.yml`` file the
    watcher validates the file via :class:`AnalysisCase` and optionally
    invokes a user callback.

    Args:
        watch_dir: Directory to monitor.
        on_change: Called with ``(path, analysis_case_or_none, error_or_none)``
            each time a case file changes.
    """

    def __init__(
        self,
        watch_dir: str | Path,
        on_change: Callable[..., None] | None = None,
    ) -> None:
        _require_watchdog()
        self._watch_dir = Path(watch_dir)
        self._on_change = on_change
        self._observer: object | None = None

    def _handle_event(self, path: Path) -> None:
        if path.suffix not in (".yaml", ".yml"):
            return
        logger.info("Case file changed: %s", path)
        try:
            from feaweld.pipeline.workflow import load_case
            case = load_case(path)
            logger.info("Validated OK: %s (%s)", path.name, case.name)
            if self._on_change:
                self._on_change(path, case, None)
        except Exception as e:
            logger.warning("Validation failed for %s: %s", path, e)
            if self._on_change:
                self._on_change(path, None, e)

    def start(self) -> None:
        """Start watching (non-blocking)."""
        from watchdog.observers import Observer  # type: ignore[import-untyped]
        from watchdog.events import FileSystemEventHandler  # type: ignore[import-untyped]

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event: object) -> None:
                if hasattr(event, "src_path") and not getattr(event, "is_directory", True):
                    watcher._handle_event(Path(event.src_path))  # type: ignore[arg-type]

            def on_created(self, event: object) -> None:
                if hasattr(event, "src_path") and not getattr(event, "is_directory", True):
                    watcher._handle_event(Path(event.src_path))  # type: ignore[arg-type]

        observer = Observer()
        observer.schedule(_Handler(), str(self._watch_dir), recursive=False)
        observer.start()
        self._observer = observer
        logger.info("Watching %s for case file changes", self._watch_dir)

    def stop(self) -> None:
        """Stop watching."""
        if self._observer is not None:
            self._observer.stop()  # type: ignore[union-attr]
            self._observer.join()  # type: ignore[union-attr]
            self._observer = None


class MaterialDBWatcher:
    """Watch the bundled materials directory and invalidate the data cache.

    Args:
        data_dir: Root data directory (containing ``materials/`` subdirectory).
    """

    def __init__(self, data_dir: str | Path) -> None:
        _require_watchdog()
        self._data_dir = Path(data_dir)
        self._observer: object | None = None

    def start(self) -> None:
        """Start watching (non-blocking)."""
        from watchdog.observers import Observer  # type: ignore[import-untyped]
        from watchdog.events import FileSystemEventHandler  # type: ignore[import-untyped]

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event: object) -> None:
                if hasattr(event, "src_path"):
                    p = Path(event.src_path)  # type: ignore[arg-type]
                    if p.suffix in (".yaml", ".yml", ".json"):
                        logger.info("Material data changed: %s — invalidating cache", p.name)
                        try:
                            from feaweld.data.cache import get_cache
                            cache = get_cache()
                            cache.clear()
                        except Exception:
                            pass

        observer = Observer()
        observer.schedule(_Handler(), str(self._data_dir), recursive=True)
        observer.start()
        self._observer = observer
        logger.info("Watching %s for material data changes", self._data_dir)

    def stop(self) -> None:
        """Stop watching."""
        if self._observer is not None:
            self._observer.stop()  # type: ignore[union-attr]
            self._observer.join()  # type: ignore[union-attr]
            self._observer = None
