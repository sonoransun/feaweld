"""Memory-mapped array support for large FEA result fields.

When stress/displacement arrays exceed a configurable threshold, they
are transparently backed by on-disk files via ``numpy.memmap``.  This
lets feaweld handle models that would otherwise exhaust RAM.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.logging import get_logger

logger = get_logger(__name__)

# Default threshold: arrays larger than this (in MB) get memory-mapped.
_DEFAULT_THRESHOLD_MB = float(os.environ.get("FEAWELD_MEMMAP_THRESHOLD_MB", "100"))


def auto_memmap(
    array: NDArray[Any],
    threshold_mb: float | None = None,
    tmpdir: str | Path | None = None,
) -> NDArray[Any]:
    """Return a memory-mapped copy of *array* if it exceeds *threshold_mb*.

    If the array is small enough, it is returned unchanged.  Large arrays
    are written to a temporary file and a ``numpy.memmap`` view is
    returned instead.

    Args:
        array: Source array.
        threshold_mb: Size threshold in megabytes.  Defaults to the
            ``FEAWELD_MEMMAP_THRESHOLD_MB`` environment variable (100 MB).
        tmpdir: Directory for temporary memmap files.  Defaults to
            ``FEAWELD_TMPDIR`` or the system default.

    Returns:
        The original array or a read-write memmap view.
    """
    if threshold_mb is None:
        threshold_mb = _DEFAULT_THRESHOLD_MB

    size_mb = array.nbytes / (1024 * 1024)
    if size_mb < threshold_mb:
        return array

    target_dir = tmpdir or os.environ.get("FEAWELD_TMPDIR") or tempfile.gettempdir()
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    fd, path = tempfile.mkstemp(suffix=".mmap", dir=str(target_dir))
    os.close(fd)

    logger.info(
        "Memory-mapping %.1f MB array (%s) to %s",
        size_mb,
        array.shape,
        path,
    )

    mm = np.memmap(path, dtype=array.dtype, mode="w+", shape=array.shape)
    mm[:] = array
    mm.flush()
    return mm


class MemmapArrayManager:
    """Context manager that tracks and cleans up temporary memmap files.

    Usage::

        with MemmapArrayManager() as mgr:
            big = mgr.wrap(some_large_array)
            # work with big ...
        # files cleaned up on exit
    """

    def __init__(
        self,
        threshold_mb: float | None = None,
        tmpdir: str | Path | None = None,
    ) -> None:
        self._threshold_mb = threshold_mb
        self._tmpdir = tmpdir
        self._paths: list[Path] = []

    def wrap(self, array: NDArray[Any]) -> NDArray[Any]:
        """Conditionally memory-map *array*; track the file for cleanup."""
        result = auto_memmap(array, self._threshold_mb, self._tmpdir)
        if isinstance(result, np.memmap) and hasattr(result, "filename"):
            self._paths.append(Path(result.filename))
        return result

    def cleanup(self) -> None:
        """Delete all managed memmap files."""
        for p in self._paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        self._paths.clear()

    def __enter__(self) -> MemmapArrayManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()
