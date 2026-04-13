"""Shared-memory transport for large numpy arrays between study workers.

When running parametric studies with :mod:`multiprocessing`, large
arrays (displacement fields, stress tensors) are normally serialized
via pickle and copied through OS pipes — a significant bottleneck for
meshes with millions of nodes.

:class:`SharedResultStore` uses :mod:`multiprocessing.shared_memory` to
place arrays in a named shared-memory segment.  Only the lightweight
:class:`SharedArrayMeta` descriptor is pickled; the receiving worker
reconstructs the array via a zero-copy view.

If shared memory is unavailable (e.g. some container runtimes restrict
``/dev/shm``), all operations fall back transparently to regular numpy
arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.logging import get_logger

logger = get_logger(__name__)

# Lazy guard — SharedMemory may not be available on all platforms.
_SHM_AVAILABLE: bool
try:
    from multiprocessing.shared_memory import SharedMemory as _SharedMemory

    _SHM_AVAILABLE = True
except ImportError:
    _SHM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metadata descriptor (small, picklable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedArrayMeta:
    """Lightweight descriptor for a shared-memory array.

    This object is safe to pickle and send across process boundaries.
    The receiving process calls :meth:`SharedResultStore.retrieve` with
    this descriptor to get back a numpy array.

    Attributes
    ----------
    shm_name:
        Name of the ``SharedMemory`` block.  Empty string when the
        fallback (embedded copy) is used.
    shape:
        Array shape.
    dtype_str:
        String representation of the numpy dtype (e.g. ``"float64"``).
    fallback_data:
        When shared memory is unavailable, the raw array bytes are
        carried inline.  ``None`` when ``shm_name`` is set.
    """

    shm_name: str
    shape: tuple[int, ...]
    dtype_str: str
    fallback_data: bytes | None = None

    @property
    def nbytes(self) -> int:
        """Total byte size of the array."""
        return int(np.prod(self.shape)) * np.dtype(self.dtype_str).itemsize

    @property
    def is_shared(self) -> bool:
        """``True`` if the data lives in a shared-memory segment."""
        return bool(self.shm_name) and self.fallback_data is None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class SharedResultStore:
    """Transfer large numpy arrays via shared memory instead of pickle.

    Typical lifecycle::

        store = SharedResultStore()
        meta = store.store("displacement", big_array)
        # ... send *meta* to another process via pickle ...
        arr = store.retrieve(meta)
        store.cleanup()

    If shared memory is unavailable, :meth:`store` transparently embeds
    the raw bytes inside :class:`SharedArrayMeta`, so callers never need
    to branch on availability.
    """

    def __init__(self) -> None:
        # Track allocated segments for cleanup.
        self._segments: dict[str, Any] = {}  # name -> SharedMemory

    # -- public API ---------------------------------------------------------

    def store(self, name: str, array: NDArray[Any]) -> SharedArrayMeta:
        """Write *array* to shared memory, returning a picklable
        descriptor.

        Parameters
        ----------
        name:
            Logical name (used only for logging / cleanup tracking).
        array:
            The numpy array to share.

        Returns
        -------
        SharedArrayMeta
            Descriptor that can be pickled and passed to
            :meth:`retrieve`.
        """
        array = np.ascontiguousarray(array)
        shape = tuple(array.shape)
        dtype_str = str(array.dtype)

        if not _SHM_AVAILABLE:
            logger.debug(
                "Shared memory unavailable — storing %r inline (%d bytes)",
                name, array.nbytes,
            )
            return SharedArrayMeta(
                shm_name="",
                shape=shape,
                dtype_str=dtype_str,
                fallback_data=array.tobytes(),
            )

        try:
            shm = _SharedMemory(create=True, size=array.nbytes)
            # Copy data into the shared segment.
            shared_arr = np.ndarray(
                shape, dtype=array.dtype, buffer=shm.buf,
            )
            np.copyto(shared_arr, array)
            self._segments[shm.name] = shm
            logger.debug(
                "Stored %r in shared memory %r (%d bytes)",
                name, shm.name, array.nbytes,
            )
            return SharedArrayMeta(
                shm_name=shm.name,
                shape=shape,
                dtype_str=dtype_str,
            )
        except Exception as exc:
            logger.warning(
                "Shared memory allocation failed for %r: %s.  "
                "Falling back to inline copy.",
                name, exc,
            )
            return SharedArrayMeta(
                shm_name="",
                shape=shape,
                dtype_str=dtype_str,
                fallback_data=array.tobytes(),
            )

    def retrieve(self, meta: SharedArrayMeta) -> NDArray[Any]:
        """Read an array back from shared memory (or inline fallback).

        Parameters
        ----------
        meta:
            Descriptor returned by a prior :meth:`store` call.

        Returns
        -------
        numpy.ndarray
            A **copy** of the shared data (the caller owns the memory).
        """
        dtype = np.dtype(meta.dtype_str)

        if meta.fallback_data is not None:
            return np.frombuffer(meta.fallback_data, dtype=dtype).reshape(
                meta.shape,
            ).copy()

        if not _SHM_AVAILABLE:
            raise RuntimeError(
                "SharedArrayMeta references shared memory, but "
                "multiprocessing.shared_memory is not available on this "
                "platform."
            )

        try:
            shm = _SharedMemory(name=meta.shm_name, create=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Shared memory segment {meta.shm_name!r} not found.  "
                f"It may have been cleaned up already."
            ) from exc

        try:
            shared_arr = np.ndarray(
                meta.shape, dtype=dtype, buffer=shm.buf,
            )
            # Return a copy so the caller does not depend on the shared
            # segment staying alive.
            return shared_arr.copy()
        finally:
            shm.close()

    def cleanup(self) -> None:
        """Release and unlink all shared-memory blocks owned by this
        store.

        Safe to call multiple times.
        """
        for name, shm in list(self._segments.items()):
            try:
                shm.close()
                shm.unlink()
                logger.debug("Cleaned up shared memory segment %r", name)
            except Exception as exc:
                logger.debug(
                    "Could not clean up segment %r: %s", name, exc,
                )
        self._segments.clear()

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> SharedResultStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.cleanup()

    def __del__(self) -> None:
        # Best-effort cleanup on GC.
        try:
            self.cleanup()
        except Exception:
            pass
