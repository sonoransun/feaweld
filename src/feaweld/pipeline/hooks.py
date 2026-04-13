"""Pipeline hooks for extensibility and observability.

Provides :class:`PipelineHooks` for registering callbacks before/after
each pipeline stage and on errors.  Also ships two built-in hook
factories:

* :func:`timing_hook` — logs wall-clock time per stage.
* :func:`memory_hook` — logs RSS memory usage after each stage.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from feaweld.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core hook dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineHooks:
    """Callback registry wired into :meth:`PipelineDAG.run`.

    Attributes
    ----------
    pre_stage:
        ``{stage_name: [callback, ...]}`` — each callback receives
        ``(stage_name: str, context: PipelineContext)``.
    post_stage:
        Same signature, invoked after the stage completes successfully.
    on_error:
        ``[callback, ...]`` — each callback receives
        ``(stage_name: str, exception: Exception)``.
    """

    pre_stage: dict[str, list[Callable]] = field(default_factory=dict)
    post_stage: dict[str, list[Callable]] = field(default_factory=dict)
    on_error: list[Callable[[str, Exception], None]] = field(
        default_factory=list,
    )

    # -- registration helpers -----------------------------------------------

    def register_pre(self, stage: str, callback: Callable) -> None:
        """Register a callback to run **before** *stage*."""
        self.pre_stage.setdefault(stage, []).append(callback)

    def register_post(self, stage: str, callback: Callable) -> None:
        """Register a callback to run **after** *stage* (on success)."""
        self.post_stage.setdefault(stage, []).append(callback)

    def register_error(
        self, callback: Callable[[str, Exception], None],
    ) -> None:
        """Register a callback invoked when **any** stage raises."""
        self.on_error.append(callback)

    def register_pre_all(self, callback: Callable) -> None:
        """Register a callback to run before **every** stage.

        Internally stores the callback under the special key ``"*"``.
        The DAG executor checks for wildcards when dispatching hooks.

        Note: This uses the wildcard convention; callers iterating
        ``pre_stage`` directly should handle ``"*"`` specially.
        """
        self.pre_stage.setdefault("*", []).append(callback)

    def register_post_all(self, callback: Callable) -> None:
        """Register a callback to run after **every** stage."""
        self.post_stage.setdefault("*", []).append(callback)

    # -- merging ------------------------------------------------------------

    def merge(self, other: PipelineHooks) -> PipelineHooks:
        """Return a **new** :class:`PipelineHooks` combining *self* and
        *other*.  Callbacks from *other* are appended after those from
        *self*."""
        merged = PipelineHooks()
        for mapping, target in [
            (self.pre_stage, merged.pre_stage),
            (other.pre_stage, merged.pre_stage),
        ]:
            for key, cbs in mapping.items():
                target.setdefault(key, []).extend(cbs)
        for mapping, target in [
            (self.post_stage, merged.post_stage),
            (other.post_stage, merged.post_stage),
        ]:
            for key, cbs in mapping.items():
                target.setdefault(key, []).extend(cbs)
        merged.on_error = list(self.on_error) + list(other.on_error)
        return merged


# ---------------------------------------------------------------------------
# Built-in hook factories
# ---------------------------------------------------------------------------


def timing_hook() -> PipelineHooks:
    """Return hooks that log elapsed wall-clock time for every stage.

    The pre-hook records the start time on a shared dict, the post-hook
    computes the delta and logs it at INFO level.  Results are also
    stored in a ``timings`` dict attached to the hooks instance as
    ``hooks._timings`` for programmatic access.
    """
    start_times: dict[str, float] = {}
    timings: dict[str, float] = {}

    def _pre(stage_name: str, _context: Any) -> None:
        start_times[stage_name] = time.perf_counter()

    def _post(stage_name: str, _context: Any) -> None:
        t0 = start_times.pop(stage_name, None)
        if t0 is not None:
            elapsed = time.perf_counter() - t0
            timings[stage_name] = elapsed
            logger.info("[timing] Stage %r: %.3fs", stage_name, elapsed)

    hooks = PipelineHooks()
    hooks.register_pre_all(_pre)
    hooks.register_post_all(_post)
    # Stash for programmatic inspection.
    hooks._timings = timings  # type: ignore[attr-defined]
    return hooks


def memory_hook() -> PipelineHooks:
    """Return hooks that log RSS memory usage after each stage.

    On Linux, reads ``/proc/self/status`` to obtain ``VmRSS``.  On other
    platforms, falls back to :mod:`resource` (``ru_maxrss``) or logs a
    debug message noting that memory tracking is unavailable.

    Results are stored in ``hooks._memory`` as ``{stage_name: rss_mb}``.
    """
    memory: dict[str, float] = {}

    def _get_rss_mb() -> float | None:
        """Return current RSS in MiB, or ``None`` if unavailable."""
        # Prefer /proc/self/status (accurate, no extra imports).
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        # Format: "VmRSS:    123456 kB"
                        parts = line.split()
                        return float(parts[1]) / 1024.0
        except (OSError, IndexError, ValueError):
            pass

        # Fallback: resource module (macOS reports bytes, Linux kB).
        try:
            import resource
            import sys

            ru = resource.getrusage(resource.RUSAGE_SELF)
            rss_kb = ru.ru_maxrss
            if sys.platform == "darwin":
                # macOS reports bytes.
                return rss_kb / (1024.0 * 1024.0)
            return rss_kb / 1024.0
        except Exception:
            pass

        return None

    def _post(stage_name: str, _context: Any) -> None:
        rss = _get_rss_mb()
        if rss is not None:
            memory[stage_name] = rss
            logger.info("[memory] After stage %r: %.1f MiB RSS", stage_name, rss)
        else:
            logger.debug(
                "[memory] RSS tracking unavailable on this platform",
            )

    hooks = PipelineHooks()
    hooks.register_post_all(_post)
    hooks._memory = memory  # type: ignore[attr-defined]
    return hooks
