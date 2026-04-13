"""DAG-based pipeline executor for feaweld analysis workflows.

Provides a directed acyclic graph abstraction over the linear analysis
pipeline so that independent stages (e.g. thermal pre-processing and
defect population) can execute concurrently, while still respecting
data-flow dependencies.  The main entry point is :class:`PipelineDAG`,
which accepts :class:`PipelineStage` nodes and runs them through a
shared :class:`PipelineContext`.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from feaweld.core.logging import get_logger

if TYPE_CHECKING:
    from feaweld.pipeline.hooks import PipelineHooks

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------


@dataclass
class PipelineStage:
    """A single stage in the analysis pipeline.

    Attributes
    ----------
    name:
        Unique stage identifier used for dependency resolution.
    callable:
        Function ``(PipelineContext) -> Any``.  If a non-``None`` value is
        returned it is stored on the context under the stage's *name*.
    depends_on:
        Names of stages that must complete before this one.
    io_bound:
        Hint for the executor — ``True`` allows the stage to share a
        thread pool with other I/O-bound work without saturating CPU.
    optional:
        If ``True``, a failure in this stage is logged as a warning and
        does not propagate to dependents as an error.
    """

    name: str
    callable: Callable[[PipelineContext], Any]
    depends_on: list[str] = field(default_factory=list)
    io_bound: bool = False
    optional: bool = False


# ---------------------------------------------------------------------------
# Shared pipeline context
# ---------------------------------------------------------------------------


class PipelineContext:
    """Shared mutable state passed through pipeline stages.

    A thin dict-like wrapper that replaces the ad-hoc local variables in
    :func:`~feaweld.pipeline.workflow.run_analysis`.  Supports attribute-
    style and bracket-style access, iteration, and ``in`` checks.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._data: dict[str, Any] = dict(kwargs)
        self._errors: list[str] = []

    # -- dict-like access ---------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return value for *key*, or *default* if absent."""
        return self._data.get(key, default)

    def keys(self) -> list[str]:
        """Return all stored keys."""
        return list(self._data.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Return all stored (key, value) pairs."""
        return list(self._data.items())

    def update(self, mapping: dict[str, Any]) -> None:
        """Merge *mapping* into the context."""
        self._data.update(mapping)

    # -- error collection ---------------------------------------------------

    @property
    def errors(self) -> list[str]:
        """Accumulated error messages from failed stages."""
        return list(self._errors)

    def add_error(self, message: str) -> None:
        """Record a non-fatal error."""
        self._errors.append(message)

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._data.keys()))
        return f"PipelineContext(keys=[{keys}], errors={len(self._errors)})"


# ---------------------------------------------------------------------------
# DAG executor
# ---------------------------------------------------------------------------


class PipelineDAG:
    """Directed acyclic graph of pipeline stages with concurrent execution.

    Build a DAG by calling :meth:`add` (chainable), then execute via
    :meth:`run`.  Stages within the same *batch* (no mutual dependencies)
    are executed concurrently in a thread pool.

    Example::

        dag = (
            PipelineDAG()
            .add(PipelineStage("materials", load_materials))
            .add(PipelineStage("geometry", build_geometry, depends_on=["materials"]))
            .add(PipelineStage("mesh", generate_mesh, depends_on=["geometry"]))
        )
        ctx = dag.run(PipelineContext(case=my_case))
    """

    def __init__(self) -> None:
        self._stages: dict[str, PipelineStage] = {}

    # -- construction -------------------------------------------------------

    def add(self, stage: PipelineStage) -> PipelineDAG:
        """Register a stage.  Returns *self* for chaining."""
        if stage.name in self._stages:
            raise ValueError(f"Duplicate stage name: {stage.name!r}")
        self._stages[stage.name] = stage
        return self

    def remove(self, name: str) -> PipelineDAG:
        """Remove a previously registered stage.  Returns *self*."""
        self._stages.pop(name, None)
        return self

    @property
    def stage_names(self) -> list[str]:
        """Registered stage names in insertion order."""
        return list(self._stages.keys())

    def __len__(self) -> int:
        return len(self._stages)

    # -- topological sort ---------------------------------------------------

    def _validate_dependencies(self) -> None:
        """Raise :class:`ValueError` if any dependency references an
        unknown stage."""
        known = set(self._stages.keys())
        for stage in self._stages.values():
            missing = set(stage.depends_on) - known
            if missing:
                raise ValueError(
                    f"Stage {stage.name!r} depends on unknown stage(s): "
                    f"{sorted(missing)}"
                )

    def execution_order(self) -> list[list[str]]:
        """Topological sort into concurrent batches (Kahn's algorithm).

        Each inner list contains stage names that can execute concurrently.
        The outer list is ordered: batch *i* must complete before batch
        *i+1* starts.

        Raises
        ------
        ValueError
            If the graph contains a cycle or references unknown stages.
        """
        self._validate_dependencies()

        in_degree: dict[str, int] = dict.fromkeys(self._stages, 0)
        # Build adjacency: for each stage, count how many deps it has.
        for stage in self._stages.values():
            for dep in stage.depends_on:
                if dep in in_degree:
                    in_degree[stage.name] += 1

        batches: list[list[str]] = []
        remaining = set(self._stages.keys())

        while remaining:
            ready = sorted(n for n in remaining if in_degree[n] == 0)
            if not ready:
                raise ValueError(
                    "Cycle detected in pipeline DAG among stages: "
                    f"{sorted(remaining)}"
                )
            batches.append(ready)
            for n in ready:
                remaining.discard(n)
                # Decrease in-degree for dependents.
                for stage in self._stages.values():
                    if n in stage.depends_on:
                        in_degree[stage.name] -= 1

        return batches

    # -- execution ----------------------------------------------------------

    def run(
        self,
        context: PipelineContext,
        *,
        max_workers: int = 4,
        hooks: PipelineHooks | None = None,
        stop_on_error: bool = False,
    ) -> PipelineContext:
        """Execute all stages respecting dependency order.

        Parameters
        ----------
        context:
            Shared mutable state for the pipeline.
        max_workers:
            Maximum thread-pool size for concurrent batches.
        hooks:
            Optional :class:`~feaweld.pipeline.hooks.PipelineHooks` for
            pre/post/error callbacks.
        stop_on_error:
            If ``True``, abort the entire pipeline on the first stage
            failure.  If ``False`` (default), failed stages are logged
            and dependents are skipped.

        Returns
        -------
        PipelineContext
            The same *context* instance, enriched with stage outputs.
        """
        batches = self.execution_order()
        failed_stages: set[str] = set()

        for batch in batches:
            # Filter out stages whose dependencies failed.
            runnable = [
                name
                for name in batch
                if not (set(self._stages[name].depends_on) & failed_stages)
            ]

            # Mark skipped stages.
            for name in batch:
                if name not in runnable:
                    msg = (
                        f"Stage {name!r} skipped: dependency failed "
                        f"({set(self._stages[name].depends_on) & failed_stages})"
                    )
                    logger.warning(msg)
                    context.add_error(msg)
                    failed_stages.add(name)

            if not runnable:
                continue

            if len(runnable) == 1:
                # Single stage — execute directly, no thread overhead.
                ok = _execute_stage(
                    self._stages[runnable[0]], context, hooks,
                )
                if not ok:
                    failed_stages.add(runnable[0])
                    if stop_on_error:
                        logger.error("Aborting pipeline after stage %r", runnable[0])
                        break
            else:
                # Concurrent batch.
                pool_size = min(max_workers, len(runnable))
                with ThreadPoolExecutor(max_workers=pool_size) as executor:
                    futures = {
                        executor.submit(
                            _execute_stage,
                            self._stages[name],
                            context,
                            hooks,
                        ): name
                        for name in runnable
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            ok = future.result()
                        except Exception:
                            # Should not happen — _execute_stage catches
                            # internally — but guard defensively.
                            ok = False
                        if not ok:
                            failed_stages.add(name)
                            if stop_on_error:
                                logger.error(
                                    "Aborting pipeline after stage %r", name,
                                )
                                # Cancel remaining futures.
                                for f in futures:
                                    f.cancel()
                                break

                if stop_on_error and failed_stages & set(runnable):
                    break

        return context


# ---------------------------------------------------------------------------
# Stage execution helper
# ---------------------------------------------------------------------------


def _execute_stage(
    stage: PipelineStage,
    context: PipelineContext,
    hooks: PipelineHooks | None,
) -> bool:
    """Execute a single stage with optional hooks.

    Returns ``True`` on success, ``False`` on failure.
    """
    # Pre-stage hooks.
    if hooks is not None:
        for cb in hooks.pre_stage.get(stage.name, []):
            try:
                cb(stage.name, context)
            except Exception:
                logger.debug(
                    "Pre-hook for stage %r raised:\n%s",
                    stage.name,
                    traceback.format_exc(),
                )

    t0 = time.perf_counter()
    logger.info("Executing stage: %s", stage.name)

    try:
        result = stage.callable(context)
        if result is not None:
            context[stage.name] = result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        msg = f"Stage {stage.name!r} failed after {elapsed:.3f}s: {exc}"
        logger.error(msg)
        context.add_error(msg)

        # Error hooks.
        if hooks is not None:
            for cb in hooks.on_error:
                try:
                    cb(stage.name, exc)
                except Exception:
                    logger.debug(
                        "Error hook for stage %r raised:\n%s",
                        stage.name,
                        traceback.format_exc(),
                    )

        if stage.optional:
            logger.warning(
                "Stage %r is optional — continuing pipeline", stage.name,
            )
            return True
        return False

    elapsed = time.perf_counter() - t0
    logger.info("Stage %s completed in %.3fs", stage.name, elapsed)

    # Post-stage hooks.
    if hooks is not None:
        for cb in hooks.post_stage.get(stage.name, []):
            try:
                cb(stage.name, context)
            except Exception:
                logger.debug(
                    "Post-hook for stage %r raised:\n%s",
                    stage.name,
                    traceback.format_exc(),
                )

    return True
