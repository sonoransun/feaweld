"""Distributed study execution with Dask or Ray backends.

Wraps the existing :class:`~feaweld.pipeline.study.Study` execution
model so that cases are dispatched to a Dask or Ray cluster instead of
a local :class:`~concurrent.futures.ProcessPoolExecutor`.

Both ``dask.distributed`` and ``ray`` are lazy-imported inside the
methods that need them, keeping ``import feaweld`` cheap when neither
is installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from feaweld.core.logging import get_logger

if TYPE_CHECKING:
    from feaweld.pipeline.workflow import AnalysisCase, WorkflowResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level worker (must be importable for remote serialization)
# ---------------------------------------------------------------------------


def _run_case_remote(case: AnalysisCase) -> WorkflowResult:
    """Top-level function executing a single analysis case.

    Defined at module scope so Dask and Ray can serialize it without
    closure issues.
    """
    from feaweld.pipeline.workflow import run_analysis

    return run_analysis(case)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class DistributedStudy:
    """Wraps Study execution with a Dask or Ray backend.

    Parameters
    ----------
    backend:
        ``"dask"`` or ``"ray"``.
    client_kwargs:
        Extra keyword arguments forwarded to ``dask.distributed.Client``
        or ``ray.init`` (e.g. ``{"address": "tcp://scheduler:8786"}``).

    Example::

        from feaweld.pipeline.distributed import DistributedStudy

        ds = DistributedStudy(backend="dask")
        results = ds.run(cases)  # dict[str, WorkflowResult]
    """

    def __init__(
        self,
        backend: str = "dask",
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        backend = backend.lower()
        if backend not in ("dask", "ray"):
            raise ValueError(
                f"Unknown distributed backend {backend!r}. "
                f"Supported backends: 'dask', 'ray'."
            )
        self._backend = backend
        self._client_kwargs = client_kwargs or {}

    # -- public API ---------------------------------------------------------

    def run(
        self,
        cases: dict[str, AnalysisCase],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, WorkflowResult]:
        """Submit all *cases* to the distributed backend.

        Parameters
        ----------
        cases:
            ``{case_name: AnalysisCase}`` mapping.
        progress_callback:
            Called as ``callback(case_name, n_completed, n_total)``
            whenever a case finishes.

        Returns
        -------
        dict[str, WorkflowResult]
            Mapping of case names to their results.  Failed cases
            produce a :class:`WorkflowResult` with a populated
            ``errors`` list rather than raising.
        """
        if not cases:
            return {}

        if self._backend == "dask":
            return self._run_dask(cases, progress_callback)
        return self._run_ray(cases, progress_callback)

    # -- Dask implementation ------------------------------------------------

    def _run_dask(
        self,
        cases: dict[str, AnalysisCase],
        callback: Callable[[str, int, int], None] | None,
    ) -> dict[str, WorkflowResult]:
        try:
            from dask.distributed import Client, as_completed as dask_as_completed
        except ImportError as exc:
            raise ImportError(
                "Dask distributed is required for the 'dask' backend.  "
                "Install it with:  pip install \"feaweld[distributed]\"  "
                "or:  pip install dask[distributed]"
            ) from exc

        total = len(cases)
        results: dict[str, WorkflowResult] = {}
        completed = 0

        logger.info(
            "Submitting %d cases to Dask cluster (%s)",
            total, self._client_kwargs.get("address", "local"),
        )

        client = Client(**self._client_kwargs)
        try:
            futures = {
                client.submit(
                    _run_case_remote, case, key=f"feaweld-{name}",
                ): name
                for name, case in cases.items()
            }

            for future in dask_as_completed(futures):
                name = futures[future]
                completed += 1
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.error("Dask case %r failed: %s", name, exc)
                    results[name] = _failed_result(cases[name], str(exc))
                if callback is not None:
                    callback(name, completed, total)
        finally:
            client.close()

        logger.info("Dask study complete: %d/%d succeeded", completed, total)
        return results

    # -- Ray implementation -------------------------------------------------

    def _run_ray(
        self,
        cases: dict[str, AnalysisCase],
        callback: Callable[[str, int, int], None] | None,
    ) -> dict[str, WorkflowResult]:
        try:
            import ray
        except ImportError as exc:
            raise ImportError(
                "Ray is required for the 'ray' backend.  "
                "Install it with:  pip install \"feaweld[distributed]\"  "
                "or:  pip install ray"
            ) from exc

        if not ray.is_initialized():
            ray.init(**self._client_kwargs)

        total = len(cases)
        results: dict[str, WorkflowResult] = {}
        completed = 0

        logger.info("Submitting %d cases to Ray cluster", total)

        # Define the remote function inside the method to keep module-
        # level imports lazy.
        remote_fn = ray.remote(_run_case_remote)

        ref_to_name: dict[ray.ObjectRef, str] = {}
        for name, case in cases.items():
            ref = remote_fn.remote(case)
            ref_to_name[ref] = name

        pending = list(ref_to_name.keys())

        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            for ref in done:
                name = ref_to_name[ref]
                completed += 1
                try:
                    results[name] = ray.get(ref)
                except Exception as exc:
                    logger.error("Ray case %r failed: %s", name, exc)
                    results[name] = _failed_result(cases[name], str(exc))
                if callback is not None:
                    callback(name, completed, total)

        logger.info("Ray study complete: %d/%d succeeded", completed, total)
        return results


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _failed_result(case: AnalysisCase, error_msg: str) -> WorkflowResult:
    """Build a :class:`WorkflowResult` representing a failed case."""
    from feaweld.pipeline.workflow import WorkflowResult

    return WorkflowResult(case=case, errors=[f"Distributed execution: {error_msg}"])
