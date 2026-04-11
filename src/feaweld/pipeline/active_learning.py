"""Acquisition-driven active learning loop over :class:`Study` (Track B2).

Wraps the existing :mod:`feaweld.pipeline.study` parameter-apply machinery in a
small Bayesian-optimization-style sampler that uses a
:class:`~feaweld.ml.bayesian_surrogate.BayesianFatigueSurrogate` to choose which
parameter vectors to evaluate next.

The module imports cleanly without Flax / JAX — the heavy deps are only
imported when :meth:`ActiveLearningLoop.run` is actually called.  That keeps
test collection fast and lets downstream code type-hint against
``ActiveLearningLoop`` in environments that do not have Flax installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from feaweld.pipeline.study import _set_nested_attr
from feaweld.pipeline.workflow import AnalysisCase, WorkflowResult, run_analysis


AcquisitionName = Literal["max_variance", "expected_improvement", "random"]


# ---------------------------------------------------------------------------
# Configuration and results dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ActiveLearningConfig:
    """Hyper-parameters controlling the active learning loop."""

    n_initial: int = 10
    n_iterations: int = 20
    n_candidates: int = 500
    acquisition: AcquisitionName = "max_variance"
    target_metric: str = "max_von_mises"
    seed: int = 0
    retrain_every: int = 1
    xi: float = 0.01  # exploration parameter for expected_improvement
    feature_extractor: Callable[[AnalysisCase], NDArray[np.float64]] | None = None
    metric_extractor: Callable[[WorkflowResult], float] | None = None


@dataclass
class ActiveLearningResults:
    """Container for everything produced by an active learning run."""

    cases_evaluated: list[dict[str, float]] = field(default_factory=list)
    metrics: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    mean_predicted: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    std_predicted: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    surrogate: Any | None = None


# ---------------------------------------------------------------------------
# Default extractors
# ---------------------------------------------------------------------------


def _default_metric_extractor(result: WorkflowResult) -> float:
    """Pull ``max_von_mises`` out of a :class:`WorkflowResult`.

    Falls back to the first ``"life"`` key found in ``fatigue_results``.
    Returns ``np.nan`` if nothing usable is present.
    """

    fea = result.fea_results
    if fea is not None and fea.stress is not None:
        try:
            vm = fea.stress.von_mises
            if vm.size:
                return float(np.max(vm))
        except Exception:  # pragma: no cover - defensive
            pass

    for entry in (result.fatigue_results or {}).values():
        if isinstance(entry, dict) and "life" in entry:
            life = entry["life"]
            if np.isfinite(life):
                return float(np.log10(max(life, 1.0)))

    return float("nan")


def _make_feature_extractor(
    parameter_keys: tuple[str, ...],
) -> Callable[[AnalysisCase], NDArray[np.float64]]:
    """Build a default feature extractor that reads ``parameter_keys``.

    The returned function walks the dot-paths on a case and concatenates the
    values into a 1-D float array in the exact order given by
    ``parameter_keys`` (already sorted by :class:`ActiveLearningLoop`).
    """

    def _extract(case: AnalysisCase) -> NDArray[np.float64]:
        vec = np.empty(len(parameter_keys), dtype=np.float64)
        for i, path in enumerate(parameter_keys):
            obj: Any = case
            for part in path.split("."):
                obj = getattr(obj, part)
            vec[i] = float(obj)
        return vec

    return _extract


# ---------------------------------------------------------------------------
# Active learning loop
# ---------------------------------------------------------------------------


class ActiveLearningLoop:
    """Acquisition-driven sampler over a parametrized :class:`AnalysisCase`.

    Parameters
    ----------
    base_case:
        A fully-specified :class:`AnalysisCase` that will be deep-copied and
        mutated via dot-path setattr for each candidate.
    parameter_ranges:
        Mapping from dot-path to ``(low, high)`` uniform sampling ranges, e.g.
        ``{"load.axial_force": (0.0, 10_000.0)}``.
    config:
        Loop hyper-parameters.  A default instance is created if omitted.
    runner:
        Callable mapping :class:`AnalysisCase` to :class:`WorkflowResult`.
        Defaults to :func:`feaweld.pipeline.workflow.run_analysis`; tests can
        inject a cheap deterministic fake.
    """

    def __init__(
        self,
        base_case: AnalysisCase,
        parameter_ranges: dict[str, tuple[float, float]],
        config: ActiveLearningConfig | None = None,
        runner: Callable[[AnalysisCase], WorkflowResult] | None = None,
    ) -> None:
        if not parameter_ranges:
            raise ValueError("parameter_ranges must not be empty")

        # Sort keys so feature vectors are deterministic regardless of caller
        # insertion order.
        self._param_keys: tuple[str, ...] = tuple(sorted(parameter_ranges.keys()))
        self._low = np.array(
            [float(parameter_ranges[k][0]) for k in self._param_keys], dtype=np.float64
        )
        self._high = np.array(
            [float(parameter_ranges[k][1]) for k in self._param_keys], dtype=np.float64
        )
        if np.any(self._high <= self._low):
            raise ValueError(
                "Each parameter range must satisfy low < high; got "
                f"{dict(zip(self._param_keys, zip(self._low, self._high)))}"
            )

        self.base_case = base_case
        self.config = config or ActiveLearningConfig()
        self.runner = runner or run_analysis

        self._feature_extractor = (
            self.config.feature_extractor
            or _make_feature_extractor(self._param_keys)
        )
        self._metric_extractor = (
            self.config.metric_extractor or _default_metric_extractor
        )

    # ---------------------------------------------------------------- helpers

    @property
    def parameter_keys(self) -> tuple[str, ...]:
        return self._param_keys

    def _sample_uniform(
        self, rng: np.random.Generator, n: int
    ) -> NDArray[np.float64]:
        return rng.uniform(
            low=self._low,
            high=self._high,
            size=(n, self._low.shape[0]),
        )

    def _vector_to_case(self, vec: NDArray[np.float64]) -> AnalysisCase:
        case = self.base_case.model_copy(deep=True)
        for path, value in zip(self._param_keys, vec):
            case = _set_nested_attr(case, path, float(value))
        return case

    def _vector_to_dict(self, vec: NDArray[np.float64]) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self._param_keys, vec)}

    def _evaluate(self, vec: NDArray[np.float64]) -> float:
        case = self._vector_to_case(vec)
        result = self.runner(case)
        return float(self._metric_extractor(result))

    # --------------------------------------------------------- acquisition

    def _score_candidates(
        self,
        candidates: NDArray[np.float64],
        surrogate: Any,
        y_best: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Return a per-candidate acquisition score (larger == better)."""

        acq = self.config.acquisition

        if acq == "random":
            return rng.uniform(size=candidates.shape[0])

        mean, epistemic_std, aleatoric_std = surrogate.predict(candidates)
        mean = np.asarray(mean, dtype=np.float64)
        epistemic_std = np.asarray(epistemic_std, dtype=np.float64)
        aleatoric_std = np.asarray(aleatoric_std, dtype=np.float64)

        if acq == "max_variance":
            # Use epistemic disagreement only — aleatoric noise is (by
            # definition) irreducible and shouldn't drive sample selection.
            return epistemic_std

        if acq == "expected_improvement":
            # Higher mean is assumed "better" (matches max-von-Mises targeting
            # worst-case or max-life targeting longest-lived).  Callers who
            # want minimization can negate the metric in their extractor.
            from scipy.stats import norm

            xi = float(self.config.xi)
            # EI uses epistemic sigma: reducible uncertainty drives exploration
            sigma = np.maximum(epistemic_std, 1e-12)
            improvement = mean - y_best - xi
            z = improvement / sigma
            ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.where(epistemic_std > 0, ei, 0.0)
            return ei

        raise ValueError(f"Unknown acquisition function: {acq!r}")

    # ------------------------------------------------------------------ run

    def run(self) -> ActiveLearningResults:
        """Execute the initial Monte-Carlo stage followed by the AL loop.

        Raises
        ------
        ImportError
            If Flax / JAX are not importable — the surrogate cannot be
            constructed without them.
        """

        from feaweld.ml.bayesian_surrogate import (
            BayesianFatigueSurrogate,
            EnsembleConfig,
        )

        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        # --- Stage 1: draw the initial batch ---------------------------------
        X_list: list[NDArray[np.float64]] = []
        y_list: list[float] = []
        cases_log: list[dict[str, float]] = []
        mean_pred_log: list[float] = []
        std_pred_log: list[float] = []

        initial_vectors = self._sample_uniform(rng, cfg.n_initial)
        for vec in initial_vectors:
            X_list.append(vec.copy())
            y_list.append(self._evaluate(vec))
            cases_log.append(self._vector_to_dict(vec))
            mean_pred_log.append(float("nan"))
            std_pred_log.append(float("nan"))

        X_train = np.asarray(X_list, dtype=np.float64)
        y_train = np.asarray(y_list, dtype=np.float64)

        # --- Stage 2: fit initial surrogate ---------------------------------
        surrogate = BayesianFatigueSurrogate(
            config=EnsembleConfig(
                n_members=5,
                hidden_sizes=(32, 32),
                n_epochs=200,
                learning_rate=5e-3,
                weight_decay=1e-4,
                seed=cfg.seed,
            )
        )
        self._fit_safely(surrogate, X_train, y_train)

        # --- Stage 3: active-learning iterations -----------------------------
        for it in range(cfg.n_iterations):
            candidates = self._sample_uniform(rng, cfg.n_candidates)
            y_best = (
                float(np.nanmax(y_train)) if y_train.size else 0.0
            )

            scores = self._score_candidates(candidates, surrogate, y_best, rng)
            idx = int(np.argmax(scores))
            chosen = candidates[idx]

            # Predicted diagnostics for the chosen candidate
            try:
                pred_mean, pred_std = surrogate.predict_total_std(chosen[None, :])
                pred_mean_val = float(pred_mean[0])
                pred_std_val = float(pred_std[0])
            except Exception:  # pragma: no cover - surrogate shouldn't fail here
                pred_mean_val = float("nan")
                pred_std_val = float("nan")

            y_obs = self._evaluate(chosen)

            X_train = np.vstack([X_train, chosen[None, :]])
            y_train = np.append(y_train, y_obs)
            cases_log.append(self._vector_to_dict(chosen))
            mean_pred_log.append(pred_mean_val)
            std_pred_log.append(pred_std_val)

            if (it + 1) % max(cfg.retrain_every, 1) == 0:
                self._fit_safely(surrogate, X_train, y_train)

        return ActiveLearningResults(
            cases_evaluated=cases_log,
            metrics=np.asarray(y_train, dtype=np.float64),
            mean_predicted=np.asarray(mean_pred_log, dtype=np.float64),
            std_predicted=np.asarray(std_pred_log, dtype=np.float64),
            surrogate=surrogate,
        )

    # ----------------------------------------------------------- internals

    @staticmethod
    def _fit_safely(
        surrogate: Any,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        """Fit the surrogate, tolerating degenerate (e.g. all-NaN) targets."""

        mask = np.isfinite(y)
        if mask.sum() < 2:
            # Not enough data yet — leave surrogate untrained; acquisition
            # will still work for "random" mode.
            return
        surrogate.fit(X[mask], y[mask])
