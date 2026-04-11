"""Tests for the Track B2 active learning loop.

All tests inject a fake deterministic runner so no real FE solves happen —
this keeps the suite fast enough to run on every push.  Heavy deps (Flax /
JAX / Optax) are guarded via ``importorskip``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("jax")

from feaweld.core.types import ElementType, FEAResults, FEMesh, StressField
from feaweld.pipeline.active_learning import (
    ActiveLearningConfig,
    ActiveLearningLoop,
    ActiveLearningResults,
)
from feaweld.pipeline.workflow import AnalysisCase, WorkflowResult


pytestmark = pytest.mark.requires_flax


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


PARAMETER_RANGES = {
    "load.axial_force": (0.0, 10_000.0),
    "load.bending_moment": (-5_000.0, 5_000.0),
}


def _placeholder_mesh() -> FEMesh:
    """Tiny 1-element TRI3 mesh, enough to satisfy FEAResults."""

    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2]], dtype=np.int64)
    return FEMesh(nodes=nodes, elements=elements, element_type=ElementType.TRI3)


def _analytic_sigma(F: float, M: float) -> float:
    """Deterministic ground-truth QoI used by the fake runner."""

    return (F + abs(M)) / 100.0


def _make_fake_runner():
    """Return a deterministic runner that produces FEAResults matching
    ``_analytic_sigma``.

    Each node's von Mises equals exactly the analytic value, so
    ``FEAResults.stress.von_mises.max()`` reproduces the ground truth.
    """

    mesh = _placeholder_mesh()

    def fake_runner(case: AnalysisCase) -> WorkflowResult:
        F = float(case.load.axial_force)
        M = float(case.load.bending_moment)
        sigma = _analytic_sigma(F, M)

        stress_vals = np.zeros((mesh.n_nodes, 6), dtype=np.float64)
        # Pure uniaxial tension so von Mises == sigma exactly.
        stress_vals[:, 0] = sigma

        fea = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )
        return WorkflowResult(case=case, mesh=mesh, fea_results=fea)

    return fake_runner


def _test_grid(n_per_dim: int = 6) -> np.ndarray:
    """A fixed (n, 2) held-out test grid spanning the parameter box."""

    F = np.linspace(500.0, 9_500.0, n_per_dim)
    M = np.linspace(-4_500.0, 4_500.0, n_per_dim)
    Fg, Mg = np.meshgrid(F, M, indexing="ij")
    return np.column_stack([Fg.ravel(), Mg.ravel()])


def _test_truth(grid: np.ndarray) -> np.ndarray:
    return np.array([_analytic_sigma(F, M) for F, M in grid])


# ---------------------------------------------------------------------------
# Configuration / record-keeping tests
# ---------------------------------------------------------------------------


def test_active_learning_config_defaults():
    cfg = ActiveLearningConfig()
    assert cfg.n_initial == 10
    assert cfg.n_iterations == 20
    assert cfg.n_candidates == 500
    assert cfg.acquisition == "max_variance"
    assert cfg.target_metric == "max_von_mises"
    assert cfg.seed == 0
    assert cfg.retrain_every == 1
    assert cfg.feature_extractor is None
    assert cfg.metric_extractor is None


def test_active_learning_records_all_evaluations():
    base = AnalysisCase(name="test")
    cfg = ActiveLearningConfig(
        n_initial=5,
        n_iterations=4,
        n_candidates=50,
        seed=123,
        retrain_every=1,
    )
    loop = ActiveLearningLoop(
        base_case=base,
        parameter_ranges=PARAMETER_RANGES,
        config=cfg,
        runner=_make_fake_runner(),
    )
    results = loop.run()

    assert isinstance(results, ActiveLearningResults)
    assert len(results.cases_evaluated) == cfg.n_initial + cfg.n_iterations
    assert results.metrics.shape == (cfg.n_initial + cfg.n_iterations,)
    assert np.all(np.isfinite(results.metrics))
    assert results.surrogate is not None

    # Every logged case dict has the expected keys.
    for entry in results.cases_evaluated:
        assert set(entry.keys()) == set(PARAMETER_RANGES.keys())


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_active_learning_deterministic_seed():
    base = AnalysisCase(name="det")
    cfg = ActiveLearningConfig(
        n_initial=4,
        n_iterations=3,
        n_candidates=40,
        seed=7,
        acquisition="max_variance",
    )

    loop_a = ActiveLearningLoop(
        base_case=base,
        parameter_ranges=PARAMETER_RANGES,
        config=cfg,
        runner=_make_fake_runner(),
    )
    loop_b = ActiveLearningLoop(
        base_case=base,
        parameter_ranges=PARAMETER_RANGES,
        config=cfg,
        runner=_make_fake_runner(),
    )

    res_a = loop_a.run()
    res_b = loop_b.run()

    assert len(res_a.cases_evaluated) == len(res_b.cases_evaluated)

    for a, b in zip(res_a.cases_evaluated, res_b.cases_evaluated):
        assert set(a.keys()) == set(b.keys())
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=0, abs=1e-12)

    np.testing.assert_allclose(res_a.metrics, res_b.metrics, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Max-variance beats random on held-out RMSE
# ---------------------------------------------------------------------------


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def test_active_learning_reduces_rmse_vs_random():
    base = AnalysisCase(name="rmse")

    cfg_mv = ActiveLearningConfig(
        n_initial=8,
        n_iterations=12,
        n_candidates=200,
        seed=42,
        acquisition="max_variance",
        retrain_every=1,
    )
    cfg_rand = ActiveLearningConfig(
        n_initial=8,
        n_iterations=12,
        n_candidates=200,
        seed=42,
        acquisition="random",
        retrain_every=1,
    )

    loop_mv = ActiveLearningLoop(
        base_case=base,
        parameter_ranges=PARAMETER_RANGES,
        config=cfg_mv,
        runner=_make_fake_runner(),
    )
    loop_rand = ActiveLearningLoop(
        base_case=base,
        parameter_ranges=PARAMETER_RANGES,
        config=cfg_rand,
        runner=_make_fake_runner(),
    )

    res_mv = loop_mv.run()
    res_rand = loop_rand.run()

    grid = _test_grid(6)
    truth = _test_truth(grid)

    mv_mean, _ = res_mv.surrogate.predict_total_std(grid)
    rand_mean, _ = res_rand.surrogate.predict_total_std(grid)

    rmse_mv = _rmse(np.asarray(mv_mean), truth)
    rmse_rand = _rmse(np.asarray(rand_mean), truth)

    # Max-variance should be no worse than random (with a small tolerance
    # for ensemble noise on a 20-sample budget).
    assert rmse_mv <= rmse_rand * 1.10, (
        f"max_variance RMSE={rmse_mv:.3f} is worse than random "
        f"RMSE={rmse_rand:.3f}"
    )
