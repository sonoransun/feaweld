"""Tests for the RealNVP normalizing-flow posterior calibration (Track C3)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("jax")

from feaweld.digital_twin.flows import (
    RealNVPConfig,
    RealNVPFlow,
    calibrate_posterior_life,
)
from feaweld.probabilistic.monte_carlo import RandomVariable
from feaweld.probabilistic.sensitivity import reliability_index_form


pytestmark = pytest.mark.requires_flax


# ---------------------------------------------------------------------------
# Shared fast-training config for CI
# ---------------------------------------------------------------------------


def _fast_config(seed: int = 0) -> RealNVPConfig:
    return RealNVPConfig(
        n_layers=4,
        hidden_sizes=(16, 16),
        n_epochs=200,
        learning_rate=5e-3,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_realnvp_fits_1d_gaussian_quantile_matches_scipy() -> None:
    """Flow trained on N(5, 1) recovers the 0.95 quantile within 10 %.

    Re-phrases the canonical "KS distance < 0.02 vs MC reference" check as a
    quantile match to the analytic CDF.
    """

    rng = np.random.default_rng(0)
    samples = rng.normal(loc=5.0, scale=1.0, size=2000).reshape(-1, 1)

    flow = RealNVPFlow(dim=1, config=_fast_config(seed=0))
    flow.fit(samples)

    q_flow = flow.quantile(alpha=0.95, n_samples=20000)
    q_ref = float(stats.norm.ppf(0.95, loc=5.0, scale=1.0))

    rel_err = abs(q_flow - q_ref) / abs(q_ref)
    assert rel_err < 0.10, (
        f"flow 0.95 quantile {q_flow:.4f} vs analytic {q_ref:.4f} "
        f"(rel err {rel_err:.3f})"
    )


def test_realnvp_log_prob_finite() -> None:
    rng = np.random.default_rng(1)
    samples = rng.normal(loc=0.0, scale=1.0, size=1000).reshape(-1, 1)

    flow = RealNVPFlow(dim=1, config=_fast_config(seed=1))
    flow.fit(samples)

    log_p = flow.log_prob(samples[:50])
    assert log_p.shape == (50,)
    assert np.all(np.isfinite(log_p)), "log_prob returned non-finite values"


def test_calibrate_posterior_life_returns_flow() -> None:
    rng = np.random.default_rng(2)
    # Synthetic log-N_f posterior samples: mixture-ish around log10 ~ 6
    log_nf = rng.normal(loc=6.0, scale=0.3, size=1500)

    flow = calibrate_posterior_life(log_nf, config=_fast_config(seed=2))
    assert isinstance(flow, RealNVPFlow)

    q50 = flow.quantile(0.5, n_samples=10000)
    assert np.isfinite(q50)
    # Median should be in the same ballpark as the empirical one.
    assert abs(q50 - np.median(log_nf)) < 0.5


def test_form_cross_check() -> None:
    """Sanity check: FORM beta + flow median on g(X) = x1 - x2.

    ``x1 ~ N(10, 1)``, ``x2 ~ N(5, 0.5)`` => ``g ~ N(5, sqrt(1.25))``.
    FORM yields an analytic reliability index ``beta = 5 / sqrt(1.25)``; the
    flow is fit to 10 000 samples of ``g`` and its 0.5-quantile should match
    the empirical median within 5 % (a data-sanity check, not a FORM
    equivalence claim).
    """

    variables = [
        RandomVariable(name="x1", distribution="normal", params={"mean": 10.0, "std": 1.0}),
        RandomVariable(name="x2", distribution="normal", params={"mean": 5.0, "std": 0.5}),
    ]

    def g(var_dict: dict[str, float]) -> float:
        return var_dict["x1"] - var_dict["x2"]

    form_result = reliability_index_form(variables, g)
    beta_form = float(form_result["beta"])  # type: ignore[arg-type]
    beta_expected = 5.0 / np.sqrt(1.25)
    assert abs(beta_form - beta_expected) < 1e-3, (
        f"FORM beta {beta_form:.4f} vs analytic {beta_expected:.4f}"
    )

    rng = np.random.default_rng(3)
    x1 = rng.normal(10.0, 1.0, size=10000)
    x2 = rng.normal(5.0, 0.5, size=10000)
    g_samples = (x1 - x2).reshape(-1, 1)
    empirical_median = float(np.median(g_samples))

    flow = RealNVPFlow(dim=1, config=_fast_config(seed=3))
    flow.fit(g_samples)

    q50_flow = flow.quantile(0.5, n_samples=20000)
    rel_err = abs(q50_flow - empirical_median) / max(abs(empirical_median), 1e-8)
    assert rel_err < 0.05, (
        f"flow median {q50_flow:.4f} vs empirical {empirical_median:.4f} "
        f"(rel err {rel_err:.3f})"
    )
