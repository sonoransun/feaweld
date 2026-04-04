"""Tests for probabilistic analysis modules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from feaweld.probabilistic.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResult,
    RandomVariable,
    sample_distribution,
)
from feaweld.probabilistic.distributions import (
    corrosion_distribution,
    geometric_tolerance_distributions,
    material_property_distributions,
)
from feaweld.probabilistic.sensitivity import (
    reliability_index_form,
    sobol_indices,
)


# ---------------------------------------------------------------------------
# sample_distribution helper
# ---------------------------------------------------------------------------


class TestSampleDistribution:
    """Verify direct sampling for each supported distribution."""

    def test_normal_mean_std(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_distribution("normal", {"mean": 100, "std": 10}, 10_000, rng)
        assert samples.shape == (10_000,)
        assert np.mean(samples) == pytest.approx(100, abs=1.0)
        assert np.std(samples, ddof=1) == pytest.approx(10, abs=1.0)

    def test_lognormal_positive(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_distribution("lognormal", {"mean": 0, "std": 0.5}, 5000, rng)
        assert np.all(samples > 0)

    def test_weibull_shape_scale(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_distribution(
            "weibull", {"shape": 2.0, "scale": 1000.0}, 5000, rng
        )
        assert np.all(samples > 0)
        # Weibull mean = scale * Gamma(1 + 1/shape)
        from scipy.special import gamma

        expected_mean = 1000.0 * gamma(1 + 1 / 2.0)
        assert np.mean(samples) == pytest.approx(expected_mean, rel=0.05)

    def test_uniform_bounds(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_distribution("uniform", {"low": 5, "high": 10}, 5000, rng)
        assert np.all(samples >= 5)
        assert np.all(samples <= 10)
        assert np.mean(samples) == pytest.approx(7.5, abs=0.2)

    def test_gumbel(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_distribution("gumbel", {"loc": 0.5, "scale": 0.3}, 5000, rng)
        assert len(samples) == 5000

    def test_unsupported_distribution_raises(self) -> None:
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unsupported distribution"):
            sample_distribution("cauchy", {}, 10, rng)


# ---------------------------------------------------------------------------
# MonteCarloEngine
# ---------------------------------------------------------------------------


class TestMonteCarloEngine:
    """Core MC engine tests."""

    def test_expected_value_x_squared_normal(self) -> None:
        """E[X^2] where X ~ N(0, 1) should be approximately 1.0."""
        var = RandomVariable(name="x", distribution="normal", params={"mean": 0, "std": 1})
        config = MonteCarloConfig(n_samples=5000, method="lhs", seed=123)
        engine = MonteCarloEngine([var], config)

        result = engine.run(lambda d: d["x"] ** 2)

        assert isinstance(result, MonteCarloResult)
        assert result.mean == pytest.approx(1.0, abs=0.1)

    def test_lhs_uniform_coverage(self) -> None:
        """LHS samples should cover each stratum at least once."""
        var = RandomVariable(name="u", distribution="uniform", params={"low": 0, "high": 1})
        config = MonteCarloConfig(n_samples=100, method="lhs", seed=99)
        engine = MonteCarloEngine([var], config)
        samples = engine.generate_samples()

        assert samples.shape == (100, 1)

        # With 100 LHS samples in [0,1], each of the 10 bins should have ~10
        hist, _ = np.histogram(samples[:, 0], bins=10, range=(0, 1))
        assert np.all(hist >= 1), "LHS should cover all bins"

    def test_random_method(self) -> None:
        var = RandomVariable(name="x", distribution="normal", params={"mean": 5, "std": 1})
        config = MonteCarloConfig(n_samples=200, method="random", seed=7)
        engine = MonteCarloEngine([var], config)
        result = engine.run(lambda d: d["x"])
        assert result.mean == pytest.approx(5.0, abs=0.5)

    def test_multiple_variables(self) -> None:
        vars_ = [
            RandomVariable(name="a", distribution="normal", params={"mean": 10, "std": 1}),
            RandomVariable(name="b", distribution="uniform", params={"low": 0, "high": 1}),
        ]
        config = MonteCarloConfig(n_samples=500, method="lhs", seed=42)
        engine = MonteCarloEngine(vars_, config)
        result = engine.run(lambda d: d["a"] + d["b"])

        # E[a+b] = 10 + 0.5 = 10.5
        assert result.mean == pytest.approx(10.5, abs=0.3)
        assert result.samples.shape == (500, 2)

    def test_percentiles(self) -> None:
        var = RandomVariable(name="x", distribution="normal", params={"mean": 0, "std": 1})
        config = MonteCarloConfig(n_samples=5000, method="lhs", seed=10)
        engine = MonteCarloEngine([var], config)
        result = engine.run(lambda d: d["x"])

        assert 5 in result.percentiles
        assert 50 in result.percentiles
        assert 95 in result.percentiles
        # Median of N(0,1) should be close to 0
        assert result.percentiles[50] == pytest.approx(0.0, abs=0.1)

    def test_convergence_flag(self) -> None:
        var = RandomVariable(name="x", distribution="normal", params={"mean": 100, "std": 1})
        config = MonteCarloConfig(
            n_samples=2000, method="lhs", seed=55, convergence_check=True, convergence_tol=0.05
        )
        engine = MonteCarloEngine([var], config)
        result = engine.run(lambda d: d["x"])
        # With low variance and many samples, should converge
        assert result.converged is True

    def test_coefficient_of_variation(self) -> None:
        var = RandomVariable(name="x", distribution="normal", params={"mean": 100, "std": 10})
        config = MonteCarloConfig(n_samples=3000, method="lhs", seed=33)
        engine = MonteCarloEngine([var], config)
        result = engine.run(lambda d: d["x"])
        # COV should be ~0.1
        assert result.cov == pytest.approx(0.1, abs=0.02)


# ---------------------------------------------------------------------------
# Pre-defined distributions
# ---------------------------------------------------------------------------


class TestDistributions:
    """Verify distribution factory functions."""

    def test_material_property_count(self) -> None:
        vars_ = material_property_distributions("S355")
        assert len(vars_) == 4
        names = {v.name for v in vars_}
        assert "yield_strength" in names
        assert "uts" in names
        assert "elastic_modulus" in names
        assert "fatigue_life_scatter" in names

    def test_material_unknown_falls_back(self) -> None:
        vars_ = material_property_distributions("unknown_steel")
        assert len(vars_) == 4

    def test_geometric_fillet(self) -> None:
        vars_ = geometric_tolerance_distributions("fillet")
        names = {v.name for v in vars_}
        assert "weld_toe_angle" in names
        assert "weld_toe_radius" in names
        assert "misalignment" in names
        assert "weld_leg_size" in names

    def test_geometric_butt_no_leg_size(self) -> None:
        vars_ = geometric_tolerance_distributions("butt")
        names = {v.name for v in vars_}
        assert "weld_leg_size" not in names

    def test_corrosion_marine(self) -> None:
        rv = corrosion_distribution("marine")
        assert rv.distribution == "gumbel"
        assert rv.name == "pit_depth"

    def test_corrosion_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown environment"):
            corrosion_distribution("outer_space")


# ---------------------------------------------------------------------------
# Sobol sensitivity indices
# ---------------------------------------------------------------------------


class TestSobolIndices:
    """Test Sobol indices for simple, analytically known functions."""

    def test_linear_single_variable_dominant(self) -> None:
        """f = 5*x1 + x2  =>  S1(x1) >> S1(x2)."""
        vars_ = [
            RandomVariable(name="x1", distribution="uniform", params={"low": 0, "high": 1}),
            RandomVariable(name="x2", distribution="uniform", params={"low": 0, "high": 1}),
        ]

        def f(d: dict[str, float]) -> float:
            return 5.0 * d["x1"] + d["x2"]

        result = sobol_indices(vars_, f, n_base=2048, seed=42)

        assert "first_order" in result
        assert "total" in result
        # x1 should have much larger first-order index
        assert result["first_order"]["x1"] > result["first_order"]["x2"]
        # Total indices should be positive
        assert result["total"]["x1"] > 0
        assert result["total"]["x2"] > 0

    def test_additive_function_no_interactions(self) -> None:
        """For purely additive f, first-order ~ total."""
        vars_ = [
            RandomVariable(name="a", distribution="uniform", params={"low": 0, "high": 1}),
            RandomVariable(name="b", distribution="uniform", params={"low": 0, "high": 1}),
        ]

        def f(d: dict[str, float]) -> float:
            return d["a"] + d["b"]

        result = sobol_indices(vars_, f, n_base=2048, seed=7)

        # For f = a + b with Uniform(0,1): V(a) = V(b) = 1/12, V(Y) = 2/12
        # S_a = S_b ~ 0.5
        assert result["first_order"]["a"] == pytest.approx(0.5, abs=0.1)
        assert result["first_order"]["b"] == pytest.approx(0.5, abs=0.1)


# ---------------------------------------------------------------------------
# FORM reliability index
# ---------------------------------------------------------------------------


class TestFORM:
    """Test FORM for a linear limit state with known analytical solution."""

    def test_linear_limit_state_r_minus_s(self) -> None:
        """g = R - S, R ~ N(200, 20), S ~ N(100, 15).

        Analytical β = (μ_R - μ_S) / sqrt(σ_R^2 + σ_S^2)
                     = 100 / sqrt(400 + 225)
                     = 100 / 25
                     = 4.0
        """
        vars_ = [
            RandomVariable(name="R", distribution="normal", params={"mean": 200, "std": 20}),
            RandomVariable(name="S", distribution="normal", params={"mean": 100, "std": 15}),
        ]

        def g(d: dict[str, float]) -> float:
            return d["R"] - d["S"]

        result = reliability_index_form(vars_, g)

        assert "beta" in result
        assert "probability_of_failure" in result
        assert "design_point" in result

        # Analytical β = 4.0
        assert result["beta"] == pytest.approx(4.0, abs=0.15)

        # P_f = Phi(-4.0) ~ 3.17e-5
        from scipy import stats

        expected_pf = stats.norm.cdf(-4.0)
        assert result["probability_of_failure"] == pytest.approx(expected_pf, rel=0.3)

    def test_design_point_is_dict(self) -> None:
        vars_ = [
            RandomVariable(name="R", distribution="normal", params={"mean": 200, "std": 20}),
            RandomVariable(name="S", distribution="normal", params={"mean": 100, "std": 15}),
        ]

        def g(d: dict[str, float]) -> float:
            return d["R"] - d["S"]

        result = reliability_index_form(vars_, g)
        dp = result["design_point"]
        assert isinstance(dp, dict)
        assert "R" in dp
        assert "S" in dp
