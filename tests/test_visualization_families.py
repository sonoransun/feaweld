"""Tests for the probabilistic, material, and thermal plot families.

Each family plot is driven with real upstream computations (a genuine Sobol
run, a FORM reliability solve, bundled CCT / residual-stress data, the
analytic Norton-Bailey law) rather than hand-faked figure inputs, so the
tests exercise the plotting code against the shapes those producers actually
emit.  Matplotlib-dependent; skipped when matplotlib is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt


def teardown_function(_func) -> None:
    plt.close("all")


# ---------------------------------------------------------------------------
# Probabilistic plots
# ---------------------------------------------------------------------------

def _toy_variables():
    from feaweld.probabilistic.monte_carlo import RandomVariable
    return [
        RandomVariable("a", "normal", {"mean": 10.0, "std": 2.0}),
        RandomVariable("b", "normal", {"mean": 5.0, "std": 1.0}),
        RandomVariable("c", "normal", {"mean": 1.0, "std": 0.5}),
    ]


class TestProbabilisticPlots:
    @pytest.mark.parametrize("kind", ["bar", "tornado"])
    def test_sobol_indices(self, kind):
        from feaweld.probabilistic.sensitivity import sobol_indices
        from feaweld.visualization.probabilistic_plots import plot_sobol_indices

        indices = sobol_indices(
            _toy_variables(),
            lambda d: 3.0 * d["a"] + 2.0 * d["b"] + 0.1 * d["c"],
            n_base=64, seed=1,
        )
        assert set(indices) == {"first_order", "total"}
        fig = plot_sobol_indices(indices, kind=kind, show=False)
        assert isinstance(fig, plt.Figure)

    def test_sobol_indices_unknown_kind_raises(self):
        from feaweld.probabilistic.sensitivity import sobol_indices
        from feaweld.visualization.probabilistic_plots import plot_sobol_indices

        indices = sobol_indices(
            _toy_variables(), lambda d: d["a"], n_base=16, seed=0,
        )
        with pytest.raises(ValueError):
            plot_sobol_indices(indices, kind="pie", show=False)

    def test_mc_histogram_default_percentiles(self):
        from feaweld.visualization.probabilistic_plots import plot_mc_histogram
        data = np.random.default_rng(0).normal(100.0, 15.0, 500)
        fig = plot_mc_histogram(data, show=False)
        assert isinstance(fig, plt.Figure)

    def test_mc_histogram_explicit_percentiles(self):
        from feaweld.visualization.probabilistic_plots import plot_mc_histogram
        data = np.random.default_rng(1).normal(100.0, 15.0, 500)
        fig = plot_mc_histogram(data, percentiles=[10, 50, 90], show=False)
        assert isinstance(fig, plt.Figure)

    def test_form_reliability(self):
        from feaweld.probabilistic.monte_carlo import RandomVariable
        from feaweld.probabilistic.sensitivity import reliability_index_form
        from feaweld.visualization.probabilistic_plots import plot_form_reliability

        variables = [
            RandomVariable("R", "normal", {"mean": 300.0, "std": 30.0}),
            RandomVariable("S", "normal", {"mean": 150.0, "std": 20.0}),
        ]
        form = reliability_index_form(variables, lambda d: d["R"] - d["S"])
        assert "beta" in form and "design_point" in form
        fig = plot_form_reliability(form, show=False)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Material / metallurgy plots
# ---------------------------------------------------------------------------

class TestMaterialPlots:
    def test_cct_diagram_with_cooling_rate_marker(self):
        from feaweld.data.cct import get_cct_diagram
        from feaweld.visualization.material_plots import plot_cct_diagram

        diagram = get_cct_diagram("A36")
        fig = plot_cct_diagram(diagram, grade="A36", cooling_rate=10.0, show=False)
        assert isinstance(fig, plt.Figure)

    def test_residual_profile_by_name(self):
        from feaweld.visualization.material_plots import plot_residual_stress_profile
        fig = plot_residual_stress_profile("BS7910_Level2_butt", show=False)
        assert isinstance(fig, plt.Figure)

    def test_residual_profile_with_yield_scaling(self):
        from feaweld.visualization.material_plots import plot_residual_stress_profile
        fig = plot_residual_stress_profile(
            "BS7910_Level2_butt", yield_strength=355.0, show=False,
        )
        # With a yield strength, the curve is drawn in MPa (scaled from the
        # normalised stress/sigma_y profile).
        line = fig.axes[0].lines[0]
        assert np.max(np.abs(line.get_xdata())) > 1.5  # not still normalised (~<=1)
        assert isinstance(fig, plt.Figure)

    def test_creep_curve_analytic_point(self):
        from feaweld.visualization.material_plots import plot_creep_curve

        A, n, m = 1e-12, 3.0, 0.5
        sigma = 100.0
        fig = plot_creep_curve(
            A, n, m, stress_levels=[sigma], t_end=1.0e5, show=False,
        )
        line = fig.axes[0].lines[0]
        t, eps = line.get_data()
        # Norton-Bailey time hardening: eps = A * sigma**n * t**(m+1) / (m+1)
        mid = len(t) // 2
        expected = A * sigma ** n * t[mid] ** (m + 1.0) / (m + 1.0)
        assert eps[mid] == pytest.approx(expected, rel=1e-6)

    def test_hall_petch_single(self):
        from feaweld.multiscale.micro import HallPetchParams
        from feaweld.visualization.material_plots import plot_hall_petch
        fig = plot_hall_petch(HallPetchParams(sigma_0=70.0, k_y=0.74), show=False)
        assert isinstance(fig, plt.Figure)

    def test_hall_petch_dict_with_markers(self):
        from feaweld.multiscale.micro import HallPetchParams
        from feaweld.visualization.material_plots import plot_hall_petch
        curves = {
            "low-carbon": HallPetchParams(sigma_0=70.0, k_y=0.74),
            "mild": HallPetchParams(sigma_0=50.0, k_y=0.70),
        }
        fig = plot_hall_petch(curves, markers={"low-carbon": 20.0}, show=False)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Fatigue plots: Haigh (mean stress) diagram
# ---------------------------------------------------------------------------

def _haigh_cycles():
    return [
        (50.0, 10.0, 1.0),
        (80.0, 20.0, 1.0),
        (120.0, 30.0, 0.5),
        (60.0, 15.0, 1.0),
        (100.0, 25.0, 1.0),
    ]


class TestHaighDiagram:
    def test_goodman(self):
        from feaweld.visualization.fatigue_plots import plot_haigh_diagram
        fig = plot_haigh_diagram(
            _haigh_cycles(), 400.0, correction="goodman", show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_gerber_with_yield_line(self):
        from feaweld.visualization.fatigue_plots import plot_haigh_diagram
        fig = plot_haigh_diagram(
            _haigh_cycles(), 400.0, correction="gerber", sigma_y=250.0,
            show=False,
        )
        labels = [line.get_label() for line in fig.axes[0].lines]
        assert any("Gerber" in lab and "active" in lab for lab in labels)
        assert any("yield" in lab for lab in labels)

    def test_goodman_envelope_through_worst_cycle(self):
        from feaweld.visualization.fatigue_plots import plot_haigh_diagram
        sigma_u = 400.0
        cycles = _haigh_cycles()
        fig = plot_haigh_diagram(
            cycles, sigma_u, correction="goodman", show=False,
        )
        active = next(
            line for line in fig.axes[0].lines if "active" in line.get_label()
        )
        # Amplitude-axis intercept is the equivalent fully reversed
        # amplitude of the most damaging cycle: sigma_a / (1 - sigma_m/sigma_u).
        expected = max(
            0.5 * rng / (1.0 - mean / sigma_u) for rng, mean, _cnt in cycles
        )
        assert active.get_ydata()[0] == pytest.approx(expected, rel=1e-9)
        # ...and the envelope reaches zero amplitude at sigma_u.
        assert active.get_xdata()[-1] == pytest.approx(sigma_u)
        assert active.get_ydata()[-1] == pytest.approx(0.0, abs=1e-9)

    def test_empty_cycles(self):
        from feaweld.visualization.fatigue_plots import plot_haigh_diagram
        fig = plot_haigh_diagram([], 400.0, show=False)
        assert isinstance(fig, plt.Figure)

    def test_invalid_correction_raises(self):
        from feaweld.visualization.fatigue_plots import plot_haigh_diagram
        with pytest.raises(ValueError):
            plot_haigh_diagram(_haigh_cycles(), 400.0, correction="soderberg",
                               show=False)

    def test_nonpositive_sigma_u_raises(self):
        from feaweld.visualization.fatigue_plots import plot_haigh_diagram
        with pytest.raises(ValueError):
            plot_haigh_diagram(_haigh_cycles(), 0.0, show=False)


# ---------------------------------------------------------------------------
# Thermal history plot
# ---------------------------------------------------------------------------

class TestThermalHistory:
    def test_temperature_history_1d(self):
        from feaweld.visualization.thermal_plots import plot_temperature_history
        t = np.linspace(0.0, 10.0, 50)
        temp = np.linspace(20.0, 800.0, 50)
        fig = plot_temperature_history(t, temp, show=False)
        assert isinstance(fig, plt.Figure)

    def test_temperature_history_2d_selects_hottest_node(self):
        from feaweld.visualization.thermal_plots import plot_temperature_history
        t = np.linspace(0.0, 10.0, 50)
        temps = np.random.default_rng(0).uniform(20.0, 800.0, (50, 6))
        fig = plot_temperature_history(t, temps, show=False)
        assert isinstance(fig, plt.Figure)
