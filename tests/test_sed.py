"""Tests for Strain Energy Density (SED) fatigue assessment method."""

import numpy as np
import pytest

from feaweld.core.types import FEMesh, FEAResults, StressField, ElementType
from feaweld.postprocess.sed import (
    compute_sed_field,
    averaged_sed,
    sed_fatigue_life,
    estimate_control_radius,
    SEDResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uniform_results(stress_yy: float, n_nodes: int = 4) -> FEAResults:
    """Create FEA results with uniform uniaxial stress on a 2D plate."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    elements = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    mesh = FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
    )
    stress_vals = np.zeros((mesh.n_nodes, 6))
    stress_vals[:, 1] = stress_yy  # sigma_yy
    return FEAResults(
        mesh=mesh,
        stress=StressField(values=stress_vals),
    )


# ---------------------------------------------------------------------------
# compute_sed_field
# ---------------------------------------------------------------------------

class TestComputeSEDField:
    """Tests for compute_sed_field()."""

    def test_uniform_stress_gives_sigma_sq_over_2E(self, uniform_stress_results):
        """Uniform sigma_yy = 100 MPa -> SED = vm^2 / (2E) at every node."""
        E = 200_000.0  # MPa
        sed = compute_sed_field(uniform_stress_results, elastic_modulus=E)

        # For uniaxial sigma_yy = 100 MPa, von Mises = 100 MPa
        expected = 100.0 ** 2 / (2.0 * E)
        assert sed.shape == (uniform_stress_results.mesh.n_nodes,)
        for val in sed:
            assert val == pytest.approx(expected, rel=1e-10)

    def test_zero_stress_gives_zero_sed(self):
        """Zero stress should produce zero SED everywhere."""
        results = _make_uniform_results(0.0)
        sed = compute_sed_field(results, elastic_modulus=210_000.0)
        np.testing.assert_allclose(sed, 0.0, atol=1e-30)

    def test_sed_scales_with_stress_squared(self):
        """Doubling stress should quadruple SED."""
        E = 200_000.0
        results_1 = _make_uniform_results(50.0)
        results_2 = _make_uniform_results(100.0)

        sed_1 = compute_sed_field(results_1, E)
        sed_2 = compute_sed_field(results_2, E)

        ratio = sed_2[0] / sed_1[0]
        assert ratio == pytest.approx(4.0, rel=1e-10)

    def test_no_stress_data_raises(self):
        """Missing stress field should raise ValueError."""
        mesh = FEMesh(
            nodes=np.zeros((3, 3)),
            elements=np.array([[0, 1, 2]]),
            element_type=ElementType.TRI3,
        )
        results = FEAResults(mesh=mesh)
        with pytest.raises(ValueError, match="No stress data"):
            compute_sed_field(results, elastic_modulus=200_000.0)

    def test_gradient_stress_field(self, gradient_stress_results):
        """Non-uniform stress should produce non-uniform SED values."""
        E = 200_000.0
        sed = compute_sed_field(gradient_stress_results, elastic_modulus=E)

        # sigma_yy varies 0..200 -> SED should vary
        assert sed.min() < sed.max()
        # Nodes at y=0 have sigma_yy=0, so SED=0
        assert sed.min() == pytest.approx(0.0, abs=1e-30)


# ---------------------------------------------------------------------------
# averaged_sed
# ---------------------------------------------------------------------------

class TestAveragedSED:
    """Tests for averaged_sed()."""

    def test_small_radius_captures_one_node(self, uniform_stress_results):
        """A very small control volume centered on a node captures only that node."""
        E = 200_000.0
        # Center on node 0 at (0,0,0) with tiny radius
        center = np.array([0.0, 0.0, 0.0])
        result = averaged_sed(
            uniform_stress_results,
            center_point=center,
            control_radius=0.01,
            elastic_modulus=E,
        )

        expected_sed = 100.0 ** 2 / (2.0 * E)
        assert result.averaged_sed == pytest.approx(expected_sed, rel=1e-10)
        assert result.control_radius == 0.01
        assert result.control_volume > 0.0

    def test_large_radius_captures_all_nodes(self, uniform_stress_results):
        """A large control radius should include all nodes and return mean SED."""
        E = 200_000.0
        center = np.array([5.0, 5.0, 0.0])
        result = averaged_sed(
            uniform_stress_results,
            center_point=center,
            control_radius=100.0,
            elastic_modulus=E,
        )

        expected_sed = 100.0 ** 2 / (2.0 * E)
        assert result.averaged_sed == pytest.approx(expected_sed, rel=1e-10)

    def test_no_nodes_in_volume_uses_nearest(self):
        """When no nodes fall within the radius, the nearest node is used."""
        results = _make_uniform_results(100.0)
        E = 200_000.0
        # Center far away from all nodes
        center = np.array([1000.0, 1000.0, 1000.0])
        result = averaged_sed(
            results,
            center_point=center,
            control_radius=0.001,
            elastic_modulus=E,
        )

        expected_sed = 100.0 ** 2 / (2.0 * E)
        assert result.averaged_sed == pytest.approx(expected_sed, rel=1e-10)

    def test_sed_field_returned(self, uniform_stress_results):
        """The full SED field should be returned in the result."""
        E = 200_000.0
        center = np.array([0.0, 0.0, 0.0])
        result = averaged_sed(
            uniform_stress_results,
            center_point=center,
            control_radius=100.0,
            elastic_modulus=E,
        )

        assert result.sed_field is not None
        assert result.sed_field.shape == (uniform_stress_results.mesh.n_nodes,)

    def test_2d_mesh_control_volume_is_circular(self, uniform_stress_results):
        """For a 3-column node array the control volume is pi*R^2 (2D approx)."""
        E = 200_000.0
        R = 5.0
        center = np.array([5.0, 5.0, 0.0])
        result = averaged_sed(
            uniform_stress_results,
            center_point=center,
            control_radius=R,
            elastic_modulus=E,
        )

        # mesh.ndim == 3 -> 3D volume formula: (4/3)*pi*R^3
        expected_volume = (4.0 / 3.0) * np.pi * R ** 3
        assert result.control_volume == pytest.approx(expected_volume, rel=1e-10)

    def test_no_stress_data_raises(self):
        """Missing stress field should raise ValueError."""
        mesh = FEMesh(
            nodes=np.zeros((3, 3)),
            elements=np.array([[0, 1, 2]]),
            element_type=ElementType.TRI3,
        )
        results = FEAResults(mesh=mesh)
        center = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="No stress data"):
            averaged_sed(results, center, 1.0, 200_000.0)

    def test_critical_sed_and_fatigue_life_are_none(self, uniform_stress_results):
        """averaged_sed does not compute fatigue life; those fields should be None."""
        E = 200_000.0
        center = np.array([0.0, 0.0, 0.0])
        result = averaged_sed(
            uniform_stress_results,
            center_point=center,
            control_radius=100.0,
            elastic_modulus=E,
        )

        assert result.critical_sed is None
        assert result.fatigue_life is None


# ---------------------------------------------------------------------------
# sed_fatigue_life
# ---------------------------------------------------------------------------

class TestSEDFatigueLife:
    """Tests for sed_fatigue_life()."""

    def test_power_law_check(self):
        """N = N_ref * (W_ref / W_bar)^(1/slope) for known inputs."""
        W_bar = 0.05  # MJ/m3
        W_ref = 0.10
        N_ref = 2e6
        slope = 1.5

        base = SEDResult(
            averaged_sed=W_bar,
            control_radius=0.28,
            control_volume=1.0,
            critical_sed=None,
            fatigue_life=None,
        )

        result = sed_fatigue_life(base, W_ref=W_ref, N_ref=N_ref, slope=slope)
        expected_life = N_ref * (W_ref / W_bar) ** (1.0 / slope)
        assert result.fatigue_life == pytest.approx(expected_life, rel=1e-10)

    def test_higher_sed_means_shorter_life(self):
        """Increasing averaged SED should decrease fatigue life."""
        W_ref = 0.10
        N_ref = 2e6
        slope = 1.5

        low = SEDResult(
            averaged_sed=0.01, control_radius=0.28,
            control_volume=1.0, critical_sed=None, fatigue_life=None,
        )
        high = SEDResult(
            averaged_sed=0.20, control_radius=0.28,
            control_volume=1.0, critical_sed=None, fatigue_life=None,
        )

        life_low = sed_fatigue_life(low, W_ref, N_ref, slope).fatigue_life
        life_high = sed_fatigue_life(high, W_ref, N_ref, slope).fatigue_life
        assert life_low > life_high

    def test_zero_sed_gives_infinite_life(self):
        """Zero averaged SED should yield infinite fatigue life."""
        base = SEDResult(
            averaged_sed=0.0, control_radius=0.28,
            control_volume=1.0, critical_sed=None, fatigue_life=None,
        )

        result = sed_fatigue_life(base, W_ref=0.10)
        assert result.fatigue_life == float("inf")

    def test_critical_sed_set_to_w_ref(self):
        """After calling sed_fatigue_life, critical_sed should equal W_ref."""
        base = SEDResult(
            averaged_sed=0.05, control_radius=0.28,
            control_volume=1.0, critical_sed=None, fatigue_life=None,
        )
        W_ref = 0.08
        result = sed_fatigue_life(base, W_ref=W_ref)
        assert result.critical_sed == pytest.approx(W_ref)

    def test_preserved_fields(self):
        """Returned SEDResult should carry over averaged_sed, radius, volume, and field."""
        field = np.array([1.0, 2.0, 3.0])
        base = SEDResult(
            averaged_sed=0.05, control_radius=0.30,
            control_volume=42.0, critical_sed=None, fatigue_life=None,
            sed_field=field,
        )

        result = sed_fatigue_life(base, W_ref=0.10)
        assert result.averaged_sed == pytest.approx(base.averaged_sed)
        assert result.control_radius == pytest.approx(base.control_radius)
        assert result.control_volume == pytest.approx(base.control_volume)
        np.testing.assert_array_equal(result.sed_field, field)

    def test_reference_life_at_w_ref(self):
        """When W_bar == W_ref, life should equal N_ref."""
        W_ref = 0.10
        N_ref = 2e6
        base = SEDResult(
            averaged_sed=W_ref, control_radius=0.28,
            control_volume=1.0, critical_sed=None, fatigue_life=None,
        )
        result = sed_fatigue_life(base, W_ref=W_ref, N_ref=N_ref)
        assert result.fatigue_life == pytest.approx(N_ref, rel=1e-10)


# ---------------------------------------------------------------------------
# estimate_control_radius
# ---------------------------------------------------------------------------

class TestEstimateControlRadius:
    """Tests for estimate_control_radius()."""

    def test_known_parameters(self):
        """R0 = (K_Ic / sigma_u)^2 / (4*pi) for known material constants."""
        K_Ic = 50.0 * np.sqrt(1000.0)  # MPa*sqrt(mm) (approx 50 MPa*sqrt(m))
        sigma_u = 400.0                 # MPa
        E = 200_000.0                   # MPa

        R0 = estimate_control_radius(K_Ic, sigma_u, E)

        expected = (K_Ic / sigma_u) ** 2 / (4.0 * np.pi)
        assert R0 == pytest.approx(expected, rel=1e-10)

    def test_positive_result(self):
        """Control radius should always be positive for positive inputs."""
        R0 = estimate_control_radius(
            fracture_toughness=100.0,
            ultimate_strength=500.0,
            elastic_modulus=200_000.0,
        )
        assert R0 > 0.0

    def test_higher_toughness_larger_radius(self):
        """Higher fracture toughness should give a larger control radius."""
        sigma_u = 400.0
        E = 200_000.0

        R0_low = estimate_control_radius(50.0, sigma_u, E)
        R0_high = estimate_control_radius(100.0, sigma_u, E)

        assert R0_high > R0_low

    def test_higher_strength_smaller_radius(self):
        """Higher ultimate strength should give a smaller control radius."""
        K_Ic = 100.0
        E = 200_000.0

        R0_low_strength = estimate_control_radius(K_Ic, 300.0, E)
        R0_high_strength = estimate_control_radius(K_Ic, 600.0, E)

        assert R0_low_strength > R0_high_strength

    def test_scales_quadratically_with_ratio(self):
        """R0 ~ (K_Ic / sigma_u)^2, so doubling the ratio quadruples R0."""
        E = 200_000.0
        R0_a = estimate_control_radius(100.0, 400.0, E)
        R0_b = estimate_control_radius(200.0, 400.0, E)  # ratio doubled

        assert R0_b / R0_a == pytest.approx(4.0, rel=1e-10)
