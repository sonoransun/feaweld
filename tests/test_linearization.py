"""Tests for through-thickness stress linearization."""

import numpy as np
import pytest

from feaweld.core.types import FEMesh, FEAResults, StressField, ElementType
from feaweld.postprocess.linearization import (
    linearize_through_thickness,
    linearize_at_weld_toe,
    LinearizationResult,
)


def _make_column_mesh(n_nodes: int = 21, thickness: float = 10.0):
    """Create a line of nodes along the y-axis for through-thickness sampling.

    Returns a mesh with ``n_nodes`` nodes from y=0 to y=thickness, connected
    by degenerate triangles (sufficient for cKDTree nearest-neighbour lookup).
    """
    nodes = np.zeros((n_nodes, 3))
    nodes[:, 1] = np.linspace(0.0, thickness, n_nodes)

    # Degenerate triangle connectivity (only geometry matters for linearization)
    elements = np.array([[i, i + 1, i + 1] for i in range(n_nodes - 1)])

    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
    )


# ── Uniform stress: membrane = sigma, bending ~ 0 ─────────────────────────

class TestUniformStress:
    """Uniform uniaxial stress should produce pure membrane, zero bending."""

    def test_membrane_equals_applied_stress(self):
        sigma_yy = 150.0
        mesh = _make_column_mesh(n_nodes=21, thickness=10.0)
        stress_vals = np.zeros((mesh.n_nodes, 6))
        stress_vals[:, 1] = sigma_yy

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 10.0, 0.0])
        lin = linearize_through_thickness(results, start, end, n_points=20)

        assert lin.membrane[1] == pytest.approx(sigma_yy, rel=1e-3)

    def test_bending_near_zero(self):
        sigma_yy = 150.0
        mesh = _make_column_mesh(n_nodes=21, thickness=10.0)
        stress_vals = np.zeros((mesh.n_nodes, 6))
        stress_vals[:, 1] = sigma_yy

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 10.0, 0.0])
        lin = linearize_through_thickness(results, start, end, n_points=20)

        assert np.allclose(lin.bending, 0.0, atol=1e-6)

    def test_peak_near_zero(self):
        sigma_yy = 150.0
        mesh = _make_column_mesh(n_nodes=21, thickness=10.0)
        stress_vals = np.zeros((mesh.n_nodes, 6))
        stress_vals[:, 1] = sigma_yy

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 10.0, 0.0])
        lin = linearize_through_thickness(results, start, end, n_points=20)

        assert np.allclose(lin.peak, 0.0, atol=1e-6)

    def test_membrane_scalar_positive(self):
        sigma_yy = 150.0
        mesh = _make_column_mesh(n_nodes=21, thickness=10.0)
        stress_vals = np.zeros((mesh.n_nodes, 6))
        stress_vals[:, 1] = sigma_yy

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 10.0, 0.0])
        lin = linearize_through_thickness(results, start, end, n_points=20)

        assert lin.membrane_scalar == pytest.approx(sigma_yy, rel=1e-3)


# ── Linearly varying stress: non-zero bending ────────────────────────────

class TestLinearGradient:
    """Stress varying linearly through thickness should produce membrane and bending."""

    @pytest.fixture
    def gradient_result(self):
        """Linearize a field that goes from 0 at y=0 to 200 at y=10."""
        thickness = 10.0
        mesh = _make_column_mesh(n_nodes=51, thickness=thickness)
        stress_vals = np.zeros((mesh.n_nodes, 6))
        # sigma_yy varies linearly: 0 at y=0, 200 at y=10
        stress_vals[:, 1] = 200.0 * mesh.nodes[:, 1] / thickness

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, thickness, 0.0])
        return linearize_through_thickness(results, start, end, n_points=50)

    def test_membrane_is_average(self, gradient_result):
        # Average of a linear 0..200 field = 100
        assert gradient_result.membrane[1] == pytest.approx(100.0, rel=1e-2)

    def test_bending_is_nonzero(self, gradient_result):
        assert abs(gradient_result.bending[1]) > 1.0

    def test_bending_value(self, gradient_result):
        # For linear stress sigma = 200*z/t, membrane = 100, bending ≈ 100
        # Numerical integration introduces ~1-2% error due to mesh interpolation
        assert gradient_result.bending[1] == pytest.approx(100.0, rel=0.02)

    def test_off_components_stay_zero(self, gradient_result):
        # Only component [1] (sigma_yy) is loaded; others should be ~0
        for comp in [0, 2, 3, 4, 5]:
            assert gradient_result.membrane[comp] == pytest.approx(0.0, abs=1e-6)
            assert gradient_result.bending[comp] == pytest.approx(0.0, abs=1e-6)


# ── linearized_stress property ────────────────────────────────────────────

class TestLinearizedStressProperty:
    """LinearizationResult.linearized_stress should reconstruct membrane + bending."""

    def test_shape_matches_total_stress(self):
        n_pts = 25
        mesh = _make_column_mesh(n_nodes=51, thickness=10.0)
        stress_vals = np.zeros((mesh.n_nodes, 6))
        stress_vals[:, 0] = 80.0  # uniform sigma_xx

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 10.0, 0.0])
        lin = linearize_through_thickness(results, start, end, n_points=n_pts)

        assert lin.linearized_stress.shape == (n_pts, 6)

    def test_uniform_linearized_equals_total(self):
        """For uniform input the linearized distribution should equal the actual."""
        mesh = _make_column_mesh(n_nodes=41, thickness=10.0)
        stress_vals = np.zeros((mesh.n_nodes, 6))
        stress_vals[:, 1] = 100.0

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 10.0, 0.0])
        lin = linearize_through_thickness(results, start, end, n_points=30)

        np.testing.assert_allclose(
            lin.linearized_stress[:, 1], 100.0, atol=0.5,
        )

    def test_linear_gradient_reconstructed(self):
        """For linear input the linearized distribution should match the original."""
        thickness = 10.0
        n_pts = 40
        mesh = _make_column_mesh(n_nodes=81, thickness=thickness)
        stress_vals = np.zeros((mesh.n_nodes, 6))
        stress_vals[:, 1] = 200.0 * mesh.nodes[:, 1] / thickness

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, thickness, 0.0])
        lin = linearize_through_thickness(results, start, end, n_points=n_pts)

        # Expected: sigma_yy varies linearly 0..200 along z_coords
        expected = 200.0 * lin.z_coords / thickness
        np.testing.assert_allclose(
            lin.linearized_stress[:, 1], expected, atol=1.0,
        )

    def test_endpoints_match_membrane_plus_minus_bending(self):
        """At the surfaces the linearized stress should be membrane +/- bending."""
        thickness = 10.0
        mesh = _make_column_mesh(n_nodes=41, thickness=thickness)
        stress_vals = np.zeros((mesh.n_nodes, 6))
        stress_vals[:, 1] = 200.0 * mesh.nodes[:, 1] / thickness

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, thickness, 0.0])
        lin = linearize_through_thickness(results, start, end, n_points=30)

        inner = lin.linearized_stress[0]   # z_rel = -1
        outer = lin.linearized_stress[-1]  # z_rel = +1

        np.testing.assert_allclose(inner, lin.membrane - lin.bending, atol=1e-10)
        np.testing.assert_allclose(outer, lin.membrane + lin.bending, atol=1e-10)


# ── Von Mises scalars non-negative ────────────────────────────────────────

class TestVonMisesNonNegative:
    """All scalar von Mises values must be >= 0."""

    @pytest.fixture(params=[
        "uniform",
        "gradient",
        "multiaxial",
    ])
    def linearization_result(self, request):
        thickness = 10.0
        mesh = _make_column_mesh(n_nodes=41, thickness=thickness)
        stress_vals = np.zeros((mesh.n_nodes, 6))

        if request.param == "uniform":
            stress_vals[:, 1] = 100.0
        elif request.param == "gradient":
            stress_vals[:, 1] = 200.0 * mesh.nodes[:, 1] / thickness
        elif request.param == "multiaxial":
            # Combined normal + shear
            stress_vals[:, 0] = 50.0
            stress_vals[:, 1] = -30.0
            stress_vals[:, 3] = 40.0 * mesh.nodes[:, 1] / thickness

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, thickness, 0.0])
        return linearize_through_thickness(results, start, end, n_points=20)

    def test_membrane_scalar_nonneg(self, linearization_result):
        assert linearization_result.membrane_scalar >= 0.0

    def test_bending_scalar_nonneg(self, linearization_result):
        assert linearization_result.bending_scalar >= 0.0

    def test_peak_scalar_nonneg(self, linearization_result):
        assert linearization_result.peak_scalar >= 0.0

    def test_membrane_plus_bending_scalar_nonneg(self, linearization_result):
        assert linearization_result.membrane_plus_bending_scalar >= 0.0


# ── Error handling ────────────────────────────────────────────────────────

class TestErrorHandling:
    """Missing stress data should raise ValueError."""

    def test_no_stress_raises(self):
        mesh = FEMesh(
            nodes=np.zeros((4, 3)),
            elements=np.array([[0, 1, 2]]),
            element_type=ElementType.TRI3,
        )
        results = FEAResults(mesh=mesh)

        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 10.0, 0.0])

        with pytest.raises(ValueError, match="No stress data"):
            linearize_through_thickness(results, start, end)

    def test_no_stress_raises_at_weld_toe(self):
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 10.0, 0.0],
        ])
        mesh = FEMesh(
            nodes=nodes,
            elements=np.array([[0, 1, 2]]),
            element_type=ElementType.TRI3,
        )
        results = FEAResults(mesh=mesh)

        with pytest.raises(ValueError, match="No stress data"):
            linearize_at_weld_toe(
                results,
                weld_toe_node=2,
                plate_thickness=10.0,
                surface_normal=np.array([0.0, 1.0, 0.0]),
            )


# ── linearize_at_weld_toe convenience wrapper ────────────────────────────

class TestLinearizeAtWeldToe:
    """Test the weld-toe convenience function delegates correctly."""

    def test_returns_result(self):
        thickness = 10.0
        n_nodes = 21
        nodes = np.zeros((n_nodes, 3))
        nodes[:, 1] = np.linspace(0.0, thickness, n_nodes)
        elements = np.array([[i, i + 1, i + 1] for i in range(n_nodes - 1)])

        mesh = FEMesh(
            nodes=nodes,
            elements=elements,
            element_type=ElementType.TRI3,
        )

        stress_vals = np.zeros((n_nodes, 6))
        stress_vals[:, 1] = 100.0

        results = FEAResults(
            mesh=mesh,
            stress=StressField(values=stress_vals),
        )

        # Weld toe is the last node (outer surface at y=10)
        # Normal points in +y, so inner = outer - t * normal = y=0
        lin = linearize_at_weld_toe(
            results,
            weld_toe_node=n_nodes - 1,
            plate_thickness=thickness,
            surface_normal=np.array([0.0, 1.0, 0.0]),
            n_points=20,
        )

        assert isinstance(lin, LinearizationResult)
        assert lin.membrane[1] == pytest.approx(100.0, rel=1e-2)
        assert np.allclose(lin.bending, 0.0, atol=1e-6)
