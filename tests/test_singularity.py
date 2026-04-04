"""Tests for the singularity detection and mesh convergence modules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from feaweld.core.types import ElementType, FEAResults, FEMesh, StressField
from feaweld.singularity.convergence import (
    ConvergenceResult,
    convergence_study,
    grid_convergence_index,
    richardson_extrapolation,
)
from feaweld.singularity.detection import (
    SingularityInfo,
    detect_singularities,
    estimate_convergence_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh(n_nodes: int, element_type: ElementType = ElementType.TRI3) -> FEMesh:
    """Create a trivial triangular mesh on a 1x1 square."""
    side = int(math.ceil(math.sqrt(n_nodes)))
    xs = np.linspace(0, 1, side)
    ys = np.linspace(0, 1, side)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(side * side)])
    nodes = nodes[:n_nodes]

    # Build triangular connectivity
    elems: list[list[int]] = []
    for j in range(side - 1):
        for i in range(side - 1):
            n0 = j * side + i
            n1 = n0 + 1
            n2 = n0 + side
            n3 = n2 + 1
            if n3 < n_nodes:
                elems.append([n0, n1, n2])
                elems.append([n1, n3, n2])
    if len(elems) == 0:
        elems.append([0, min(1, n_nodes - 1), min(2, n_nodes - 1)])

    return FEMesh(
        nodes=nodes,
        elements=np.array(elems, dtype=np.int64),
        element_type=element_type,
    )


def _make_results(
    n_nodes: int,
    vm_stress: np.ndarray | float,
) -> FEAResults:
    """Create FEA results with a prescribed von-Mises-like stress field."""
    mesh = _make_mesh(n_nodes)
    if isinstance(vm_stress, (int, float)):
        vm_stress = np.full(n_nodes, vm_stress, dtype=np.float64)
    # Build a stress tensor whose von Mises equals the desired value.
    # Uniaxial tension: sigma_xx = vm, rest zero → VM = sigma_xx.
    stress_vals = np.zeros((n_nodes, 6), dtype=np.float64)
    stress_vals[:, 0] = vm_stress  # sigma_xx
    return FEAResults(
        mesh=mesh,
        stress=StressField(values=stress_vals),
    )


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------


class TestRichardsonExtrapolation:

    def test_known_values(self) -> None:
        """Use values [1.0, 1.1, 1.3] at sizes [0.5, 1.0, 2.0]."""
        values = [1.0, 1.1, 1.3]
        sizes = [0.5, 1.0, 2.0]
        extrap, p = richardson_extrapolation(values, sizes)

        # Manual:
        # r = 1.0 / 0.5 = 2.0
        # ratio = (1.3 - 1.1) / (1.1 - 1.0) = 2.0
        # p = ln(2) / ln(2) = 1.0
        # extrap = 1.0 + (1.0 - 1.1) / (2^1 - 1) = 1.0 - 0.1 = 0.9
        assert p == pytest.approx(1.0, rel=1e-10)
        assert extrap == pytest.approx(0.9, rel=1e-10)

    def test_second_order_convergence(self) -> None:
        """Quadratic convergence: error ~ h^2."""
        # Exact = 1.0, error = C*h^2 with C = 1
        sizes = [0.5, 1.0, 2.0]
        exact = 1.0
        values = [exact + s ** 2 for s in sizes]  # [1.25, 2.0, 5.0]
        extrap, p = richardson_extrapolation(values, sizes)
        assert p == pytest.approx(2.0, rel=1e-6)
        assert extrap == pytest.approx(exact, rel=1e-2)

    def test_requires_three_levels(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            richardson_extrapolation([1.0, 2.0], [0.5, 1.0])

    def test_wrong_ordering_raises(self) -> None:
        """Sizes must increase from finest to coarsest."""
        with pytest.raises(ValueError, match="increase"):
            richardson_extrapolation([1.0, 1.1, 1.3], [2.0, 1.0, 0.5])

    def test_identical_finest_values_returns_converged(self) -> None:
        extrap, p = richardson_extrapolation([5.0, 5.0, 5.5], [0.25, 0.5, 1.0])
        assert extrap == pytest.approx(5.0)
        assert p == float("inf")


# ---------------------------------------------------------------------------
# Grid Convergence Index
# ---------------------------------------------------------------------------


class TestGridConvergenceIndex:

    def test_basic_computation(self) -> None:
        # f_fine=1.0, f_coarse=1.1, r=2, p=2
        gci = grid_convergence_index(1.0, 1.1, r=2.0, p=2.0)
        # epsilon = (1.1 - 1.0) / 1.0 = 0.1
        # GCI = 1.25 * 0.1 / (4 - 1) = 0.125 / 3 ≈ 0.04167
        assert gci == pytest.approx(1.25 * 0.1 / 3.0, rel=1e-10)

    def test_zero_fine_value(self) -> None:
        gci = grid_convergence_index(0.0, 1.0, 2.0, 2.0)
        assert math.isinf(gci)

    def test_identical_values_gives_zero(self) -> None:
        gci = grid_convergence_index(5.0, 5.0, 2.0, 2.0)
        assert gci == pytest.approx(0.0)

    def test_custom_safety_factor(self) -> None:
        gci_125 = grid_convergence_index(1.0, 1.2, 2.0, 1.0, safety_factor=1.25)
        gci_300 = grid_convergence_index(1.0, 1.2, 2.0, 1.0, safety_factor=3.0)
        assert gci_300 > gci_125


# ---------------------------------------------------------------------------
# Full convergence study
# ---------------------------------------------------------------------------


class TestConvergenceStudy:

    def test_converged_case(self) -> None:
        """A well-converged sequence should yield is_converged=True."""
        # Stress converging to 100 MPa with second-order convergence.
        exact = 100.0
        sizes = [0.25, 0.5, 1.0]
        values = [exact + 0.5 * s ** 2 for s in sizes]
        result = convergence_study(values, sizes)
        assert isinstance(result, ConvergenceResult)
        assert result.is_converged is True
        assert result.gci < 0.05
        assert result.convergence_order == pytest.approx(2.0, rel=1e-3)
        assert result.extrapolated_value == pytest.approx(exact, rel=1e-2)

    def test_non_converged_case(self) -> None:
        """Stress that keeps growing signals non-convergence."""
        sizes = [0.25, 0.5, 1.0]
        # Stress grows rapidly as mesh is refined — singularity-like.
        values = [800.0, 200.0, 100.0]
        result = convergence_study(values, sizes)
        # GCI should be large → not converged.
        assert result.is_converged is False

    def test_requires_three_levels(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            convergence_study([1.0, 2.0], [0.5, 1.0])

    def test_auto_sorts_by_mesh_size(self) -> None:
        """Regardless of input order, the study should sort by mesh size."""
        sizes = [1.0, 0.25, 0.5]
        values = [102.0, 100.0625, 100.25]  # exact=100, error = 2*h^2
        result = convergence_study(values, sizes)
        assert result.mesh_sizes == sorted(sizes)
        assert result.is_converged is True


# ---------------------------------------------------------------------------
# Singularity detection
# ---------------------------------------------------------------------------


class TestDetectSingularities:

    def test_no_singularity_when_stress_unchanged(self) -> None:
        """Identical results on coarse and fine → no flags."""
        res_coarse = _make_results(4, 100.0)
        res_fine = _make_results(16, 100.0)
        detections = detect_singularities(res_coarse, res_fine)
        assert len(detections) == 0

    def test_flag_stress_increase(self) -> None:
        """A 30 % stress increase should be flagged at the default 20 % threshold."""
        res_coarse = _make_results(4, 100.0)
        # Fine mesh: most nodes at 100, but one at 130 (30 % increase).
        fine_stress = np.full(16, 100.0)
        fine_stress[4] = 130.0
        res_fine = _make_results(16, fine_stress)
        detections = detect_singularities(res_coarse, res_fine, threshold=0.20)
        assert len(detections) >= 1
        flagged_nodes = {d.node_id for d in detections}
        assert 4 in flagged_nodes

    def test_singularity_marked_correctly(self) -> None:
        """Stress that grows without converging should be marked is_singular=True."""
        # Coarse mesh: 4 nodes, all at 100 MPa.
        res_coarse = _make_results(4, 100.0)
        # Fine mesh: 16 nodes, one blows up to 500 MPa (5x increase).
        fine_stress = np.full(16, 100.0)
        fine_stress[0] = 500.0
        res_fine = _make_results(16, fine_stress)
        detections = detect_singularities(res_coarse, res_fine, threshold=0.20)
        singular_flags = [d for d in detections if d.node_id == 0]
        assert len(singular_flags) == 1
        # With such a large jump the convergence rate should be low/negative → singular.
        assert singular_flags[0].is_singular is True

    def test_raises_without_stress(self) -> None:
        mesh = _make_mesh(4)
        res_no_stress = FEAResults(mesh=mesh)
        res_ok = _make_results(4, 100.0)
        with pytest.raises(ValueError, match="stress field"):
            detect_singularities(res_no_stress, res_ok)


# ---------------------------------------------------------------------------
# Convergence rate estimator
# ---------------------------------------------------------------------------


class TestEstimateConvergenceRate:

    def test_converging_values(self) -> None:
        """Stress converging to a limit should give positive rate."""
        # Values at meshes h=0.5, h=1.0: 100.25, 101.0 (error ~ h^2)
        rate = estimate_convergence_rate(
            stress_values=[100.25, 101.0],
            mesh_sizes=[0.5, 1.0],
        )
        assert rate > 0

    def test_diverging_values(self) -> None:
        """Stress growing as mesh is refined should give low/negative rate."""
        rate = estimate_convergence_rate(
            stress_values=[500.0, 100.0],
            mesh_sizes=[0.5, 1.0],
        )
        # Fine mesh has much higher stress → divergent behaviour.
        assert rate < 0.5

    def test_three_levels(self) -> None:
        """Three levels should still work."""
        rate = estimate_convergence_rate(
            stress_values=[100.0625, 100.25, 101.0],
            mesh_sizes=[0.25, 0.5, 1.0],
        )
        assert rate > 0

    def test_requires_at_least_two(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            estimate_convergence_rate([100.0], [1.0])
