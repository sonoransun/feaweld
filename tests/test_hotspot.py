"""Tests for hot-spot stress extrapolation methods."""

import numpy as np
import pytest

from feaweld.core.types import FEMesh, FEAResults, StressField, ElementType, WeldLineDefinition
from feaweld.postprocess.hotspot import (
    hotspot_stress_linear,
    hotspot_stress_quadratic,
    max_hotspot_stress,
    HotSpotType,
)


def _make_stress_gradient_mesh():
    """Create a mesh with stress that decreases away from weld toe."""
    # 10 nodes in a line along x, y=0, z=0
    n = 10
    nodes = np.zeros((n, 3))
    nodes[:, 0] = np.linspace(0, 50, n)  # x from 0 to 50mm

    elements = np.array([[i, i + 1, i + 1] for i in range(n - 1)])  # dummy connectivity

    stress_vals = np.zeros((n, 6))
    # Stress decreases from weld toe (x=0): σ = 300 - 4*x
    stress_vals[:, 1] = 300.0 - 4.0 * nodes[:, 0]

    mesh = FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
        node_sets={"weld_toe": np.array([0])},
    )
    results = FEAResults(
        mesh=mesh,
        stress=StressField(values=stress_vals),
    )
    return mesh, results


def test_hotspot_linear_type_a():
    """Test Type A linear extrapolation returns a result."""
    mesh, results = _make_stress_gradient_mesh()

    weld_line = WeldLineDefinition(
        name="test",
        node_ids=np.array([0]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 0.0, 1.0]),
    )

    hs_results = hotspot_stress_linear(results, weld_line, HotSpotType.TYPE_A)
    assert len(hs_results) == 1
    assert hs_results[0].hot_spot_stress > 0
    assert hs_results[0].extrapolation_type == HotSpotType.TYPE_A


def test_hotspot_result_has_reference_stresses():
    """Test that reference stresses are captured."""
    mesh, results = _make_stress_gradient_mesh()

    weld_line = WeldLineDefinition(
        name="test",
        node_ids=np.array([0]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 0.0, 1.0]),
    )

    hs_results = hotspot_stress_linear(results, weld_line, HotSpotType.TYPE_A)
    assert len(hs_results[0].reference_stresses) == 2  # Type A uses 2 points
    assert len(hs_results[0].reference_distances) == 2


def test_max_hotspot_stress():
    """Test max_hotspot_stress selects the highest."""
    mesh, results = _make_stress_gradient_mesh()

    # Add another weld toe node
    weld_line = WeldLineDefinition(
        name="test",
        node_ids=np.array([0, 1]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 0.0, 1.0]),
    )

    hs_results = hotspot_stress_linear(results, weld_line)
    if len(hs_results) > 1:
        max_result = max_hotspot_stress(hs_results)
        assert max_result.hot_spot_stress >= hs_results[1].hot_spot_stress


def test_hotspot_no_stress_raises():
    """Test that missing stress data raises ValueError."""
    mesh = FEMesh(
        nodes=np.zeros((4, 3)),
        elements=np.array([[0, 1, 2]]),
        element_type=ElementType.TRI3,
    )
    results = FEAResults(mesh=mesh)

    weld_line = WeldLineDefinition(
        name="test",
        node_ids=np.array([0]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )

    with pytest.raises(ValueError, match="No stress data"):
        hotspot_stress_linear(results, weld_line)


def test_hotspot_parallel_vectors_raises():
    """Normal parallel to weld tangent should raise ValueError."""
    mesh, results = _make_stress_gradient_mesh()

    # Normal along x-axis — same direction as weld line (all nodes along x)
    weld_line = WeldLineDefinition(
        name="test",
        node_ids=np.array([0, 1]),
        plate_thickness=10.0,
        normal_direction=np.array([1.0, 0.0, 0.0]),  # parallel to weld!
    )

    with pytest.raises(ValueError, match="parallel"):
        hotspot_stress_linear(results, weld_line)
