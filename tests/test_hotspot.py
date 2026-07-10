"""Tests for hot-spot stress extrapolation methods."""

import numpy as np
import pytest

from feaweld.core.types import FEMesh, FEAResults, StressField, ElementType, WeldLineDefinition
from feaweld.postprocess.hotspot import (
    hotspot_stress_linear,
    hotspot_stress_quadratic,
    max_hotspot_stress,
    order_weld_line_nodes,
    HotSpotType,
    _estimate_weld_tangent,
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


# ---------------------------------------------------------------------------
# order_weld_line_nodes
# ---------------------------------------------------------------------------

def _line_mesh(nodes):
    """Wrap a coordinate array in a minimal FEMesh (dummy connectivity)."""
    return FEMesh(
        nodes=np.asarray(nodes, dtype=np.float64),
        elements=np.array([[0, 1, 2]]),
        element_type=ElementType.TRI3,
    )


def test_order_weld_line_nodes_shuffled_line():
    """Shuffled ids along a straight line come back spatially ordered."""
    nodes = np.zeros((6, 3))
    nodes[:, 0] = np.arange(6) * 10.0
    mesh = _line_mesh(nodes)

    ordered = order_weld_line_nodes(mesh, np.array([3, 0, 5, 1, 4, 2]))

    # The principal-axis sign is arbitrary, so either traversal is valid
    assert ordered.tolist() in ([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0])


def test_order_weld_line_nodes_diagonal_line():
    """PCA ordering works for a line not aligned with any coordinate axis."""
    t = np.array([30.0, 0.0, 50.0, 10.0, 60.0, 20.0, 40.0])
    nodes = np.zeros((len(t), 3))
    nodes[:, 0] = t
    nodes[:, 1] = t                     # 45-degree line in the xy-plane
    nodes[:, 2] = 0.05 * np.sin(t)      # small out-of-plane jitter
    mesh = _line_mesh(nodes)

    ordered = order_weld_line_nodes(mesh, np.arange(len(t)))

    diffs = np.diff(t[ordered])
    assert np.all(diffs > 0) or np.all(diffs < 0)


def test_order_weld_line_nodes_single_node():
    """A single node is returned unchanged."""
    mesh = _line_mesh(np.zeros((3, 3)))
    ordered = order_weld_line_nodes(mesh, np.array([2]))
    assert ordered.tolist() == [2]


# ---------------------------------------------------------------------------
# Weld tangent estimation
# ---------------------------------------------------------------------------

def test_tangent_estimation_on_unsorted_node_ids():
    """The tangent must use the node's position in the given (spatially
    ordered) array, not assume the ids are numerically sorted."""
    nodes = np.zeros((10, 3))
    # Spatial order along the bent line: 5 -> 2 -> 9
    nodes[5] = [0.0, 0.0, 0.0]
    nodes[2] = [10.0, 0.0, 0.0]
    nodes[9] = [10.0, 10.0, 0.0]
    mesh = FEMesh(
        nodes=nodes,
        elements=np.array([[5, 2, 9]]),
        element_type=ElementType.TRI3,
    )
    weld_line = WeldLineDefinition(
        name="test",
        node_ids=np.array([5, 2, 9]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 0.0, 1.0]),
    )

    # Node 2 is the middle of the array: tangent = chord between nodes 5 and 9
    tangent = _estimate_weld_tangent(mesh, weld_line, 2)
    expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(tangent, expected)


# ---------------------------------------------------------------------------
# Surface direction sign selection
# ---------------------------------------------------------------------------

def _asymmetric_weld_setup(with_weld_group):
    """Toe at the origin with a tagged weld region on the -x side.

    Reference nodes lie exactly at 0.4t and 1.0t (t = 10) on both sides of
    the toe; the weld side carries a wildly different stress so that the
    sampled side is unambiguous.
    """
    nodes = np.array([
        [0.0, 0.0, 0.0],     # weld toe
        [0.0, 5.0, 0.0],     # second weld-line node (tangent +y)
        [4.0, 0.0, 0.0],     # 0.4t on the plate side
        [10.0, 0.0, 0.0],    # 1.0t on the plate side
        [-4.0, 0.0, 0.0],    # weld side
        [-10.0, 0.0, 0.0],   # weld side
    ])
    elements = np.array([
        [0, 2, 3],   # plate side
        [0, 4, 5],   # weld side
    ])
    stress_vals = np.zeros((6, 6))
    stress_vals[:, 1] = [0.0, 0.0, 250.0, 100.0, 999.0, 999.0]

    mesh = FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
        physical_groups={"weld": np.array([1])} if with_weld_group else {},
        node_sets={"weld_toe": np.array([0, 1])},
    )
    results = FEAResults(mesh=mesh, stress=StressField(values=stress_vals))
    weld_line = WeldLineDefinition(
        name="test",
        node_ids=np.array([0, 1]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 0.0, 1.0]),
    )
    return results, weld_line


def test_surface_direction_points_away_from_weld_region():
    """With a "weld" element group, extrapolation samples the far side."""
    results, weld_line = _asymmetric_weld_setup(with_weld_group=True)

    hs = hotspot_stress_linear(results, weld_line, HotSpotType.TYPE_A)
    assert hs[0].reference_stresses == pytest.approx([250.0, 100.0])
    assert hs[0].hot_spot_stress == pytest.approx(1.67 * 250.0 - 0.67 * 100.0)
    assert hs[0].well_resolved

    hs_q = hotspot_stress_quadratic(results, weld_line)
    assert hs_q[0].reference_stresses[0] == pytest.approx(250.0)


def test_surface_direction_unchanged_without_weld_group():
    """Without a "weld" group the raw cross-product direction is kept."""
    results, weld_line = _asymmetric_weld_setup(with_weld_group=False)

    hs = hotspot_stress_linear(results, weld_line, HotSpotType.TYPE_A)
    # cross(normal, tangent) = cross(+z, +y) = -x: the weld-metal side
    assert hs[0].reference_stresses == pytest.approx([999.0, 999.0])


# ---------------------------------------------------------------------------
# well_resolved diagnostics
# ---------------------------------------------------------------------------

def test_well_resolved_false_when_reference_points_collapse():
    """On a very coarse mesh both reference points snap to the toe node."""
    mesh, results = _make_stress_gradient_mesh()

    weld_line = WeldLineDefinition(
        name="test",
        node_ids=np.array([0]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 0.0, 1.0]),
    )

    hs = hotspot_stress_linear(results, weld_line, HotSpotType.TYPE_A)
    assert hs[0].well_resolved is False


# ---------------------------------------------------------------------------
# 3D stations along an extruded toe line
# ---------------------------------------------------------------------------

def test_hotspot_3d_stations_along_toe_line(grid_solid_mesh):
    """Per-station extrapolation on a 3D toe line; max = governing station.

    The synthetic field decays away from the toe line (x = 20 on the top
    surface) and rises along the weld direction z, so every station yields
    a different hot-spot value and the z = 40 end governs.
    """
    n = grid_solid_mesh.n_nodes
    x = grid_solid_mesh.nodes[:, 0]
    z = grid_solid_mesh.nodes[:, 2]
    stress_vals = np.zeros((n, 6))
    stress_vals[:, 1] = 200.0 + 2.0 * z - 2.0 * x
    results = FEAResults(
        mesh=grid_solid_mesh,
        stress=StressField(values=stress_vals),
    )

    toe_ids = order_weld_line_nodes(
        grid_solid_mesh, grid_solid_mesh.node_sets["weld_toe_0"]
    )
    weld_line = WeldLineDefinition(
        name="weld_toe_0",
        node_ids=toe_ids,
        plate_thickness=20.0,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )

    hs = hotspot_stress_linear(results, weld_line, HotSpotType.TYPE_A)
    assert len(hs) == 5
    assert all(r.well_resolved for r in hs)

    # On the top surface sigma(x, z) = 200 + 2z - 2x; the reference points
    # at 0.4t and 1.0t snap to x = 30 and x = 40 (away from the tagged weld
    # region), so sigma_hs(z) = 1.67*(140 + 2z) - 0.67*(120 + 2z) = 153.4 + 2z.
    station_z = np.array([r.weld_toe_location[2] for r in hs])
    expected = 153.4 + 2.0 * station_z
    np.testing.assert_allclose(
        [r.hot_spot_stress for r in hs], expected, rtol=1e-9
    )

    governing = max(hs, key=lambda r: r.hot_spot_stress)
    assert governing.weld_toe_location[2] == pytest.approx(40.0)
    assert max(r.hot_spot_stress for r in hs) == pytest.approx(
        governing.hot_spot_stress
    )
    assert governing.hot_spot_stress == pytest.approx(153.4 + 80.0)


def test_well_resolved_true_on_refined_grid(grid_plate_mesh):
    """Reference points land near distinct nodes on the structured grid."""
    n = grid_plate_mesh.n_nodes
    stress_vals = np.zeros((n, 6))
    stress_vals[:, 1] = 100.0 + grid_plate_mesh.nodes[:, 0]
    results = FEAResults(
        mesh=grid_plate_mesh,
        stress=StressField(values=stress_vals),
    )

    # Toe line: the x = 20 column (grid nodes are numbered j*5 + i)
    toe_ids = np.array([1 * 5 + 2, 2 * 5 + 2, 3 * 5 + 2])
    weld_line = WeldLineDefinition(
        name="test",
        node_ids=toe_ids,
        plate_thickness=20.0,
        normal_direction=np.array([0.0, 0.0, 1.0]),
    )

    hs = hotspot_stress_linear(results, weld_line, HotSpotType.TYPE_A)
    assert len(hs) == 3
    assert all(r.well_resolved for r in hs)
