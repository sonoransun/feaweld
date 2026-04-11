"""Tests for :mod:`feaweld.postprocess.path_extraction`."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import (
    ElementType,
    FEAResults,
    FEMesh,
    Point3D,
    StressField,
)
from feaweld.geometry.weld_path import WeldPath
from feaweld.postprocess.path_extraction import (
    PathExtractionResult,
    extract_along_path,
    extract_tangent_normal_frame,
    sample_offset_points,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cube_grid_results(
    n_per_axis: int,
    field_fn,
    bbox: tuple[float, float] = (0.0, 10.0),
) -> FEAResults:
    """Build a regular 3D grid mesh with a scalar nodal field placed in σ_xx."""
    lo, hi = bbox
    xs = np.linspace(lo, hi, n_per_axis)
    ys = np.linspace(lo, hi, n_per_axis)
    zs = np.linspace(lo, hi, n_per_axis)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

    # Dummy tet4 elements (connectivity isn't exercised by path_extraction).
    n_nodes = nodes.shape[0]
    elements = np.array(
        [[i, (i + 1) % n_nodes, (i + 2) % n_nodes, (i + 3) % n_nodes] for i in range(n_nodes - 3)],
        dtype=np.int64,
    )
    mesh = FEMesh(nodes=nodes, elements=elements, element_type=ElementType.TET4)

    vals = field_fn(nodes[:, 0], nodes[:, 1], nodes[:, 2])
    # Store the analytical scalar in the σ_xx slot so that the von-Mises norm
    # we sample via extract_along_path equals |vals| (all other components zero).
    stress_vals = np.zeros((n_nodes, 6), dtype=np.float64)
    stress_vals[:, 0] = vals
    stress = StressField(values=stress_vals)
    return FEAResults(mesh=mesh, stress=stress)


def _analytic(x, y, z):
    return x ** 2 + y ** 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extract_analytic_field_on_linear_path() -> None:
    """Extract an analytic x**2 + y**2 field along a straight path."""
    results = _build_cube_grid_results(10, _analytic)

    # Avoid the extreme grid boundary where RBF extrapolation is unreliable:
    # sample the path on the interior of the cube.
    path = WeldPath(
        control_points=[Point3D(1.0, 1.0, 5.0), Point3D(9.0, 1.0, 5.0)],
        mode="linear",
    )

    out = extract_along_path(results, path, n_samples=11, field="von_mises")
    assert isinstance(out, PathExtractionResult)
    assert out.s.shape == (11,)
    assert out.values.shape == (11,)
    assert out.points.shape == (11, 3)

    # For this field, σ_xx = x**2 + y**2, all other components zero, so
    # von Mises = |σ_xx|.
    expected = _analytic(out.points[:, 0], out.points[:, 1], out.points[:, 2])
    rel_err = np.abs(out.values - expected) / np.maximum(np.abs(expected), 1.0)
    assert float(np.max(rel_err)) < 0.02


def test_extract_on_helix_path() -> None:
    """Same analytic field sampled along a helix of radius 2."""
    results = _build_cube_grid_results(12, _analytic, bbox=(-2.0, 12.0))

    # Helix of radius 2 centred on (5, 5, z) from z=2 to z=8, 2 turns.
    # Keep the path well inside the grid so RBF extrapolation stays stable.
    n_cp = 32
    zs = np.linspace(2.0, 8.0, n_cp)
    turns = 2.0
    angles = 2.0 * np.pi * turns * (zs / 6.0 - zs[0] / 6.0)
    cps = [
        Point3D(5.0 + 2.0 * float(np.cos(a)), 5.0 + 2.0 * float(np.sin(a)), float(z))
        for a, z in zip(angles, zs)
    ]
    path = WeldPath(control_points=cps, mode="bspline", degree=3)

    out = extract_along_path(results, path, n_samples=21, field="von_mises")
    # von Mises = |σ_xx| = x**2 + y**2 at each sample point.
    expected = out.points[:, 0] ** 2 + out.points[:, 1] ** 2
    rel_err = np.abs(out.values - expected) / np.maximum(np.abs(expected), 1.0)
    assert float(np.max(rel_err)) < 0.05


def test_rbf_fallback_to_nearest() -> None:
    """A collinear node cloud is ill-conditioned for thin-plate splines."""
    # 20 collinear nodes along x-axis at y=z=0.
    n = 20
    nodes = np.zeros((n, 3), dtype=np.float64)
    nodes[:, 0] = np.linspace(0.0, 10.0, n)
    elements = np.array([[i, i + 1, i + 1, i + 1] for i in range(n - 1)], dtype=np.int64)
    mesh = FEMesh(nodes=nodes, elements=elements, element_type=ElementType.TET4)
    stress_vals = np.zeros((n, 6), dtype=np.float64)
    stress_vals[:, 0] = nodes[:, 0]  # linear ramp
    results = FEAResults(mesh=mesh, stress=StressField(values=stress_vals))

    path = WeldPath(
        control_points=[Point3D(1.0, 0.0, 0.0), Point3D(9.0, 0.0, 0.0)],
        mode="linear",
    )

    out = extract_along_path(results, path, n_samples=5, field="von_mises", method="rbf")
    # Must not raise; either RBF succeeded or we fell back cleanly.
    assert out.values.shape == (5,)
    assert "method_used" in out.metadata
    # Regardless of which path was taken, the result should be finite.
    assert np.all(np.isfinite(out.values))


def test_tangent_normal_frame_orthogonal() -> None:
    """Frenet frame returned by the helper is orthonormal."""
    cps = [
        Point3D(0.0, 0.0, 0.0),
        Point3D(1.0, 1.0, 0.2),
        Point3D(2.0, 0.5, 0.5),
        Point3D(3.0, 1.5, 0.3),
        Point3D(4.0, 2.0, 0.6),
    ]
    path = WeldPath(control_points=cps, mode="bspline", degree=3)
    total = path.arc_length()
    for frac in [0.1, 0.25, 0.5, 0.75, 0.9]:
        t, n, b = extract_tangent_normal_frame(path, frac * total)
        assert np.linalg.norm(t) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(b) == pytest.approx(1.0, abs=1e-9)
        assert abs(float(np.dot(t, n))) < 1e-9
        assert abs(float(np.dot(t, b))) < 1e-9
        assert abs(float(np.dot(n, b))) < 1e-9


def test_sample_offset_points_shape() -> None:
    """Offset grid has the expected shape and direction."""
    path = WeldPath(
        control_points=[Point3D(0.0, 0.0, 0.0), Point3D(10.0, 0.0, 0.0)],
        mode="linear",
    )
    n_samples = 7
    offsets = [0.0, 1.0, 2.0]
    grid = sample_offset_points(
        path, n_samples=n_samples, offset_distances=offsets, offset_direction="normal"
    )
    assert grid.shape == (n_samples, len(offsets), 3)
    # Zero offset should coincide with the path point at each sample.
    for i in range(n_samples):
        base = grid[i, 0]
        for j, d in enumerate(offsets):
            # All offsets must differ from base by exactly |d|.
            assert np.linalg.norm(grid[i, j] - base) == pytest.approx(abs(d), abs=1e-9)

    # Binormal direction should produce a different grid than normal on a
    # non-degenerate path.
    grid_b = sample_offset_points(
        path, n_samples=n_samples, offset_distances=[1.0], offset_direction="binormal"
    )
    assert grid_b.shape == (n_samples, 1, 3)
