"""Tests for volumetric (Monte Carlo) SED post-processing (Track F5)."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import ElementType, FEAResults, FEMesh, Point3D, StressField
from feaweld.defects.types import PoreDefect
from feaweld.postprocess.volumetric_sed import (
    VolumetricSEDResult,
    averaged_sed_over_volume,
    cylindrical_control_volume,
    defect_wrapping_volume,
    ellipsoidal_control_volume,
    spherical_control_volume,
)


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def _uniform_cube_results(
    sigma_vm: float,
    n_per_side: int = 8,
    half_extent: float = 5.0,
) -> FEAResults:
    """Build a structured TET4 cube mesh with uniform σ_yy = sigma_vm on every node."""
    xs = np.linspace(-half_extent, half_extent, n_per_side)
    nodes = np.array(
        [[x, y, z] for z in xs for y in xs for x in xs], dtype=np.float64
    )
    # A single fake tet connectivity — the nodes carry the SED field for
    # this test; we just need a valid TET4 mesh object.
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    mesh = FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TET4,
    )

    stress_vals = np.zeros((mesh.n_nodes, 6))
    stress_vals[:, 1] = sigma_vm  # σ_yy uniform -> σ_vm = |σ_yy|
    return FEAResults(
        mesh=mesh,
        stress=StressField(values=stress_vals),
    )


# ---------------------------------------------------------------------------
# Predicate factory tests
# ---------------------------------------------------------------------------


def test_cylindrical_predicate():
    """Points inside / outside a unit cylinder should return correct counts."""
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    pred = cylindrical_control_volume(origin, direction, radius=1.0, height=2.0)

    pts = np.array(
        [
            [0.0, 0.0, 1.0],    # inside (centre)
            [0.5, 0.5, 1.0],    # inside (r = 0.707)
            [0.9, 0.0, 0.1],    # inside, near bottom
            [1.1, 0.0, 1.0],    # outside radially
            [0.0, 0.0, 2.5],    # outside above
            [0.0, 0.0, -0.5],   # outside below
            [0.7, 0.7, 0.0],    # on the base disk (r ~ 0.99)
        ]
    )
    mask = pred(pts)
    expected = np.array([True, True, True, False, False, False, True])
    np.testing.assert_array_equal(mask, expected)


def test_ellipsoidal_predicate():
    """Points inside / outside an axis-aligned ellipsoid."""
    center = np.array([1.0, 2.0, 3.0])
    semi_axes = np.array([2.0, 1.0, 0.5])
    pred = ellipsoidal_control_volume(center, semi_axes)

    pts = np.array(
        [
            [1.0, 2.0, 3.0],    # center
            [2.9, 2.0, 3.0],    # just inside x (rel = 0.95)
            [3.1, 2.0, 3.0],    # just outside x (rel = 1.05)
            [1.0, 2.9, 3.0],    # just inside y
            [1.0, 3.1, 3.0],    # just outside y
            [1.0, 2.0, 3.4],    # just inside z
            [1.0, 2.0, 3.6],    # just outside z
        ]
    )
    mask = pred(pts)
    expected = np.array([True, True, False, True, False, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_spherical_predicate():
    center = np.array([0.0, 0.0, 0.0])
    pred = spherical_control_volume(center, radius=2.0)
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # sqrt(3) ~= 1.73 -> inside
            [2.0, 0.0, 0.0],  # on boundary -> inside
            [3.0, 0.0, 0.0],  # outside
        ]
    )
    mask = pred(pts)
    np.testing.assert_array_equal(mask, np.array([True, True, True, False]))


# ---------------------------------------------------------------------------
# Monte Carlo averaged SED
# ---------------------------------------------------------------------------


def test_averaged_sed_on_uniform_stress_field():
    """Uniform σ_vm = 100 MPa, spherical control volume r = 2 -> SED = 100² / (2E)."""
    sigma_vm = 100.0
    E = 210_000.0
    results = _uniform_cube_results(sigma_vm=sigma_vm, n_per_side=10, half_extent=5.0)

    center = np.array([0.0, 0.0, 0.0])
    radius = 2.0
    pred = spherical_control_volume(center, radius)
    bbox = (center - radius, center + radius)

    result = averaged_sed_over_volume(
        results,
        volume_predicate=pred,
        bounding_box=bbox,
        n_samples=2000,
        seed=123,
        E=E,
    )

    assert isinstance(result, VolumetricSEDResult)
    expected = sigma_vm**2 / (2.0 * E)
    assert result.sed_average == pytest.approx(expected, rel=0.10)

    # Monte Carlo volume estimate should be close to (4/3) π r^3.
    expected_volume = (4.0 / 3.0) * np.pi * radius**3
    assert result.volume == pytest.approx(expected_volume, rel=0.10)
    assert result.n_samples > 0


def test_mc_std_decreases_with_more_samples():
    """More Monte Carlo samples -> smaller standard error of the mean."""
    sigma_vm = 100.0
    E = 210_000.0
    # Use a gradient stress field so sed has real variance (otherwise
    # σ_vm is constant and the MC std is zero).
    n_side = 12
    xs = np.linspace(-5.0, 5.0, n_side)
    nodes = np.array(
        [[x, y, z] for z in xs for y in xs for x in xs], dtype=np.float64
    )
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    mesh = FEMesh(nodes=nodes, elements=elements, element_type=ElementType.TET4)

    stress_vals = np.zeros((mesh.n_nodes, 6))
    stress_vals[:, 1] = sigma_vm + 10.0 * nodes[:, 0]  # varies with x
    results = FEAResults(mesh=mesh, stress=StressField(values=stress_vals))

    center = np.array([0.0, 0.0, 0.0])
    radius = 2.0
    pred = spherical_control_volume(center, radius)
    bbox = (center - radius, center + radius)

    r200 = averaged_sed_over_volume(
        results, pred, bbox, n_samples=200, seed=0, E=E
    )
    r2000 = averaged_sed_over_volume(
        results, pred, bbox, n_samples=2000, seed=0, E=E
    )

    assert r200.sed_std > 0.0
    assert r2000.sed_std < r200.sed_std


def test_defect_wrapping_volume_covers_defect():
    """A PoreDefect's wrapping volume must contain the defect center."""
    pore = PoreDefect(center=Point3D(1.0, 2.0, 3.0), diameter=0.5)
    pred, bbox = defect_wrapping_volume(pore, padding=1.0)

    center = np.array([[1.0, 2.0, 3.0]])
    mask = pred(center)
    assert bool(mask[0])

    # Bounding box should contain the center too.
    bmin, bmax = bbox
    assert np.all(bmin <= center[0])
    assert np.all(center[0] <= bmax)


def test_averaged_sed_with_strain_fallback_path():
    """When strain is populated, SED uses full σ:ε contraction."""
    E = 210_000.0
    sigma_vm = 150.0

    # Build a simple uniform cube with consistent strain (elastic uniaxial).
    results = _uniform_cube_results(sigma_vm=sigma_vm, n_per_side=8, half_extent=3.0)
    n_nodes = results.mesh.n_nodes
    strain_vals = np.zeros((n_nodes, 6))
    # Uniaxial plane-stress-like: ε_yy = σ_yy / E
    strain_vals[:, 1] = sigma_vm / E
    results = FEAResults(
        mesh=results.mesh,
        stress=results.stress,
        strain=strain_vals,
    )

    center = np.array([0.0, 0.0, 0.0])
    pred = spherical_control_volume(center, radius=2.0)
    bbox = (center - 2.0, center + 2.0)

    res = averaged_sed_over_volume(
        results, pred, bbox, n_samples=1000, seed=0, E=E
    )
    expected = 0.5 * sigma_vm * (sigma_vm / E)
    assert res.sed_average == pytest.approx(expected, rel=0.10)
    assert res.metadata["fallback"] == "full_voigt"
