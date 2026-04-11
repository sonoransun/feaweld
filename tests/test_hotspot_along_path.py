"""Tests for :func:`feaweld.postprocess.hotspot.hotspot_stress_along_path`."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import (
    ElementType,
    FEAResults,
    FEMesh,
    Point3D,
    StressField,
    WeldLineDefinition,
)
from feaweld.geometry.weld_path import WeldPath
from feaweld.postprocess.hotspot import (
    HotSpotType,
    hotspot_stress_along_path,
    hotspot_stress_linear,
)


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------

def _build_plate_with_y_gradient(
    t: float = 10.0, length: float = 60.0, width: float = 40.0, n: int = 15
) -> FEAResults:
    """Plate in the x-y plane (z=0). Stress gradient in y approaches the weld toe at y=0.

    σ_yy = 300 - 4*y    (decreases away from the toe)
    σ_xx = σ_zz = τ = 0

    Weld toe is the line (0..length, 0, 0); offsets into the plate surface are
    along +y (i.e. the `-normal` direction from the path's perspective once we
    orient the WeldPath so its principal normal points back toward -y).
    """
    xs = np.linspace(0.0, length, n)
    ys = np.linspace(0.0, width, n)
    # A single layer at z=0 suffices for the discrete hot-spot API, but the
    # RBF needs a 3-D point cloud to avoid being trivially planar.  Add a
    # second layer at z=1 (far outside the weld toe plane) — the analytical
    # stress is defined by y only so both layers agree.
    zs = np.array([0.0, 1.0])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

    # Dummy element connectivity — not used by the post-processors.
    n_nodes = nodes.shape[0]
    elements = np.array(
        [[i, min(i + 1, n_nodes - 1), min(i + 2, n_nodes - 1), min(i + 3, n_nodes - 1)]
         for i in range(n_nodes - 3)],
        dtype=np.int64,
    )
    mesh = FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TET4,
        node_sets={"weld_toe": np.array(
            [i for i, p in enumerate(nodes) if p[1] == 0.0 and p[2] == 0.0],
            dtype=np.int64,
        )},
    )

    # σ_yy = 300 - 4*y ; store in component 1 (σ_yy).
    stress_vals = np.zeros((n_nodes, 6), dtype=np.float64)
    stress_vals[:, 1] = 300.0 - 4.0 * nodes[:, 1]

    return FEAResults(mesh=mesh, stress=StressField(values=stress_vals))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_along_path_matches_discrete_api_on_straight_line() -> None:
    """The path-based API should agree with the discrete-node API on a straight weld."""
    t = 10.0
    results = _build_plate_with_y_gradient(t=t, length=60.0, width=40.0, n=17)

    # Weld toe line: all mesh nodes with y==0 and z==0, ordered by x.
    toe_ids = np.array(
        [i for i, p in enumerate(results.mesh.nodes) if p[1] == 0.0 and p[2] == 0.0],
        dtype=np.int64,
    )
    toe_ids = toe_ids[np.argsort(results.mesh.nodes[toe_ids][:, 0])]

    weld_line = WeldLineDefinition(
        name="toe",
        node_ids=toe_ids,
        plate_thickness=t,
        normal_direction=np.array([0.0, 0.0, 1.0]),  # plate normal = +z
    )

    discrete_results = hotspot_stress_linear(results, weld_line, HotSpotType.TYPE_A)
    # Discrete API finds reference points along `cross(normal, tangent)` which
    # for normal=+z and tangent=+x gives -y: but the discrete API applies +
    # offset, so reference points end up at y<0 (outside the plate).  In this
    # setup, we bracket by offsetting along +y instead.  To get comparable
    # reference values, we re-interpret the discrete API's choice: with the
    # stress field symmetric about y=0 (not in this problem), they'd match.
    # Here, since σ_yy only changes with y, both signs give the same ref
    # stresses because nearest-neighbour snaps to interior nodes.
    disc_mean = float(np.mean([r.hot_spot_stress for r in discrete_results]))

    # Path-based API — build a straight WeldPath through the same toe points.
    path = WeldPath(
        control_points=[Point3D(0.0, 0.0, 0.0), Point3D(60.0, 0.0, 0.0)],
        mode="linear",
    )
    path_results = hotspot_stress_along_path(
        results, path, plate_thickness=t, hotspot_type=HotSpotType.TYPE_A, n_samples=15
    )
    assert len(path_results) == 15
    path_mean = float(np.mean([r.hot_spot_stress for r in path_results]))

    # Check within 10% on this simple linear gradient case.
    assert path_mean > 0.0
    assert disc_mean > 0.0
    rel_err = abs(path_mean - disc_mean) / max(abs(disc_mean), 1e-12)
    assert rel_err < 0.10, f"path_mean={path_mean}, disc_mean={disc_mean}, rel_err={rel_err}"


def test_along_path_returns_n_samples_results() -> None:
    results = _build_plate_with_y_gradient()
    path = WeldPath(
        control_points=[Point3D(0.0, 0.0, 0.0), Point3D(60.0, 0.0, 0.0)],
        mode="linear",
    )
    out = hotspot_stress_along_path(
        results, path, plate_thickness=10.0, hotspot_type=HotSpotType.TYPE_A, n_samples=21
    )
    assert len(out) == 21
    for r in out:
        assert r.extrapolation_type == HotSpotType.TYPE_A
        assert len(r.reference_stresses) == 2
        assert len(r.reference_distances) == 2


def test_along_path_handles_curved_path_without_error() -> None:
    results = _build_plate_with_y_gradient(t=10.0, length=60.0, width=40.0, n=13)
    # A bspline path that curves in x-y but stays near the weld toe line.
    cps = [
        Point3D(5.0, 0.0, 0.0),
        Point3D(15.0, 1.0, 0.0),
        Point3D(30.0, -0.5, 0.0),
        Point3D(45.0, 0.8, 0.0),
        Point3D(55.0, 0.0, 0.0),
    ]
    path = WeldPath(control_points=cps, mode="bspline", degree=3)
    out = hotspot_stress_along_path(
        results,
        path,
        plate_thickness=10.0,
        hotspot_type=HotSpotType.TYPE_A,
        n_samples=12,
    )
    assert len(out) == 12
    # All results should be finite numbers.
    for r in out:
        assert np.isfinite(r.hot_spot_stress)
        assert all(np.isfinite(v) for v in r.reference_stresses)
