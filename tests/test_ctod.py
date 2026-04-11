"""Tests for CTOD extraction methods (Track F3)."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import ElementType, FEAResults, FEMesh, StressField
from feaweld.fracture import (
    CTODResult,
    ctod_90_degree,
    ctod_displacement_extrapolation,
)


def _structured_tri_plate(
    lx: float,
    ly: float,
    nx: int,
    ny: int,
    origin: tuple[float, float] = (0.0, 0.0),
) -> FEMesh:
    x0, y0 = origin
    xs = np.linspace(x0, x0 + lx, nx + 1)
    ys = np.linspace(y0, y0 + ly, ny + 1)
    nodes = np.array(
        [[x, y, 0.0] for y in ys for x in xs], dtype=np.float64
    )

    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    elements: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            n0 = nid(i, j)
            n1 = nid(i + 1, j)
            n2 = nid(i + 1, j + 1)
            n3 = nid(i, j + 1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

    return FEMesh(
        nodes=nodes,
        elements=np.array(elements, dtype=np.int64),
        element_type=ElementType.TRI3,
    )


def _bare_results(mesh: FEMesh, disp: np.ndarray) -> FEAResults:
    n = mesh.n_nodes
    stress_values = np.zeros((n, 6))
    return FEAResults(
        mesh=mesh,
        displacement=disp,
        stress=StressField(values=stress_values),
    )


def test_ctod_displacement_extrapolation_linear_field():
    mesh = _structured_tri_plate(lx=10.0, ly=10.0, nx=20, ny=20, origin=(-5.0, -5.0))
    delta = 0.02
    nodes_xy = mesh.nodes[:, :2]

    disp = np.zeros((mesh.n_nodes, 3))
    # Uniform +delta/2 above y=0, -delta/2 below.
    disp[nodes_xy[:, 1] > 0, 1] = +0.5 * delta
    disp[nodes_xy[:, 1] < 0, 1] = -0.5 * delta

    results = _bare_results(mesh, disp)

    tip = np.array([3.0, 0.0])
    crack_axis = np.array([1.0, 0.0])  # points toward tip from mouth

    for offset in (1.0, 2.0, 3.0):
        res = ctod_displacement_extrapolation(
            results, crack_tip=tip, crack_axis=crack_axis, offset_distance=offset
        )
        assert isinstance(res, CTODResult)
        assert res.ctod == pytest.approx(delta, rel=1e-6)


def test_ctod_90_degree_matches_extrapolation_for_linear():
    """45-degree-opening crack -> 90-degree and extrapolation agree analytically.

    If the crack flanks open at a constant 45-degree angle (i.e.
    |u_y(x)| = -x for x<0), then the 45-degree intercept with any point
    on the flank sits at distance r* and CTOD = 2*|u_y(x*)| = 2*|x*|.
    For a fixed offset d, the displacement-extrapolation CTOD is 2*d.
    We test that both are nonzero and within 20% of each other when the
    extrapolation offset is chosen to coincide with the 45-degree
    intercept point (a distance = sqrt(2) from the tip here).
    """
    mesh = _structured_tri_plate(lx=10.0, ly=10.0, nx=40, ny=40, origin=(-5.0, -5.0))
    # 45-degree flank opening — unit slope so the 45-degree intercept
    # exists away from the tip and is easy to verify.
    slope = 1.0

    nodes_xy = mesh.nodes[:, :2]
    disp = np.zeros((mesh.n_nodes, 3))
    behind = nodes_xy[:, 0] < 0.0
    upper = behind & (nodes_xy[:, 1] > 0.0)
    lower = behind & (nodes_xy[:, 1] < 0.0)
    disp[upper, 1] = +slope * (-nodes_xy[upper, 0])
    disp[lower, 1] = -slope * (-nodes_xy[lower, 0])

    results = _bare_results(mesh, disp)
    tip = np.array([0.0, 0.0])
    crack_axis = np.array([1.0, 0.0])

    res_90 = ctod_90_degree(results, crack_tip=tip, crack_axis=crack_axis)
    assert res_90.ctod > 0.0

    # For the displacement extrapolation comparison, pick an offset so the
    # analytical opening is 2 * slope * offset.
    offset = 2.0
    res_ext = ctod_displacement_extrapolation(
        results, crack_tip=tip, crack_axis=crack_axis, offset_distance=offset
    )
    expected_ext = 2.0 * slope * offset
    assert res_ext.ctod == pytest.approx(expected_ext, rel=0.2)
    # Both methods should deliver a positive CTOD of comparable magnitude.
    assert res_90.ctod == pytest.approx(res_ext.ctod, rel=0.5)


def test_ctod_zero_for_closed_crack():
    mesh = _structured_tri_plate(lx=10.0, ly=10.0, nx=20, ny=20, origin=(-5.0, -5.0))
    delta = 0.02
    nodes_xy = mesh.nodes[:, :2]

    disp = np.zeros((mesh.n_nodes, 3))
    # Compressive: upper flank moves *down*, lower moves *up* — overlap.
    disp[nodes_xy[:, 1] > 0, 1] = -0.5 * delta
    disp[nodes_xy[:, 1] < 0, 1] = +0.5 * delta

    results = _bare_results(mesh, disp)
    tip = np.array([3.0, 0.0])
    crack_axis = np.array([1.0, 0.0])

    res = ctod_displacement_extrapolation(
        results, crack_tip=tip, crack_axis=crack_axis, offset_distance=2.0
    )
    assert res.ctod == pytest.approx(0.0, abs=1e-12)
