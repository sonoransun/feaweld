"""Tests for the :mod:`feaweld.geometry.weld_path` module."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import Point3D
from feaweld.geometry.weld_path import WeldPath


def _pt(x: float, y: float, z: float = 0.0) -> Point3D:
    return Point3D(x, y, z)


def test_linear_path_arc_length() -> None:
    pts = [_pt(0.0, 0.0), _pt(3.0, 0.0), _pt(3.0, 4.0)]  # 3 + 4 = 7
    path = WeldPath(pts, mode="linear")
    assert path.arc_length() == pytest.approx(7.0, abs=1e-10)


def test_bspline_path_smoothness() -> None:
    pts = [
        _pt(0.0, 0.0),
        _pt(1.0, 1.0),
        _pt(2.0, 0.5),
        _pt(3.0, -0.5),
        _pt(4.0, 0.25),
    ]
    path = WeldPath(pts, mode="bspline", degree=3)
    us = np.linspace(0.0, 1.0, 200)
    tangents = path.tangent(us)
    # Tangent changes smoothly: consecutive angles differ by a small amount
    cos_diffs = np.einsum("ij,ij->i", tangents[:-1], tangents[1:])
    cos_diffs = np.clip(cos_diffs, -1.0, 1.0)
    angle_steps = np.arccos(cos_diffs)
    assert np.max(angle_steps) < 0.2  # < ~11 degrees between neighbors


def test_circle_approximation() -> None:
    n = 8
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    # Duplicate first point at the end so path closes (unit circle in XY)
    pts = [_pt(float(np.cos(a)), float(np.sin(a)), 0.0) for a in angles]
    path = WeldPath(pts, mode="bspline", degree=3)
    assert path.arc_length() == pytest.approx(2.0 * np.pi, rel=0.02)


def test_evaluate_s_and_u_consistent() -> None:
    pts = [_pt(0.0, 0.0), _pt(10.0, 0.0)]
    path = WeldPath(pts, mode="linear")
    mid_s = path.evaluate_s(path.arc_length() / 2.0)
    mid_u = path.evaluate_u(0.5)
    np.testing.assert_allclose(np.asarray(mid_s), np.asarray(mid_u), atol=1e-9)


def test_sample_shape() -> None:
    pts = [_pt(0.0, 0.0, 0.0), _pt(1.0, 2.0, 3.0), _pt(4.0, 0.0, 1.0)]
    path = WeldPath(pts, mode="bspline", degree=2)
    arr = path.sample(10)
    assert arr.shape == (10, 3)


def test_frenet_frame_orthogonal() -> None:
    pts = [
        _pt(0.0, 0.0, 0.0),
        _pt(1.0, 1.0, 0.1),
        _pt(2.0, 0.5, 0.3),
        _pt(3.0, 2.0, 0.2),
        _pt(4.0, 1.5, 0.4),
    ]
    path = WeldPath(pts, mode="bspline", degree=3)
    for u in [0.15, 0.3, 0.5, 0.7, 0.85]:
        t, n, b = path.frenet_frame(u)
        assert np.linalg.norm(t) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-9)
        assert np.linalg.norm(b) == pytest.approx(1.0, abs=1e-9)
        assert abs(float(np.dot(t, n))) < 1e-9
        assert abs(float(np.dot(t, b))) < 1e-9
        assert abs(float(np.dot(n, b))) < 1e-9


def test_catmull_rom_passes_through_control_points() -> None:
    pts = [_pt(0.0, 0.0), _pt(1.0, 2.0), _pt(3.0, 1.0), _pt(4.0, 4.0)]
    path = WeldPath(pts, mode="catmull_rom")
    # Control points lie at parameter values k / (n - 1)
    for i, cp in enumerate(pts):
        u = i / (len(pts) - 1)
        got = path.evaluate_u(u)
        np.testing.assert_allclose(got, cp.to_array(), atol=1e-10)


def test_to_gmsh_wire_skipped_without_gmsh() -> None:
    gmsh = pytest.importorskip("gmsh")
    pts = [_pt(0.0, 0.0, 0.0), _pt(1.0, 0.0, 0.0), _pt(2.0, 1.0, 0.0)]
    path = WeldPath(pts, mode="bspline", degree=2)
    if not gmsh.is_initialized():
        gmsh.initialize()
    try:
        gmsh.model.add("wp_test")
        wire_tag = path.to_gmsh_wire(gmsh.model)
        assert wire_tag > 0
    finally:
        gmsh.finalize()
