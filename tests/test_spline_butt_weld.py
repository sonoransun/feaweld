"""Tests for :class:`SplineButtWeld` (Track D4)."""

from __future__ import annotations

import math

import pytest

gmsh = pytest.importorskip("gmsh")
pytestmark = pytest.mark.requires_gmsh

from feaweld.core.types import Point3D
from feaweld.geometry.groove import VGroove
from feaweld.geometry.spline_joints import SplineButtWeld
from feaweld.geometry.weld_path import WeldPath


@pytest.fixture
def gmsh_session():
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    try:
        yield gmsh
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def test_straight_path_builds(gmsh_session):
    gmsh.model.add("test_spline_butt_straight")
    path = WeldPath(
        control_points=[
            Point3D(0.0, 0.0, 0.0),
            Point3D(5.0, 0.0, 0.0),
            Point3D(10.0, 0.0, 0.0),
            Point3D(15.0, 0.0, 0.0),
            Point3D(20.0, 0.0, 0.0),
        ],
        mode="bspline",
    )
    groove = VGroove(
        plate_thickness=10.0, root_gap=2.0, angle=60.0, root_face=2.0
    )
    weld = SplineButtWeld(
        plate_width=40.0,
        plate_thickness=10.0,
        path=path,
        groove=groove,
    )
    result = weld.build()

    assert "volume_tags" in result
    assert "physical_groups" in result
    assert len(result["volume_tags"]) >= 1
    assert "weld_metal" in result["physical_groups"]


def test_curved_path_builds(gmsh_session):
    gmsh.model.add("test_spline_butt_curved")
    # Gentle arc of radius ~20 mm spanning roughly a quarter circle.
    r = 20.0
    control_points = [
        Point3D(r * math.cos(theta), r * math.sin(theta), 0.0)
        for theta in [0.0, math.pi / 16, math.pi / 8, 3 * math.pi / 16, math.pi / 4]
    ]
    path = WeldPath(control_points=control_points, mode="bspline")
    groove = VGroove(
        plate_thickness=8.0, root_gap=1.5, angle=60.0, root_face=1.5
    )
    weld = SplineButtWeld(
        plate_width=30.0,
        plate_thickness=8.0,
        path=path,
        groove=groove,
    )
    result = weld.build()

    assert len(result["volume_tags"]) >= 1
    assert "weld_metal" in result["physical_groups"]


def test_too_sharp_curvature_raises(gmsh_session):
    gmsh.model.add("test_spline_butt_sharp")
    # Very tight circle of radius 2 mm
    r = 2.0
    control_points = [
        Point3D(r * math.cos(theta), r * math.sin(theta), 0.0)
        for theta in [
            0.0,
            math.pi / 6,
            math.pi / 3,
            math.pi / 2,
            2 * math.pi / 3,
            5 * math.pi / 6,
            math.pi,
        ]
    ]
    path = WeldPath(control_points=control_points, mode="bspline")
    # V-groove with half-width ~ 3 mm (angle 60, plate 8 -> top half ~ plate*tan(30))
    groove = VGroove(
        plate_thickness=8.0, root_gap=0.0, angle=60.0, root_face=0.0
    )
    weld = SplineButtWeld(
        plate_width=20.0,
        plate_thickness=8.0,
        path=path,
        groove=groove,
    )
    with pytest.raises(ValueError):
        weld.check_curvature()


def test_get_weld_toe_points_nonempty(gmsh_session):
    gmsh.model.add("test_spline_butt_toe_points")
    path = WeldPath(
        control_points=[
            Point3D(0.0, 0.0, 0.0),
            Point3D(5.0, 0.0, 0.0),
            Point3D(10.0, 0.0, 0.0),
            Point3D(15.0, 0.0, 0.0),
            Point3D(20.0, 0.0, 0.0),
        ],
        mode="bspline",
    )
    groove = VGroove(
        plate_thickness=10.0, root_gap=2.0, angle=60.0, root_face=2.0
    )
    weld = SplineButtWeld(
        plate_width=40.0,
        plate_thickness=10.0,
        path=path,
        groove=groove,
    )
    points = weld.get_weld_toe_points(n_samples=16)
    # At least n_samples points (two toes per station = 32 for V-groove).
    assert len(points) >= 16
    for p in points:
        assert len(p) == 3
