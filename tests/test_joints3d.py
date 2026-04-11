"""Tests for 3D solid weld joint builders (Track D2)."""

from __future__ import annotations

import pytest

gmsh = pytest.importorskip("gmsh")
pytestmark = pytest.mark.requires_gmsh

from feaweld.geometry.groove import VGroove
from feaweld.geometry.joints3d import (
    VolumetricButtJoint,
    VolumetricFilletTJoint,
)


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


# ---------------------------------------------------------------------------
# VolumetricButtJoint
# ---------------------------------------------------------------------------


def test_volumetric_butt_straight_path_builds(gmsh_session):
    gmsh.model.add("test_butt_build")
    groove = VGroove(plate_thickness=10.0, root_gap=2.0, angle=60.0, root_face=2.0)
    joint = VolumetricButtJoint(
        plate_width=30.0,
        plate_thickness=10.0,
        length=50.0,
        groove=groove,
    )
    result = joint.build()

    assert "volume_tags" in result
    assert "physical_groups" in result
    assert len(result["volume_tags"]) >= 1
    assert len(result["physical_groups"]) >= 1

    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    assert len(volumes) >= 3  # left plate + right plate + weld metal


def test_volumetric_butt_returns_weld_metal_group(gmsh_session):
    gmsh.model.add("test_butt_group")
    groove = VGroove(plate_thickness=8.0, root_gap=1.5, angle=60.0, root_face=1.5)
    joint = VolumetricButtJoint(
        plate_width=20.0,
        plate_thickness=8.0,
        length=30.0,
        groove=groove,
    )
    result = joint.build()

    pg = result["physical_groups"]
    assert "plate_left" in pg
    assert "plate_right" in pg
    assert "weld_metal" in pg

    toes = joint.get_weld_toe_points()
    assert len(toes) >= 2
    # Each toe should lie at y == plate_thickness
    for _, y, _ in toes:
        assert y == pytest.approx(8.0)


def test_volumetric_butt_curved_path_raises(gmsh_session):
    from feaweld.core.types import Point3D
    from feaweld.geometry.weld_path import WeldPath

    gmsh.model.add("test_butt_curved")
    groove = VGroove(plate_thickness=10.0)
    path = WeldPath(
        control_points=[
            Point3D(0.0, 0.0, 0.0),
            Point3D(10.0, 0.0, 5.0),
            Point3D(20.0, 0.0, 0.0),
        ],
        mode="bspline",
    )
    joint = VolumetricButtJoint(
        plate_width=30.0,
        plate_thickness=10.0,
        length=50.0,
        groove=groove,
        path=path,
    )
    with pytest.raises(NotImplementedError):
        joint.build()


# ---------------------------------------------------------------------------
# VolumetricFilletTJoint
# ---------------------------------------------------------------------------


def test_volumetric_fillet_t_builds(gmsh_session):
    gmsh.model.add("test_fillet_t_build")
    joint = VolumetricFilletTJoint(
        base_width=40.0,
        base_thickness=8.0,
        web_height=30.0,
        web_thickness=6.0,
        weld_leg_size=5.0,
        length=50.0,
    )
    result = joint.build()

    assert "volume_tags" in result
    assert "physical_groups" in result
    assert len(result["volume_tags"]) >= 1
    assert len(result["physical_groups"]) >= 1

    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    # base + web + 2 welds = 4 (may be more if fragment splits)
    assert len(volumes) >= 4


def test_volumetric_fillet_t_toe_points_count(gmsh_session):
    gmsh.model.add("test_fillet_t_toes")
    joint = VolumetricFilletTJoint(
        base_width=40.0,
        base_thickness=8.0,
        web_height=30.0,
        web_thickness=6.0,
        weld_leg_size=5.0,
        length=50.0,
    )
    result = joint.build()
    assert "base_plate" in result["physical_groups"]
    assert "web" in result["physical_groups"]
    assert "weld_left" in result["physical_groups"]
    assert "weld_right" in result["physical_groups"]

    toes = joint.get_weld_toe_points()
    # 4 toe cross-section locations * 2 ends (z=0 and z=length) = 8
    assert len(toes) == 8
