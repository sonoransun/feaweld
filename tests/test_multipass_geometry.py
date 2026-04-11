"""Tests for multi-pass weld bead geometry build (Track G)."""

from __future__ import annotations

import pytest

gmsh = pytest.importorskip("gmsh")
pytestmark = pytest.mark.requires_gmsh

from feaweld.core.types import Point3D, WeldPass, WeldSequence
from feaweld.geometry.groove import VGroove
from feaweld.geometry.multipass import build_multipass_joint
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


def test_build_multipass_joint_yields_pass_physical_groups(gmsh_session):
    """Three passes → three named physical groups pass_1..pass_3."""
    sequence = WeldSequence(
        passes=[
            WeldPass(order=1, pass_type="root", start_time=0.0, duration=10.0),
            WeldPass(order=2, pass_type="fill", start_time=11.0, duration=10.0),
            WeldPass(order=3, pass_type="cap", start_time=22.0, duration=10.0),
        ]
    )
    path = WeldPath(
        control_points=[
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.0, 0.0, 100.0),
        ],
        mode="linear",
    )
    # Three thin bead profiles stacked through the groove.
    profiles = [
        VGroove(plate_thickness=3.0, angle=60.0, root_face=1.0),
        VGroove(plate_thickness=3.0, angle=60.0, root_face=1.0),
        VGroove(plate_thickness=3.0, angle=60.0, root_face=1.0),
    ]

    result = build_multipass_joint(
        base_joint=None,
        sequence=sequence,
        path=path,
        per_pass_profiles=profiles,
    )

    assert "volume_tags" in result
    assert "physical_groups" in result
    assert len(result["volume_tags"]) == 3

    pg = result["physical_groups"]
    assert set(pg.keys()) == {"pass_1", "pass_2", "pass_3"}

    # Physical group tags should be distinct positive integers.
    tags = list(pg.values())
    assert len(set(tags)) == 3
    assert all(t > 0 for t in tags)
