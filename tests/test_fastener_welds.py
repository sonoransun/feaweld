"""Tests for fastener weld builders (Track D3)."""

from __future__ import annotations

import pytest

gmsh = pytest.importorskip("gmsh")
pytestmark = pytest.mark.requires_gmsh

from feaweld.geometry.fastener_welds import (
    PlugWeld,
    SlotWeld,
    SpotWeld,
    StudWeld,
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
# PlugWeld
# ---------------------------------------------------------------------------


def test_plug_weld_builds(gmsh_session):
    gmsh.model.add("test_plug_build")
    weld = PlugWeld(hole_diameter=6.0, plate_thickness=5.0)
    result = weld.build()

    assert "volume_tags" in result
    assert "physical_groups" in result
    assert len(result["volume_tags"]) >= 1
    assert "plate" in result["physical_groups"]
    assert "weld" in result["physical_groups"]


def test_plug_weld_volume_count(gmsh_session):
    gmsh.model.add("test_plug_volumes")
    weld = PlugWeld(hole_diameter=8.0, plate_thickness=10.0, fill_depth=6.0)
    weld.build()
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(dim=3)
    # plate with hole + weld cylinder
    assert len(vols) >= 2


# ---------------------------------------------------------------------------
# SlotWeld
# ---------------------------------------------------------------------------


def test_slot_weld_builds(gmsh_session):
    gmsh.model.add("test_slot_build")
    weld = SlotWeld(slot_length=20.0, slot_width=6.0, plate_thickness=5.0)
    result = weld.build()

    assert "volume_tags" in result
    assert "physical_groups" in result
    assert len(result["volume_tags"]) >= 1
    assert "plate" in result["physical_groups"]
    assert "weld" in result["physical_groups"]


def test_slot_weld_volume_count(gmsh_session):
    gmsh.model.add("test_slot_volumes")
    weld = SlotWeld(
        slot_length=15.0, slot_width=5.0, plate_thickness=4.0, fill_depth=3.0
    )
    weld.build()
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(dim=3)
    assert len(vols) >= 2


# ---------------------------------------------------------------------------
# StudWeld
# ---------------------------------------------------------------------------


def test_stud_weld_builds(gmsh_session):
    gmsh.model.add("test_stud_build")
    weld = StudWeld(stud_diameter=10.0, stud_height=25.0, plate_thickness=8.0)
    result = weld.build()

    assert "volume_tags" in result
    assert "physical_groups" in result
    assert len(result["volume_tags"]) >= 1
    assert "plate" in result["physical_groups"]
    assert "weld" in result["physical_groups"]


def test_stud_weld_volume_count(gmsh_session):
    gmsh.model.add("test_stud_volumes")
    weld = StudWeld(stud_diameter=6.0, stud_height=15.0, plate_thickness=5.0)
    weld.build()
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(dim=3)
    # plate + stud cylinder
    assert len(vols) >= 2


# ---------------------------------------------------------------------------
# SpotWeld
# ---------------------------------------------------------------------------


def test_spot_weld_builds(gmsh_session):
    gmsh.model.add("test_spot_build")
    weld = SpotWeld(
        spot_diameter=8.0,
        plate_thickness_1=3.0,
        plate_thickness_2=3.0,
    )
    result = weld.build()

    assert "volume_tags" in result
    assert "physical_groups" in result
    assert len(result["volume_tags"]) >= 1
    assert "plate_1" in result["physical_groups"]
    assert "plate_2" in result["physical_groups"]
    assert "weld" in result["physical_groups"]


def test_spot_weld_volume_count(gmsh_session):
    gmsh.model.add("test_spot_volumes")
    weld = SpotWeld(
        spot_diameter=6.0,
        plate_thickness_1=2.0,
        plate_thickness_2=2.5,
    )
    weld.build()
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(dim=3)
    # two plates + one nugget
    assert len(vols) >= 3
