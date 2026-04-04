"""Tests for feaweld.geometry.joints — weld joint geometry builders."""

from __future__ import annotations

import pytest

# Guard: skip the entire module if gmsh is not importable
gmsh = pytest.importorskip("gmsh")

from feaweld.geometry.joints import (
    ButtWeld,
    CornerJoint,
    CruciformJoint,
    FilletTJoint,
    LapJoint,
)

requires_gmsh = pytest.mark.requires_gmsh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _gmsh_session():
    """Ensure a fresh Gmsh session for each test."""
    if gmsh.is_initialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # suppress output
    yield
    if gmsh.is_initialized():
        gmsh.finalize()


# ---------------------------------------------------------------------------
# FilletTJoint
# ---------------------------------------------------------------------------

@requires_gmsh
class TestFilletTJoint:
    def _make(self) -> FilletTJoint:
        return FilletTJoint(
            base_width=100.0,
            base_thickness=10.0,
            web_height=50.0,
            web_thickness=10.0,
            weld_leg_size=6.0,
        )

    def test_build_succeeds(self):
        joint = self._make()
        joint.build()  # should not raise

    def test_physical_groups_created(self):
        joint = self._make()
        joint.build()
        pg = joint.get_physical_groups()
        assert "base_plate" in pg
        assert "web" in pg
        assert "weld_left" in pg
        assert "weld_right" in pg

    def test_weld_toe_points(self):
        joint = self._make()
        toes = joint.get_weld_toe_points()
        assert len(toes) == 4
        for pt in toes:
            assert len(pt) == 3  # (x, y, z)

    def test_boundary_groups(self):
        joint = self._make()
        joint.build()
        pg = joint.get_physical_groups()
        assert "bottom" in pg
        assert "top" in pg


# ---------------------------------------------------------------------------
# ButtWeld
# ---------------------------------------------------------------------------

@requires_gmsh
class TestButtWeld:
    def _make(self) -> ButtWeld:
        return ButtWeld(
            plate_width=80.0,
            plate_thickness=12.0,
            groove_angle=60.0,
            root_gap=2.0,
        )

    def test_build_succeeds(self):
        self._make().build()

    def test_physical_groups(self):
        j = self._make()
        j.build()
        pg = j.get_physical_groups()
        for name in ("plate_left", "plate_right", "weld_metal"):
            assert name in pg, f"Missing physical group '{name}'"

    def test_toe_points(self):
        j = self._make()
        toes = j.get_weld_toe_points()
        assert len(toes) == 2


# ---------------------------------------------------------------------------
# LapJoint
# ---------------------------------------------------------------------------

@requires_gmsh
class TestLapJoint:
    def _make(self) -> LapJoint:
        return LapJoint(
            plate_thickness=8.0,
            overlap_length=40.0,
            weld_leg_size=5.0,
        )

    def test_build_succeeds(self):
        self._make().build()

    def test_physical_groups(self):
        j = self._make()
        j.build()
        pg = j.get_physical_groups()
        assert "plate_lower" in pg
        assert "plate_upper" in pg
        assert "weld" in pg


# ---------------------------------------------------------------------------
# CornerJoint
# ---------------------------------------------------------------------------

@requires_gmsh
class TestCornerJoint:
    def _make(self) -> CornerJoint:
        return CornerJoint(
            plate_thickness_h=10.0,
            plate_thickness_v=10.0,
            weld_leg_size=6.0,
        )

    def test_build_succeeds(self):
        self._make().build()

    def test_physical_groups(self):
        j = self._make()
        j.build()
        pg = j.get_physical_groups()
        assert "plate_horizontal" in pg
        assert "plate_vertical" in pg
        assert "weld" in pg


# ---------------------------------------------------------------------------
# CruciformJoint
# ---------------------------------------------------------------------------

@requires_gmsh
class TestCruciformJoint:
    def _make(self) -> CruciformJoint:
        return CruciformJoint(
            plate_thickness=10.0,
            web_thickness=10.0,
            weld_leg_size=6.0,
        )

    def test_build_succeeds(self):
        self._make().build()

    def test_physical_groups(self):
        j = self._make()
        j.build()
        pg = j.get_physical_groups()
        for name in ("base_plate", "web_upper", "web_lower",
                      "weld_upper_left", "weld_upper_right",
                      "weld_lower_left", "weld_lower_right"):
            assert name in pg, f"Missing '{name}'"

    def test_toe_points(self):
        j = self._make()
        toes = j.get_weld_toe_points()
        assert len(toes) == 8
