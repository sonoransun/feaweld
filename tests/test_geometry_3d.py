"""Tests for 3D (extruded) weld joint geometry builds."""

from __future__ import annotations

import numpy as np
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

LENGTH = 30.0
TOL = 1e-5

JOINTS = {
    "fillet_t": {
        "cls": FilletTJoint,
        "params": dict(base_width=60.0, base_thickness=10.0, web_height=30.0,
                       web_thickness=8.0, weld_leg_size=6.0),
        "regions": {"base_plate", "web", "weld_left", "weld_right"},
        "boundaries": ["bottom", "top"],
    },
    "butt": {
        "cls": ButtWeld,
        "params": dict(plate_width=40.0, plate_thickness=10.0),
        "regions": {"plate_left", "plate_right", "weld_metal"},
        "boundaries": ["bottom", "top"],
    },
    "lap": {
        "cls": LapJoint,
        "params": dict(plate_thickness=8.0, overlap_length=30.0,
                       weld_leg_size=5.0),
        "regions": {"plate_lower", "plate_upper", "weld"},
        "boundaries": ["bottom"],
    },
    "corner": {
        "cls": CornerJoint,
        "params": dict(plate_thickness_h=10.0, plate_thickness_v=10.0,
                       weld_leg_size=6.0, plate_length=40.0),
        "regions": {"plate_horizontal", "plate_vertical", "weld"},
        "boundaries": ["bottom"],
    },
    "cruciform": {
        "cls": CruciformJoint,
        "params": dict(plate_thickness=10.0, web_thickness=8.0,
                       weld_leg_size=6.0, base_width=60.0, web_height=25.0),
        "regions": {"base_plate", "web_upper", "web_lower",
                    "weld_upper_left", "weld_upper_right",
                    "weld_lower_left", "weld_lower_right"},
        "boundaries": ["top", "bottom"],
    },
}

ALL_JOINTS = sorted(JOINTS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(name: str, **kwargs):
    info = JOINTS[name]
    return info["cls"](**info["params"], **kwargs)


def _physical_names(dim: int) -> dict[str, int]:
    """Map physical group names of dimension *dim* to their tags."""
    return {
        gmsh.model.getPhysicalName(d, tag): tag
        for d, tag in gmsh.model.getPhysicalGroups(dim)
    }


def _group_bbox(dim: int, phys_tag: int) -> tuple[list[float], list[float]]:
    """Union bounding box over all entities of a physical group."""
    lo = [np.inf] * 3
    hi = [-np.inf] * 3
    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
    assert len(entities) > 0
    for ent in entities:
        bb = gmsh.model.getBoundingBox(dim, ent)
        lo = [min(lo[k], bb[k]) for k in range(3)]
        hi = [max(hi[k], bb[k + 3]) for k in range(3)]
    return lo, hi


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
# 3D builds
# ---------------------------------------------------------------------------

@requires_gmsh
@pytest.mark.parametrize("name", ALL_JOINTS)
class TestBuild3D:
    def test_volumes_and_region_groups(self, name):
        joint = _make(name, dimension=3, length=LENGTH)
        joint.build("j3d")  # passing implies the COM cross-check succeeded

        assert len(gmsh.model.getEntities(3)) > 0
        names_3d = _physical_names(3)
        assert set(names_3d) == JOINTS[name]["regions"]
        # One volume group per 2D region
        assert len(names_3d) == len(JOINTS[name]["regions"])

    def test_region_names_match_2d_build(self, name):
        joint_2d = _make(name)
        joint_2d.build("j2d")
        regions_2d = set(_physical_names(2))

        joint_3d = _make(name, dimension=3, length=LENGTH)
        joint_3d.build("j3d")
        regions_3d = set(_physical_names(3))

        assert regions_3d == regions_2d

    def test_boundary_face_groups(self, name):
        joint = _make(name, dimension=3, length=LENGTH)
        joint.build("j3d")
        faces = _physical_names(2)

        for bname in JOINTS[name]["boundaries"] + ["front", "back"]:
            assert bname in faces, f"Missing face group '{bname}'"

        lo, hi = _group_bbox(2, faces["front"])
        assert abs(lo[2]) < TOL and abs(hi[2]) < TOL

        lo, hi = _group_bbox(2, faces["back"])
        assert abs(lo[2] - LENGTH) < TOL and abs(hi[2] - LENGTH) < TOL

        # Loaded/fixed boundary faces span the full extrusion depth
        for bname in JOINTS[name]["boundaries"]:
            lo, hi = _group_bbox(2, faces[bname])
            assert lo[2] < TOL
            assert hi[2] > LENGTH - TOL

    def test_weld_toe_edge_groups(self, name):
        joint = _make(name, dimension=3, length=LENGTH)
        joint.build("j3d")
        edges = _physical_names(1)
        pg = joint.get_physical_groups()

        assert "weld_toe" in edges
        assert "weld_toe" in pg

        toes = joint.get_weld_toe_points()
        for i, (px, py, _pz) in enumerate(toes):
            gname = f"weld_toe_{i}"
            assert gname in edges, f"Missing toe edge group '{gname}'"
            assert gname in pg
            lo, hi = _group_bbox(1, edges[gname])
            # Degenerate at the analytic (x, y), spanning z = [0, LENGTH]
            assert abs(lo[0] - px) < TOL and abs(hi[0] - px) < TOL
            assert abs(lo[1] - py) < TOL and abs(hi[1] - py) < TOL
            assert abs(lo[2]) < TOL
            assert abs(hi[2] - LENGTH) < TOL


# ---------------------------------------------------------------------------
# Weld-toe line / normal API (dimension-independent)
# ---------------------------------------------------------------------------

@requires_gmsh
@pytest.mark.parametrize("name", ALL_JOINTS)
@pytest.mark.parametrize("dimension", [2, 3])
class TestWeldToeAPI:
    def test_toe_lines_match_points(self, name, dimension):
        joint = _make(name, dimension=dimension, length=LENGTH)
        points = joint.get_weld_toe_points()
        lines = joint.get_weld_toe_lines()

        assert len(lines) == len(points)
        for (px, py, _pz), (start, end) in zip(points, lines):
            assert start == pytest.approx((px, py, 0.0))
            assert end == pytest.approx((px, py, LENGTH))

    def test_toe_normals_unit_and_count(self, name, dimension):
        joint = _make(name, dimension=dimension, length=LENGTH)
        normals = joint.get_weld_toe_normals()

        assert len(normals) == len(joint.get_weld_toe_points())
        for n in normals:
            assert len(n) == 3
            assert np.linalg.norm(n) == pytest.approx(1.0)


@requires_gmsh
def test_fillet_toe_normal_convention():
    joint = _make("fillet_t")
    assert joint.get_weld_toe_normals() == [
        (0.0, 1.0, 0.0),     # base-plate top surface
        (-1.0, 0.0, 0.0),    # web left face
        (1.0, 0.0, 0.0),     # web right face
        (0.0, 1.0, 0.0),     # base-plate top surface
    ]


@requires_gmsh
def test_cruciform_lower_toe_normals():
    normals = _make("cruciform").get_weld_toe_normals()
    assert normals[4] == (0.0, -1.0, 0.0)   # base-plate bottom surface
    assert normals[7] == (0.0, -1.0, 0.0)   # base-plate bottom surface
    assert normals[5] == (-1.0, 0.0, 0.0)   # lower web left face
    assert normals[6] == (1.0, 0.0, 0.0)    # lower web right face


# ---------------------------------------------------------------------------
# 2D default unchanged
# ---------------------------------------------------------------------------

@requires_gmsh
def test_dimension2_default_build_unchanged():
    joint = _make("fillet_t")
    assert joint.dimension == 2
    joint.build()

    pg = joint.get_physical_groups()
    for gname in ("base_plate", "web", "weld_left", "weld_right",
                  "bottom", "top"):
        assert gname in pg, f"Missing group '{gname}'"

    assert gmsh.model.getEntities(3) == []
    assert set(_physical_names(2)) == JOINTS["fillet_t"]["regions"]
    names_1d = _physical_names(1)
    assert "bottom" in names_1d and "top" in names_1d


# ---------------------------------------------------------------------------
# Notch guard
# ---------------------------------------------------------------------------

@requires_gmsh
def test_notched_model_rejects_3d():
    from feaweld.geometry.notch import create_notched_model

    joint = _make("fillet_t", dimension=3, length=LENGTH)
    with pytest.raises(NotImplementedError, match="2D"):
        create_notched_model(joint)
