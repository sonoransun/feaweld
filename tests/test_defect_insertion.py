"""Gmsh OCC boolean insertion tests for feaweld.defects.insertion.

Gmsh is an optional dependency.  If it is not importable the whole
module is skipped via :func:`pytest.importorskip`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from feaweld.core.types import Point3D  # noqa: E402
from feaweld.defects.insertion import (  # noqa: E402
    insert_all,
    insert_defect,
    insert_pore,
    insert_undercut,
)
from feaweld.defects.types import (  # noqa: E402
    PoreDefect,
    SlagInclusion,
    SurfaceCrack,
    UndercutDefect,
)

pytestmark = pytest.mark.requires_gmsh


@pytest.fixture
def gmsh_plate():
    """Yield a freshly-initialized gmsh model with a 40x40x20 mm plate.

    The fixture tears gmsh down unconditionally so tests can leave the
    OCC kernel in whatever state they like.
    """
    gmsh.initialize()
    try:
        gmsh.model.add("defect_insertion_test")
        tag = gmsh.model.occ.addBox(0.0, 0.0, 0.0, 40.0, 40.0, 20.0)
        gmsh.model.occ.synchronize()
        yield tag
    finally:
        gmsh.finalize()


def _volume(tag: int) -> float:
    return float(gmsh.model.occ.getMass(3, tag))


def test_insert_pore_reduces_volume(gmsh_plate):
    tag = gmsh_plate
    v0 = _volume(tag)
    expected_plate_volume = 40.0 * 40.0 * 20.0
    assert math.isclose(v0, expected_plate_volume, rel_tol=1e-6)

    pore = PoreDefect(center=Point3D(20.0, 20.0, 10.0), diameter=1.0)
    record = insert_pore(tag, pore)

    new_tag = record["parent_tag"]
    v1 = _volume(new_tag)
    pore_vol = (4.0 / 3.0) * math.pi * (0.5) ** 3
    assert math.isclose(v1, v0 - pore_vol, rel_tol=1e-2)
    assert record["defect_type"] == "pore"


def test_insert_many_pores(gmsh_plate):
    tag = gmsh_plate
    v0 = _volume(tag)

    defects = [
        PoreDefect(center=Point3D(10.0, 10.0, 10.0), diameter=1.0),
        PoreDefect(center=Point3D(20.0, 20.0, 10.0), diameter=1.2),
        PoreDefect(center=Point3D(30.0, 30.0, 10.0), diameter=0.8),
    ]
    current = tag
    for d in defects:
        rec = insert_pore(current, d)
        current = rec["parent_tag"]

    v1 = _volume(current)
    expected_removed = sum(
        (4.0 / 3.0) * math.pi * (d.diameter / 2.0) ** 3 for d in defects
    )
    assert v1 < v0
    assert math.isclose(v1, v0 - expected_removed, rel_tol=5e-2)


def test_insert_undercut_reduces_volume(gmsh_plate):
    tag = gmsh_plate
    v0 = _volume(tag)

    undercut = UndercutDefect(
        start=Point3D(5.0, 20.0, 20.0),
        end=Point3D(35.0, 20.0, 20.0),
        depth=1.0,
        profile="V",
    )
    rec = insert_undercut(tag, undercut)
    v1 = _volume(rec["parent_tag"])

    # MVP undercut is a prism length x depth x depth; V-profile uses
    # width = depth so the box volume is length * depth * depth.
    length = 30.0
    expected_removed = length * 1.0 * 1.0
    # Allow ~50% slack: the prism may clip the top surface.  The goal is
    # simply that the volume DOES shrink by a meaningful amount.
    assert v1 < v0
    assert (v0 - v1) > 0.25 * expected_removed
    assert rec["defect_type"] == "undercut"


def test_insert_all_dispatches(gmsh_plate):
    tag = gmsh_plate
    v0 = _volume(tag)

    defects = [
        PoreDefect(center=Point3D(10.0, 10.0, 10.0), diameter=1.0),
        SurfaceCrack(
            start=Point3D(20.0, 5.0, 20.0),
            end=Point3D(20.0, 35.0, 20.0),
            depth=1.0,
            aspect_ratio=0.3,
        ),
        PoreDefect(center=Point3D(30.0, 30.0, 10.0), diameter=1.5),
    ]
    result = insert_all(tag, defects)

    assert "parent_tag" in result
    assert "insertion_records" in result
    assert len(result["insertion_records"]) == len(defects)
    types = [rec["defect_type"] for rec in result["insertion_records"]]
    assert types == ["pore", "surface_crack", "pore"]

    v1 = _volume(result["parent_tag"])
    assert v1 < v0


def test_insert_defect_rejects_unknown_type(gmsh_plate):
    class _Bogus:
        defect_type = "not_a_defect"

    with pytest.raises(ValueError, match="Unknown defect type"):
        insert_defect(gmsh_plate, _Bogus())  # type: ignore[arg-type]


def test_insert_slag_inclusion_creates_second_volume(gmsh_plate):
    tag = gmsh_plate
    slag = SlagInclusion(
        center=Point3D(20.0, 20.0, 10.0),
        semi_axes=(2.0, 1.5, 1.0),
    )
    from feaweld.defects.insertion import insert_slag_inclusion

    rec = insert_slag_inclusion(tag, slag)
    assert rec["defect_type"] == "slag"
    # Fragment should produce both a parent and an inclusion tag.
    assert "inclusion_tag" in rec
    all_volumes = [t for d, t in gmsh.model.occ.getEntities(3)]
    assert len(all_volumes) >= 2
