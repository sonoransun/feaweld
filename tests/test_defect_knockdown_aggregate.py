"""Tests for the multi-defect FAT aggregation helper (Track H)."""

from __future__ import annotations

import math

import numpy as np

from feaweld.core.types import Point3D
from feaweld.defects import (
    LackOfFusionDefect,
    PoreDefect,
)
from feaweld.fatigue.knockdown import defect_knockdown


def test_defect_knockdown_empty_list():
    out = defect_knockdown([], base_fat=90.0)
    assert out["worst_fat"] == 90.0
    assert out["knockdown_factor"] == 1.0
    assert out["per_defect"] == []


def test_defect_knockdown_worst_wins():
    # Small pore (no downgrade) + LoF (FAT 36 floor). LoF must dominate.
    defects = [
        PoreDefect(center=Point3D(0.0, 0.0, 0.0), diameter=0.1),
        LackOfFusionDefect(
            plane_origin=Point3D(1.0, 0.0, 0.0),
            plane_normal=np.array([0.0, 1.0, 0.0]),
            extent_u=0.5,
            extent_v=0.5,
        ),
    ]
    out = defect_knockdown(defects, base_fat=90.0, plate_thickness=20.0)
    assert math.isclose(out["worst_fat"], 36.0, rel_tol=1e-9)
    assert len(out["per_defect"]) == 2


def test_defect_knockdown_per_defect_list():
    defects = [
        PoreDefect(center=Point3D(0.0, 0.0, 0.0), diameter=0.1),
        PoreDefect(center=Point3D(0.0, 0.0, 0.0), diameter=0.2),
        LackOfFusionDefect(
            plane_origin=Point3D(0.0, 0.0, 0.0),
            plane_normal=np.array([0.0, 1.0, 0.0]),
            extent_u=0.2,
            extent_v=0.2,
        ),
    ]
    out = defect_knockdown(defects, base_fat=100.0, plate_thickness=15.0)
    assert len(out["per_defect"]) == 3
    for entry in out["per_defect"]:
        assert "type" in entry
        assert "fat" in entry
        assert "rationale" in entry
