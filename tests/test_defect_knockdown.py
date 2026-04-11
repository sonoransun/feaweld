"""Tests for BS 7910 / IIW FAT downgrade helpers."""

from __future__ import annotations

import math

import numpy as np

from feaweld.core.types import Point3D
from feaweld.defects import (
    ClusterPorosity,
    KnockdownResult,
    LackOfFusionDefect,
    PoreDefect,
    RootGapDefect,
    SlagInclusion,
    SurfaceCrack,
    UndercutDefect,
    defect_fat_downgrade,
    lof_fat_downgrade,
    porosity_fat_downgrade,
)


def _p(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Point3D:
    return Point3D(x, y, z)


def test_porosity_below_threshold_no_downgrade():
    result = porosity_fat_downgrade(
        pore_diameter=0.1, plate_thickness=20.0, base_fat=90.0
    )
    assert isinstance(result, KnockdownResult)
    assert result.knockdown_factor == 1.0
    assert math.isclose(result.downgraded_fat, 90.0, rel_tol=1e-12)


def test_porosity_floor_at_0_5():
    # d/t = 0.5 -> 1 - 5*(0.5 - 0.02) = 1 - 2.4 = -1.4 -> floored at 0.5.
    result = porosity_fat_downgrade(
        pore_diameter=10.0, plate_thickness=20.0, base_fat=100.0
    )
    assert math.isclose(result.knockdown_factor, 0.5, rel_tol=1e-12)
    assert math.isclose(result.downgraded_fat, 50.0, rel_tol=1e-12)


def test_lof_floors_to_fat36():
    result = lof_fat_downgrade(lof_depth=1.5, base_fat=100.0)
    assert math.isclose(result.downgraded_fat, 36.0, rel_tol=1e-12)
    assert "floor" in result.rationale.lower()


def test_dispatch_handles_all_types():
    normal = np.array([0.0, 0.0, 1.0])
    defects = [
        PoreDefect(center=_p(), diameter=1.0),
        ClusterPorosity(
            center=_p(), radius=2.0, n_pores=5, size_mean=0.4, size_std=0.1
        ),
        SlagInclusion(center=_p(), semi_axes=(0.5, 0.7, 1.0)),
        UndercutDefect(
            start=_p(0.0, 0.0, 0.0), end=_p(5.0, 0.0, 0.0), depth=0.4
        ),
        LackOfFusionDefect(
            plane_origin=_p(), plane_normal=normal, extent_u=3.0, extent_v=1.0
        ),
        SurfaceCrack(
            start=_p(0.0, 0.0, 0.0), end=_p(4.0, 0.0, 0.0), depth=0.6
        ),
        RootGapDefect(
            start=_p(0.0, 0.0, 0.0),
            end=_p(10.0, 0.0, 0.0),
            gap_width=0.5,
            plate_thickness=20.0,
        ),
    ]
    for d in defects:
        res = defect_fat_downgrade(d, base_fat=100.0, plate_thickness=20.0)
        assert isinstance(res, KnockdownResult)
        assert 0.0 < res.knockdown_factor <= 1.0
        assert res.downgraded_fat > 0.0
        assert res.rationale
