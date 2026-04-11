"""Tests for defect dataclass geometry helpers."""

from __future__ import annotations

import math

import numpy as np

from feaweld.core.types import Point3D
from feaweld.defects import (
    ClusterPorosity,
    LackOfFusionDefect,
    PoreDefect,
    SlagInclusion,
    SurfaceCrack,
    UndercutDefect,
)


def _p(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Point3D:
    return Point3D(x, y, z)


def test_pore_volume():
    pore = PoreDefect(center=_p(), diameter=2.0)
    assert math.isclose(pore.volume(), (4.0 / 3.0) * math.pi, rel_tol=1e-12)
    assert pore.critical_dimension() == 2.0


def test_undercut_v_profile_volume():
    uc = UndercutDefect(
        start=_p(0.0, 0.0, 0.0),
        end=_p(10.0, 0.0, 0.0),
        depth=1.0,
        profile="V",
    )
    assert math.isclose(uc.volume(), 0.5 * 10.0 * 1.0 ** 2, rel_tol=1e-12)


def test_undercut_u_profile_volume():
    uc = UndercutDefect(
        start=_p(0.0, 0.0, 0.0),
        end=_p(10.0, 0.0, 0.0),
        depth=1.0,
        profile="U",
    )
    assert math.isclose(uc.volume(), 10.0 * 1.0 * 1.0, rel_tol=1e-12)


def test_slag_inclusion_volume():
    slag = SlagInclusion(center=_p(), semi_axes=(1.0, 2.0, 3.0))
    expected = (4.0 / 3.0) * math.pi * 1.0 * 2.0 * 3.0
    assert math.isclose(slag.volume(), expected, rel_tol=1e-12)
    assert math.isclose(slag.critical_dimension(), 6.0, rel_tol=1e-12)


def test_cluster_porosity_description():
    cluster = ClusterPorosity(
        center=_p(),
        radius=3.0,
        n_pores=7,
        size_mean=0.5,
        size_std=0.1,
    )
    desc = cluster.description()
    assert "cluster_porosity" in desc
    assert cluster.volume() > 0.0


def test_lof_volume_is_zero():
    lof = LackOfFusionDefect(
        plane_origin=_p(),
        plane_normal=np.array([0.0, 0.0, 1.0]),
        extent_u=5.0,
        extent_v=2.0,
    )
    assert lof.volume() == 0.0
    assert lof.critical_dimension() == 5.0


def test_surface_crack_volume_is_zero():
    crack = SurfaceCrack(
        start=_p(0.0, 0.0, 0.0),
        end=_p(5.0, 0.0, 0.0),
        depth=0.8,
    )
    assert crack.volume() == 0.0
    assert crack.critical_dimension() == 0.8
