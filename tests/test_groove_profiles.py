"""Tests for the :mod:`feaweld.geometry.groove` module."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.geometry.groove import (
    GrooveProfile,
    JGroove,
    KGroove,
    UGroove,
    VGroove,
    XGroove,
)


def _polygon_is_closed(pts: np.ndarray) -> bool:
    """Shoelace non-zero is sufficient for a closed simple polygon."""
    x = pts[:, 0]
    y = pts[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 1e-9


def test_v_groove_area() -> None:
    t = 20.0
    rg = 2.0
    rf = 2.0
    angle = 60.0
    vg = VGroove(plate_thickness=t, root_gap=rg, angle=angle, root_face=rf)
    half = np.radians(angle / 2.0)
    bev_h = t - rf
    # Closed-form: rectangular root gap band (rg * t) plus the two triangular
    # bevel flanks (2 * 0.5 * bev_h * (bev_h * tan(half))) = bev_h^2 * tan(half)
    expected = rg * t + bev_h * bev_h * np.tan(half)
    assert vg.area() == pytest.approx(expected, rel=1e-6)


def test_u_groove_area_positive() -> None:
    ug = UGroove(
        plate_thickness=20.0,
        root_gap=1.0,
        root_radius=4.0,
        bevel_angle=10.0,
        root_face=2.0,
    )
    poly = ug.cross_section_polygon()
    area = ug.area()
    assert area > 0.0
    t_extent = poly[:, 0].max() - poly[:, 0].min()
    z_extent = poly[:, 1].max() - poly[:, 1].min()
    assert area < t_extent * z_extent


def test_j_groove_asymmetry() -> None:
    jg = JGroove(
        plate_thickness=20.0,
        root_gap=1.0,
        bevel_angle=30.0,
        root_radius=4.0,
        root_face=2.0,
    )
    poly = jg.cross_section_polygon()
    # Compute centroid of the t-coordinate
    cx = float(np.mean(poly[:, 0]))
    # For a symmetric profile centroid would sit at t = 0; for J it should
    # lean toward the beveled (right) side.
    assert cx > 0.1


def test_x_groove_symmetric() -> None:
    xg = XGroove(
        plate_thickness=30.0,
        root_gap=2.0,
        angle_top=60.0,
        angle_bottom=60.0,
        root_face=4.0,
    )
    poly = xg.cross_section_polygon()
    t_mid = xg.plate_thickness / 2.0
    # Every vertex should have a mirror partner across z = t/2 and t = 0
    reflected = np.stack([-poly[:, 0], 2.0 * t_mid - poly[:, 1]], axis=1)
    # For each reflected vertex there exists a matching vertex in the polygon
    for p in reflected:
        diffs = np.linalg.norm(poly - p, axis=1)
        assert diffs.min() < 1e-9


def test_k_groove_one_sided() -> None:
    kg = KGroove(
        plate_thickness=20.0,
        root_gap=1.0,
        angle=45.0,
        root_face=2.0,
    )
    poly = kg.cross_section_polygon()
    # Square side is at t = -half_gap; bevel opens to +t. So the only
    # negative-t vertices should equal -half_gap (the square face).
    neg = poly[poly[:, 0] < -1e-12]
    assert np.allclose(neg[:, 0], -kg.root_gap / 2.0)


def test_all_grooves_closed_polygon() -> None:
    grooves: list[GrooveProfile] = [
        VGroove(plate_thickness=20.0, root_gap=2.0),
        UGroove(plate_thickness=20.0, root_gap=1.0),
        JGroove(plate_thickness=20.0, root_gap=1.0),
        XGroove(plate_thickness=30.0, root_gap=2.0),
        KGroove(plate_thickness=20.0, root_gap=1.0),
    ]
    for g in grooves:
        poly = g.cross_section_polygon()
        assert poly.ndim == 2 and poly.shape[1] == 2
        assert poly.shape[0] >= 3
        assert _polygon_is_closed(poly)
        assert g.area() > 0.0
