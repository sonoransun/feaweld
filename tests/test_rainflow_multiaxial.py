"""Tests for multi-axial rainflow cycle counting (Track G)."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.fatigue.rainflow import rainflow_count, rainflow_multiaxial


def _make_uniaxial_history(sigma_xx: np.ndarray) -> np.ndarray:
    """Build a (n_t, 6) Voigt history with only σ_xx populated."""
    n = len(sigma_xx)
    out = np.zeros((n, 6), dtype=np.float64)
    out[:, 0] = sigma_xx
    return out


def test_principal_method_on_uniaxial_matches_scalar_rainflow():
    """Pure-σ_xx history: principal-method multi-axial should match 1-D."""
    sigma = np.array(
        [0.0, 100.0, -50.0, 80.0, -30.0, 60.0, 0.0],
        dtype=np.float64,
    )
    history = _make_uniaxial_history(sigma)

    ma_cycles = rainflow_multiaxial(history, method="principal")
    scalar_cycles = rainflow_count(sigma)

    assert len(ma_cycles) == len(scalar_cycles)
    # Sort both by (range, mean) to compare independent of ordering.
    ma_sorted = sorted((r, m) for r, m, _ in ma_cycles)
    sc_sorted = sorted((r, m) for r, m, _ in scalar_cycles)
    for (r_ma, m_ma), (r_sc, m_sc) in zip(ma_sorted, sc_sorted):
        assert r_ma == pytest.approx(r_sc, rel=1e-9, abs=1e-9)
        assert m_ma == pytest.approx(m_sc, rel=1e-9, abs=1e-9)


def test_projection_method_returns_plane_scalar():
    """Projection onto n=[1,0,0] should give σ_xx cycle counts."""
    sigma = np.array(
        [0.0, 120.0, -40.0, 90.0, -20.0, 50.0, 0.0],
        dtype=np.float64,
    )
    # Embed σ_xx plus some decorrelated shear to make sure projection
    # really picks out the normal component (shear contributes nothing
    # when n = [1,0,0]).
    history = _make_uniaxial_history(sigma)
    history[:, 3] = np.linspace(0.0, 25.0, len(sigma))  # τ_xy

    ma_cycles = rainflow_multiaxial(
        history,
        method="projection",
        plane_normal=np.array([1.0, 0.0, 0.0]),
    )
    scalar_cycles = rainflow_count(sigma)

    assert len(ma_cycles) == len(scalar_cycles)
    ma_sorted = sorted((r, m) for r, m, _ in ma_cycles)
    sc_sorted = sorted((r, m) for r, m, _ in scalar_cycles)
    for (r_ma, m_ma), (r_sc, m_sc) in zip(ma_sorted, sc_sorted):
        assert r_ma == pytest.approx(r_sc, rel=1e-9, abs=1e-9)
        assert m_ma == pytest.approx(m_sc, rel=1e-9, abs=1e-9)


def test_projection_requires_plane_normal():
    history = _make_uniaxial_history(np.array([0.0, 10.0, 0.0]))
    with pytest.raises(ValueError, match="plane_normal"):
        rainflow_multiaxial(history, method="projection")


def test_invalid_method_raises():
    history = _make_uniaxial_history(np.array([0.0, 10.0, 0.0]))
    with pytest.raises(ValueError, match="method"):
        rainflow_multiaxial(history, method="bogus")  # type: ignore[arg-type]


def test_bad_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        rainflow_multiaxial(np.zeros((5, 3)))
