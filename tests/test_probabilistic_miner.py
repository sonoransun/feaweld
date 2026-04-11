"""Tests for probabilistic Miner damage propagation (Track B3)."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.fatigue.miner import miner_damage_with_uq
from feaweld.fatigue.sn_curves import (
    iiw_fat,
    life_with_scatter,
    life_with_scatter_stress,
)


SEED = 1234


def test_life_with_scatter_returns_required_keys() -> None:
    curve = iiw_fat(90)
    band = life_with_scatter(curve, 100.0, n_samples=500, seed=SEED)
    for key in ("mean", "std", "p05", "p50", "p95"):
        assert key in band
        assert np.isfinite(band[key])


def test_life_with_scatter_mean_matches_deterministic_within_tolerance() -> None:
    """The lognormal mean should be within 20% of deterministic life."""
    curve = iiw_fat(90)
    N_det = curve.life(100.0)
    band = life_with_scatter(
        curve,
        100.0,
        scatter_std_log10_N=0.2,
        n_samples=5000,
        seed=SEED,
    )
    rel_err = abs(band["mean"] - N_det) / N_det
    assert rel_err < 0.20


def test_life_with_scatter_stress_keys() -> None:
    curve = iiw_fat(90)
    band = life_with_scatter_stress(
        curve,
        stress_mean=100.0,
        stress_std=5.0,
        n_samples=500,
        seed=SEED,
    )
    for key in ("mean", "std", "p05", "p50", "p95"):
        assert key in band


def test_miner_uq_bands_ordering() -> None:
    """p05 < mean < p95 strictly."""
    curve = iiw_fat(90)
    stress_mean = np.array([120.0, 80.0])
    stress_std = np.array([6.0, 4.0])
    counts = np.array([500.0, 2000.0])

    result = miner_damage_with_uq(
        stress_mean,
        stress_std,
        counts,
        curve,
        scatter_std_log10_N=0.2,
        n_mc=1000,
        seed=SEED,
    )
    assert result["damage_p05"] < result["damage_mean"] < result["damage_p95"]
    assert result["life_p05"] < result["life_mean"] < result["life_p95"]


def test_miner_uq_widens_with_larger_scatter() -> None:
    """Higher scatter_std_log10_N must increase the std of sampled damage."""
    curve = iiw_fat(90)
    stress_mean = np.array([120.0, 80.0])
    stress_std = np.array([6.0, 4.0])
    counts = np.array([500.0, 2000.0])

    low = miner_damage_with_uq(
        stress_mean,
        stress_std,
        counts,
        curve,
        scatter_std_log10_N=0.1,
        n_mc=1000,
        seed=SEED,
    )
    high = miner_damage_with_uq(
        stress_mean,
        stress_std,
        counts,
        curve,
        scatter_std_log10_N=0.3,
        n_mc=1000,
        seed=SEED,
    )
    assert high["damage_std"] > low["damage_std"]


def test_miner_uq_deterministic_seed() -> None:
    """Same seed must produce identical output."""
    curve = iiw_fat(90)
    stress_mean = np.array([100.0])
    stress_std = np.array([5.0])
    counts = np.array([1000.0])

    a = miner_damage_with_uq(
        stress_mean, stress_std, counts, curve, n_mc=500, seed=SEED,
    )
    b = miner_damage_with_uq(
        stress_mean, stress_std, counts, curve, n_mc=500, seed=SEED,
    )
    assert a == b
