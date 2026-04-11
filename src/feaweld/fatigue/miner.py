"""Palmgren-Miner cumulative damage rule and fatigue life estimation."""

from __future__ import annotations

import math

import numpy as np

from feaweld.core.types import LoadHistory, SNCurve


def miner_damage(
    cycles: list[tuple[float, float]],
    sn_curve: SNCurve,
) -> float:
    """Compute cumulative Miner's damage.

    Parameters
    ----------
    cycles : list[tuple[float, float]]
        Each tuple is ``(stress_range, count)`` where *count* can be
        fractional (e.g. 0.5 for a half-cycle from rainflow counting).
    sn_curve : SNCurve
        S-N curve used for life calculation.

    Returns
    -------
    float
        Total damage D = sum(n_i / N_i).  D >= 1 implies failure.
    """
    D = 0.0
    for stress_range, n_i in cycles:
        if stress_range <= 0:
            continue
        N_i = sn_curve.life(stress_range)
        if math.isinf(N_i):
            continue
        D += n_i / N_i
    return D


def fatigue_life(
    load_history: LoadHistory,
    sn_curve: SNCurve,
) -> dict[str, float]:
    """Estimate fatigue life from a load history.

    This function performs rainflow counting (imported locally to avoid
    circular imports), computes Miner's damage per block, and returns
    the damage, equivalent stress range, and estimated life in blocks.

    Parameters
    ----------
    load_history : LoadHistory
        Stress-range time history.
    sn_curve : SNCurve
        S-N curve for the detail category.

    Returns
    -------
    dict
        ``damage`` -- Miner damage per single block (one pass of the history).
        ``equivalent_stress_range`` -- constant-amplitude stress range that
            would produce the same damage in the same number of cycles.
        ``estimated_life`` -- number of blocks to failure (1 / damage).
    """
    from feaweld.fatigue.rainflow import rainflow_count

    # Run rainflow counting on the stress_ranges signal
    rf_cycles = rainflow_count(load_history.stress_ranges)

    # Build (stress_range, count) pairs for Miner
    cycle_pairs: list[tuple[float, float]] = [
        (sr, cnt) for sr, _mean, cnt in rf_cycles
    ]

    D = miner_damage(cycle_pairs, sn_curve)

    # Equivalent stress range
    # D = sum(n_i / N_i) = sum(n_i * S_i^m / C)  for the dominant segment
    # S_eq = (sum(n_i * S_i^m) / sum(n_i))^(1/m)
    total_count = sum(cnt for _, cnt in cycle_pairs)
    if total_count > 0 and len(sn_curve.segments) > 0:
        m = sn_curve.segments[0].m
        weighted_sum = sum(
            cnt * sr ** m for sr, cnt in cycle_pairs if sr > 0
        )
        S_eq = (weighted_sum / total_count) ** (1.0 / m) if weighted_sum > 0 else 0.0
    else:
        S_eq = 0.0

    estimated_life = 1.0 / D if D > 0 else float("inf")

    return {
        "damage": D,
        "equivalent_stress_range": S_eq,
        "estimated_life": estimated_life,
    }


def miner_damage_with_uq(
    stress_ranges_mean: np.ndarray,
    stress_ranges_std: np.ndarray,
    counts: np.ndarray,
    curve: SNCurve,
    scatter_std_log10_N: float = 0.2,
    n_mc: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Monte Carlo propagation of stress + S-N scatter through Miner's rule.

    Returns dict with keys: damage_mean, damage_std, damage_p05, damage_p95,
    life_mean, life_std, life_p05, life_p95.
    """
    stress_mean = np.asarray(stress_ranges_mean, dtype=float)
    stress_std = np.asarray(stress_ranges_std, dtype=float)
    counts_arr = np.asarray(counts, dtype=float)

    if stress_mean.shape != stress_std.shape or stress_mean.shape != counts_arr.shape:
        raise ValueError(
            "stress_ranges_mean, stress_ranges_std and counts must share shape"
        )

    n_bins = stress_mean.size
    rng = np.random.default_rng(seed)
    sigma_ln = scatter_std_log10_N * math.log(10.0)

    damage_samples = np.zeros(n_mc)
    for i in range(n_mc):
        s_samples = rng.normal(stress_mean, stress_std)
        s_samples = np.clip(s_samples, 1e-12, None)
        D = 0.0
        for j in range(n_bins):
            s = float(s_samples[j])
            if s <= 0 or counts_arr[j] <= 0:
                continue
            N_det = curve.life(s)
            if not math.isfinite(N_det) or N_det <= 0:
                continue
            # Per-bin lognormal scatter on N.
            N_scatter = N_det * float(rng.lognormal(mean=0.0, sigma=sigma_ln))
            if N_scatter > 0:
                D += counts_arr[j] / N_scatter
        damage_samples[i] = D

    finite_D = damage_samples > 0
    life_samples = np.where(finite_D, 1.0 / np.where(finite_D, damage_samples, 1.0), np.inf)

    def _percentile_finite(arr: np.ndarray, q: float) -> float:
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            return float("inf")
        return float(np.percentile(vals, q))

    def _mean_finite(arr: np.ndarray) -> float:
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            return float("inf")
        return float(np.mean(vals))

    def _std_finite(arr: np.ndarray) -> float:
        vals = arr[np.isfinite(arr)]
        if vals.size == 0:
            return 0.0
        return float(np.std(vals))

    return {
        "damage_mean": float(np.mean(damage_samples)),
        "damage_std": float(np.std(damage_samples)),
        "damage_p05": float(np.percentile(damage_samples, 5.0)),
        "damage_p95": float(np.percentile(damage_samples, 95.0)),
        "life_mean": _mean_finite(life_samples),
        "life_std": _std_finite(life_samples),
        "life_p05": _percentile_finite(life_samples, 5.0),
        "life_p95": _percentile_finite(life_samples, 95.0),
    }
