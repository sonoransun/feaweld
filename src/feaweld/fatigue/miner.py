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
