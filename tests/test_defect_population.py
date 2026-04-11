"""Pure-numpy tests for the ISO 5817 defect population sampler."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.defects.population import (
    sample_iso5817_population,
    validate_population,
)
from feaweld.defects.types import PoreDefect


def test_iso5817_B_population_respects_bounds():
    """20 independent draws at level B should all pass validation."""
    for seed in range(20):
        defects = sample_iso5817_population(
            level="B",
            weld_length=500.0,
            weld_width=10.0,
            plate_thickness=10.0,
            seed=seed,
        )
        violations = validate_population(defects, "B", plate_thickness=10.0)
        assert violations == [], (
            f"seed={seed} produced violations: {violations}"
        )


@pytest.mark.parametrize("level", ["B", "C", "D"])
def test_iso5817_all_levels_respect_bounds(level):
    for seed in range(10):
        defects = sample_iso5817_population(
            level=level,
            weld_length=800.0,
            weld_width=12.0,
            plate_thickness=15.0,
            seed=seed,
        )
        violations = validate_population(defects, level, plate_thickness=15.0)
        assert violations == []


def test_iso5817_deterministic_seed():
    """Identical seed + inputs must produce identical populations."""
    a = sample_iso5817_population(
        level="C", weld_length=400.0, weld_width=8.0, seed=42
    )
    b = sample_iso5817_population(
        level="C", weld_length=400.0, weld_width=8.0, seed=42
    )
    assert len(a) == len(b)
    for da, db in zip(a, b):
        assert da.defect_type == db.defect_type
        if da.defect_type == "pore":
            assert da.center == db.center
            assert da.diameter == db.diameter
        elif da.defect_type == "undercut":
            assert da.start == db.start
            assert da.end == db.end
            assert da.depth == db.depth
        elif da.defect_type == "slag":
            assert da.center == db.center
            assert da.semi_axes == db.semi_axes


def test_poisson_rate_scaling():
    """A 10x longer weld yields ~10x more defects on average."""
    short_counts: list[int] = []
    long_counts: list[int] = []
    for seed in range(20):
        short_counts.append(
            len(sample_iso5817_population(
                level="B", weld_length=100.0, weld_width=10.0, seed=seed
            ))
        )
        long_counts.append(
            len(sample_iso5817_population(
                level="B", weld_length=1000.0, weld_width=10.0, seed=seed
            ))
        )
    short_mean = float(np.mean(short_counts))
    long_mean = float(np.mean(long_counts))
    # Allow a wide tolerance on the ratio: Poisson with small lambda
    # has large relative variance.  We only require the long weld to be
    # at least 5x the short weld on average (it should be ~10x).
    assert short_mean >= 1.0
    assert long_mean / short_mean > 5.0
    assert long_mean / short_mean < 20.0


def test_level_D_allows_more_than_B():
    """Max pore diameter at level D is >= max at level B (same rate)."""
    max_d_B = 0.0
    max_d_D = 0.0
    for seed in range(30):
        defs_B = sample_iso5817_population(
            level="B",
            weld_length=2000.0,
            weld_width=10.0,
            plate_thickness=20.0,
            seed=seed,
        )
        defs_D = sample_iso5817_population(
            level="D",
            weld_length=2000.0,
            weld_width=10.0,
            plate_thickness=20.0,
            seed=seed,
        )
        for d in defs_B:
            if isinstance(d, PoreDefect):
                max_d_B = max(max_d_B, d.diameter)
        for d in defs_D:
            if isinstance(d, PoreDefect):
                max_d_D = max(max_d_D, d.diameter)
    # Level D accepts larger pores (absolute max 5.0 vs 3.0 at level B),
    # so the observed maximum across 30 draws should be larger or at
    # worst equal.
    assert max_d_D >= max_d_B


def test_validate_population_flags_oversize_pore():
    """Directly constructed oversized pores must be flagged."""
    from feaweld.core.types import Point3D

    bad = [PoreDefect(center=Point3D(0.0, 0.0, 0.0), diameter=10.0)]
    violations = validate_population(bad, "B", plate_thickness=10.0)
    assert len(violations) >= 1
    assert any("pore" in v for v in violations)
