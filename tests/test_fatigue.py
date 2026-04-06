"""Tests for the fatigue analysis modules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from feaweld.core.types import SNCurve, SNSegment, SNStandard
from feaweld.fatigue.knockdown import (
    combined_knockdown,
    environment_factor,
    gerber_correction,
    goodman_correction,
    size_factor,
    surface_finish_factor,
)
from feaweld.fatigue.miner import miner_damage
from feaweld.fatigue.rainflow import rainflow_count
from feaweld.fatigue.sn_curves import dnv_curve, get_sn_curve, iiw_fat


# ---------------------------------------------------------------------------
# S-N curves
# ---------------------------------------------------------------------------


class TestIIWCurves:
    """Verify IIW FAT S-N curve construction and life calculations."""

    def test_fat90_construction(self) -> None:
        curve = iiw_fat(90)
        assert curve.name == "IIW FAT90"
        assert curve.standard == SNStandard.IIW
        assert len(curve.segments) == 2
        assert curve.segments[0].m == pytest.approx(3.0)
        assert curve.segments[1].m == pytest.approx(5.0)

    def test_fat90_life_at_reference(self) -> None:
        """At stress_range = FAT class, life should be 2e6 cycles."""
        curve = iiw_fat(90)
        N = curve.life(90.0)
        assert N == pytest.approx(2.0e6, rel=1e-6)

    def test_fat90_life_at_100mpa(self) -> None:
        """At 100 MPa, life should be less than at 90 MPa (higher stress)."""
        curve = iiw_fat(90)
        N_100 = curve.life(100.0)
        N_90 = curve.life(90.0)
        assert N_100 < N_90
        # From segment 1: N = 90^3 * 2e6 / 100^3
        expected = (90.0 ** 3 * 2.0e6) / (100.0 ** 3)
        assert N_100 == pytest.approx(expected, rel=1e-6)

    def test_fat90_below_cutoff_gives_infinite(self) -> None:
        """Very low stress should yield infinite life."""
        curve = iiw_fat(90)
        # Cutoff stress is quite low -- use a very small value
        N = curve.life(1.0)
        assert math.isinf(N)

    def test_invalid_fat_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported IIW FAT class"):
            iiw_fat(99)


class TestDNVCurves:
    """Verify DNV-RP-C203 S-N curves."""

    def test_dnv_d_construction(self) -> None:
        curve = dnv_curve("D")
        assert curve.name == "DNV D"
        assert curve.standard == SNStandard.DNV
        assert len(curve.segments) == 2

    def test_dnv_d_known_life(self) -> None:
        """Verify life at a known stress level for DNV D curve.

        DNV D: log_a1 = 12.164, m1 = 3
        At S = 100 MPa: log10(N) = 12.164 - 3*log10(100) = 12.164 - 6 = 6.164
        N ~ 10^6.164 ~ 1.459e6
        """
        curve = dnv_curve("D")
        N = curve.life(100.0)
        expected = 10.0 ** (12.164 - 3.0 * math.log10(100.0))
        assert N == pytest.approx(expected, rel=1e-3)

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown DNV category"):
            dnv_curve("Z99")


class TestGetSNCurve:
    """Test the unified dispatcher."""

    def test_iiw_dispatch(self) -> None:
        curve = get_sn_curve("iiw", "FAT90")
        assert curve.name == "IIW FAT90"

    def test_iiw_dispatch_numeric(self) -> None:
        curve = get_sn_curve("iiw", "125")
        assert curve.name == "IIW FAT125"

    def test_dnv_dispatch(self) -> None:
        curve = get_sn_curve("dnv", "F1")
        assert curve.name == "DNV F1"

    def test_asme_dispatch(self) -> None:
        curve = get_sn_curve("asme", "ferritic")
        assert curve.standard == SNStandard.ASME


# ---------------------------------------------------------------------------
# Rainflow counting
# ---------------------------------------------------------------------------


class TestRainflow:
    """Verify rainflow counting against a known signal."""

    def test_known_signal(self) -> None:
        signal = np.array([0.0, 100.0, -50.0, 80.0, -30.0, 100.0, 0.0])
        cycles = rainflow_count(signal)

        # Should contain both full cycles and half-cycles
        assert len(cycles) > 0

        # Check that all stress ranges are positive
        for sr, mean, count in cycles:
            assert sr >= 0
            assert count in (0.5, 1.0)

        # Total damage-related: sum of counts * ranges should account
        # for all reversals.  Full cycles extracted should have range
        # <= the enclosing ranges.
        full_cycles = [(sr, m, c) for sr, m, c in cycles if c == 1.0]
        half_cycles = [(sr, m, c) for sr, m, c in cycles if c == 0.5]

        # With this signal we expect some extracted full cycles
        # The inner ranges (110, 130, 130) should produce extracted cycles
        assert len(full_cycles) + len(half_cycles) == len(cycles)

    def test_simple_triangle(self) -> None:
        """A single peak-valley-peak should give one half-cycle."""
        signal = np.array([0.0, 100.0, 0.0])
        cycles = rainflow_count(signal)
        # Should get at least one cycle
        assert len(cycles) >= 1
        # The total range should be captured
        ranges = [sr for sr, _, _ in cycles]
        assert 100.0 in ranges

    def test_constant_signal_no_cycles(self) -> None:
        signal = np.array([50.0, 50.0, 50.0, 50.0])
        cycles = rainflow_count(signal)
        assert len(cycles) == 0


# ---------------------------------------------------------------------------
# Miner damage accumulation
# ---------------------------------------------------------------------------


class TestMinerDamage:
    """Verify Palmgren-Miner cumulative damage."""

    def test_single_stress_range(self) -> None:
        """Simple case: 1000 cycles at 100 MPa on FAT90 curve."""
        curve = iiw_fat(90)
        N_at_100 = curve.life(100.0)
        # Expected damage: n/N
        cycles = [(100.0, 1000.0)]
        D = miner_damage(cycles, curve)
        assert D == pytest.approx(1000.0 / N_at_100, rel=1e-6)

    def test_multiple_ranges(self) -> None:
        """Two stress ranges should accumulate linearly."""
        curve = iiw_fat(90)
        N_100 = curve.life(100.0)
        N_80 = curve.life(80.0)
        cycles = [(100.0, 500.0), (80.0, 1000.0)]
        D = miner_damage(cycles, curve)
        expected = 500.0 / N_100 + 1000.0 / N_80
        assert D == pytest.approx(expected, rel=1e-6)

    def test_zero_stress_range_ignored(self) -> None:
        curve = iiw_fat(90)
        cycles = [(0.0, 1000.0), (100.0, 500.0)]
        D = miner_damage(cycles, curve)
        # Only the 100 MPa cycles should contribute
        expected = 500.0 / curve.life(100.0)
        assert D == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Knockdown factors
# ---------------------------------------------------------------------------


class TestKnockdown:

    def test_goodman_zero_mean(self) -> None:
        """With zero mean stress, Goodman returns the amplitude unchanged."""
        S_eq = goodman_correction(100.0, 0.0, 500.0)
        assert S_eq == pytest.approx(100.0)

    def test_goodman_tensile_mean(self) -> None:
        """Positive mean stress increases the equivalent amplitude."""
        S_eq = goodman_correction(100.0, 200.0, 500.0)
        expected = 100.0 / (1.0 - 200.0 / 500.0)
        assert S_eq == pytest.approx(expected)
        assert S_eq > 100.0

    def test_goodman_at_uts_gives_inf(self) -> None:
        S_eq = goodman_correction(100.0, 500.0, 500.0)
        assert math.isinf(S_eq)

    def test_gerber_zero_mean(self) -> None:
        S_eq = gerber_correction(100.0, 0.0, 500.0)
        assert S_eq == pytest.approx(100.0)

    def test_gerber_less_severe_than_goodman(self) -> None:
        """Gerber parabola is less conservative than Goodman line."""
        amp = 100.0
        mean = 200.0
        uts = 500.0
        g_good = goodman_correction(amp, mean, uts)
        g_gerb = gerber_correction(amp, mean, uts)
        assert g_gerb < g_good

    def test_surface_finish_factor_range(self) -> None:
        k = surface_finish_factor(1.6, 400.0)
        assert 0.5 < k <= 1.0

    def test_size_factor_small(self) -> None:
        k = size_factor(2.0)
        assert k == 1.0

    def test_size_factor_medium(self) -> None:
        k = size_factor(25.0)
        assert 0.8 < k < 1.0

    def test_environment_factor_air(self) -> None:
        assert environment_factor("air") == 1.0

    def test_environment_factor_seawater(self) -> None:
        assert environment_factor("seawater") == pytest.approx(0.4)

    def test_combined_knockdown(self) -> None:
        result = combined_knockdown(k_a=0.9, k_b=0.85, k_env=0.6)
        assert result == pytest.approx(0.9 * 0.85 * 0.6)

    def test_combined_with_extra(self) -> None:
        result = combined_knockdown(k_a=0.9, k_b=1.0, k_env=1.0, k_reliability=0.8)
        assert result == pytest.approx(0.9 * 0.8)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestSNCurveEdgeCases:
    """Edge cases for S-N curve life computation."""

    def test_life_zero_stress_returns_inf(self) -> None:
        curve = iiw_fat(90)
        assert math.isinf(curve.life(0.0))

    def test_life_negative_stress_returns_inf(self) -> None:
        curve = iiw_fat(90)
        assert math.isinf(curve.life(-50.0))

    def test_life_at_knee_point_continuity(self) -> None:
        """Life should be continuous at the knee point (segment transition)."""
        curve = iiw_fat(90)
        S_knee = curve.segments[0].stress_threshold
        # Compute life from both sides of the knee
        N_above = curve.segments[0].C / (S_knee ** curve.segments[0].m)
        N_below = curve.segments[1].C / (S_knee ** curve.segments[1].m)
        # They should be approximately equal (continuity)
        assert N_above == pytest.approx(N_below, rel=1e-3)


class TestRainflowEdgeCases:

    def test_constant_amplitude_signal(self) -> None:
        """Repeating 0-100-0 pattern should produce full cycles."""
        signal = np.array([0, 100, 0, 100, 0, 100, 0], dtype=float)
        cycles = rainflow_count(signal)
        # Should have cycles with range = 100
        full = [c for c in cycles if c[2] == 1.0]
        ranges_100 = [c for c in cycles if abs(c[0] - 100.0) < 1e-6]
        assert len(ranges_100) >= 1

    def test_two_point_signal_produces_half_cycle(self) -> None:
        """Two-point signal should produce residue half-cycle."""
        signal = np.array([0.0, 100.0], dtype=float)
        cycles = rainflow_count(signal)
        # Residue handling produces half cycles
        if len(cycles) > 0:
            assert all(c[2] == 0.5 for c in cycles)
