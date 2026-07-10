"""Tests for the spectrum fatigue assessment engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from feaweld.core.types import SNCurve, SNSegment, SNStandard
from feaweld.fatigue.assessment import (
    assess_spectrum,
    build_cycle_set,
    scale_cycles,
    spectrum_life_power_law,
)
from feaweld.fatigue.knockdown import (
    gerber_correction,
    goodman_correction,
    thickness_correction,
)
from feaweld.fatigue.miner import fatigue_life_from_damage
from feaweld.fatigue.rainflow import rainflow_count
from feaweld.fatigue.sn_curves import iiw_fat


def single_slope_curve(m: float = 3.0, fat: float = 90.0) -> SNCurve:
    """Single-segment power-law curve with no knee or cutoff."""
    return SNCurve(
        name=f"single m={m}",
        standard=SNStandard.IIW,
        segments=[SNSegment(m=m, C=fat ** m * 2.0e6, stress_threshold=0.0)],
        cutoff_cycles=1e12,
    )


# ---------------------------------------------------------------------------
# build_cycle_set
# ---------------------------------------------------------------------------


class TestBuildCycleSet:

    def test_no_definition_returns_none(self) -> None:
        assert build_cycle_set() is None

    def test_empty_containers_return_none(self) -> None:
        assert build_cycle_set(blocks=(), history=[]) is None

    def test_two_styles_raises(self) -> None:
        with pytest.raises(ValueError, match="one cyclic-loading style"):
            build_cycle_set(r_ratio=0.1, blocks=[{"range_factor": 1.0}])

    def test_history_and_history_file_raises(self) -> None:
        with pytest.raises(ValueError, match="one cyclic-loading style"):
            build_cycle_set(history=[0.0, 1.0, 0.0], history_file="signal.csv")

    def test_r_ratio_zero(self) -> None:
        triples, kind, is_absolute = build_cycle_set(r_ratio=0.0, cycles=2e6)
        assert kind == "constant_amplitude"
        assert is_absolute is False
        assert triples == [(1.0, 0.5, 2e6)]

    def test_r_ratio_fully_reversed(self) -> None:
        """R = -1 gives range 2 and zero mean in factor space."""
        triples, _, _ = build_cycle_set(r_ratio=-1.0, cycles=1000.0)
        (rng, mean, count), = triples
        assert rng == pytest.approx(2.0)
        assert mean == pytest.approx(0.0)
        assert count == pytest.approx(1000.0)

    def test_r_ratio_default_count(self) -> None:
        triples, _, _ = build_cycle_set(r_ratio=0.5)
        assert triples[0][2] == pytest.approx(1.0)

    def test_r_ratio_at_one_raises(self) -> None:
        with pytest.raises(ValueError, match="r_ratio"):
            build_cycle_set(r_ratio=1.0)

    def test_blocks_factor_space(self) -> None:
        blocks = [
            {"range_factor": 1.0, "mean_factor": 0.5, "cycles": 100.0},
            {"range_factor": 0.5, "cycles": 1000.0},
        ]
        triples, kind, is_absolute = build_cycle_set(blocks=blocks)
        assert kind == "blocks"
        assert is_absolute is False
        assert triples == [(1.0, 0.5, 100.0), (0.5, 0.0, 1000.0)]

    def test_blocks_absolute(self) -> None:
        blocks = [
            {"stress_range": 120.0, "mean_stress": 60.0, "cycles": 5e4},
            {"stress_range": 80.0, "cycles": 2e5},
        ]
        triples, kind, is_absolute = build_cycle_set(blocks=blocks)
        assert kind == "blocks"
        assert is_absolute is True
        assert triples == [(120.0, 60.0, 5e4), (80.0, 0.0, 2e5)]

    def test_block_defaults(self) -> None:
        triples, _, _ = build_cycle_set(blocks=[{"range_factor": 0.8}])
        assert triples == [(0.8, 0.0, 1.0)]

    def test_block_mixed_keys_raises(self) -> None:
        with pytest.raises(ValueError, match="mixes factor keys"):
            build_cycle_set(blocks=[{"range_factor": 1.0, "mean_stress": 50.0}])

    def test_blocks_mixed_styles_across_blocks_raises(self) -> None:
        with pytest.raises(ValueError, match="same key style"):
            build_cycle_set(
                blocks=[{"range_factor": 1.0}, {"stress_range": 100.0}]
            )

    def test_block_missing_range_raises(self) -> None:
        with pytest.raises(ValueError, match="range_factor"):
            build_cycle_set(blocks=[{"cycles": 100.0}])

    def test_block_mean_only_missing_range_raises(self) -> None:
        with pytest.raises(ValueError, match="missing 'stress_range'"):
            build_cycle_set(blocks=[{"mean_stress": 50.0}])

    def test_history_inline_matches_rainflow(self) -> None:
        signal = np.array([0.0, 100.0, -50.0, 80.0, -30.0, 100.0, 0.0])
        triples, kind, is_absolute = build_cycle_set(history=signal)
        assert kind == "history"
        assert is_absolute is False
        assert triples == rainflow_count(signal)

    def test_history_units_stress_is_absolute(self) -> None:
        _, _, is_absolute = build_cycle_set(
            history=[0.0, 100.0, 0.0], history_units="stress"
        )
        assert is_absolute is True

    def test_history_units_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="history_units"):
            build_cycle_set(history=[0.0, 1.0, 0.0], history_units="ksi")

    def test_history_file_comma_delimited(self, tmp_path) -> None:
        signal = np.array([0.0, 1.0, -0.5, 0.8, -0.3, 1.0, 0.0])
        path = tmp_path / "signal.csv"
        path.write_text("".join(f"{v},{i}\n" for i, v in enumerate(signal)))
        triples, kind, _ = build_cycle_set(history_file=path)
        assert kind == "history"
        assert triples == rainflow_count(signal)

    def test_history_file_single_column(self, tmp_path) -> None:
        signal = np.array([0.0, 100.0, -50.0, 80.0, 0.0])
        path = tmp_path / "signal.txt"
        path.write_text("".join(f"{v}\n" for v in signal))
        triples, _, is_absolute = build_cycle_set(
            history_file=path, history_units="stress"
        )
        assert is_absolute is True
        assert triples == rainflow_count(signal)

    def test_history_file_relative_to_base_dir(self, tmp_path) -> None:
        signal = np.array([0.0, 1.0, 0.0])
        (tmp_path / "signal.csv").write_text("".join(f"{v}\n" for v in signal))
        triples, _, _ = build_cycle_set(
            history_file="signal.csv", base_dir=tmp_path
        )
        assert triples == rainflow_count(signal)

    def test_history_file_missing_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            build_cycle_set(history_file=tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# scale_cycles
# ---------------------------------------------------------------------------


class TestScaleCycles:

    def test_range_and_mean_scaled_count_unchanged(self) -> None:
        cycles = [(1.0, 0.5, 100.0), (0.5, -0.25, 1000.0)]
        scaled = scale_cycles(cycles, 150.0)
        assert scaled == [(150.0, 75.0, 100.0), (75.0, -37.5, 1000.0)]

    def test_empty(self) -> None:
        assert scale_cycles([], 150.0) == []


# ---------------------------------------------------------------------------
# assess_spectrum
# ---------------------------------------------------------------------------


class TestAssessSpectrum:

    def test_single_block_identity(self) -> None:
        """D = n / N for one block with no corrections."""
        curve = iiw_fat(90)
        result = assess_spectrum([(100.0, 50.0, 1000.0)], curve)
        assert result["damage"] == pytest.approx(1000.0 / curve.life(100.0))
        assert result["equivalent_stress_range"] == pytest.approx(100.0)
        assert result["max_corrected_range"] == pytest.approx(100.0)
        assert result["n_cycles_per_repeat"] == pytest.approx(1000.0)

    def test_life_relationships(self) -> None:
        curve = iiw_fat(90)
        cycles = [(120.0, 0.0, 500.0), (90.0, 0.0, 2000.0)]
        result = assess_spectrum(cycles, curve)
        assert result["life_repeats"] == pytest.approx(1.0 / result["damage"])
        assert result["life_cycles"] == pytest.approx(
            result["life_repeats"] * 2500.0
        )

    def test_goodman_matches_hand_calculation(self) -> None:
        """amp 50, mean 200, sigma_u 500: S_eq = 50 / (1 - 0.4)."""
        curve = iiw_fat(90)
        result = assess_spectrum(
            [(100.0, 200.0, 1000.0)],
            curve,
            mean_correction="goodman",
            sigma_u=500.0,
        )
        amp_eq = 50.0 / (1.0 - 200.0 / 500.0)
        expected_range = 2.0 * amp_eq
        assert result["max_corrected_range"] == pytest.approx(expected_range)
        assert result["damage"] == pytest.approx(
            1000.0 / curve.life(expected_range)
        )

    def test_gerber_less_damaging_than_goodman(self) -> None:
        curve = iiw_fat(90)
        cycles = [(100.0, 200.0, 1000.0)]
        d_goodman = assess_spectrum(
            cycles, curve, mean_correction="goodman", sigma_u=500.0
        )["damage"]
        d_gerber = assess_spectrum(
            cycles, curve, mean_correction="gerber", sigma_u=500.0
        )["damage"]
        assert 0.0 < d_gerber < d_goodman

    def test_correction_uses_knockdown_functions(self) -> None:
        """Corrected ranges reproduce goodman/gerber_correction exactly."""
        curve = single_slope_curve()
        for name, fn in (
            ("goodman", goodman_correction),
            ("gerber", gerber_correction),
        ):
            result = assess_spectrum(
                [(80.0, 120.0, 1.0)], curve, mean_correction=name, sigma_u=450.0
            )
            expected = 2.0 * fn(40.0, 120.0, 450.0)
            assert result["max_corrected_range"] == pytest.approx(expected)

    def test_sigma_u_required_for_goodman(self) -> None:
        curve = iiw_fat(90)
        with pytest.raises(ValueError, match="sigma_u"):
            assess_spectrum([(100.0, 50.0, 1.0)], curve, mean_correction="goodman")

    def test_invalid_correction_raises(self) -> None:
        curve = iiw_fat(90)
        with pytest.raises(ValueError, match="mean_correction"):
            assess_spectrum([(100.0, 0.0, 1.0)], curve, mean_correction="soderberg")

    def test_strength_factor_reduces_life(self) -> None:
        curve = iiw_fat(90)
        cycles = [(100.0, 0.0, 1000.0)]
        base = assess_spectrum(cycles, curve)
        knocked = assess_spectrum(cycles, curve, strength_factor=0.8)
        assert knocked["max_corrected_range"] == pytest.approx(100.0 / 0.8)
        assert knocked["life_cycles"] < base["life_cycles"]

    def test_strength_factor_nonpositive_raises(self) -> None:
        curve = iiw_fat(90)
        with pytest.raises(ValueError, match="strength_factor"):
            assess_spectrum([(100.0, 0.0, 1.0)], curve, strength_factor=0.0)

    def test_residual_mean_inert_without_correction(self) -> None:
        """Residual mean enters only through the mean-stress correction."""
        curve = iiw_fat(90)
        cycles = [(100.0, 50.0, 1000.0)]
        base = assess_spectrum(cycles, curve, mean_correction="none")
        shifted = assess_spectrum(
            cycles, curve, mean_correction="none", residual_mean=150.0
        )
        assert shifted == base

    def test_residual_mean_enters_goodman(self) -> None:
        curve = iiw_fat(90)
        cycles = [(100.0, 100.0, 1000.0)]
        without = assess_spectrum(
            cycles, curve, mean_correction="goodman", sigma_u=500.0
        )
        with_residual = assess_spectrum(
            cycles,
            curve,
            mean_correction="goodman",
            sigma_u=500.0,
            residual_mean=100.0,
        )
        assert with_residual["damage"] > without["damage"]
        # Residual shift is equivalent to the same mean applied directly
        direct = assess_spectrum(
            [(100.0, 200.0, 1000.0)],
            curve,
            mean_correction="goodman",
            sigma_u=500.0,
        )
        assert with_residual["damage"] == pytest.approx(direct["damage"])

    def test_below_cutoff_contributes_zero_damage(self) -> None:
        """Cycles under the CAFL add count but no damage."""
        curve = iiw_fat(90)
        loud = [(100.0, 0.0, 1000.0)]
        quiet = loud + [(1.0, 0.0, 1e9)]
        d_loud = assess_spectrum(loud, curve)
        d_quiet = assess_spectrum(quiet, curve)
        assert d_quiet["damage"] == pytest.approx(d_loud["damage"])
        assert d_quiet["n_cycles_per_repeat"] == pytest.approx(1000.0 + 1e9)

    def test_all_below_cutoff_infinite_life(self) -> None:
        curve = iiw_fat(90)
        result = assess_spectrum([(1.0, 0.0, 1e6)], curve)
        assert result["damage"] == 0.0
        assert math.isinf(result["life_repeats"])
        assert math.isinf(result["life_cycles"])

    def test_mean_at_ultimate_gives_infinite_damage(self) -> None:
        curve = iiw_fat(90)
        result = assess_spectrum(
            [(100.0, 500.0, 1.0)],
            curve,
            mean_correction="goodman",
            sigma_u=500.0,
        )
        assert math.isinf(result["damage"])
        assert result["life_repeats"] == 0.0

    def test_empty_cycle_set(self) -> None:
        curve = iiw_fat(90)
        result = assess_spectrum([], curve)
        assert result["damage"] == 0.0
        assert math.isinf(result["life_cycles"])
        assert result["n_cycles_per_repeat"] == 0.0
        assert result["equivalent_stress_range"] == 0.0

    def test_equivalent_range_two_blocks(self) -> None:
        """S_eq = (sum(n S^m) / sum(n))^(1/m) at the first-segment slope."""
        curve = iiw_fat(90)
        cycles = [(120.0, 0.0, 100.0), (80.0, 0.0, 900.0)]
        result = assess_spectrum(cycles, curve)
        m = curve.segments[0].m
        expected = (
            (100.0 * 120.0 ** m + 900.0 * 80.0 ** m) / 1000.0
        ) ** (1.0 / m)
        assert result["equivalent_stress_range"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# spectrum_life_power_law
# ---------------------------------------------------------------------------


class TestSpectrumLifePowerLaw:

    def test_exact_against_miner_on_single_slope_curve(self) -> None:
        """Power-law spectrum life equals Miner life for a one-slope law."""
        m = 3.0
        curve = single_slope_curve(m=m)
        reference_stress = 150.0
        factor_cycles = [(1.0, 0.0, 10.0), (0.6, 0.0, 200.0), (0.3, 0.0, 5000.0)]
        miner_result = assess_spectrum(
            scale_cycles(factor_cycles, reference_stress), curve
        )
        life = spectrum_life_power_law(
            curve.life(reference_stress), factor_cycles, m
        )
        assert life == pytest.approx(miner_result["life_repeats"], rel=1e-9)

    def test_single_cycle_at_reference(self) -> None:
        """One cycle at factor 1.0 gives exactly the reference life."""
        life = spectrum_life_power_law(2.0e6, [(1.0, 0.0, 1.0)], 3.13)
        assert life == pytest.approx(2.0e6)

    def test_empty_set_infinite(self) -> None:
        assert math.isinf(spectrum_life_power_law(2.0e6, [], 3.0))

    def test_infinite_reference_life(self) -> None:
        cycles = [(1.0, 0.0, 100.0)]
        assert math.isinf(spectrum_life_power_law(float("inf"), cycles, 3.0))

    def test_zero_range_cycles_ignored(self) -> None:
        life_with = spectrum_life_power_law(
            2.0e6, [(1.0, 0.0, 1.0), (0.0, 0.0, 1e9)], 3.0
        )
        life_without = spectrum_life_power_law(2.0e6, [(1.0, 0.0, 1.0)], 3.0)
        assert life_with == pytest.approx(life_without)


# ---------------------------------------------------------------------------
# thickness_correction / fatigue_life_from_damage
# ---------------------------------------------------------------------------


class TestThicknessCorrection:

    def test_below_reference_no_correction(self) -> None:
        assert thickness_correction(10.0) == 1.0

    def test_at_reference_no_correction(self) -> None:
        assert thickness_correction(25.0) == 1.0

    def test_above_reference(self) -> None:
        assert thickness_correction(50.0) == pytest.approx((25.0 / 50.0) ** 0.3)

    def test_custom_exponent(self) -> None:
        f = thickness_correction(100.0, reference_thickness=25.0, exponent=0.2)
        assert f == pytest.approx(0.25 ** 0.2)


class TestFatigueLifeFromDamage:

    def test_reciprocal(self) -> None:
        assert fatigue_life_from_damage(0.25) == pytest.approx(4.0)

    def test_zero_damage_infinite(self) -> None:
        assert math.isinf(fatigue_life_from_damage(0.0))

    def test_negative_damage_infinite(self) -> None:
        assert math.isinf(fatigue_life_from_damage(-0.1))
