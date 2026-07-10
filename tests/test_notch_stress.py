"""Tests for the effective notch stress method (postprocess/notch_stress).

Pins the FAT225 two-slope S-N curve, notch stress extraction and SCF
computation on canned FEA results, the parametric SCF formulas (inline
coefficients and the SCF database dispatch), and the fatigue assessment
dict contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import FEAResults, SNCurve, SNSegment, SNStandard
from feaweld.postprocess.notch_stress import (
    FAT225_CURVE,
    FAT225_KNEE_STRESS,
    FICTITIOUS_RADIUS_STEEL,
    FICTITIOUS_RADIUS_THIN,
    NotchStressResult,
    assess_notch_fatigue,
    effective_notch_stress,
    notch_stress_scf_parametric,
)


# ---------------------------------------------------------------------------
# Tests: FAT225 S-N curve
# ---------------------------------------------------------------------------

class TestFAT225Curve:
    def test_detail_class_life_at_225(self):
        # Detail class definition: 225 MPa at N = 2e6.
        assert FAT225_CURVE.life(225.0) == pytest.approx(2e6)

    def test_knee_stress_value(self):
        assert FAT225_KNEE_STRESS == pytest.approx(225.0 * 0.2 ** (1.0 / 3.0))
        assert FAT225_KNEE_STRESS == pytest.approx(131.5808, abs=1e-3)

    def test_knee_continuity(self):
        # Both slopes give N = 1e7 at the knee stress by construction.
        at_knee = FAT225_CURVE.life(FAT225_KNEE_STRESS)
        just_below = FAT225_CURVE.life(FAT225_KNEE_STRESS - 1e-9)
        assert at_knee == pytest.approx(1e7)
        assert just_below == pytest.approx(1e7)

    def test_slope_m3_above_knee(self):
        assert FAT225_CURVE.life(200.0) == pytest.approx(2847656.25)
        # Doubling the range above the knee cuts life by 2^3.
        ratio = FAT225_CURVE.life(150.0) / FAT225_CURVE.life(300.0)
        assert ratio == pytest.approx(8.0)

    def test_slope_m5_below_knee(self):
        assert FAT225_CURVE.life(100.0) == pytest.approx(3.94423319e7, rel=1e-8)
        # Doubling the range below the knee cuts life by 2^5.
        ratio = FAT225_CURVE.life(50.0) / FAT225_CURVE.life(100.0)
        assert ratio == pytest.approx(32.0)

    def test_cutoff_attribute(self):
        assert FAT225_CURVE.cutoff_cycles == 1e9

    def test_life_not_clamped_at_cutoff(self):
        # SNCurve.life applies no cutoff clamp — very low ranges return
        # finite lives beyond cutoff_cycles.
        life = FAT225_CURVE.life(10.0)
        assert np.isfinite(life)
        assert life > FAT225_CURVE.cutoff_cycles

    def test_nonpositive_stress_infinite_life(self):
        assert FAT225_CURVE.life(0.0) == float("inf")
        assert FAT225_CURVE.life(-50.0) == float("inf")

    def test_curve_metadata(self):
        assert FAT225_CURVE.name == "FAT225"
        assert FAT225_CURVE.standard == SNStandard.IIW
        assert len(FAT225_CURVE.segments) == 2
        assert FAT225_CURVE.segments[0].m == 3.0
        assert FAT225_CURVE.segments[1].m == 5.0


# ---------------------------------------------------------------------------
# Tests: effective_notch_stress
# ---------------------------------------------------------------------------

class TestEffectiveNotchStress:
    def test_uniform_field_max_stress(self, uniform_stress_results):
        result = effective_notch_stress(
            uniform_stress_results, np.array([1, 2]), nominal_stress=100.0,
        )
        assert result.max_notch_stress == pytest.approx(100.0)

    def test_scf_vs_nominal(self, uniform_stress_results):
        ids = np.array([1, 2])
        r1 = effective_notch_stress(uniform_stress_results, ids, 100.0)
        r2 = effective_notch_stress(uniform_stress_results, ids, 50.0)
        assert r1.stress_concentration_factor == pytest.approx(1.0)
        assert r2.stress_concentration_factor == pytest.approx(2.0)

    def test_gradient_field_max_over_ids(self, gradient_stress_results):
        # σ_yy runs 0 → 200 with y; nodes 2/3 sit at y = 10.
        top = effective_notch_stress(gradient_stress_results, np.array([1, 2]), 100.0)
        bottom = effective_notch_stress(gradient_stress_results, np.array([0, 1]), 100.0)
        assert top.max_notch_stress == pytest.approx(200.0)
        assert bottom.max_notch_stress == pytest.approx(0.0)

    def test_range_equals_max(self, gradient_stress_results):
        # R = 0 convention: the range is the maximum notch stress itself.
        result = effective_notch_stress(gradient_stress_results, np.array([1, 2]), 100.0)
        assert result.notch_stress_range == result.max_notch_stress

    def test_fatigue_life_from_fat225(self, uniform_stress_results,
                                      gradient_stress_results):
        r100 = effective_notch_stress(uniform_stress_results, np.array([1, 2]), 100.0)
        r200 = effective_notch_stress(gradient_stress_results, np.array([1, 2]), 100.0)
        assert r100.fatigue_life == pytest.approx(FAT225_CURVE.life(100.0))
        assert r200.fatigue_life == pytest.approx(2847656.25)

    def test_zero_nominal_gives_inf_scf(self, uniform_stress_results):
        result = effective_notch_stress(uniform_stress_results, np.array([1, 2]), 0.0)
        assert result.stress_concentration_factor == float("inf")

    def test_negative_nominal_gives_inf_scf(self, uniform_stress_results):
        # Only nominal_stress > 0 yields a finite SCF.
        result = effective_notch_stress(uniform_stress_results, np.array([1, 2]), -50.0)
        assert result.stress_concentration_factor == float("inf")

    def test_missing_stress_raises(self, simple_plate_mesh):
        results = FEAResults(mesh=simple_plate_mesh)
        with pytest.raises(ValueError, match="No stress data"):
            effective_notch_stress(results, np.array([1, 2]), 100.0)

    def test_field_slicing(self, gradient_stress_results):
        ids = np.array([0, 2, 3])
        result = effective_notch_stress(gradient_stress_results, ids, 100.0)
        assert result.notch_stress_field is not None
        assert result.notch_stress_field.shape == (3,)
        expected = gradient_stress_results.stress.von_mises[ids]
        np.testing.assert_allclose(result.notch_stress_field, expected)

    def test_fictitious_radius_passthrough(self, uniform_stress_results):
        default = effective_notch_stress(uniform_stress_results, np.array([1]), 100.0)
        thin = effective_notch_stress(
            uniform_stress_results, np.array([1]), 100.0,
            fictitious_radius=FICTITIOUS_RADIUS_THIN,
        )
        assert default.fictitious_radius == FICTITIOUS_RADIUS_STEEL == 1.0
        assert thin.fictitious_radius == 0.05


# ---------------------------------------------------------------------------
# Tests: parametric SCF (inline coefficients)
# ---------------------------------------------------------------------------

class TestParametricSCF:
    def test_fillet_reference_value(self):
        # Anthes coefficients a=0.469, b=0.572, c=0.469 at ρ=1, θ=45°, t=10.
        kt = notch_stress_scf_parametric(1.0, 45.0, 10.0, weld_toe_type="fillet")
        assert kt == pytest.approx(1.9137065, rel=1e-6)

    def test_butt_reference_value(self):
        kt = notch_stress_scf_parametric(1.0, 45.0, 10.0, weld_toe_type="butt")
        assert kt == pytest.approx(1.7132537, rel=1e-6)

    def test_formula_matches_definition(self):
        rho, theta, t = 2.0, 60.0, 25.0
        kt = notch_stress_scf_parametric(rho, theta, t, weld_toe_type="fillet")
        expected = 1.0 + 0.469 * (t / rho) ** 0.572 * (np.radians(theta) / np.pi) ** 0.469
        assert kt == pytest.approx(expected)
        assert isinstance(kt, float)

    def test_increases_with_angle(self):
        kts = [notch_stress_scf_parametric(1.0, theta, 10.0)
               for theta in (30.0, 45.0, 60.0, 90.0)]
        assert kts == sorted(kts)
        assert kts[0] < kts[-1]

    def test_increases_with_thickness(self):
        kts = [notch_stress_scf_parametric(1.0, 45.0, t)
               for t in (5.0, 10.0, 25.0, 50.0)]
        assert kts == sorted(kts)
        assert kts[0] < kts[-1]

    def test_decreases_with_radius(self):
        kts = [notch_stress_scf_parametric(rho, 45.0, 10.0)
               for rho in (0.5, 1.0, 2.0, 4.0)]
        assert kts == sorted(kts, reverse=True)
        assert kts[0] > kts[-1]

    def test_radius_floor_at_0p01(self):
        # ρ is clamped to 0.01 mm — zero and sub-floor radii match the floor.
        at_floor = notch_stress_scf_parametric(0.01, 45.0, 10.0)
        assert notch_stress_scf_parametric(0.0, 45.0, 10.0) == pytest.approx(at_floor)
        assert notch_stress_scf_parametric(0.001, 45.0, 10.0) == pytest.approx(at_floor)

    def test_unknown_toe_type_uses_butt_coefficients(self):
        # Anything other than "fillet" falls into the butt branch.
        kt = notch_stress_scf_parametric(1.0, 45.0, 10.0, weld_toe_type="other")
        butt = notch_stress_scf_parametric(1.0, 45.0, 10.0, weld_toe_type="butt")
        assert kt == pytest.approx(butt)


# ---------------------------------------------------------------------------
# Tests: geometry= dispatch into the SCF database
# ---------------------------------------------------------------------------

class TestSCFGeometryDispatch:
    def test_geometry_key_uses_scf_database(self):
        from feaweld.data.scf import compute_scf

        kt = notch_stress_scf_parametric(1.0, 45.0, 10.0, geometry="cruciform")
        assert kt == pytest.approx(compute_scf("cruciform", 1.0, 45.0, 10.0))
        # a=0.520, b=0.550, c=0.480 from the bundled dataset.
        assert kt == pytest.approx(1.9484502, rel=1e-6)

    def test_fillet_toe_key_matches_inline_fillet(self):
        # The "fillet_toe" database entry carries the same Anthes coefficients
        # as the inline fillet branch.
        from_db = notch_stress_scf_parametric(1.5, 50.0, 12.0, geometry="fillet_toe")
        inline = notch_stress_scf_parametric(1.5, 50.0, 12.0, weld_toe_type="fillet")
        assert from_db == pytest.approx(inline)

    def test_geometry_overrides_toe_type(self):
        # geometry= wins even when weld_toe_type says fillet.
        kt = notch_stress_scf_parametric(
            1.0, 45.0, 10.0, weld_toe_type="fillet", geometry="butt_toe",
        )
        butt = notch_stress_scf_parametric(1.0, 45.0, 10.0, weld_toe_type="butt")
        assert kt == pytest.approx(butt)

    def test_unknown_geometry_raises_keyerror(self):
        with pytest.raises(KeyError, match="SCF geometry not found"):
            notch_stress_scf_parametric(1.0, 45.0, 10.0, geometry="no_such_joint")


# ---------------------------------------------------------------------------
# Tests: assess_notch_fatigue
# ---------------------------------------------------------------------------

class TestAssessNotchFatigue:
    @staticmethod
    def _notch_result(stress_range: float = 150.0) -> NotchStressResult:
        return NotchStressResult(
            max_notch_stress=stress_range,
            notch_stress_range=stress_range,
            stress_concentration_factor=1.5,
            fatigue_life=FAT225_CURVE.life(stress_range),
        )

    def test_default_fat225_keys_and_values(self):
        assessment = assess_notch_fatigue(self._notch_result(150.0))
        assert set(assessment) == {
            "notch_stress_range", "scf", "sn_curve",
            "fatigue_life_cycles", "fictitious_radius_mm",
        }
        assert assessment["sn_curve"] == "FAT225"
        assert assessment["notch_stress_range"] == 150.0
        assert assessment["scf"] == 1.5
        assert assessment["fictitious_radius_mm"] == 1.0
        assert assessment["fatigue_life_cycles"] == pytest.approx(
            FAT225_CURVE.life(150.0)
        )

    def test_life_matches_notch_result(self):
        nr = self._notch_result(220.0)
        assessment = assess_notch_fatigue(nr)
        assert assessment["fatigue_life_cycles"] == pytest.approx(nr.fatigue_life)

    def test_custom_curve(self, simple_sn_curve):
        assessment = assess_notch_fatigue(self._notch_result(100.0),
                                          sn_curve=simple_sn_curve)
        assert assessment["sn_curve"] == "TestFAT90"
        assert assessment["fatigue_life_cycles"] == pytest.approx(
            90.0**3 * 2e6 / 100.0**3
        )

    def test_custom_single_slope_curve(self):
        curve = SNCurve(
            name="FAT160",
            standard=SNStandard.IIW,
            segments=[SNSegment(m=3.0, C=160.0**3 * 2e6, stress_threshold=0.0)],
        )
        assessment = assess_notch_fatigue(self._notch_result(160.0), sn_curve=curve)
        assert assessment["sn_curve"] == "FAT160"
        assert assessment["fatigue_life_cycles"] == pytest.approx(2e6)
