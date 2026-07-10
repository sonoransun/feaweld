"""Tests for the EC3 (EN 1993-1-9), BS 7608, and AWS D1.1 S-N standards."""

from __future__ import annotations

import math

import pytest

from feaweld.core.types import SNStandard
from feaweld.fatigue.sn_curves import (
    aws_curve,
    bs7608_curve,
    ec3_curve,
    get_sn_curve,
    parse_sn_spec,
)

EC3_CATEGORIES = (36, 40, 45, 50, 56, 63, 71, 80, 90, 100, 112, 125, 140, 160)
BS7608_CLASSES = ("B", "C", "D", "E", "F", "F2", "G", "W1")
AWS_CATEGORIES = ("A", "B", "BP", "C", "D", "E", "EP")


# ---------------------------------------------------------------------------
# EC3 (EN 1993-1-9)
# ---------------------------------------------------------------------------


class TestEC3Curves:
    """Verify Eurocode 3 detail category curve construction."""

    def test_category_90_construction(self) -> None:
        curve = ec3_curve(90)
        assert curve.name == "EC3 detail category 90"
        assert curve.standard == SNStandard.EC3
        assert len(curve.segments) == 2
        assert curve.segments[0].m == pytest.approx(3.0)
        assert curve.segments[1].m == pytest.approx(5.0)

    def test_category_90_life_at_reference(self) -> None:
        """At stress_range = detail category, life is 2e6 cycles."""
        curve = ec3_curve(90)
        assert curve.life(90.0) == pytest.approx(2.0e6, rel=1e-9)

    def test_cafl_position(self) -> None:
        """CAFL at N_D = 5e6 with delta_sigma_D = (2/5)^(1/3) * delta_sigma_C."""
        curve = ec3_curve(90)
        S_D = curve.segments[0].stress_threshold
        assert S_D == pytest.approx(90.0 * (2.0 / 5.0) ** (1.0 / 3.0), rel=1e-9)
        assert curve.life(S_D) == pytest.approx(5.0e6, rel=1e-9)

    def test_knee_continuity(self) -> None:
        """Both segment laws give the same life at the CAFL knee."""
        curve = ec3_curve(90)
        S_D = curve.segments[0].stress_threshold
        N1 = curve.segments[0].C / S_D ** curve.segments[0].m
        N2 = curve.segments[1].C / S_D ** curve.segments[1].m
        assert N1 == pytest.approx(N2, rel=1e-9)

    def test_cutoff_stress(self) -> None:
        """Cut-off limit delta_sigma_L ~= 0.4047 * delta_sigma_C at N_L = 1e8."""
        curve = ec3_curve(90)
        S_L = curve.segments[1].stress_threshold
        assert S_L == pytest.approx(0.4047 * 90.0, rel=1e-3)
        assert curve.life(S_L) == pytest.approx(1.0e8, rel=1e-9)

    def test_below_cutoff_infinite_life(self) -> None:
        curve = ec3_curve(90)
        S_L = curve.segments[1].stress_threshold
        assert math.isinf(curve.life(0.99 * S_L))

    @pytest.mark.parametrize("category", EC3_CATEGORIES)
    def test_all_categories_construct(self, category: int) -> None:
        curve = ec3_curve(category)
        assert curve.life(float(category)) == pytest.approx(2.0e6, rel=1e-9)

    def test_monotonic_across_categories(self) -> None:
        """Higher detail category gives longer life at a fixed stress range."""
        lives = [ec3_curve(c).life(100.0) for c in EC3_CATEGORIES]
        assert lives == sorted(lives)

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported EC3 detail category"):
            ec3_curve(91)


# ---------------------------------------------------------------------------
# BS 7608
# ---------------------------------------------------------------------------


class TestBS7608Curves:
    """Verify BS 7608 basic class curves (design = mean - 2 SD)."""

    def test_class_d_construction(self) -> None:
        curve = bs7608_curve("D")
        assert curve.name == "BS 7608 class D"
        assert curve.standard == SNStandard.BS7608
        assert len(curve.segments) == 2
        assert curve.segments[0].m == pytest.approx(3.0)
        assert curve.segments[1].m == pytest.approx(5.0)

    def test_class_d_design_life_at_100mpa(self) -> None:
        """Class D design: log_a = 12.6007 - 2*0.2095, so N(100) ~= 1.52e6."""
        curve = bs7608_curve("D")
        expected = 10.0 ** (12.6007 - 2.0 * 0.2095 - 3.0 * 2.0)
        assert curve.life(100.0) == pytest.approx(expected, rel=1e-9)
        assert curve.life(100.0) == pytest.approx(1.52e6, rel=1e-2)

    def test_mean_curve_uses_log_a_mean(self) -> None:
        curve = bs7608_curve("D", mean_curve=True)
        expected = 10.0 ** (12.6007 - 3.0 * 2.0)
        assert curve.life(100.0) == pytest.approx(expected, rel=1e-9)

    def test_mean_curve_above_design_curve(self) -> None:
        design = bs7608_curve("D")
        mean = bs7608_curve("D", mean_curve=True)
        assert mean.life(100.0) > design.life(100.0)

    def test_haibach_slope_is_m_plus_2(self) -> None:
        """Class B has m = 4, so the Haibach extension has slope 6."""
        curve = bs7608_curve("B")
        assert curve.segments[0].m == pytest.approx(4.0)
        assert curve.segments[1].m == pytest.approx(6.0)

    def test_knee_continuity_at_1e7(self) -> None:
        """Both segment laws give exactly 1e7 cycles at the knee stress."""
        curve = bs7608_curve("D")
        S_knee = curve.segments[0].stress_threshold
        N1 = curve.segments[0].C / S_knee ** curve.segments[0].m
        N2 = curve.segments[1].C / S_knee ** curve.segments[1].m
        assert N1 == pytest.approx(1.0e7, rel=1e-9)
        assert N2 == pytest.approx(1.0e7, rel=1e-9)

    @pytest.mark.parametrize("class_name", BS7608_CLASSES)
    def test_all_classes_construct(self, class_name: str) -> None:
        curve = bs7608_curve(class_name)
        assert curve.name == f"BS 7608 class {class_name}"
        assert 0.0 < curve.life(100.0) < float("inf")

    def test_w1_class_lookup(self) -> None:
        curve = bs7608_curve("W1")
        expected = 10.0 ** (11.5662 - 2.0 * 0.1846 - 3.0 * 2.0)
        assert curve.life(100.0) == pytest.approx(expected, rel=1e-9)

    def test_f2_class_lowercase_lookup(self) -> None:
        curve = bs7608_curve("f2")
        assert curve.name == "BS 7608 class F2"

    def test_unknown_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown BS 7608 class"):
            bs7608_curve("Z9")


# ---------------------------------------------------------------------------
# AWS D1.1 / AISC
# ---------------------------------------------------------------------------


class TestAWSCurves:
    """Verify AWS D1.1 / AISC category curves."""

    def test_category_c_construction(self) -> None:
        curve = aws_curve("C")
        assert curve.name == "AWS D1.1 category C"
        assert curve.standard == SNStandard.AWS
        assert len(curve.segments) == 1
        assert curve.segments[0].m == pytest.approx(3.0)

    def test_category_c_stress_at_2e6(self) -> None:
        """SI form F_SR = (Cf * 329e8 / N)^(1/3) gives ~90 MPa at N = 2e6."""
        curve = aws_curve("C")
        S = (44.0 * 3.29e10 / 2.0e6) ** (1.0 / 3.0)
        assert S == pytest.approx(90.0, rel=1e-2)
        assert curve.life(S) == pytest.approx(2.0e6, rel=1e-9)

    def test_below_threshold_infinite_life(self) -> None:
        """Category C threshold F_TH = 69 MPa."""
        curve = aws_curve("C")
        assert math.isinf(curve.life(68.9))
        assert curve.life(69.0) < float("inf")

    def test_primed_alias_forms_equal(self) -> None:
        base = aws_curve("BP")
        assert base.name == "AWS D1.1 category B'"
        for alias in ("B'", "Bp", "bp"):
            curve = aws_curve(alias)
            assert curve.name == base.name
            assert curve.segments[0].C == pytest.approx(base.segments[0].C)
            assert curve.segments[0].stress_threshold == pytest.approx(
                base.segments[0].stress_threshold
            )

    @pytest.mark.parametrize("category", AWS_CATEGORIES)
    def test_all_categories_construct(self, category: str) -> None:
        curve = aws_curve(category)
        assert curve.standard == SNStandard.AWS
        assert len(curve.segments) == 1
        assert curve.segments[0].stress_threshold > 0.0

    def test_thresholds_decrease_from_a_to_ep(self) -> None:
        thresholds = [
            aws_curve(c).segments[0].stress_threshold for c in AWS_CATEGORIES
        ]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_unknown_category_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown AWS category"):
            aws_curve("Z")


# ---------------------------------------------------------------------------
# Dispatcher and spec parsing
# ---------------------------------------------------------------------------


class TestDispatcher:
    """Test the unified dispatcher branches for the new standards."""

    def test_get_sn_curve_ec3(self) -> None:
        curve = get_sn_curve("ec3", "90")
        assert curve.name == "EC3 detail category 90"

    def test_get_sn_curve_eurocode3_alias(self) -> None:
        curve = get_sn_curve("eurocode3", "71")
        assert curve.name == "EC3 detail category 71"

    def test_get_sn_curve_bs7608(self) -> None:
        curve = get_sn_curve("bs7608", "F2")
        assert curve.name == "BS 7608 class F2"

    def test_get_sn_curve_bs_alias(self) -> None:
        curve = get_sn_curve("bs", "D")
        assert curve.name == "BS 7608 class D"

    def test_get_sn_curve_aws(self) -> None:
        curve = get_sn_curve("aws", "C")
        assert curve.name == "AWS D1.1 category C"

    def test_parse_spec_ec3(self) -> None:
        curve = parse_sn_spec("EC3_90")
        assert curve.standard == SNStandard.EC3
        assert curve.life(90.0) == pytest.approx(2.0e6, rel=1e-9)

    def test_parse_spec_bs7608(self) -> None:
        curve = parse_sn_spec("BS7608_F2")
        assert curve.name == "BS 7608 class F2"

    def test_parse_spec_aws(self) -> None:
        curve = parse_sn_spec("AWS_C")
        assert curve.name == "AWS D1.1 category C"

    def test_parse_spec_aws_primed(self) -> None:
        curve = parse_sn_spec("AWS_BP")
        assert curve.name == "AWS D1.1 category B'"

    def test_unknown_standard_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown standard"):
            get_sn_curve("iso", "90")


class TestSNStandardEnum:
    """New SNStandard members and case-insensitive lookup."""

    def test_new_members(self) -> None:
        assert SNStandard.EC3.value == "ec3"
        assert SNStandard.BS7608.value == "bs7608"
        assert SNStandard.AWS.value == "aws"

    def test_case_insensitive_lookup(self) -> None:
        assert SNStandard("EC3") is SNStandard.EC3
        assert SNStandard("Bs7608") is SNStandard.BS7608
        assert SNStandard("AWS") is SNStandard.AWS
