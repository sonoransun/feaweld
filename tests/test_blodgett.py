"""Tests for the Blodgett weld-as-a-line calculations."""

from __future__ import annotations

import math

import pytest

from feaweld.core.types import WeldGroupProperties, WeldGroupShape
from feaweld.postprocess.blodgett import (
    asd_capacity,
    lrfd_capacity,
    weld_group_properties,
    weld_stress,
)


# ---------------------------------------------------------------------------
# Weld group section properties
# ---------------------------------------------------------------------------


class TestWeldGroupProperties:
    """Verify section properties against hand-calculated values."""

    def test_line(self) -> None:
        d = 200.0
        props = weld_group_properties(WeldGroupShape.LINE, d)
        assert props.A_w == pytest.approx(200.0)
        assert props.I_x == pytest.approx(200.0 ** 3 / 12.0)
        assert props.S_x == pytest.approx(200.0 ** 2 / 6.0)
        assert props.J_w == pytest.approx(200.0 ** 3 / 12.0)

    def test_box(self) -> None:
        d = 200.0
        b = 100.0
        props = weld_group_properties(WeldGroupShape.BOX, d, b)

        assert props.A_w == pytest.approx(2 * 100 + 2 * 200)
        assert props.I_x == pytest.approx(200 ** 2 * (3 * 100 + 200) / 6.0)
        expected_Sx = props.I_x / (d / 2.0)
        assert props.S_x == pytest.approx(expected_Sx)
        assert props.J_w == pytest.approx((100 + 200) ** 3 / 6.0)

    def test_circular(self) -> None:
        d = 150.0
        props = weld_group_properties(WeldGroupShape.CIRCULAR, d)

        assert props.A_w == pytest.approx(math.pi * 150.0)
        assert props.I_x == pytest.approx(math.pi * 150.0 ** 3 / 8.0)
        assert props.S_x == pytest.approx(math.pi * 150.0 ** 2 / 4.0)
        assert props.J_w == pytest.approx(math.pi * 150.0 ** 3 / 4.0)

    def test_parallel(self) -> None:
        d = 200.0
        b = 100.0
        props = weld_group_properties(WeldGroupShape.PARALLEL, d, b)
        assert props.A_w == pytest.approx(400.0)
        assert props.I_x == pytest.approx(200 * 100 ** 2 / 2.0)
        assert props.S_x == pytest.approx(200 * 100)

    def test_i_shape(self) -> None:
        d = 300.0
        b = 150.0
        props = weld_group_properties(WeldGroupShape.I_SHAPE, d, b)
        assert props.A_w == pytest.approx(300.0)
        assert props.I_x == pytest.approx(150 * 300 ** 2 / 2.0)
        assert props.S_x == pytest.approx(150 * 300)

    def test_unknown_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown weld group shape"):
            weld_group_properties("not_a_shape", 100.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Weld stress
# ---------------------------------------------------------------------------


class TestWeldStress:
    """Verify stress calculations."""

    def test_pure_axial(self) -> None:
        props = weld_group_properties(WeldGroupShape.LINE, d=200.0)
        throat = 5.0
        P = 10_000.0  # N

        result = weld_stress(props, throat, P=P)
        expected_fa = P / (200.0 * 5.0)
        assert result["f_a"] == pytest.approx(expected_fa)
        assert result["f_v"] == pytest.approx(0.0)
        assert result["f_b"] == pytest.approx(0.0)
        # f_n = f_a + f_b
        assert result["f_n"] == pytest.approx(expected_fa)
        # f_r = sqrt(f_n^2 + f_v^2) = f_n for zero shear
        assert result["f_r"] == pytest.approx(expected_fa)

    def test_combined_loads(self) -> None:
        """Axial + shear should give non-zero resultant."""
        props = weld_group_properties(WeldGroupShape.BOX, d=200.0, b=100.0)
        throat = 6.0
        result = weld_stress(props, throat, P=5000.0, V=3000.0, M=1e6)

        assert result["f_a"] > 0
        assert result["f_v"] > 0
        assert result["f_b"] > 0
        assert result["f_r"] > result["f_a"]
        assert result["von_mises"] > 0


# ---------------------------------------------------------------------------
# LRFD and ASD capacity
# ---------------------------------------------------------------------------


class TestCapacity:
    """Check AISC weld capacity formulas."""

    def test_lrfd_capacity(self) -> None:
        # 8mm fillet, throat = 8/sqrt(2) ~ 5.657
        throat = 8.0 / math.sqrt(2.0)
        A_w = 200.0  # 200mm weld length
        F_EXX = 483.0  # E70XX

        cap = lrfd_capacity(throat, A_w, F_EXX)
        expected = 0.75 * 0.60 * 483.0 * 200.0 * throat
        assert cap == pytest.approx(expected)
        assert cap > 0

    def test_asd_capacity(self) -> None:
        throat = 8.0 / math.sqrt(2.0)
        A_w = 200.0
        F_EXX = 483.0

        cap = asd_capacity(throat, A_w, F_EXX)
        expected = 0.60 * 483.0 * 200.0 * throat / 2.0
        assert cap == pytest.approx(expected)
        assert cap > 0

    def test_lrfd_greater_than_asd(self) -> None:
        """LRFD capacity should exceed ASD for same parameters."""
        throat = 6.0 / math.sqrt(2.0)
        A_w = 300.0
        lrfd = lrfd_capacity(throat, A_w)
        asd = asd_capacity(throat, A_w)
        # LRFD/ASD ratio = 0.75/0.5 = 1.5
        assert lrfd / asd == pytest.approx(1.5)
