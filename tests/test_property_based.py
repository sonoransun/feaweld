"""Property-based tests using hypothesis for stress tensor invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from feaweld.core.types import StressField, SNCurve, SNSegment, SNStandard


# ---------------------------------------------------------------------------
# Strategy: generate a single stress tensor as 6-component Voigt vector
# ---------------------------------------------------------------------------

_stress_component = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)

_stress_tensor_strategy = st.tuples(
    _stress_component,  # sigma_xx
    _stress_component,  # sigma_yy
    _stress_component,  # sigma_zz
    _stress_component,  # tau_xy
    _stress_component,  # tau_yz
    _stress_component,  # tau_xz
)


def _make_stress_field(tensor_tuple: tuple[float, ...]) -> StressField:
    """Wrap a single 6-tuple into a StressField with one point."""
    vals = np.array([list(tensor_tuple)], dtype=np.float64)
    return StressField(values=vals)


# ---------------------------------------------------------------------------
# Von Mises properties
# ---------------------------------------------------------------------------


class TestVonMises:

    @given(tensor=_stress_tensor_strategy)
    @settings(max_examples=200)
    def test_von_mises_nonnegative(self, tensor):
        """von Mises equivalent stress must always be >= 0."""
        sf = _make_stress_field(tensor)
        vm = sf.von_mises
        assert vm[0] >= 0.0

    @given(p=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_von_mises_hydrostatic_zero(self, p):
        """For a pure hydrostatic stress state (all normal equal, no shear),
        the von Mises stress should be approximately zero."""
        sf = _make_stress_field((p, p, p, 0.0, 0.0, 0.0))
        vm = sf.von_mises[0]
        np.testing.assert_allclose(vm, 0.0, atol=1e-8)

    @given(tensor=_stress_tensor_strategy)
    @settings(max_examples=200)
    def test_von_mises_deviatoric_relation(self, tensor):
        """von Mises should equal sqrt(3/2) * ||deviatoric||_F for normal components."""
        sf = _make_stress_field(tensor)
        vm = sf.von_mises[0]
        # Octahedral shear = sqrt(2)/3 * vm, so vm = 3/sqrt(2) * tau_oct
        tau_oct = sf.octahedral_shear[0]
        np.testing.assert_allclose(vm, 3.0 / np.sqrt(2.0) * tau_oct, rtol=1e-10)


# ---------------------------------------------------------------------------
# Principal stress properties
# ---------------------------------------------------------------------------


class TestPrincipalStress:

    @given(tensor=_stress_tensor_strategy)
    @settings(max_examples=200)
    def test_principal_stress_sum_equals_trace(self, tensor):
        """Sum of principal stresses must equal the trace sigma_xx + sigma_yy + sigma_zz."""
        sf = _make_stress_field(tensor)
        principals = sf.principal[0]  # (3,) sorted ascending
        trace = tensor[0] + tensor[1] + tensor[2]
        np.testing.assert_allclose(np.sum(principals), trace, rtol=1e-8, atol=1e-8)

    @given(tensor=_stress_tensor_strategy)
    @settings(max_examples=200)
    def test_principal_stresses_sorted_ascending(self, tensor):
        """Principal stresses should be sorted in ascending order."""
        sf = _make_stress_field(tensor)
        principals = sf.principal[0]
        assert principals[0] <= principals[1] + 1e-10
        assert principals[1] <= principals[2] + 1e-10


# ---------------------------------------------------------------------------
# Tresca properties
# ---------------------------------------------------------------------------


class TestTresca:

    @given(tensor=_stress_tensor_strategy)
    @settings(max_examples=200)
    def test_tresca_nonnegative(self, tensor):
        """Tresca stress (max shear) must always be >= 0."""
        sf = _make_stress_field(tensor)
        tresca = sf.tresca[0]
        assert tresca >= -1e-10  # allow tiny float rounding

    @given(tensor=_stress_tensor_strategy)
    @settings(max_examples=200)
    def test_von_mises_bounded_by_tresca(self, tensor):
        """Von Mises is bounded: sqrt(3)/2 * tresca <= vm <= tresca."""
        sf = _make_stress_field(tensor)
        tresca = sf.tresca[0]
        vm = sf.von_mises[0]
        # Upper bound: von Mises never exceeds Tresca
        assert vm <= tresca + 1e-8
        # Lower bound: von Mises >= sqrt(3)/2 * Tresca
        assert vm >= (np.sqrt(3.0) / 2.0) * tresca - 1e-8

    @given(p=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_tresca_hydrostatic_zero(self, p):
        """For pure hydrostatic stress, Tresca should be zero."""
        sf = _make_stress_field((p, p, p, 0.0, 0.0, 0.0))
        tresca = sf.tresca[0]
        np.testing.assert_allclose(tresca, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# S-N curve monotonicity
# ---------------------------------------------------------------------------


class TestSNCurveMonotonicity:

    @given(
        s1=st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
        s2=st.floats(min_value=10.0, max_value=500.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_sn_life_monotonically_decreasing(self, s1, s2):
        """Higher stress range should give lower (or equal) fatigue life."""
        assume(abs(s1 - s2) > 0.01)
        curve = SNCurve(
            name="TestFAT90",
            standard=SNStandard.IIW,
            segments=[
                SNSegment(m=3.0, C=90.0**3 * 2e6, stress_threshold=0.0),
            ],
            cutoff_cycles=1e7,
        )
        n1 = curve.life(s1)
        n2 = curve.life(s2)

        if s1 > s2:
            assert n1 <= n2
        else:
            assert n1 >= n2

    @given(s=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False))
    @settings(max_examples=100)
    def test_sn_life_positive(self, s):
        """Fatigue life for a positive stress range should be positive and finite."""
        curve = SNCurve(
            name="TestFAT90",
            standard=SNStandard.IIW,
            segments=[
                SNSegment(m=3.0, C=90.0**3 * 2e6, stress_threshold=0.0),
            ],
            cutoff_cycles=1e7,
        )
        life = curve.life(s)
        assert life > 0
        assert np.isfinite(life)
