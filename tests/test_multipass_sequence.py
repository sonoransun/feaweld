"""Tests for multi-pass weld sequencing and dispatching heat source (Track G)."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import WeldPass, WeldSequence
from feaweld.solver.thermal import MultiPassHeatSource


def _make_three_pass_sequence() -> WeldSequence:
    """Three 10s passes separated by 1s gaps: [0,10) [11,21) [22,32)."""
    return WeldSequence(
        passes=[
            WeldPass(order=1, pass_type="root", start_time=0.0, duration=10.0),
            WeldPass(order=2, pass_type="fill", start_time=11.0, duration=10.0),
            WeldPass(order=3, pass_type="cap", start_time=22.0, duration=10.0),
        ]
    )


def test_weldsequence_monotone_start_times_required():
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        WeldSequence(
            passes=[
                WeldPass(order=1, start_time=10.0, duration=5.0),
                WeldPass(order=2, start_time=5.0, duration=5.0),
            ]
        )


def test_total_duration():
    seq = _make_three_pass_sequence()
    # Latest pass ends at 22 + 10 = 32.
    assert seq.total_duration() == pytest.approx(32.0)


def test_total_duration_empty_sequence():
    assert WeldSequence(passes=[]).total_duration() == 0.0


def test_active_pass_at():
    seq = _make_three_pass_sequence()

    # Before any pass
    assert seq.active_pass_at(-1.0) is None

    # During pass 1
    p = seq.active_pass_at(5.0)
    assert p is not None and p.order == 1

    # Gap between passes 1 and 2
    assert seq.active_pass_at(10.5) is None

    # During pass 2
    p = seq.active_pass_at(15.0)
    assert p is not None and p.order == 2

    # Gap between passes 2 and 3
    assert seq.active_pass_at(21.5) is None

    # During pass 3
    p = seq.active_pass_at(25.0)
    assert p is not None and p.order == 3

    # After all passes
    assert seq.active_pass_at(100.0) is None

    # Exactly at the end of pass 1 (half-open interval) → None
    assert seq.active_pass_at(10.0) is None
    # Exactly at the start of pass 2 → pass 2
    p = seq.active_pass_at(11.0)
    assert p is not None and p.order == 2


def test_multipass_heat_source_power_history():
    seq = _make_three_pass_sequence()
    src = MultiPassHeatSource(sequence=seq)

    # Three 10s passes each with default 25 V * 200 A * 0.8 = 4000 W.
    expected_power = 25.0 * 200.0 * 0.8

    times = np.array(
        [
            -0.5,   # before
            0.0,    # start of pass 1
            5.0,    # mid pass 1
            10.0,   # end of pass 1 (inactive)
            10.5,   # gap
            11.0,   # start of pass 2
            15.0,   # mid pass 2
            21.0,   # end of pass 2 (inactive)
            22.0,   # start of pass 3
            31.99,  # end of pass 3 (active)
            40.0,   # after
        ]
    )
    powers = src.power_history(times)

    assert powers[0] == 0.0
    assert powers[1] == pytest.approx(expected_power)
    assert powers[2] == pytest.approx(expected_power)
    assert powers[3] == 0.0
    assert powers[4] == 0.0
    assert powers[5] == pytest.approx(expected_power)
    assert powers[6] == pytest.approx(expected_power)
    assert powers[7] == 0.0
    assert powers[8] == pytest.approx(expected_power)
    assert powers[9] == pytest.approx(expected_power)
    assert powers[10] == 0.0


def _scalarize(q) -> float:
    """Convert a Goldak evaluate result (scalar or 1-element array) to float."""
    arr = np.asarray(q)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.ravel()[0])


def test_multipass_heat_source_evaluate_dispatches():
    seq = _make_three_pass_sequence()
    src = MultiPassHeatSource(sequence=seq)

    x = np.array([0.0])
    y = np.array([0.0])
    z = np.array([0.0])

    # Before the first pass → zero.
    assert _scalarize(src.evaluate(x, y, z, t=-0.5)) == 0.0

    # Mid-pass-1 at the time-shifted source origin → non-zero.
    assert _scalarize(src.evaluate(x, y, z, t=0.0)) > 0.0

    # In the gap between pass 1 and pass 2 → zero.
    assert _scalarize(src.evaluate(x, y, z, t=10.5)) == 0.0

    # At the start of pass 2, the source origin resets; query at the
    # pass-start location (which for the default Goldak parameters is
    # the global origin again) should be non-zero.
    assert _scalarize(src.evaluate(x, y, z, t=11.0)) > 0.0

    # After the final pass → zero.
    assert _scalarize(src.evaluate(x, y, z, t=100.0)) == 0.0
