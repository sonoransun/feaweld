"""Tests for thermal solver utilities."""

import numpy as np
import pytest


def test_goldak_heat_source_peak():
    """Test Goldak heat source has peak at center."""
    from feaweld.solver.thermal import GoldakHeatSource

    source = GoldakHeatSource(
        power=5000.0,  # 5 kW
        a_f=5.0,
        a_r=10.0,
        b=5.0,
        c=5.0,
        travel_speed=5.0,
    )

    # At t=0, center should be at start_position (origin)
    x = np.array([0.0])
    y = np.array([0.0])
    z = np.array([0.0])

    Q_center = source.evaluate(x, y, z, t=0.0)
    Q_offset = source.evaluate(np.array([10.0]), y, z, t=0.0)

    assert Q_center[0] > 0
    assert Q_center[0] > Q_offset[0]  # Peak at center


def test_goldak_heat_source_moves():
    """Test heat source moves with travel speed."""
    from feaweld.solver.thermal import GoldakHeatSource

    source = GoldakHeatSource(
        power=5000.0,
        a_f=5.0,
        a_r=10.0,
        b=5.0,
        c=5.0,
        travel_speed=10.0,
        start_position=np.zeros(3),
        direction=np.array([1.0, 0.0, 0.0]),
    )

    # At t=1s, center should be at x=10
    x_at_10 = np.array([10.0])
    y = np.array([0.0])
    z = np.array([0.0])

    Q_at_center_t1 = source.evaluate(x_at_10, y, z, t=1.0)
    Q_at_origin_t1 = source.evaluate(np.array([0.0]), y, z, t=1.0)

    assert Q_at_center_t1[0] > Q_at_origin_t1[0]


def test_goldak_symmetry():
    """Test heat source is symmetric about y and z axes."""
    from feaweld.solver.thermal import GoldakHeatSource

    source = GoldakHeatSource(
        power=5000.0, a_f=5.0, a_r=10.0, b=5.0, c=5.0,
    )

    y_pos = source.evaluate(np.array([0.0]), np.array([2.0]), np.array([0.0]), t=0.0)
    y_neg = source.evaluate(np.array([0.0]), np.array([-2.0]), np.array([0.0]), t=0.0)

    np.testing.assert_allclose(y_pos, y_neg, rtol=1e-10)


def test_pwht_schedule_profile():
    """Test PWHT temperature profile generation."""
    from feaweld.core.loads import PWHTSchedule

    schedule = PWHTSchedule(
        heating_rate=100.0,   # C/hour
        holding_temperature=620.0,
        holding_time=2.0,     # hours
        cooling_rate=50.0,    # C/hour
    )

    times, temps = schedule.temperature_profile(dt=300.0)  # 5-minute steps

    assert len(times) == len(temps)
    assert temps[0] == pytest.approx(20.0, abs=1.0)  # starts at ambient
    assert np.max(temps) == pytest.approx(620.0, abs=5.0)  # reaches holding
    assert temps[-1] < 100.0  # cools back down


def test_norton_bailey_rate():
    """Test Norton-Bailey creep rate computation."""
    from feaweld.solver.creep import norton_bailey_rate

    # σ = 100 MPa, t = 1 hour, A = 1e-20, n = 5
    rate = norton_bailey_rate(
        stress=np.array([100.0]),
        time=3600.0,
        A=1e-20,
        n=5.0,
        m=0.0,
    )
    assert rate[0] > 0
    assert np.isfinite(rate[0])

    # Higher stress → higher rate
    rate_high = norton_bailey_rate(
        stress=np.array([200.0]),
        time=3600.0,
        A=1e-20,
        n=5.0,
        m=0.0,
    )
    assert rate_high[0] > rate[0]
