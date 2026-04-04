"""Tests for digital twin modules."""

import time
import numpy as np
import pytest

from feaweld.digital_twin.ingest import (
    SensorReading, SensorDataBuffer, parse_thermocouple_array, parse_arc_waveform,
)
from feaweld.digital_twin.dashboard import (
    AlertEngine, ThresholdConfig, TrendConfig, WeldStateMachine, WeldLifecycleState, Alert,
)


class TestSensorDataBuffer:
    def test_add_and_retrieve(self):
        buffer = SensorDataBuffer(window_seconds=60.0)
        reading = SensorReading(
            timestamp=time.time(),
            sensor_id="tc1",
            channel="temperature",
            value=350.0,
            unit="C",
        )
        buffer.add(reading)
        result = buffer.get_channel("tc1", "temperature")
        assert len(result) == 1
        assert result[0].value == 350.0

    def test_window_trimming(self):
        buffer = SensorDataBuffer(window_seconds=10.0)
        now = time.time()

        # Add old reading
        buffer.add(SensorReading(timestamp=now - 20, sensor_id="s1", channel="ch1", value=1.0))
        # Add recent reading
        buffer.add(SensorReading(timestamp=now, sensor_id="s1", channel="ch1", value=2.0))

        result = buffer.get_channel("s1", "ch1")
        assert len(result) == 1
        assert result[0].value == 2.0

    def test_aligned_data(self):
        buffer = SensorDataBuffer(window_seconds=60.0)
        now = time.time()

        for i in range(10):
            buffer.add(SensorReading(timestamp=now + i, sensor_id="s1", channel="temp", value=float(100 + i)))
            buffer.add(SensorReading(timestamp=now + i, sensor_id="s2", channel="strain", value=float(0.001 * i)))

        times, values = buffer.get_aligned([("s1", "temp"), ("s2", "strain")], dt=1.0)
        assert len(times) > 0
        assert values.shape[1] == 2

    def test_callback(self):
        buffer = SensorDataBuffer()
        received = []
        buffer.on_data(lambda r: received.append(r))
        buffer.add(SensorReading(timestamp=time.time(), sensor_id="s1", channel="c1", value=42.0))
        assert len(received) == 1


class TestAlertEngine:
    def test_threshold_alert(self):
        engine = AlertEngine()
        engine.add_threshold(ThresholdConfig(
            channel_pattern="temperature",
            warning_high=500.0,
            critical_high=800.0,
        ))

        reading = SensorReading(
            timestamp=time.time(), sensor_id="s1", channel="temperature", value=600.0,
        )
        alerts = engine.check(reading)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"

    def test_critical_threshold(self):
        engine = AlertEngine()
        engine.add_threshold(ThresholdConfig(
            channel_pattern="temperature",
            critical_high=800.0,
        ))
        reading = SensorReading(
            timestamp=time.time(), sensor_id="s1", channel="temperature", value=900.0,
        )
        alerts = engine.check(reading)
        assert any(a.severity == "critical" for a in alerts)

    def test_no_alert_within_range(self):
        engine = AlertEngine()
        engine.add_threshold(ThresholdConfig(
            channel_pattern="temperature",
            warning_high=500.0,
        ))
        reading = SensorReading(
            timestamp=time.time(), sensor_id="s1", channel="temperature", value=300.0,
        )
        alerts = engine.check(reading)
        assert len(alerts) == 0


class TestWeldStateMachine:
    def test_initial_state(self):
        sm = WeldStateMachine()
        assert sm.state == WeldLifecycleState.IDLE

    def test_welding_transition(self):
        sm = WeldStateMachine()
        sm.transition("start_welding")
        assert sm.state == WeldLifecycleState.WELDING

    def test_full_lifecycle(self):
        sm = WeldStateMachine()
        sm.transition("start_welding")
        assert sm.state == WeldLifecycleState.WELDING

        sm.transition("stop_welding")
        assert sm.state == WeldLifecycleState.COOLING

        sm.transition("start_pwht")
        assert sm.state == WeldLifecycleState.PWHT

        sm.transition("pwht_complete")
        assert sm.state == WeldLifecycleState.MONITORING

    def test_invalid_transition_stays(self):
        sm = WeldStateMachine()
        sm.transition("stop_welding")  # invalid from IDLE
        assert sm.state == WeldLifecycleState.IDLE


class TestParsers:
    def test_thermocouple_array(self):
        result = parse_thermocouple_array(
            raw_values=[100, 300, 500, 400, 200],
            positions_mm=[0, 5, 10, 15, 20],
        )
        assert result["peak_temperature"] == 500.0
        assert result["peak_position"] == 10.0
        assert len(result["gradient"]) == 5

    def test_arc_waveform(self):
        current = np.array([250.0, 255.0, 248.0, 252.0])
        voltage = np.array([25.0, 25.5, 24.8, 25.2])
        result = parse_arc_waveform(current, voltage, sample_rate=1000.0)
        assert result["current_mean"] > 0
        assert result["voltage_mean"] > 0
        assert result["power_mean"] > 0
