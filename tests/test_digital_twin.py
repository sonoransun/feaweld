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


# ---------------------------------------------------------------------------
# Bayesian model updating (pure-python core + one slow MCMC recovery test)
# ---------------------------------------------------------------------------

from feaweld.digital_twin.bayesian import (
    BayesianUpdater, ObservedData, PriorSpec, _gelman_rubin,
)


def _linear_forward(params):
    """Trivial forward model: observations = [k, 2k]."""
    return np.array([params["k"], 2.0 * params["k"]])


class TestPriorSpec:
    def test_log_prior_inside_support_is_finite(self):
        prior = PriorSpec("k", "normal", {"mean": 1.0, "std": 1.0}, bounds=(0.0, 5.0))
        assert np.isfinite(prior.log_prior(1.0))

    def test_log_prior_outside_support_is_neg_inf(self):
        prior = PriorSpec("k", "normal", {"mean": 1.0, "std": 1.0}, bounds=(0.0, 5.0))
        assert prior.log_prior(-1.0) == -np.inf
        assert prior.log_prior(6.0) == -np.inf


class TestBayesianProbabilities:
    def _updater(self):
        prior = PriorSpec("k", "normal", {"mean": 1.0, "std": 1.0}, bounds=(0.0, 5.0))
        return BayesianUpdater(
            priors=[prior], forward_model=_linear_forward,
            n_walkers=8, n_steps=40, n_burnin=10,
        )

    def _obs(self):
        return ObservedData(
            measurement_type="strain",
            positions=np.zeros((2, 3)),
            values=np.array([2.0, 4.0]),
            uncertainty=0.2,
        )

    def test_log_likelihood_finite(self):
        ll = self._updater().log_likelihood(np.array([2.0]), [self._obs()])
        assert np.isfinite(ll)

    def test_log_posterior_is_prior_plus_likelihood(self):
        updater = self._updater()
        theta = np.array([2.0])
        obs = [self._obs()]
        expected = updater.log_prior(theta) + updater.log_likelihood(theta, obs)
        assert updater.log_posterior(theta, obs) == pytest.approx(expected)

    def test_log_posterior_neg_inf_outside_prior(self):
        updater = self._updater()
        # k = 10 is outside the (0, 5) support -> prior kills the posterior.
        assert updater.log_posterior(np.array([10.0]), [self._obs()]) == -np.inf


class TestGelmanRubin:
    def test_rhat_near_one_for_iid_normal(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, 1.0, 8000)
        rhat = _gelman_rubin(samples, n_chains=8)
        assert abs(rhat - 1.0) < 0.05


@pytest.mark.slow
def test_bayesian_update_recovers_parameter():
    """A short MCMC run recovers a known parameter within 2σ."""
    pytest.importorskip("emcee")
    np.random.seed(42)

    prior = PriorSpec("k", "normal", {"mean": 1.0, "std": 2.0}, bounds=(0.0, 5.0))
    updater = BayesianUpdater(
        priors=[prior], forward_model=_linear_forward,
        n_walkers=8, n_steps=40, n_burnin=10,
    )
    obs = ObservedData(
        measurement_type="strain",
        positions=np.zeros((2, 3)),
        values=np.array([2.0, 4.0]),  # consistent with k = 2
        uncertainty=0.2,
    )

    summary = updater.update(obs)
    assert abs(summary.means["k"] - 2.0) < 2.0 * summary.stds["k"]


# ---------------------------------------------------------------------------
# Dashboard web page + serving (HTTP static page, WebSocket contract, demo feed)
# ---------------------------------------------------------------------------

import asyncio
import json
import urllib.error
import urllib.request
from importlib.resources import files


class TestDashboardPage:
    def test_page_resource_exists_and_has_contract(self):
        page = (files("feaweld.digital_twin") / "static" / "dashboard.html").read_text()
        assert page.lstrip().lower().startswith("<!doctype")
        # The browser client must speak the four server message types.
        for message_type in ("state", "sensor", "alert", "predictions"):
            assert f'"{message_type}"' in page, message_type
        assert "WebSocket" in page


class TestHTTPServing:
    def test_serves_page_and_404(self):
        from feaweld.digital_twin.serve import _start_http_server

        httpd = _start_http_server(0, host="localhost")
        try:
            port = httpd.server_address[1]
            with urllib.request.urlopen(
                f"http://localhost:{port}/dashboard.html", timeout=5
            ) as resp:
                assert resp.status == 200
                body = resp.read().decode()
            assert body.lstrip().lower().startswith("<!doctype")

            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(f"http://localhost:{port}/nope", timeout=5)
            assert excinfo.value.code == 404
        finally:
            httpd.shutdown()


class TestWebSocketContract:
    def test_connect_predictions_and_sensor_push(self):
        websockets = pytest.importorskip("websockets")
        from feaweld.digital_twin.dashboard import DashboardServer
        from feaweld.digital_twin.ingest import SensorReading

        async def scenario():
            server = DashboardServer(host="localhost", port=0)
            await server.start()
            port = server._server.sockets[0].getsockname()[1]
            try:
                async with websockets.connect(f"ws://localhost:{port}") as ws:
                    # On connect the server pushes the current lifecycle state.
                    first = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    assert first["type"] == "state"
                    assert "state" in first

                    # Request predictions -> typed predictions message.
                    await ws.send(json.dumps({"type": "get_predictions"}))
                    reply = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    assert reply["type"] == "predictions"

                    # A reading added to the buffer (from within the loop) is
                    # broadcast as a fully-typed sensor message.
                    server.buffer.add(SensorReading(
                        timestamp=time.time(), sensor_id="s1",
                        channel="temperature", value=42.0, unit="C",
                    ))
                    sensor_msg = None
                    for _ in range(10):
                        msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
                        if msg.get("type") == "sensor":
                            sensor_msg = msg
                            break
                    assert sensor_msg is not None
                    assert sensor_msg["sensor_id"] == "s1"
                    assert sensor_msg["channel"] == "temperature"
                    assert sensor_msg["value"] == 42.0
                    assert "timestamp" in sensor_msg
            finally:
                await server.stop()

        asyncio.run(scenario())


class TestDemoFeed:
    def test_demo_feed_populates_buffer(self):
        pytest.importorskip("websockets")
        from feaweld.digital_twin.dashboard import DashboardServer
        from feaweld.digital_twin.serve import _demo_feed

        async def scenario():
            server = DashboardServer(host="localhost", port=0)
            await server.start()
            task = asyncio.ensure_future(_demo_feed(server))
            try:
                await asyncio.sleep(1.0)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            assert len(server.buffer.channels) > 0
            await server.stop()

        asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Concurrency / memory-bound regression tests
# ---------------------------------------------------------------------------

import threading


class TestAlertEngineBounds:
    """AlertEngine retains bounded history/alerts regardless of trend config."""

    def test_history_bounded_for_non_trend_channel(self):
        # No trend rule matches "vibration", so the time-window trim never runs;
        # the unconditional per-channel cap must still bound the history.
        engine = AlertEngine()
        for i in range(2500):
            engine.check(SensorReading(
                timestamp=float(i), sensor_id="s1", channel="vibration", value=float(i),
            ))
        key = "s1:vibration"
        assert key in engine._history
        assert len(engine._history[key]) <= engine._max_history_per_channel
        # The most recent samples are the ones retained.
        assert engine._history[key][-1] == (2499.0, 2499.0)

    def test_alerts_bounded(self):
        # A threshold that fires on every reading would otherwise grow _alerts
        # without bound over a long monitoring run.
        engine = AlertEngine()
        engine.add_threshold(ThresholdConfig(channel_pattern="temperature", warning_high=0.0))
        for i in range(2500):
            engine.check(SensorReading(
                timestamp=float(i), sensor_id="s1", channel="temperature", value=100.0,
            ))
        assert len(engine._alerts) <= engine._max_alerts
        # Public read still slices the retained alerts.
        assert len(engine.recent_alerts) == 100


class TestDashboardThreadSafety:
    """Broadcasts scheduled from foreign threads must reach connected clients."""

    def test_sensor_callback_from_foreign_thread_broadcasts(self):
        websockets = pytest.importorskip("websockets")
        from feaweld.digital_twin.dashboard import DashboardServer
        from feaweld.digital_twin.ingest import SensorReading

        async def scenario():
            server = DashboardServer(host="localhost", port=0)
            await server.start()
            port = server._server.sockets[0].getsockname()[1]
            try:
                async with websockets.connect(f"ws://localhost:{port}") as ws:
                    # Drain the on-connect state message.
                    first = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    assert first["type"] == "state"

                    # Deliver a reading from a *foreign* thread with no running
                    # loop, exactly as paho-mqtt's network thread would. This is
                    # the case the old asyncio.get_event_loop() path dropped.
                    reading = SensorReading(
                        timestamp=time.time(), sensor_id="mqtt1",
                        channel="temperature", value=123.4, unit="C",
                    )
                    threading.Thread(
                        target=server.buffer.add, args=(reading,)
                    ).start()

                    sensor_msg = None
                    for _ in range(10):
                        msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
                        if msg.get("type") == "sensor":
                            sensor_msg = msg
                            break
                    assert sensor_msg is not None
                    assert sensor_msg["sensor_id"] == "mqtt1"
                    assert sensor_msg["channel"] == "temperature"
                    assert sensor_msg["value"] == 123.4
            finally:
                await server.stop()

        asyncio.run(scenario())

    def test_update_predictions_from_foreign_thread(self):
        pytest.importorskip("websockets")
        from feaweld.digital_twin.dashboard import DashboardServer

        async def scenario():
            server = DashboardServer(host="localhost", port=0)
            await server.start()
            try:
                errors: list[BaseException] = []

                def do_update():
                    try:
                        server.update_predictions({"fatigue_life_cycles": 1.0e6})
                    except BaseException as exc:  # pragma: no cover - failure path
                        errors.append(exc)

                t = threading.Thread(target=do_update)
                t.start()
                t.join(5)
                # Let the scheduled broadcast run on the loop.
                await asyncio.sleep(0.05)

                assert not errors, errors
                assert server._model_predictions == {"fatigue_life_cycles": 1.0e6}
            finally:
                await server.stop()

        asyncio.run(scenario())

    def test_update_predictions_before_start_does_not_raise(self):
        # No loop captured yet: the broadcast is dropped (not raised) and the
        # prediction is still stored for the next get_predictions request.
        from feaweld.digital_twin.dashboard import DashboardServer

        server = DashboardServer(host="localhost", port=0)
        server.update_predictions({"reliability_beta": 3.1})
        assert server._model_predictions == {"reliability_beta": 3.1}
