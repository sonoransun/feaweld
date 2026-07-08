"""Serve the digital-twin dashboard: static page over HTTP + live WebSocket feed.

This module wires together the [DashboardServer][feaweld.digital_twin.dashboard.DashboardServer] WebSocket service with a
minimal HTTP server that hands out the single-page dashboard (``static/dashboard.html``).
It also provides a self-contained demo feed so the dashboard can be exercised end to
end without any physical sensors, MQTT broker, or OPC-UA server.

The public entry point is [run_dashboard][feaweld.digital_twin.serve.run_dashboard], which the CLI calls.
"""

from __future__ import annotations

import asyncio
import math
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any

from feaweld.digital_twin.dashboard import (
    AlertEngine,
    DashboardServer,
    ThresholdConfig,
    TrendConfig,
    WeldLifecycleState,
    WeldStateMachine,
)
from feaweld.digital_twin.ingest import SensorDataBuffer, SensorReading

__all__ = ["run_dashboard"]

# Compressed weld-lifecycle timeline used by the demo feed (seconds).
_DEMO_CYCLE = 44.0
_PHASE_BOUNDS = (
    (2.0, "idle"),
    (10.0, "welding"),
    (20.0, "cooling"),
    (30.0, "pwht"),
    (_DEMO_CYCLE, "monitoring"),
)
# Explicit transition event to fire when entering each phase from the previous one.
_PHASE_EVENTS = {
    "welding": "start_welding",
    "cooling": "stop_welding",
    "pwht": "start_pwht",
    "monitoring": "pwht_complete",
}


def _load_page() -> bytes:
    """Read the bundled dashboard HTML page.

    Returns
    -------
    bytes
        UTF-8 encoded contents of ``digital_twin/static/dashboard.html``.
    """
    return (files("feaweld.digital_twin") / "static" / "dashboard.html").read_bytes()


def _make_http_handler(page_bytes: bytes) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class that serves only the dashboard page.

    Parameters
    ----------
    page_bytes : bytes
        The dashboard HTML to return for ``/`` and ``/dashboard.html``.

    Returns
    -------
    type[BaseHTTPRequestHandler]
        A handler class bound to ``page_bytes``. Any other path returns 404.
    """

    class DashboardHTTPHandler(BaseHTTPRequestHandler):
        server_version = "feaweld-twin/1.0"

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/dashboard.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page_bytes)))
                self.end_headers()
                self.wfile.write(page_bytes)
            else:
                self.send_error(404, "Not found")

        def log_message(self, *args: Any) -> None:  # silence per-request logging
            pass

    return DashboardHTTPHandler


def _start_http_server(
    http_port: int,
    host: str = "localhost",
    page_bytes: bytes | None = None,
) -> ThreadingHTTPServer:
    """Start the dashboard HTTP server on a daemon thread.

    Parameters
    ----------
    http_port : int
        Port to bind. Pass ``0`` to let the OS choose a free port (useful in tests);
        read the chosen port back from ``server.server_address[1]``.
    host : str, optional
        Interface to bind, by default ``"localhost"``.
    page_bytes : bytes, optional
        Pre-loaded page contents. Loaded from the package if omitted.

    Returns
    -------
    ThreadingHTTPServer
        The running server. Call ``.shutdown()`` to stop it.
    """
    if page_bytes is None:
        page_bytes = _load_page()
    handler = _make_http_handler(page_bytes)
    httpd = ThreadingHTTPServer((host, http_port), handler)
    thread = threading.Thread(target=httpd.serve_forever, name="feaweld-twin-http", daemon=True)
    thread.start()
    return httpd


def _build_server(
    ws_host: str,
    ws_port: int,
) -> DashboardServer:
    """Construct a [DashboardServer][feaweld.digital_twin.dashboard.DashboardServer] with sensible alert rules wired in.

    Registers temperature threshold (warning 900 C / critical 1200 C) and a rapid
    rate-of-change trend rule on any channel whose name contains ``"temperature"``,
    and arranges for weld-lifecycle transitions to be broadcast to clients (the stock
    server only pushes state on connect / on request).

    Parameters
    ----------
    ws_host, ws_port : str, int
        Bind address for the WebSocket server.

    Returns
    -------
    DashboardServer
        A configured, not-yet-started server.
    """
    buffer = SensorDataBuffer(window_seconds=120.0)

    alert_engine = AlertEngine()
    alert_engine.add_threshold(
        ThresholdConfig(channel_pattern="temperature", warning_high=900.0, critical_high=1200.0)
    )
    alert_engine.add_trend(
        TrendConfig(channel_pattern="temperature", window_seconds=10.0, max_rate_per_second=5000.0)
    )

    state_machine = WeldStateMachine()
    server = DashboardServer(
        host=ws_host,
        port=ws_port,
        buffer=buffer,
        alert_engine=alert_engine,
        state_machine=state_machine,
    )

    def _broadcast_transition(old: WeldLifecycleState, new: WeldLifecycleState) -> None:
        # Runs inside the event loop (transitions are driven on-loop), so ensure_future is safe.
        asyncio.ensure_future(
            server._broadcast({"type": "state", "state": new.value, "timestamp": time.time()})
        )

    state_machine.on_transition(_broadcast_transition)
    return server


def _phase_for(t: float) -> str:
    """Return the demo lifecycle phase for cycle-relative time ``t`` (seconds)."""
    for bound, name in _PHASE_BOUNDS:
        if t < bound:
            return name
    return "monitoring"


def _demo_temperature(t: float) -> float:
    """Process-thermocouple temperature (C) for the demo weld thermal lifecycle."""
    jitter = random.uniform(-6.0, 6.0)
    if t < 2.0:
        return 25.0 + random.uniform(-1.5, 1.5)
    if t < 10.0:  # welding: ramp to weld-pool temperature and hold
        return 25.0 + (1500.0 - 25.0) * min(1.0, (t - 2.0) / 3.0) + jitter
    if t < 20.0:  # cooling: exponential decay toward the PWHT hold
        return 620.0 + (1500.0 - 620.0) * math.exp(-(t - 10.0) / 2.5) + jitter
    if t < 30.0:  # post-weld heat treatment plateau
        return 620.0 + random.uniform(-9.0, 9.0)
    return 30.0 + (620.0 - 30.0) * math.exp(-(t - 30.0) / 3.5) + random.uniform(-3.0, 3.0)


def _demo_predictions(elapsed: float) -> dict[str, float]:
    """Synthetic model-prediction tiles that drift slowly with accumulated exposure."""
    wear = (elapsed % (4 * _DEMO_CYCLE)) / (4 * _DEMO_CYCLE)
    return {
        "fatigue_life_cycles": 2.45e6 * (1.0 - 0.05 * wear),
        "accumulated_damage": 0.012 + 0.05 * wear,
        "reliability_beta": 3.62 - 0.25 * wear,
    }


async def _demo_feed(server: DashboardServer, rate_hz: float = 5.0) -> None:
    """Push a synthetic weld lifecycle into the server, in the event loop.

    Readings must be added from within the running loop because the server's sensor
    callback schedules broadcasts with `asyncio.create_task`. This coroutine
    drives the same thermal lifecycle as the browser demo mode, advances the weld
    state machine through its transitions, and refreshes model predictions every 5 s.

    Alerts arise naturally from the registered thresholds: the structural
    ``temperature`` channel is held near ambient and spiked once per cycle to a
    warning level during welding and to a critical level during monitoring, so the
    real [AlertEngine][feaweld.digital_twin.dashboard.AlertEngine] produces a clean, occasional alert stream. The dramatic
    ``thermocouple_1`` process curve is deliberately not alarmed.

    Parameters
    ----------
    server : DashboardServer
        The running server whose buffer/state machine/predictions are driven.
    rate_hz : float, optional
        Sample rate for the feed, by default ``5.0``.
    """
    buffer = server.buffer
    sm = server.state_machine
    period = 1.0 / rate_hz
    start = time.time()
    last_phase: str | None = None
    last_pred = 0.0
    warned = False
    critted = False

    def push(sensor_id: str, channel: str, value: float, unit: str = "") -> None:
        buffer.add(
            SensorReading(
                timestamp=time.time(),
                sensor_id=sensor_id,
                channel=channel,
                value=float(value),
                unit=unit,
            )
        )

    while True:
        now = time.time()
        elapsed = now - start
        t = elapsed % _DEMO_CYCLE
        phase = _phase_for(t)

        if phase != last_phase:
            if phase == "idle":
                # Cycle wrap: MONITORING has no event back to IDLE, so reset directly.
                sm.state = WeldLifecycleState.IDLE
                await server._broadcast(
                    {"type": "state", "state": sm.state.value, "timestamp": now}
                )
                warned = False
                critted = False
            elif phase in _PHASE_EVENTS:
                sm.transition(_PHASE_EVENTS[phase])  # on_transition callback broadcasts it
            last_phase = phase

        welding = phase == "welding"
        # Process thermocouple (dramatic, not alarmed) + arc signals during welding.
        push("weld_torch", "thermocouple_1", _demo_temperature(t), "C")
        push(
            "weld_torch",
            "arc_current",
            220.0 + 18.0 * math.sin(t * 5.0) + random.uniform(-9.0, 9.0) if welding
            else abs(random.uniform(-1.5, 1.5)),
            "A",
        )
        push(
            "weld_torch",
            "arc_voltage",
            26.0 + 1.1 * math.sin(t * 3.0) + random.uniform(-0.6, 0.6) if welding
            else abs(random.uniform(-0.2, 0.2)),
            "V",
        )
        # Structural health channels: mild baseline temperature + strain.
        push("hss_node", "strain", 40.0 + 8.0 * math.sin(t * 0.8) + random.uniform(-4.0, 4.0), "ue")
        push("hss_node", "temperature", 65.0 + random.uniform(-3.0, 3.0), "C")

        # One warning spike during welding, one critical anomaly during monitoring.
        if welding and t >= 6.0 and not warned:
            warned = True
            push("hss_node", "temperature", 952.4, "C")
        if phase == "monitoring" and t >= 37.0 and not critted:
            critted = True
            push("hss_node", "temperature", 1304.7, "C")

        if now - last_pred >= 5.0:
            server.update_predictions(_demo_predictions(elapsed))
            last_pred = now

        await asyncio.sleep(period)


async def _serve(
    ws_host: str,
    ws_port: int,
    http_port: int,
    demo: bool,
    open_browser: bool,
) -> None:
    """Async orchestrator: start WS + HTTP servers, optionally the demo, run forever."""
    server = _build_server(ws_host, ws_port)
    await server.start()

    httpd = _start_http_server(http_port, host=ws_host)
    actual_http = httpd.server_address[1]

    if open_browser:
        url = f"http://{ws_host}:{actual_http}/dashboard.html?ws=ws://{ws_host}:{ws_port}"
        webbrowser.open(url)

    demo_task: asyncio.Task | None = None
    if demo:
        demo_task = asyncio.ensure_future(_demo_feed(server))

    try:
        await asyncio.Future()  # run until cancelled (KeyboardInterrupt / task cancel)
    except asyncio.CancelledError:
        pass
    finally:
        if demo_task is not None:
            demo_task.cancel()
        await server.stop()
        httpd.shutdown()


def run_dashboard(
    ws_host: str = "localhost",
    ws_port: int = 8765,
    http_port: int = 8766,
    demo: bool = False,
    open_browser: bool = True,
) -> None:
    """Run the digital-twin dashboard (HTTP page + WebSocket feed) until interrupted.

    Serves ``static/dashboard.html`` over HTTP and starts the live
    [DashboardServer][feaweld.digital_twin.dashboard.DashboardServer] WebSocket endpoint. With
    ``demo=True`` a synthetic weld lifecycle is streamed so the dashboard is fully
    populated without any real sensor sources.

    Parameters
    ----------
    ws_host : str, optional
        Host/interface for both servers, by default ``"localhost"``.
    ws_port : int, optional
        WebSocket port, by default ``8765``.
    http_port : int, optional
        HTTP port for the dashboard page, by default ``8766``.
    demo : bool, optional
        Stream a synthetic weld lifecycle instead of waiting for real sensors,
        by default ``False``.
    open_browser : bool, optional
        Open the dashboard in the default web browser once servers are up,
        by default ``True``.

    Raises
    ------
    ImportError
        If the optional ``websockets`` dependency is not installed.

    Notes
    -----
    Blocks the calling thread until ``KeyboardInterrupt``. Both servers are shut down
    cleanly on exit.
    """
    try:
        import websockets  # noqa: F401
    except ImportError:
        raise ImportError("websockets required: pip install feaweld[digital-twin]")

    try:
        asyncio.run(_serve(ws_host, ws_port, http_port, demo, open_browser))
    except KeyboardInterrupt:
        pass
