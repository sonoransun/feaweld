"""Real-time monitoring dashboard for digital twin weld analysis.

Provides WebSocket server for pushing live updates to monitoring
dashboards, with anomaly detection and lifecycle state management.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from feaweld.digital_twin.ingest import SensorDataBuffer, SensorReading


class WeldLifecycleState(str, Enum):
    """State machine for weld monitoring lifecycle."""
    IDLE = "idle"
    WELDING = "welding"
    COOLING = "cooling"
    PWHT = "pwht"
    MONITORING = "monitoring"
    ALERT = "alert"


@dataclass
class Alert:
    """An alert triggered by anomaly detection."""
    timestamp: float
    severity: str         # "info", "warning", "critical"
    alert_type: str       # "threshold", "trend", "anomaly"
    message: str
    sensor_id: str = ""
    channel: str = ""
    value: float = 0.0
    threshold: float = 0.0


@dataclass
class ThresholdConfig:
    """Threshold-based alert configuration."""
    channel_pattern: str     # e.g., "temperature" matches any temp channel
    warning_high: float = float("inf")
    critical_high: float = float("inf")
    warning_low: float = float("-inf")
    critical_low: float = float("-inf")


@dataclass
class TrendConfig:
    """Trend-based anomaly detection configuration."""
    channel_pattern: str
    window_seconds: float = 30.0
    max_rate_per_second: float = float("inf")  # max allowed rate of change
    min_rate_per_second: float = 0.0           # expected minimum rate


class AlertEngine:
    """Threshold and trend-based anomaly detection on sensor data."""

    def __init__(self) -> None:
        self.thresholds: list[ThresholdConfig] = []
        self.trends: list[TrendConfig] = []
        self._history: dict[str, list[tuple[float, float]]] = {}
        self._alerts: list[Alert] = []
        self._callbacks: list[Callable[[Alert], None]] = []

    def add_threshold(self, config: ThresholdConfig) -> None:
        self.thresholds.append(config)

    def add_trend(self, config: TrendConfig) -> None:
        self.trends.append(config)

    def on_alert(self, callback: Callable[[Alert], None]) -> None:
        self._callbacks.append(callback)

    def check(self, reading: SensorReading) -> list[Alert]:
        """Check a reading against all configured thresholds and trends."""
        alerts = []
        key = f"{reading.sensor_id}:{reading.channel}"
        value = float(reading.value) if np.isscalar(reading.value) else float(reading.value[0])

        # Threshold checks
        for tc in self.thresholds:
            if tc.channel_pattern in reading.channel:
                if value >= tc.critical_high:
                    alerts.append(Alert(
                        timestamp=reading.timestamp,
                        severity="critical",
                        alert_type="threshold",
                        message=f"{reading.channel} = {value:.1f} exceeds critical high {tc.critical_high:.1f}",
                        sensor_id=reading.sensor_id,
                        channel=reading.channel,
                        value=value,
                        threshold=tc.critical_high,
                    ))
                elif value >= tc.warning_high:
                    alerts.append(Alert(
                        timestamp=reading.timestamp,
                        severity="warning",
                        alert_type="threshold",
                        message=f"{reading.channel} = {value:.1f} exceeds warning high {tc.warning_high:.1f}",
                        sensor_id=reading.sensor_id,
                        channel=reading.channel,
                        value=value,
                        threshold=tc.warning_high,
                    ))
                if value <= tc.critical_low:
                    alerts.append(Alert(
                        timestamp=reading.timestamp,
                        severity="critical",
                        alert_type="threshold",
                        message=f"{reading.channel} = {value:.1f} below critical low {tc.critical_low:.1f}",
                        sensor_id=reading.sensor_id,
                        channel=reading.channel,
                        value=value,
                        threshold=tc.critical_low,
                    ))

        # Trend checks
        if key not in self._history:
            self._history[key] = []
        self._history[key].append((reading.timestamp, value))

        for tc in self.trends:
            if tc.channel_pattern in reading.channel:
                # Get recent history within window
                cutoff = reading.timestamp - tc.window_seconds
                recent = [(t, v) for t, v in self._history[key] if t >= cutoff]
                self._history[key] = recent  # trim

                if len(recent) >= 2:
                    t0, v0 = recent[0]
                    t1, v1 = recent[-1]
                    dt = t1 - t0
                    if dt > 0:
                        rate = abs(v1 - v0) / dt
                        if rate > tc.max_rate_per_second:
                            alerts.append(Alert(
                                timestamp=reading.timestamp,
                                severity="warning",
                                alert_type="trend",
                                message=f"{reading.channel} changing at {rate:.2f}/s exceeds max {tc.max_rate_per_second:.2f}/s",
                                sensor_id=reading.sensor_id,
                                channel=reading.channel,
                                value=rate,
                                threshold=tc.max_rate_per_second,
                            ))

        self._alerts.extend(alerts)
        for alert in alerts:
            for cb in self._callbacks:
                cb(alert)

        return alerts

    @property
    def recent_alerts(self) -> list[Alert]:
        return self._alerts[-100:]  # last 100 alerts


class WeldStateMachine:
    """Lifecycle state machine for weld monitoring."""

    def __init__(self) -> None:
        self.state = WeldLifecycleState.IDLE
        self._transitions: dict[tuple, WeldLifecycleState] = {
            (WeldLifecycleState.IDLE, "start_welding"): WeldLifecycleState.WELDING,
            (WeldLifecycleState.WELDING, "stop_welding"): WeldLifecycleState.COOLING,
            (WeldLifecycleState.COOLING, "start_pwht"): WeldLifecycleState.PWHT,
            (WeldLifecycleState.COOLING, "cooled"): WeldLifecycleState.MONITORING,
            (WeldLifecycleState.PWHT, "pwht_complete"): WeldLifecycleState.MONITORING,
            (WeldLifecycleState.MONITORING, "alert_triggered"): WeldLifecycleState.ALERT,
            (WeldLifecycleState.ALERT, "alert_resolved"): WeldLifecycleState.MONITORING,
        }
        self._callbacks: list[Callable[[WeldLifecycleState, WeldLifecycleState], None]] = []

    def transition(self, event: str) -> WeldLifecycleState:
        key = (self.state, event)
        if key in self._transitions:
            old = self.state
            self.state = self._transitions[key]
            for cb in self._callbacks:
                cb(old, self.state)
        return self.state

    def on_transition(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def auto_detect_state(self, buffer: SensorDataBuffer) -> WeldLifecycleState:
        """Attempt to auto-detect lifecycle state from sensor data."""
        temp_channels = [ch for ch in buffer.channels if "temp" in ch.lower()]

        if not temp_channels:
            return self.state

        # Check if welding is active (high temperatures)
        for ch_key in temp_channels:
            parts = ch_key.split(":")
            if len(parts) == 2:
                readings = buffer.get_channel(parts[0], parts[1])
                if readings:
                    recent_temp = float(readings[-1].value) if np.isscalar(readings[-1].value) else float(readings[-1].value[0])
                    if recent_temp > 500:  # High temp = welding
                        if self.state == WeldLifecycleState.IDLE:
                            self.transition("start_welding")
                    elif recent_temp > 100:  # Moderate = cooling or PWHT
                        if self.state == WeldLifecycleState.WELDING:
                            self.transition("stop_welding")
                    elif recent_temp < 50:  # Cool = monitoring
                        if self.state == WeldLifecycleState.COOLING:
                            self.transition("cooled")

        return self.state


class DashboardServer:
    """WebSocket server for real-time dashboard updates."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        buffer: SensorDataBuffer | None = None,
        alert_engine: AlertEngine | None = None,
        state_machine: WeldStateMachine | None = None,
    ):
        self.host = host
        self.port = port
        self.buffer = buffer or SensorDataBuffer()
        self.alert_engine = alert_engine or AlertEngine()
        self.state_machine = state_machine or WeldStateMachine()
        self._clients: set = set()
        self._server = None
        self._model_predictions: dict[str, Any] = {}

    async def start(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets required: pip install feaweld[digital-twin]")

        # Register sensor data callback
        self.buffer.on_data(self._on_sensor_data)
        self.alert_engine.on_alert(self._on_alert)

        self._server = await websockets.serve(
            self._handler, self.host, self.port
        )

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handler(self, websocket, path=None) -> None:
        """Handle WebSocket connection."""
        self._clients.add(websocket)
        try:
            # Send current state on connect
            await websocket.send(json.dumps({
                "type": "state",
                "state": self.state_machine.state.value,
                "timestamp": time.time(),
            }))

            async for message in websocket:
                # Handle incoming commands
                try:
                    cmd = json.loads(message)
                    await self._handle_command(cmd, websocket)
                except json.JSONDecodeError:
                    pass
        finally:
            self._clients.discard(websocket)

    async def _handle_command(self, cmd: dict, websocket: Any) -> None:
        """Handle incoming commands from dashboard clients."""
        cmd_type = cmd.get("type", "")
        if cmd_type == "get_state":
            await websocket.send(json.dumps({
                "type": "state",
                "state": self.state_machine.state.value,
            }))
        elif cmd_type == "get_predictions":
            await websocket.send(json.dumps({
                "type": "predictions",
                "data": self._model_predictions,
            }))

    def update_predictions(self, predictions: dict[str, Any]) -> None:
        """Update model predictions for broadcast."""
        self._model_predictions = predictions
        asyncio.ensure_future(self._broadcast({
            "type": "predictions",
            "data": predictions,
            "timestamp": time.time(),
        }))

    def _on_sensor_data(self, reading: SensorReading) -> None:
        """Callback when new sensor data arrives."""
        # Check for alerts
        self.alert_engine.check(reading)

        # Auto-detect state
        self.state_machine.auto_detect_state(self.buffer)

        # Broadcast to connected clients
        msg = {
            "type": "sensor",
            "sensor_id": reading.sensor_id,
            "channel": reading.channel,
            "value": float(reading.value) if np.isscalar(reading.value) else reading.value.tolist(),
            "timestamp": reading.timestamp,
        }
        try:
            asyncio.get_event_loop().create_task(self._broadcast(msg))
        except RuntimeError:
            pass  # No event loop running

    def _on_alert(self, alert: Alert) -> None:
        """Callback when alert is triggered."""
        msg = {
            "type": "alert",
            "severity": alert.severity,
            "alert_type": alert.alert_type,
            "message": alert.message,
            "timestamp": alert.timestamp,
        }
        try:
            asyncio.get_event_loop().create_task(self._broadcast(msg))
        except RuntimeError:
            pass

    async def _broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        if self._clients:
            msg_str = json.dumps(message)
            await asyncio.gather(
                *(client.send(msg_str) for client in self._clients),
                return_exceptions=True,
            )
