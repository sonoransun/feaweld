"""Sensor data ingestion for digital twin weld monitoring.

Supports MQTT (thermal cameras, arc monitors) and OPC-UA (PLC data)
sensor sources with time-aligned buffering.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


@dataclass
class SensorReading:
    """A single sensor measurement."""
    timestamp: float          # UNIX timestamp
    sensor_id: str
    channel: str              # e.g., "thermocouple_1", "arc_current"
    value: float | NDArray
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorDataBuffer:
    """Time-windowed ring buffer for sensor data with timestamp alignment."""
    window_seconds: float = 60.0
    max_samples: int = 10000

    def __post_init__(self) -> None:
        self._buffers: dict[str, deque[SensorReading]] = {}
        self._callbacks: list[Callable[[SensorReading], None]] = []

    def add(self, reading: SensorReading) -> None:
        """Add a reading and trim old data outside the window."""
        key = f"{reading.sensor_id}:{reading.channel}"
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.max_samples)
        self._buffers[key].append(reading)

        # Trim by time window
        cutoff = reading.timestamp - self.window_seconds
        buf = self._buffers[key]
        while buf and buf[0].timestamp < cutoff:
            buf.popleft()

        for cb in self._callbacks:
            cb(reading)

    def get_channel(self, sensor_id: str, channel: str) -> list[SensorReading]:
        """Get all readings for a sensor channel within the buffer window."""
        key = f"{sensor_id}:{channel}"
        return list(self._buffers.get(key, []))

    def get_aligned(
        self,
        channels: list[tuple[str, str]],
        dt: float = 1.0,
    ) -> tuple[NDArray, NDArray]:
        """Get time-aligned data across multiple channels.

        Interpolates to common time grid with spacing dt.

        Args:
            channels: List of (sensor_id, channel) tuples
            dt: Time step for alignment (seconds)

        Returns:
            (times, values) where values is (n_times, n_channels)
        """
        all_data = []
        for sid, ch in channels:
            readings = self.get_channel(sid, ch)
            if readings:
                all_data.append(readings)

        if not all_data:
            return np.array([]), np.array([])

        # Find common time range
        t_min = max(d[0].timestamp for d in all_data)
        t_max = min(d[-1].timestamp for d in all_data)

        if t_min >= t_max:
            return np.array([]), np.array([])

        times = np.arange(t_min, t_max, dt)
        values = np.empty((len(times), len(channels)))

        for j, readings in enumerate(all_data):
            ts = np.array([r.timestamp for r in readings])
            vs = np.array([
                float(r.value) if np.isscalar(r.value) else float(r.value[0])
                for r in readings
            ])
            values[:, j] = np.interp(times, ts, vs)

        return times, values

    def on_data(self, callback: Callable[[SensorReading], None]) -> None:
        """Register a callback for new data."""
        self._callbacks.append(callback)

    @property
    def channels(self) -> list[str]:
        """List all active channel keys."""
        return list(self._buffers.keys())


class SensorSource(ABC):
    """Abstract sensor data source."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""

    @abstractmethod
    async def start_streaming(self, buffer: SensorDataBuffer) -> None:
        """Start streaming data into the buffer."""


class MQTTSensorSource(SensorSource):
    """MQTT-based sensor source for thermal cameras, arc monitors, strain gauges.

    Subscribes to topics matching sensor data patterns and parses incoming
    JSON messages into SensorReading objects.
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topics: list[str] | None = None,
        client_id: str = "feaweld_twin",
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topics = topics or ["feaweld/sensors/#"]
        self.client_id = client_id
        self._client = None
        self._running = False

    async def connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise ImportError("paho-mqtt required: pip install feaweld[digital-twin]")

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
        )
        self._client.connect(self.broker_host, self.broker_port)

    async def disconnect(self) -> None:
        if self._client:
            self._client.disconnect()
            self._client = None

    async def start_streaming(self, buffer: SensorDataBuffer) -> None:
        if not self._client:
            await self.connect()

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                reading = SensorReading(
                    timestamp=payload.get("timestamp", time.time()),
                    sensor_id=payload.get("sensor_id", msg.topic.split("/")[-2]),
                    channel=payload.get("channel", msg.topic.split("/")[-1]),
                    value=payload.get("value", 0.0),
                    unit=payload.get("unit", ""),
                    metadata=payload.get("metadata", {}),
                )
                buffer.add(reading)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass  # Skip malformed messages

        self._client.on_message = on_message
        for topic in self.topics:
            self._client.subscribe(topic)

        self._running = True
        self._client.loop_start()

    async def stop(self) -> None:
        self._running = False
        if self._client:
            self._client.loop_stop()


class OPCUASensorSource(SensorSource):
    """OPC-UA sensor source for PLC data in weld cells."""

    def __init__(
        self,
        server_url: str = "opc.tcp://localhost:4840",
        node_ids: list[str] | None = None,
        poll_interval: float = 0.5,
    ):
        self.server_url = server_url
        self.node_ids = node_ids or []
        self.poll_interval = poll_interval
        self._client = None
        self._running = False

    async def connect(self) -> None:
        try:
            from asyncua import Client
        except ImportError:
            raise ImportError("asyncua required: pip install feaweld[digital-twin]")

        self._client = Client(url=self.server_url)
        await self._client.connect()

    async def disconnect(self) -> None:
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def start_streaming(self, buffer: SensorDataBuffer) -> None:
        if not self._client:
            await self.connect()

        from asyncua import ua

        self._running = True
        while self._running:
            for node_id in self.node_ids:
                try:
                    node = self._client.get_node(node_id)
                    value = await node.read_value()
                    reading = SensorReading(
                        timestamp=time.time(),
                        sensor_id="opcua",
                        channel=node_id,
                        value=float(value),
                        unit="",
                    )
                    buffer.add(reading)
                except Exception:
                    pass
            await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        self._running = False


def parse_thermocouple_array(raw_values: list[float], positions_mm: list[float]) -> dict:
    """Parse thermocouple array data.

    Args:
        raw_values: Temperature readings (C) from each thermocouple
        positions_mm: Position of each thermocouple from weld center (mm)

    Returns:
        Dict with temperatures, positions, peak_temp, peak_position, gradient
    """
    temps = np.array(raw_values)
    pos = np.array(positions_mm)
    peak_idx = np.argmax(temps)

    gradient = np.gradient(temps, pos) if len(temps) > 1 else np.zeros_like(temps)

    return {
        "temperatures": temps,
        "positions": pos,
        "peak_temperature": float(temps[peak_idx]),
        "peak_position": float(pos[peak_idx]),
        "gradient": gradient,
    }


def parse_arc_waveform(
    current: NDArray,
    voltage: NDArray,
    sample_rate: float,
) -> dict:
    """Parse welding arc current/voltage waveforms.

    Args:
        current: Arc current samples (A)
        voltage: Arc voltage samples (V)
        sample_rate: Samples per second

    Returns:
        Dict with mean/rms current and voltage, power, heat input estimate
    """
    i_mean = float(np.mean(current))
    i_rms = float(np.sqrt(np.mean(current**2)))
    v_mean = float(np.mean(voltage))
    v_rms = float(np.sqrt(np.mean(voltage**2)))
    power_mean = float(np.mean(current * voltage))

    return {
        "current_mean": i_mean,
        "current_rms": i_rms,
        "voltage_mean": v_mean,
        "voltage_rms": v_rms,
        "power_mean": power_mean,
        "duration_s": len(current) / sample_rate,
    }
