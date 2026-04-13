"""Digital twin daemon for systemd service deployment.

Runs as a long-lived process that:
1. Connects to an MQTT broker for sensor data ingestion.
2. Maintains a Bayesian model updater.
3. Periodically sends sd_notify watchdog pings.
4. Handles SIGTERM for graceful shutdown.

Usage::

    python -m feaweld.digital_twin.daemon

Or via systemd::

    systemctl start feaweld-twin
"""

from __future__ import annotations

import os
import signal
import threading

from feaweld.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _sd_notify(state: str) -> None:
    """Send a notification to systemd (best-effort)."""
    try:
        from systemd.daemon import notify  # type: ignore[import-untyped]
        notify(state)
    except ImportError:
        pass


def _run_daemon() -> None:
    """Main daemon loop."""
    setup_logging("INFO", use_journal=True)
    logger.info("feaweld digital twin daemon starting")

    shutdown = threading.Event()

    def _handle_signal(signum: int, frame: object) -> None:
        logger.info("Received signal %d, shutting down", signum)
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # --- Initialise components ---
    broker_host = os.environ.get("FEAWELD_MQTT_HOST", "localhost")
    broker_port = int(os.environ.get("FEAWELD_MQTT_PORT", "1883"))

    try:
        from feaweld.digital_twin.ingest import MQTTSensorSource, SensorDataBuffer

        source = MQTTSensorSource(
            broker_host=broker_host,
            broker_port=broker_port,
            topics=["feaweld/sensors/#"],
        )
        buffer = SensorDataBuffer(window_seconds=300.0)

        # Wire sensor callback.
        def _on_reading(reading: object) -> None:
            buffer.add(reading)  # type: ignore[arg-type]

        source.on_reading = _on_reading
        source.start()
        logger.info("MQTT source connected to %s:%d", broker_host, broker_port)
    except ImportError:
        logger.warning(
            "paho-mqtt not available — running without MQTT. "
            "Install with: pip install feaweld[digital-twin]"
        )
        source = None  # type: ignore[assignment]

    # --- Notify systemd we are ready ---
    _sd_notify("READY=1")
    logger.info("Daemon ready")

    # --- Main loop: watchdog + periodic tasks ---
    watchdog_usec = int(os.environ.get("WATCHDOG_USEC", "0"))
    ping_interval = (watchdog_usec / 1_000_000 / 2) if watchdog_usec > 0 else 15.0

    while not shutdown.wait(timeout=ping_interval):
        _sd_notify("WATCHDOG=1")

    # --- Graceful shutdown ---
    logger.info("Shutting down")
    _sd_notify("STOPPING=1")

    if source is not None:
        try:
            source.stop()
        except Exception:
            pass

    logger.info("Daemon stopped")


if __name__ == "__main__":
    _run_daemon()
