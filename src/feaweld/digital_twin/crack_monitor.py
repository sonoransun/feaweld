"""Crack growth monitoring orchestrator.

Connects the MQTT sensor source to the Ensemble Kalman Filter for
real-time crack length estimation and remaining life prediction,
then pushes updates to the WebSocket dashboard.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CrackMonitorConfig:
    """Configuration for the crack growth monitor."""

    # Paris law defaults for structural steel.
    C: float = 1.65e-11           # m/cycle (SI)
    m: float = 3.0
    stress_range_mpa: float = 100.0
    geometry_factor: float = 1.12

    # EnKF settings.
    n_ensemble: int = 50
    initial_crack_mm: float = 0.5
    initial_std_mm: float = 0.2
    process_noise_std: float = 0.001
    observation_noise_std: float = 0.05  # mm

    # Critical crack length for remaining-life estimate.
    a_critical_mm: float = 25.0

    # Cycle increment per sensor update (how many cycles between readings).
    cycles_per_update: int = 1000

    seed: int = 42


class CrackMonitor:
    """Orchestrates sensor data -> EnKF -> dashboard for crack growth.

    Usage::

        monitor = CrackMonitor(config)
        monitor.on_prediction(lambda pred: dashboard.update_predictions(pred))
        monitor.feed_observation(measured_crack_mm)
    """

    def __init__(self, config: CrackMonitorConfig | None = None) -> None:
        self._config = config or CrackMonitorConfig()
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []

        from feaweld.digital_twin.assimilation import (
            CrackEnKF,
            ParisLawModel,
            paris_law_sif,
        )

        dK_fn = paris_law_sif(
            self._config.stress_range_mpa,
            self._config.geometry_factor,
        )
        model = ParisLawModel(
            C=self._config.C,
            m=self._config.m,
            dK_fn=dK_fn,
        )
        self._enkf = CrackEnKF(
            model=model,
            n_ensemble=self._config.n_ensemble,
            initial_mean=self._config.initial_crack_mm,
            initial_std=self._config.initial_std_mm,
            process_noise_std=self._config.process_noise_std,
            seed=self._config.seed,
        )
        self._step_count = 0
        logger.info(
            "CrackMonitor initialised: a0=%.2f mm, a_crit=%.1f mm",
            self._config.initial_crack_mm,
            self._config.a_critical_mm,
        )

    def on_prediction(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked after each prediction update."""
        self._callbacks.append(callback)

    def feed_observation(self, crack_length_mm: float) -> dict[str, Any]:
        """Process one crack-length observation.

        1. Predict forward by ``cycles_per_update`` cycles.
        2. Update with the observation.
        3. Estimate remaining life.
        4. Notify callbacks.

        Returns the prediction dict.
        """
        cfg = self._config
        self._step_count += 1

        # Predict
        self._enkf.predict(dn=float(cfg.cycles_per_update))

        # Update
        self._enkf.update(
            observation=crack_length_mm,
            obs_std=cfg.observation_noise_std,
        )

        # Remaining-life estimate
        remaining = self.remaining_life_cycles()

        prediction = {
            "step": self._step_count,
            "total_cycles": self._step_count * cfg.cycles_per_update,
            "crack_mean_mm": float(self._enkf.mean),
            "crack_std_mm": float(self._enkf.std),
            "remaining_life_cycles": remaining,
            "a_critical_mm": cfg.a_critical_mm,
            "timestamp": time.time(),
        }

        logger.debug(
            "Step %d: a=%.3f +/- %.3f mm, remaining=%s cycles",
            self._step_count,
            prediction["crack_mean_mm"],
            prediction["crack_std_mm"],
            f"{remaining:.0f}" if np.isfinite(remaining) else "inf",
        )

        for cb in self._callbacks:
            try:
                cb(prediction)
            except Exception as e:
                logger.warning("Prediction callback error: %s", e)

        return prediction

    def remaining_life_cycles(
        self,
        a_critical_mm: float | None = None,
        max_steps: int = 100_000,
    ) -> float:
        """Estimate remaining fatigue life by propagating the ensemble mean.

        Advances a copy of the current mean crack length forward until it
        reaches ``a_critical_mm``, counting the cycles needed.

        Returns ``float('inf')`` if the crack doesn't reach critical length
        within ``max_steps * cycles_per_update`` cycles.
        """
        a_crit = a_critical_mm or self._config.a_critical_mm
        cfg = self._config

        a = float(self._enkf.mean)
        if a >= a_crit:
            return 0.0

        from feaweld.digital_twin.assimilation import ParisLawModel, paris_law_sif

        dK_fn = paris_law_sif(cfg.stress_range_mpa, cfg.geometry_factor)
        model = ParisLawModel(C=cfg.C, m=cfg.m, dK_fn=dK_fn)

        total_cycles = 0.0
        for _ in range(max_steps):
            a = model.step(a, float(cfg.cycles_per_update))
            total_cycles += cfg.cycles_per_update
            if a >= a_crit:
                return total_cycles

        return float("inf")

    @property
    def ensemble_state(self) -> NDArray[np.floating]:
        """Current ensemble of crack lengths (mm)."""
        return np.array(self._enkf.ensemble)

    @property
    def history(self) -> list[tuple[float, float]]:
        """History of (mean, std) after each update."""
        return list(self._enkf.history)
