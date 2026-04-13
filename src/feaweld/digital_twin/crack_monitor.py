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

    # Multi-state mode (joint state-parameter estimation)
    joint_estimation: bool = False
    initial_logC_mean: float = -24.83   # ln(1.65e-11)
    initial_logC_std: float = 1.0
    initial_m_mean: float = 3.0
    initial_m_std: float = 0.3
    process_noise_logC: float = 0.001
    process_noise_m: float = 0.001

    # Adaptive inflation
    adaptive_inflation: bool = True
    inflation_factor: float = 1.05
    min_spread_factor: float = 0.01


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
        self._is_multistate = False

        from feaweld.digital_twin.assimilation import paris_law_sif

        dK_fn = paris_law_sif(
            self._config.stress_range_mpa,
            self._config.geometry_factor,
        )

        if self._config.joint_estimation:
            from feaweld.digital_twin.assimilation import (
                MultiStateCrackEnKF,
                MultiStateParisModel,
            )

            model = MultiStateParisModel(dK_fn=dK_fn)
            initial_means = np.array([
                self._config.initial_crack_mm,
                self._config.initial_logC_mean,
                self._config.initial_m_mean,
            ])
            initial_stds = np.array([
                self._config.initial_std_mm,
                self._config.initial_logC_std,
                self._config.initial_m_std,
            ])
            process_noise_stds = np.array([
                self._config.process_noise_std,
                self._config.process_noise_logC,
                self._config.process_noise_m,
            ])
            self._enkf = MultiStateCrackEnKF(
                model=model,
                n_ensemble=self._config.n_ensemble,
                initial_means=initial_means,
                initial_stds=initial_stds,
                process_noise_stds=process_noise_stds,
                seed=self._config.seed,
                inflation_factor=self._config.inflation_factor,
                min_spread_factor=self._config.min_spread_factor,
            )
            self._is_multistate = True
        else:
            from feaweld.digital_twin.assimilation import (
                CrackEnKF,
                ParisLawModel,
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
            "CrackMonitor initialised: a0=%.2f mm, a_crit=%.1f mm, multistate=%s",
            self._config.initial_crack_mm,
            self._config.a_critical_mm,
            self._is_multistate,
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
        if self._is_multistate:
            from feaweld.digital_twin.observation_operators import ultrasonic_crack_operator
            ops = [ultrasonic_crack_operator()]
            self._enkf.update(
                observations=np.array([crack_length_mm]),
                obs_stds=np.array([cfg.observation_noise_std]),
                operators=ops,
            )
        else:
            self._enkf.update(
                observation=crack_length_mm,
                obs_std=cfg.observation_noise_std,
            )

        # Remaining-life estimate
        remaining = self.remaining_life_cycles()

        mean_val = float(self._enkf.mean) if not self._is_multistate else float(self._enkf.mean[0])
        std_val = float(self._enkf.std) if not self._is_multistate else float(self._enkf.std[0])

        prediction = {
            "step": self._step_count,
            "total_cycles": self._step_count * cfg.cycles_per_update,
            "crack_mean_mm": mean_val,
            "crack_std_mm": std_val,
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

        if self._is_multistate:
            a = float(self._enkf.mean[0])
        else:
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

    def remaining_life_distribution(
        self, a_critical_mm: float | None = None, dn_step: float = 1000.0,
        max_steps: int = 100_000,
    ) -> dict[str, Any]:
        """Remaining life distribution from full ensemble propagation."""
        a_crit = a_critical_mm or self._config.a_critical_mm
        if not self._is_multistate:
            return {"mean": self.remaining_life_cycles(a_crit), "std": 0.0}
        lives = self._enkf.remaining_life_distribution(a_crit, dn_step, max_steps)
        finite = lives[np.isfinite(lives)]
        if len(finite) == 0:
            return {"mean": float("inf"), "std": 0.0, "median": float("inf"),
                    "p05": float("inf"), "p95": float("inf"), "fraction_survived": 1.0}
        return {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "median": float(np.median(finite)),
            "p05": float(np.percentile(finite, 5)),
            "p95": float(np.percentile(finite, 95)),
            "fraction_survived": float(np.sum(np.isinf(lives))) / len(lives),
        }

    @property
    def ensemble_state(self) -> NDArray[np.floating]:
        """Current ensemble of crack lengths (mm)."""
        return np.array(self._enkf.ensemble)

    @property
    def history(self) -> list[tuple[float, float]]:
        """History of (mean, std) after each update."""
        return list(self._enkf.history)
