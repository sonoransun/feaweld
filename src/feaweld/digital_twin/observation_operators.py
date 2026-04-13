"""Multi-sensor observation operators for the EnKF."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class ObservationSpec:
    """Specification of one observation channel."""
    name: str
    obs_std: float
    operator: Callable[[NDArray], float]


@dataclass
class MultiObservation:
    """Bundle of observations from multiple sensors at one time step."""
    values: NDArray        # (n_obs,)
    noise_stds: NDArray    # (n_obs,)
    operators: list[Callable[[NDArray], float]]


def ultrasonic_crack_operator() -> Callable[[NDArray], float]:
    """H(state) = state[0] = crack length a."""
    return lambda state: float(state[0])


def strain_gauge_operator(
    sif_fn: Callable[[float], float],
    gauge_distance_mm: float,
    E: float = 210_000.0,
) -> Callable[[NDArray], float]:
    """H(state) -> predicted strain at gauge location via LEFM."""
    def H(state: NDArray) -> float:
        a = float(state[0])
        dk = sif_fn(a)
        return dk / (E * np.sqrt(2.0 * np.pi * gauge_distance_mm))
    return H


def acpd_operator(
    calibration_slope: float = 1.0,
    calibration_intercept: float = 0.0,
) -> Callable[[NDArray], float]:
    """H(state) = slope * a + intercept (ACPD linear calibration)."""
    def H(state: NDArray) -> float:
        return calibration_slope * float(state[0]) + calibration_intercept
    return H
