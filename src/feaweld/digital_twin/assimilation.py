"""Ensemble Kalman Filter for online crack-length estimation.

Pure-NumPy EnKF specialized for fatigue crack propagation driven by a
Paris-law forward model. Units are SI-mm: crack length in mm, stress in
MPa, stress intensity range in MPa*sqrt(mm).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass
class ParisLawModel:
    """Paris-law crack growth forward model.

    da/dN = C * (dK(a))^m
    """

    C: float
    m: float
    dK_fn: Callable[[float], float]

    def step(self, a: float, dn: float) -> float:
        """Advance crack length by ``dn`` cycles via explicit Euler."""
        dk = self.dK_fn(a)
        if dk <= 0.0:
            return a
        return a + self.C * (dk ** self.m) * dn


def paris_law_sif(
    stress_range: float, geometry_factor: float = 1.12
) -> Callable[[float], float]:
    """Return a closure dK(a) = Y * S * sqrt(pi * a) (MPa*sqrt(mm))."""

    def dK(a: float) -> float:
        if a <= 0.0:
            return 0.0
        return geometry_factor * stress_range * float(np.sqrt(np.pi * a))

    return dK


class CrackEnKF:
    """Stochastic Ensemble Kalman Filter for scalar crack length state."""

    def __init__(
        self,
        model: ParisLawModel,
        n_ensemble: int = 50,
        initial_mean: float = 0.1,
        initial_std: float = 0.05,
        process_noise_std: float = 0.001,
        seed: int = 0,
    ):
        self.model = model
        self.n_ensemble = int(n_ensemble)
        self.process_noise_std = float(process_noise_std)
        self._rng = np.random.default_rng(seed)
        self.ensemble: NDArray = self._rng.normal(
            loc=initial_mean, scale=initial_std, size=self.n_ensemble
        )
        self.ensemble = np.maximum(self.ensemble, 1e-6)
        self.history: list[tuple[float, float]] = [(self.mean, self.std)]

    @property
    def mean(self) -> float:
        return float(np.mean(self.ensemble))

    @property
    def std(self) -> float:
        return float(np.std(self.ensemble, ddof=1))

    def predict(self, dn: float) -> None:
        """Propagate each ensemble member by ``dn`` cycles with process noise."""
        propagated = np.array(
            [self.model.step(float(a), dn) for a in self.ensemble]
        )
        noise = self._rng.normal(
            loc=0.0, scale=self.process_noise_std, size=self.n_ensemble
        )
        self.ensemble = np.maximum(propagated + noise, 1e-6)
        self.history.append((self.mean, self.std))

    def update(
        self,
        observation: float,
        obs_std: float,
        H: Callable[[float], float] | None = None,
    ) -> None:
        """Stochastic EnKF analysis step with perturbed observations."""
        if H is None:
            H = lambda a: a  # noqa: E731

        x = self.ensemble
        y_pred = np.array([H(float(a)) for a in x])

        x_mean = np.mean(x)
        y_mean = np.mean(y_pred)
        dx = x - x_mean
        dy = y_pred - y_mean

        n = self.n_ensemble
        p_xy = float(np.sum(dx * dy) / (n - 1))
        p_yy = float(np.sum(dy * dy) / (n - 1)) + obs_std ** 2

        if p_yy <= 0.0:
            self.history.append((self.mean, self.std))
            return

        k = p_xy / p_yy
        perturbed_obs = observation + self._rng.normal(
            loc=0.0, scale=obs_std, size=n
        )
        self.ensemble = np.maximum(x + k * (perturbed_obs - y_pred), 1e-6)
        self.history.append((self.mean, self.std))
