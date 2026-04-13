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


@dataclass
class MultiStateParisModel:
    """Paris-law crack growth with joint estimation of C and m.

    State vector x = [a, log(C), m] where:
    - a: crack length (mm)
    - log(C): natural log of Paris coefficient
    - m: Paris exponent

    da/dN = exp(logC) * dK(a)^m
    """

    dK_fn: Callable[[float], float]

    def step(self, state: NDArray, dn: float) -> NDArray:
        """Propagate state by ``dn`` cycles; logC and m are unchanged."""
        state = np.asarray(state, dtype=np.float64)
        a = state[0]
        logC = state[1]
        m = state[2]
        dk = self.dK_fn(a)
        if dk <= 0.0:
            return state.copy()
        a_new = a + np.exp(logC) * (dk ** m) * dn
        return np.array([a_new, logC, m])


class MultiStateCrackEnKF:
    """Ensemble Kalman Filter for joint crack-length / Paris-parameter estimation.

    State vector per member: [a, log(C), m].
    Supports multiple observation channels via operator list.
    """

    def __init__(
        self,
        model: MultiStateParisModel,
        n_ensemble: int = 50,
        initial_means: NDArray | None = None,
        initial_stds: NDArray | None = None,
        process_noise_stds: NDArray | None = None,
        seed: int = 0,
        inflation_factor: float = 1.05,
        min_spread_factor: float = 0.01,
    ):
        self.model = model
        self.n_ensemble = int(n_ensemble)
        self.inflation_factor = float(inflation_factor)
        self.min_spread_factor = float(min_spread_factor)
        self._rng = np.random.default_rng(seed)

        if initial_means is None:
            initial_means = np.array([0.1, -24.83, 3.0])
        else:
            initial_means = np.asarray(initial_means, dtype=np.float64)

        if initial_stds is None:
            initial_stds = np.array([0.05, 1.0, 0.3])
        else:
            initial_stds = np.asarray(initial_stds, dtype=np.float64)

        if process_noise_stds is None:
            self.process_noise_stds = np.array([0.001, 0.001, 0.001])
        else:
            self.process_noise_stds = np.asarray(process_noise_stds, dtype=np.float64)

        # Generate ensemble: (n_ensemble, 3)
        self.ensemble = self._rng.normal(
            loc=initial_means, scale=initial_stds, size=(self.n_ensemble, 3)
        )
        # Clamp: a >= 1e-6, m in [1.0, 6.0]
        self._clamp()
        self.history: list[tuple[NDArray, NDArray]] = [
            (self.mean.copy(), self.std.copy())
        ]

    def _clamp(self) -> None:
        """Enforce physical constraints on ensemble members."""
        self.ensemble[:, 0] = np.maximum(self.ensemble[:, 0], 1e-6)
        self.ensemble[:, 2] = np.clip(self.ensemble[:, 2], 1.0, 6.0)

    @property
    def mean(self) -> NDArray:
        """Ensemble mean, shape (3,)."""
        return np.mean(self.ensemble, axis=0)

    @property
    def std(self) -> NDArray:
        """Ensemble standard deviation, shape (3,)."""
        return np.std(self.ensemble, axis=0, ddof=1)

    def predict(self, dn: float) -> None:
        """Propagate each ensemble member by ``dn`` cycles with process noise."""
        propagated = np.array(
            [self.model.step(self.ensemble[i], dn) for i in range(self.n_ensemble)]
        )
        noise = self._rng.normal(
            loc=0.0,
            scale=self.process_noise_stds,
            size=(self.n_ensemble, 3),
        )
        self.ensemble = propagated + noise
        self._clamp()
        self.history.append((self.mean.copy(), self.std.copy()))

    def update(
        self,
        observations: NDArray,
        obs_stds: NDArray,
        operators: list[Callable],
    ) -> None:
        """Matrix EnKF analysis step with multiple observation channels.

        Parameters
        ----------
        observations
            Observed values, shape (n_obs,).
        obs_stds
            Observation noise standard deviations, shape (n_obs,).
        operators
            List of callables, each mapping state (3,) -> float.
        """
        observations = np.asarray(observations, dtype=np.float64)
        obs_stds = np.asarray(obs_stds, dtype=np.float64)
        n_obs = len(observations)
        N = self.n_ensemble

        # Predicted observations: (n_ensemble, n_obs)
        Y_pred = np.zeros((N, n_obs))
        for i in range(N):
            for j, op in enumerate(operators):
                Y_pred[i, j] = op(self.ensemble[i])

        # Anomalies
        x_mean = np.mean(self.ensemble, axis=0)  # (3,)
        y_mean = np.mean(Y_pred, axis=0)  # (n_obs,)
        dX = self.ensemble - x_mean  # (N, 3)
        dY = Y_pred - y_mean  # (N, n_obs)

        # Cross-covariance and observation-space covariance
        P_xy = dX.T @ dY / (N - 1)  # (3, n_obs)
        P_yy = dY.T @ dY / (N - 1) + np.diag(obs_stds ** 2)  # (n_obs, n_obs)

        # Kalman gain
        K = P_xy @ np.linalg.inv(P_yy)  # (3, n_obs)

        # Perturbed observations
        perturbed_obs = observations + self._rng.normal(
            loc=0.0, scale=obs_stds, size=(N, n_obs)
        )

        # Update ensemble
        innovation = perturbed_obs - Y_pred  # (N, n_obs)
        self.ensemble = self.ensemble + innovation @ K.T  # (N, 3)

        self._check_and_inflate()
        self._clamp()
        self.history.append((self.mean.copy(), self.std.copy()))

    def _check_and_inflate(self) -> None:
        """Inflate ensemble if any component's relative spread is too small."""
        ens_mean = np.mean(self.ensemble, axis=0)
        ens_std = np.std(self.ensemble, axis=0, ddof=1)
        abs_mean = np.abs(ens_mean)
        # Use absolute mean for relative spread; guard against near-zero mean
        ref = np.where(abs_mean > 1e-12, abs_mean, 1.0)
        relative_spread = ens_std / ref
        if np.any(relative_spread < self.min_spread_factor):
            deviations = self.ensemble - ens_mean
            self.ensemble = ens_mean + deviations * self.inflation_factor

    def remaining_life_distribution(
        self,
        a_critical: float,
        dn_step: float = 1000.0,
        max_steps: int = 100_000,
    ) -> NDArray:
        """Propagate each member to failure, return remaining lives.

        Returns
        -------
        NDArray
            Shape (n_ensemble,) of remaining cycle counts. ``np.inf`` if
            a_critical is not reached within ``max_steps * dn_step`` cycles.
        """
        lives = np.full(self.n_ensemble, np.inf)
        for i in range(self.n_ensemble):
            state = self.ensemble[i].copy()
            for step in range(max_steps):
                state = self.model.step(state, dn_step)
                if state[0] >= a_critical:
                    lives[i] = (step + 1) * dn_step
                    break
        return lives
