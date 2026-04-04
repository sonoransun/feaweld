"""Bayesian model updating for digital twin weld monitoring.

Uses MCMC (via emcee) to update posterior distributions of material
properties as sensor data arrives, enabling real-time fatigue life prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class PriorSpec:
    """Specification of a prior distribution for a model parameter."""
    name: str
    distribution: str  # "normal", "lognormal", "uniform"
    params: dict[str, float]
    bounds: tuple[float, float] = (-np.inf, np.inf)

    def log_prior(self, value: float) -> float:
        """Evaluate log-prior density at value."""
        if value < self.bounds[0] or value > self.bounds[1]:
            return -np.inf

        if self.distribution == "normal":
            return float(stats.norm.logpdf(
                value, loc=self.params["mean"], scale=self.params["std"]
            ))
        elif self.distribution == "lognormal":
            if value <= 0:
                return -np.inf
            return float(stats.lognorm.logpdf(
                value,
                s=self.params["std"],
                scale=np.exp(self.params["mean"]),
            ))
        elif self.distribution == "uniform":
            return float(stats.uniform.logpdf(
                value,
                loc=self.params["low"],
                scale=self.params["high"] - self.params["low"],
            ))
        return 0.0

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> NDArray:
        """Draw samples from the prior."""
        rng = rng or np.random.default_rng()
        if self.distribution == "normal":
            samples = rng.normal(self.params["mean"], self.params["std"], n)
        elif self.distribution == "lognormal":
            samples = rng.lognormal(self.params["mean"], self.params["std"], n)
        elif self.distribution == "uniform":
            samples = rng.uniform(self.params["low"], self.params["high"], n)
        else:
            samples = rng.normal(0, 1, n)
        return np.clip(samples, self.bounds[0], self.bounds[1])


@dataclass
class ObservedData:
    """Sensor observations for Bayesian updating."""
    measurement_type: str     # "temperature", "strain", "displacement"
    positions: NDArray         # (n_sensors, 3) — sensor positions
    values: NDArray            # (n_sensors,) — measured values
    uncertainty: float = 0.05  # measurement noise std (relative or absolute)
    timestamp: float = 0.0


@dataclass
class PosteriorSummary:
    """Summary statistics of the posterior distribution."""
    parameter_names: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    medians: dict[str, float]
    ci_95: dict[str, tuple[float, float]]
    samples: NDArray              # (n_samples, n_params) MCMC chain
    log_probability: NDArray      # (n_samples,) log-posterior values
    r_hat: dict[str, float]       # Gelman-Rubin convergence diagnostic
    ess: dict[str, float]         # Effective sample size


class BayesianUpdater:
    """Bayesian parameter estimation via MCMC.

    Updates material property distributions as sensor data arrives,
    using a physics model (FEA or surrogate) as the forward model.
    """

    def __init__(
        self,
        priors: list[PriorSpec],
        forward_model: Callable[[dict[str, float]], NDArray],
        n_walkers: int = 32,
        n_steps: int = 1000,
        n_burnin: int = 200,
    ):
        """
        Args:
            priors: Prior distributions for each parameter
            forward_model: Function mapping parameter dict → predicted observations
                           Must return array matching ObservedData.values shape
            n_walkers: Number of MCMC walkers (emcee ensemble sampler)
            n_steps: Number of MCMC steps after burn-in
            n_burnin: Number of burn-in steps to discard
        """
        self.priors = priors
        self.forward_model = forward_model
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_burnin = n_burnin
        self.param_names = [p.name for p in priors]
        self._posterior_samples: NDArray | None = None
        self._observations: list[ObservedData] = []

    def log_prior(self, theta: NDArray) -> float:
        """Evaluate total log-prior for parameter vector theta."""
        lp = 0.0
        for i, prior in enumerate(self.priors):
            lp += prior.log_prior(theta[i])
            if not np.isfinite(lp):
                return -np.inf
        return lp

    def log_likelihood(self, theta: NDArray, observations: list[ObservedData]) -> float:
        """Evaluate log-likelihood given parameters and observations."""
        params = {name: theta[i] for i, name in enumerate(self.param_names)}

        try:
            predicted = self.forward_model(params)
        except Exception:
            return -np.inf

        ll = 0.0
        for obs in observations:
            residuals = obs.values - predicted[:len(obs.values)]
            sigma = obs.uncertainty
            if sigma <= 0:
                sigma = 1e-6
            ll += -0.5 * np.sum((residuals / sigma) ** 2 + np.log(2 * np.pi * sigma**2))

        return float(ll)

    def log_posterior(self, theta: NDArray, observations: list[ObservedData]) -> float:
        """Log-posterior = log-prior + log-likelihood."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta, observations)
        return lp + ll

    def update(self, new_data: ObservedData) -> PosteriorSummary:
        """Perform Bayesian update with new sensor data.

        If previous posterior exists, use it as the starting point
        (sequential updating).
        """
        try:
            import emcee
        except ImportError:
            raise ImportError("emcee required: pip install feaweld[digital-twin]")

        self._observations.append(new_data)
        n_params = len(self.priors)

        # Initialize walkers
        if self._posterior_samples is not None:
            # Sequential update: start from previous posterior
            indices = np.random.randint(
                0, len(self._posterior_samples), self.n_walkers
            )
            p0 = self._posterior_samples[indices]
            # Add small perturbation
            p0 += 1e-4 * np.random.randn(*p0.shape)
        else:
            # Initial run: sample from priors
            rng = np.random.default_rng()
            p0 = np.column_stack([
                prior.sample(self.n_walkers, rng) for prior in self.priors
            ])

        # Run MCMC
        sampler = emcee.EnsembleSampler(
            self.n_walkers, n_params,
            self.log_posterior,
            args=[self._observations],
        )

        # Burn-in
        state = sampler.run_mcmc(p0, self.n_burnin, progress=False)
        sampler.reset()

        # Production
        sampler.run_mcmc(state, self.n_steps, progress=False)
        chain = sampler.get_chain(flat=True)  # (n_walkers * n_steps, n_params)
        log_prob = sampler.get_log_prob(flat=True)

        self._posterior_samples = chain

        return self._summarize_posterior(chain, log_prob, sampler)

    def predict_remaining_life(
        self,
        fatigue_model: Callable[[dict[str, float]], float],
        n_predictions: int = 500,
    ) -> dict[str, Any]:
        """Propagate posterior uncertainty through fatigue model.

        Args:
            fatigue_model: Maps material parameters → fatigue life (cycles)
            n_predictions: Number of posterior samples to use

        Returns:
            Dict with life statistics: mean, std, median, ci_95, distribution
        """
        if self._posterior_samples is None:
            raise RuntimeError("No posterior available. Run update() first.")

        indices = np.random.choice(
            len(self._posterior_samples), n_predictions, replace=False
        )
        lives = []

        for idx in indices:
            theta = self._posterior_samples[idx]
            params = {name: theta[i] for i, name in enumerate(self.param_names)}
            try:
                life = fatigue_model(params)
                if np.isfinite(life) and life > 0:
                    lives.append(life)
            except Exception:
                continue

        if not lives:
            return {"mean": float("nan"), "std": float("nan")}

        lives_arr = np.array(lives)
        log_lives = np.log10(lives_arr)

        return {
            "mean_life": float(np.mean(lives_arr)),
            "std_life": float(np.std(lives_arr)),
            "median_life": float(np.median(lives_arr)),
            "ci_95": (float(np.percentile(lives_arr, 2.5)),
                      float(np.percentile(lives_arr, 97.5))),
            "mean_log_life": float(np.mean(log_lives)),
            "std_log_life": float(np.std(log_lives)),
            "n_valid": len(lives),
        }

    def _summarize_posterior(
        self,
        chain: NDArray,
        log_prob: NDArray,
        sampler: Any,
    ) -> PosteriorSummary:
        """Compute summary statistics from MCMC chain."""
        means = {}
        stds = {}
        medians = {}
        ci_95 = {}
        r_hat = {}
        ess = {}

        for i, name in enumerate(self.param_names):
            col = chain[:, i]
            means[name] = float(np.mean(col))
            stds[name] = float(np.std(col))
            medians[name] = float(np.median(col))
            ci_95[name] = (
                float(np.percentile(col, 2.5)),
                float(np.percentile(col, 97.5)),
            )
            # Gelman-Rubin R-hat (simplified: split-chain)
            r_hat[name] = _gelman_rubin(col, self.n_walkers)
            # Effective sample size
            ess[name] = _effective_sample_size(col)

        return PosteriorSummary(
            parameter_names=self.param_names,
            means=means,
            stds=stds,
            medians=medians,
            ci_95=ci_95,
            samples=chain,
            log_probability=log_prob,
            r_hat=r_hat,
            ess=ess,
        )


def _gelman_rubin(samples: NDArray, n_chains: int) -> float:
    """Simplified Gelman-Rubin R-hat diagnostic."""
    n = len(samples)
    chain_len = n // n_chains
    if chain_len < 2:
        return float("nan")

    chains = samples[:chain_len * n_chains].reshape(n_chains, chain_len)
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)

    W = np.mean(chain_vars)  # within-chain variance
    B = chain_len * np.var(chain_means, ddof=1)  # between-chain variance

    if W < 1e-12:
        return 1.0

    var_hat = (1 - 1.0 / chain_len) * W + B / chain_len
    return float(np.sqrt(var_hat / W))


def _effective_sample_size(samples: NDArray) -> float:
    """Estimate effective sample size using autocorrelation."""
    n = len(samples)
    if n < 10:
        return float(n)

    mean = np.mean(samples)
    var = np.var(samples)
    if var < 1e-12:
        return float(n)

    # Compute autocorrelation up to lag n/2
    max_lag = min(n // 2, 100)
    acf = np.correlate(samples - mean, samples - mean, mode="full")
    acf = acf[n - 1:n - 1 + max_lag] / (var * n)

    # Sum positive autocorrelations
    tau = 1.0
    for lag in range(1, max_lag):
        if acf[lag] <= 0:
            break
        tau += 2.0 * acf[lag]

    return float(n / tau)
