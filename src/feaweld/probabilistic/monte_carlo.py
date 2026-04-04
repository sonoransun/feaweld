"""Monte Carlo engine with Latin Hypercube Sampling for probabilistic weld analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats.qmc import LatinHypercube


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RandomVariable:
    """A random variable with a named distribution and parameters.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"yield_strength"``).
    distribution : str
        One of ``"normal"``, ``"lognormal"``, ``"weibull"``, ``"uniform"``,
        ``"gumbel"``.
    params : dict
        Distribution parameters.  Expected keys depend on *distribution*:

        * ``"normal"`` -- ``{"mean": ..., "std": ...}``
        * ``"lognormal"`` -- ``{"mean": ..., "std": ...}`` (of the underlying normal)
        * ``"weibull"`` -- ``{"shape": ..., "scale": ...}``
        * ``"uniform"`` -- ``{"low": ..., "high": ...}``
        * ``"gumbel"`` -- ``{"loc": ..., "scale": ...}``
    """

    name: str
    distribution: str
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {"normal", "lognormal", "weibull", "uniform", "gumbel"}
        if self.distribution not in allowed:
            raise ValueError(
                f"Unsupported distribution '{self.distribution}'. "
                f"Choose from {sorted(allowed)}."
            )


@dataclass
class MonteCarloConfig:
    """Configuration for a Monte Carlo simulation run."""

    n_samples: int = 1000
    method: str = "lhs"  # "lhs" or "random"
    seed: int | None = None
    convergence_check: bool = True
    convergence_tol: float = 0.01

    def __post_init__(self) -> None:
        if self.method not in ("lhs", "random"):
            raise ValueError(
                f"Unknown sampling method '{self.method}'. Use 'lhs' or 'random'."
            )


@dataclass
class MonteCarloResult:
    """Results produced by :class:`MonteCarloEngine.run`."""

    samples: NDArray[np.float64]        # (n_samples, n_vars)
    results: NDArray[np.float64]        # (n_samples,) or (n_samples, n_outputs)
    mean: float | NDArray[np.float64]
    std: float | NDArray[np.float64]
    cov: float                          # coefficient of variation
    percentiles: dict[int, float]       # {5: ..., 50: ..., 95: ...}
    converged: bool
    n_effective: int


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_distribution(
    dist_name: str,
    params: dict[str, float],
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate *n* samples from a named distribution.

    Parameters
    ----------
    dist_name : str
        Distribution name (``"normal"``, ``"lognormal"``, ``"weibull"``,
        ``"uniform"``, ``"gumbel"``).
    params : dict
        Distribution parameters (see :class:`RandomVariable`).
    n : int
        Number of samples.
    rng : numpy.random.Generator
        NumPy random generator instance.

    Returns
    -------
    NDArray
        1-D array of shape ``(n,)`` with samples.
    """

    if dist_name == "normal":
        return rng.normal(loc=params["mean"], scale=params["std"], size=n)
    elif dist_name == "lognormal":
        # params["mean"] and params["std"] are for the underlying normal
        return rng.lognormal(mean=params["mean"], sigma=params["std"], size=n)
    elif dist_name == "weibull":
        # Weibull with shape (a) and scale (lambda):
        # X = scale * weibull(a)
        return params["scale"] * rng.weibull(params["shape"], size=n)
    elif dist_name == "uniform":
        return rng.uniform(low=params["low"], high=params["high"], size=n)
    elif dist_name == "gumbel":
        return rng.gumbel(loc=params["loc"], scale=params["scale"], size=n)
    else:
        raise ValueError(f"Unsupported distribution: {dist_name}")


def _ppf(dist_name: str, params: dict[str, float], u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Percent-point function (inverse CDF) mapping uniform [0,1] -> target distribution."""
    from scipy import stats

    if dist_name == "normal":
        return stats.norm.ppf(u, loc=params["mean"], scale=params["std"])
    elif dist_name == "lognormal":
        # scipy lognormal: shape s = sigma, scale = exp(mu)
        s = params["std"]
        scale = np.exp(params["mean"])
        return stats.lognorm.ppf(u, s=s, scale=scale)
    elif dist_name == "weibull":
        # scipy weibull_min: shape c, scale
        return stats.weibull_min.ppf(u, c=params["shape"], scale=params["scale"])
    elif dist_name == "uniform":
        return stats.uniform.ppf(u, loc=params["low"], scale=params["high"] - params["low"])
    elif dist_name == "gumbel":
        return stats.gumbel_r.ppf(u, loc=params["loc"], scale=params["scale"])
    else:
        raise ValueError(f"Unsupported distribution: {dist_name}")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MonteCarloEngine:
    """Run Monte Carlo simulations over random variables.

    Parameters
    ----------
    variables : list[RandomVariable]
        The uncertain inputs.
    config : MonteCarloConfig
        Simulation settings.
    """

    def __init__(
        self,
        variables: list[RandomVariable],
        config: MonteCarloConfig | None = None,
    ) -> None:
        self.variables = variables
        self.config = config or MonteCarloConfig()
        self._rng = np.random.default_rng(self.config.seed)

    # ----- public -----------------------------------------------------------

    def generate_samples(self) -> NDArray[np.float64]:
        """Generate a sample matrix of shape ``(n_samples, n_vars)``.

        For *LHS* sampling the unit hypercube samples are mapped through each
        variable's inverse CDF (ppf).  For *random* sampling the distributions
        are sampled directly.
        """

        n = self.config.n_samples
        k = len(self.variables)

        if self.config.method == "lhs":
            sampler = LatinHypercube(d=k, seed=self._rng)
            u = sampler.random(n=n)  # (n, k) in [0, 1]
            samples = np.empty_like(u)
            for j, var in enumerate(self.variables):
                samples[:, j] = _ppf(var.distribution, var.params, u[:, j])
        else:
            samples = np.column_stack(
                [
                    sample_distribution(var.distribution, var.params, n, self._rng)
                    for var in self.variables
                ]
            )

        return samples

    def run(
        self,
        analysis_func: Callable[[dict[str, float]], float | NDArray[np.float64]],
    ) -> MonteCarloResult:
        """Run *analysis_func* over every sample row.

        Parameters
        ----------
        analysis_func : callable
            Accepts ``dict[str, float]`` mapping variable names to sampled
            values and returns a scalar result (or a 1-D array of outputs).

        Returns
        -------
        MonteCarloResult
        """

        samples = self.generate_samples()
        n = samples.shape[0]

        results_list: list[Any] = []
        converged = False

        for i in range(n):
            var_dict = {
                var.name: float(samples[i, j])
                for j, var in enumerate(self.variables)
            }
            results_list.append(analysis_func(var_dict))

            # Convergence check (only meaningful for scalar results)
            if (
                self.config.convergence_check
                and i >= 50
                and i % 50 == 0
            ):
                partial = np.asarray(results_list, dtype=np.float64)
                if partial.ndim == 1 and self._check_convergence(partial):
                    converged = True

        results = np.asarray(results_list, dtype=np.float64)

        # If we never set converged but didn't bail out early, do a final check
        if self.config.convergence_check and not converged and results.ndim == 1:
            converged = self._check_convergence(results)

        mean = float(np.mean(results)) if results.ndim == 1 else np.mean(results, axis=0)
        std = float(np.std(results, ddof=1)) if results.ndim == 1 else np.std(results, axis=0, ddof=1)
        mean_abs = np.abs(mean) if np.isscalar(mean) else np.mean(np.abs(mean))
        cov = float(std / mean_abs) if mean_abs > 0 else float("inf") if np.isscalar(std) else float(np.mean(std) / mean_abs)

        if results.ndim == 1:
            percentiles = {
                5: float(np.percentile(results, 5)),
                50: float(np.percentile(results, 50)),
                95: float(np.percentile(results, 95)),
            }
        else:
            percentiles = {
                p: np.percentile(results, p, axis=0) for p in (5, 50, 95)
            }

        # Effective sample size based on autocorrelation lag-1
        n_effective = self._compute_n_effective(results)

        return MonteCarloResult(
            samples=samples,
            results=results,
            mean=mean,
            std=std,
            cov=cov,
            percentiles=percentiles,
            converged=converged,
            n_effective=n_effective,
        )

    # ----- private ----------------------------------------------------------

    def _check_convergence(self, results_so_far: NDArray[np.float64]) -> bool:
        """Check if the running mean has stabilised within tolerance.

        Compares the mean of the first half with the mean of the full sample.
        """

        n = len(results_so_far)
        if n < 100:
            return False

        half = n // 2
        mean_first = np.mean(results_so_far[:half])
        mean_full = np.mean(results_so_far)

        if mean_full == 0:
            return abs(mean_first) < self.config.convergence_tol

        rel_change = abs((mean_full - mean_first) / mean_full)
        return bool(rel_change < self.config.convergence_tol)

    @staticmethod
    def _compute_n_effective(results: NDArray[np.float64]) -> int:
        """Estimate effective sample size from autocorrelation."""
        if results.ndim != 1:
            # For multi-output, use first output
            vals = results[:, 0] if results.ndim == 2 else results.ravel()
        else:
            vals = results

        n = len(vals)
        if n < 10:
            return n

        vals_centered = vals - np.mean(vals)
        var = np.var(vals_centered)
        if var == 0:
            return n

        # Lag-1 autocorrelation
        rho1 = np.sum(vals_centered[:-1] * vals_centered[1:]) / ((n - 1) * var)
        rho1 = np.clip(rho1, -0.99, 0.99)

        # Effective sample size formula: n_eff = n * (1 - rho1) / (1 + rho1)
        n_eff = n * (1 - rho1) / (1 + rho1)
        return max(1, int(round(n_eff)))
