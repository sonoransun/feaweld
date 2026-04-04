"""Sobol sensitivity analysis and first-order reliability method (FORM).

Provides variance-based global sensitivity indices via the Saltelli sampling
scheme and a FORM implementation for structural reliability assessment.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from feaweld.probabilistic.monte_carlo import RandomVariable, _ppf


# ---------------------------------------------------------------------------
# Sobol sensitivity indices
# ---------------------------------------------------------------------------


def sobol_indices(
    variables: list[RandomVariable],
    analysis_func: Callable[[dict[str, float]], float],
    n_base: int = 1024,
    seed: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compute Sobol first-order and total sensitivity indices.

    Uses the Saltelli sampling scheme (2002) which requires
    ``n_base * (2 * k + 2)`` model evaluations for *k* variables.

    Parameters
    ----------
    variables : list[RandomVariable]
        Uncertain input variables.
    analysis_func : callable
        Scalar-valued model ``f(dict[str, float]) -> float``.
    n_base : int
        Base sample size (before Saltelli expansion).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"first_order": {name: S_i, ...}, "total": {name: ST_i, ...}}``
    """

    rng = np.random.default_rng(seed)
    k = len(variables)

    # --- Generate two independent uniform sample matrices A, B in [0,1] ----
    u_A = rng.random((n_base, k))
    u_B = rng.random((n_base, k))

    # --- Map to physical space via inverse CDF ---
    def _to_physical(u: NDArray) -> NDArray:
        x = np.empty_like(u)
        for j, var in enumerate(variables):
            x[:, j] = _ppf(var.distribution, var.params, u[:, j])
        return x

    def _evaluate(x_matrix: NDArray) -> NDArray:
        n = x_matrix.shape[0]
        y = np.empty(n)
        for i in range(n):
            var_dict = {var.name: float(x_matrix[i, j]) for j, var in enumerate(variables)}
            y[i] = analysis_func(var_dict)
        return y

    x_A = _to_physical(u_A)
    x_B = _to_physical(u_B)

    f_A = _evaluate(x_A)
    f_B = _evaluate(x_B)

    # --- For each variable i, create AB_i = A with column i from B ----------
    first_order: dict[str, float] = {}
    total: dict[str, float] = {}

    f0 = np.mean(f_A)
    var_y = np.var(f_A, ddof=0)

    if var_y == 0:
        # Deterministic model -- all indices are zero
        for var in variables:
            first_order[var.name] = 0.0
            total[var.name] = 0.0
        return {"first_order": first_order, "total": total}

    for i, var in enumerate(variables):
        # AB_i: A with column i replaced by B's column i
        u_AB_i = u_A.copy()
        u_AB_i[:, i] = u_B[:, i]
        x_AB_i = _to_physical(u_AB_i)
        f_AB_i = _evaluate(x_AB_i)

        # First-order: S_i = V_i / V(Y)
        # V_i = (1/N) * sum(f_B * (f_AB_i - f_A))  [Jansen estimator]
        V_i = np.mean(f_B * (f_AB_i - f_A))
        S_i = V_i / var_y

        # Total: ST_i = E_i / V(Y)
        # E_i = (1/(2N)) * sum((f_A - f_AB_i)^2)  [Jansen estimator]
        E_i = 0.5 * np.mean((f_A - f_AB_i) ** 2)
        ST_i = E_i / var_y

        first_order[var.name] = float(S_i)
        total[var.name] = float(ST_i)

    return {"first_order": first_order, "total": total}


# ---------------------------------------------------------------------------
# FORM -- First-Order Reliability Method
# ---------------------------------------------------------------------------


def reliability_index_form(
    variables: list[RandomVariable],
    limit_state_func: Callable[[dict[str, float]], float],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float | dict[str, float]]:
    """Compute the Hasofer-Lind reliability index using FORM.

    Finds the Most Probable Point (MPP) on the limit-state surface
    ``g(X) = 0`` in standard normal space using the HL-RF algorithm.

    Parameters
    ----------
    variables : list[RandomVariable]
        Independent random variables.
    limit_state_func : callable
        Limit-state function ``g(dict[str, float]) -> float``.  Failure is
        defined as ``g < 0``.
    max_iter : int
        Maximum HL-RF iterations.
    tol : float
        Convergence tolerance on the design-point movement.

    Returns
    -------
    dict
        ``{"beta": float, "probability_of_failure": float,
          "design_point": dict[str, float]}``
    """

    k = len(variables)

    # Marginal CDF and inverse CDF wrappers for each variable
    def _cdf(j: int, x: float) -> float:
        """CDF of variable j evaluated at x."""
        var = variables[j]
        p = var.params
        if var.distribution == "normal":
            return float(stats.norm.cdf(x, loc=p["mean"], scale=p["std"]))
        elif var.distribution == "lognormal":
            s = p["std"]
            scale = np.exp(p["mean"])
            return float(stats.lognorm.cdf(x, s=s, scale=scale))
        elif var.distribution == "weibull":
            return float(stats.weibull_min.cdf(x, c=p["shape"], scale=p["scale"]))
        elif var.distribution == "uniform":
            return float(stats.uniform.cdf(x, loc=p["low"], scale=p["high"] - p["low"]))
        elif var.distribution == "gumbel":
            return float(stats.gumbel_r.cdf(x, loc=p["loc"], scale=p["scale"]))
        else:
            raise ValueError(f"Unsupported distribution: {var.distribution}")

    def _ppf_single(j: int, u: float) -> float:
        """Inverse CDF of variable j."""
        var = variables[j]
        p = var.params
        if var.distribution == "normal":
            return float(stats.norm.ppf(u, loc=p["mean"], scale=p["std"]))
        elif var.distribution == "lognormal":
            s = p["std"]
            scale = np.exp(p["mean"])
            return float(stats.lognorm.ppf(u, s=s, scale=scale))
        elif var.distribution == "weibull":
            return float(stats.weibull_min.ppf(u, c=p["shape"], scale=p["scale"]))
        elif var.distribution == "uniform":
            return float(stats.uniform.ppf(u, loc=p["low"], scale=p["high"] - p["low"]))
        elif var.distribution == "gumbel":
            return float(stats.gumbel_r.ppf(u, loc=p["loc"], scale=p["scale"]))
        else:
            raise ValueError(f"Unsupported distribution: {var.distribution}")

    def _x_to_u(x: NDArray) -> NDArray:
        """Nataf / Rosenblatt transform: physical -> standard normal."""
        u = np.empty(k)
        for j in range(k):
            p = np.clip(_cdf(j, float(x[j])), 1e-15, 1.0 - 1e-15)
            u[j] = stats.norm.ppf(p)
        return u

    def _u_to_x(u: NDArray) -> NDArray:
        """Inverse transform: standard normal -> physical."""
        x = np.empty(k)
        for j in range(k):
            p = float(stats.norm.cdf(u[j]))
            p = np.clip(p, 1e-15, 1.0 - 1e-15)
            x[j] = _ppf_single(j, p)
        return x

    def _g_u(u: NDArray) -> float:
        """Limit-state in standard normal space."""
        x = _u_to_x(u)
        var_dict = {variables[j].name: float(x[j]) for j in range(k)}
        return limit_state_func(var_dict)

    def _grad_g_u(u: NDArray, h: float = 1e-5) -> NDArray:
        """Numerical gradient of g in u-space (central difference)."""
        grad = np.empty(k)
        g0 = _g_u(u)
        for j in range(k):
            u_fwd = u.copy()
            u_fwd[j] += h
            u_bwd = u.copy()
            u_bwd[j] -= h
            grad[j] = (_g_u(u_fwd) - _g_u(u_bwd)) / (2.0 * h)
        return grad

    # HL-RF iteration
    u = np.zeros(k)  # start at origin (mean point)

    for _iteration in range(max_iter):
        g_val = _g_u(u)
        grad = _grad_g_u(u)
        grad_norm_sq = np.dot(grad, grad)

        if grad_norm_sq < 1e-30:
            # Zero gradient -- cannot proceed
            break

        # HL-RF update: u_new = (grad^T u - g) / |grad|^2 * grad
        alpha = grad / np.sqrt(grad_norm_sq)  # unit gradient
        u_new = ((np.dot(grad, u) - g_val) / grad_norm_sq) * grad

        if np.linalg.norm(u_new - u) < tol:
            u = u_new
            break
        u = u_new

    beta = float(np.linalg.norm(u))
    pf = float(stats.norm.cdf(-beta))

    # Map design point back to physical space
    x_star = _u_to_x(u)
    design_point = {variables[j].name: float(x_star[j]) for j in range(k)}

    return {
        "beta": beta,
        "probability_of_failure": pf,
        "design_point": design_point,
    }
