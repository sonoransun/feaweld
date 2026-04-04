"""Fatigue knockdown and mean-stress correction factors.

Provides Marin surface-finish and size factors, Goodman / Gerber
mean-stress corrections, environment factors, and a combined knockdown
multiplier.
"""

from __future__ import annotations

import math


def surface_finish_factor(roughness_um: float, sigma_u: float) -> float:
    """Marin surface-finish factor k_a.

    Uses the empirical relationship:
        k_a = a * sigma_u^b
    where *a* and *b* depend on surface roughness class.

    A simplified model is used here:
        k_a = 1.0 - 0.22 * log10(roughness_um) * (272 / sigma_u)^0.2

    For machined surfaces (roughness ~ 1.6 um), k_a ~ 0.9 for mild steel.

    Parameters
    ----------
    roughness_um : float
        Arithmetic-mean surface roughness R_a in micrometres.
    sigma_u : float
        Ultimate tensile strength (MPa).

    Returns
    -------
    float
        Surface factor k_a (0 < k_a <= 1).
    """
    if roughness_um <= 0 or sigma_u <= 0:
        return 1.0

    # Marin coefficients (approximation covering ground to as-forged)
    log_r = math.log10(max(roughness_um, 0.05))
    k_a = 1.0 - 0.22 * log_r * (272.0 / sigma_u) ** 0.2
    return max(min(k_a, 1.0), 0.1)


def size_factor(diameter_mm: float) -> float:
    """Marin size factor k_b.

    Based on Shigley's formulation:
    - For 2.79 <= d <= 51 mm:  k_b = 1.24 * d^(-0.107)
    - For 51 < d <= 254 mm:   k_b = 1.51 * d^(-0.157)
    - For d < 2.79 mm:        k_b = 1.0

    Parameters
    ----------
    diameter_mm : float
        Effective diameter or equivalent dimension (mm).

    Returns
    -------
    float
        Size factor k_b.
    """
    d = diameter_mm
    if d <= 0:
        return 1.0
    if d < 2.79:
        return 1.0
    if d <= 51.0:
        return 1.24 * d ** (-0.107)
    if d <= 254.0:
        return 1.51 * d ** (-0.157)
    # Extrapolate for very large sections
    return 1.51 * 254.0 ** (-0.157) * (254.0 / d) ** 0.1


def goodman_correction(
    stress_amp: float,
    mean_stress: float,
    sigma_u: float,
) -> float:
    """Modified Goodman mean-stress correction.

    Returns the equivalent fully-reversed stress amplitude:
        S_eq = stress_amp / (1 - mean_stress / sigma_u)

    If mean_stress >= sigma_u the material is expected to yield/fail
    under static load; returns infinity.

    Parameters
    ----------
    stress_amp : float
        Alternating (half-range) stress amplitude (MPa).
    mean_stress : float
        Mean stress (MPa).
    sigma_u : float
        Ultimate tensile strength (MPa).

    Returns
    -------
    float
        Equivalent fully-reversed stress amplitude.
    """
    if sigma_u <= 0:
        raise ValueError("Ultimate strength must be positive.")
    if mean_stress >= sigma_u:
        return float("inf")
    if mean_stress <= 0:
        # Compressive mean stress -- Goodman is non-conservative;
        # conservatively return the amplitude unchanged.
        return stress_amp
    return stress_amp / (1.0 - mean_stress / sigma_u)


def gerber_correction(
    stress_amp: float,
    mean_stress: float,
    sigma_u: float,
) -> float:
    """Gerber parabola mean-stress correction.

    Returns the equivalent fully-reversed stress amplitude:
        S_eq = stress_amp / (1 - (mean_stress / sigma_u)^2)

    Parameters
    ----------
    stress_amp : float
        Alternating stress amplitude (MPa).
    mean_stress : float
        Mean stress (MPa).
    sigma_u : float
        Ultimate tensile strength (MPa).

    Returns
    -------
    float
        Equivalent fully-reversed stress amplitude.
    """
    if sigma_u <= 0:
        raise ValueError("Ultimate strength must be positive.")
    ratio = mean_stress / sigma_u
    if abs(ratio) >= 1.0:
        return float("inf")
    return stress_amp / (1.0 - ratio ** 2)


def environment_factor(environment: str) -> float:
    """Return a fatigue environment knockdown factor.

    Parameters
    ----------
    environment : str
        ``"air"`` (1.0), ``"corrosive"`` (0.6), or ``"seawater"`` (0.4).

    Returns
    -------
    float
        Knockdown factor k_env.
    """
    env = environment.lower().strip()
    factors = {
        "air": 1.0,
        "corrosive": 0.6,
        "seawater": 0.4,
    }
    if env not in factors:
        raise ValueError(
            f"Unknown environment '{environment}'. "
            f"Choose from {sorted(factors.keys())}."
        )
    return factors[env]


def combined_knockdown(
    k_a: float = 1.0,
    k_b: float = 1.0,
    k_env: float = 1.0,
    **extra_factors: float,
) -> float:
    """Product of all knockdown factors.

    Parameters
    ----------
    k_a : float
        Surface finish factor.
    k_b : float
        Size factor.
    k_env : float
        Environment factor.
    **extra_factors
        Any additional named factors to include (e.g. ``k_reliability``).

    Returns
    -------
    float
        Combined knockdown factor (product of all inputs).
    """
    result = k_a * k_b * k_env
    for v in extra_factors.values():
        result *= v
    return result
