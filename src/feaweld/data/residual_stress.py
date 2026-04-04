"""Residual stress profile lookup and interpolation.

Provides through-thickness residual stress distributions from major
fitness-for-service standards (BS 7910, API 579, R6, FITNET, DNV).
Profiles are normalised as *stress / sigma_y* versus *z / t*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from feaweld.data.cache import get_cache


@dataclass
class ResidualStressProfile:
    """A through-thickness residual stress profile."""

    name: str
    standard: str
    weld_type: str
    z_over_t: list[float] = field(default_factory=list)
    stress_over_sy: list[float] = field(default_factory=list)
    notes: str = ""


def _to_profile(entry: dict) -> ResidualStressProfile:
    """Convert a raw dict from the JSON data to a ResidualStressProfile."""
    return ResidualStressProfile(
        name=entry["name"],
        standard=entry["standard"],
        weld_type=entry["weld_type"],
        z_over_t=entry["z_over_t"],
        stress_over_sy=entry["stress_over_sy"],
        notes=entry.get("notes", ""),
    )


def get_residual_profile(name: str) -> ResidualStressProfile:
    """Look up a residual stress profile by name.

    Args:
        name: Profile name (e.g. ``"BS7910_Level2_butt"``).

    Returns:
        ResidualStressProfile dataclass.

    Raises:
        KeyError: If the profile name is not found.
    """
    data = get_cache().get("residual_stress/profiles")
    for entry in data:
        if entry["name"] == name:
            return _to_profile(entry)
    available = [e["name"] for e in data]
    raise KeyError(f"Residual stress profile not found: {name!r}. Available: {available}")


def evaluate_residual_stress(
    name: str,
    z_over_t: float | np.ndarray,
    yield_strength: float,
) -> float | np.ndarray:
    """Evaluate residual stress at given through-thickness position(s).

    Linearly interpolates the stored profile and scales by the yield
    strength.

    Args:
        name: Profile name (e.g. ``"BS7910_Level2_butt"``).
        z_over_t: Normalised through-thickness position(s) in [0, 1].
            Scalar or array.
        yield_strength: Material yield strength in MPa.

    Returns:
        Residual stress in MPa (same shape as *z_over_t*).
    """
    profile = get_residual_profile(name)
    z_arr = np.asarray(profile.z_over_t, dtype=np.float64)
    s_arr = np.asarray(profile.stress_over_sy, dtype=np.float64)

    normalised = np.interp(z_over_t, z_arr, s_arr)
    result = normalised * yield_strength

    # Return scalar if input was scalar
    if np.ndim(z_over_t) == 0:
        return float(result)
    return result


def list_residual_profiles(
    *,
    standard: str | None = None,
    weld_type: str | None = None,
) -> list[ResidualStressProfile]:
    """List available residual stress profiles with optional filtering.

    Args:
        standard: Filter by standard (e.g. ``"BS7910"``, ``"API579"``).
        weld_type: Filter by weld type (e.g. ``"butt"``, ``"fillet"``).

    Returns:
        List of matching profiles.
    """
    data = get_cache().get("residual_stress/profiles")
    results: list[ResidualStressProfile] = []

    for entry in data:
        if standard is not None and entry["standard"] != standard:
            continue
        if weld_type is not None and entry["weld_type"] != weld_type:
            continue
        results.append(_to_profile(entry))

    return results
