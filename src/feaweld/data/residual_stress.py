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

    Parameters
    ----------
    name : str
        Profile name (e.g. ``"BS7910_Level2_butt"``).

    Returns
    -------
    ResidualStressProfile
        ResidualStressProfile dataclass.

    Raises
    ------
    KeyError
        If the profile name is not found.
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

    Parameters
    ----------
    name : str
        Profile name (e.g. ``"BS7910_Level2_butt"``).
    z_over_t : float | np.ndarray
        Normalised through-thickness position(s) in [0, 1].
        Scalar or array.
    yield_strength : float
        Material yield strength in MPa.

    Returns
    -------
    float | np.ndarray
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


def surface_residual_stress(name: str, yield_strength: float) -> float:
    """Residual stress at the weld surface (z/t = 0) for a named profile.

    Parameters
    ----------
    name : str
        Profile name (e.g. ``"BS7910_Level2_butt"``).
    yield_strength : float
        Material yield strength in MPa.

    Returns
    -------
    float
        Surface residual stress in MPa.
    """
    return float(evaluate_residual_stress(name, 0.0, yield_strength))


def residual_stress_field(
    name: str,
    mesh,
    thickness: float,
    yield_strength: float,
    thickness_axis: int = 1,
    component: int = 1,
    surface_coordinate: float | None = None,
) -> np.ndarray:
    """Nodal residual stress field from a through-thickness profile.

    Maps each node's depth below the welded plate's surface to a
    normalised position $z/t$, evaluates the profile there, and places
    the result in a single Voigt component.

    With *surface_coordinate* given, the profile applies only within the
    plate band ``surface_coordinate - thickness <= coord <=
    surface_coordinate`` along *thickness_axis*; nodes outside the band
    (an attachment web above the plate, a second plate below it) get
    zero residual.  Without it, the surface is anchored at the mesh's
    maximum coordinate along the axis and depths beyond the thickness
    clamp to $z/t = 1$ (the legacy behavior, appropriate for meshes that
    consist of the plate alone).

    Parameters
    ----------
    name : str
        Profile name (e.g. ``"BS7910_Level2_butt"``).
    mesh : FEMesh
        Mesh whose nodes the field is evaluated at.
    thickness : float
        Plate thickness t (mm).
    yield_strength : float
        Material yield strength in MPa.
    thickness_axis : int
        Coordinate axis of the through-thickness direction (default 1,
        the y axis).
    component : int
        Voigt component index receiving the profile stress (default 1,
        $\\sigma_{yy}$).
    surface_coordinate : float | None
        Coordinate of the welded plate's surface (the $z/t = 0$ face)
        along *thickness_axis*.  *None* anchors at ``coords.max()`` and
        clamps every node into [0, 1].

    Returns
    -------
    np.ndarray
        ``(n_nodes, 6)`` stress field in Voigt notation (MPa); all other
        components are zero.
    """
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    coords = np.asarray(mesh.nodes, dtype=np.float64)[:, thickness_axis]
    if surface_coordinate is None:
        depth = coords.max() - coords
        z_over_t = np.clip(depth / thickness, 0.0, 1.0)
        values = np.asarray(
            evaluate_residual_stress(name, z_over_t, yield_strength)
        )
    else:
        depth = float(surface_coordinate) - coords
        tol = 1e-9 * thickness
        in_band = (depth >= -tol) & (depth <= thickness + tol)
        values = np.zeros_like(coords)
        z_over_t = np.clip(depth[in_band] / thickness, 0.0, 1.0)
        values[in_band] = evaluate_residual_stress(
            name, z_over_t, yield_strength,
        )
    field_arr = np.zeros((coords.shape[0], 6))
    field_arr[:, component] = values
    return field_arr


def list_residual_profiles(
    *,
    standard: str | None = None,
    weld_type: str | None = None,
) -> list[ResidualStressProfile]:
    """List available residual stress profiles with optional filtering.

    Parameters
    ----------
    standard : str | None
        Filter by standard (e.g. ``"BS7910"``, ``"API579"``).
    weld_type : str | None
        Filter by weld type (e.g. ``"butt"``, ``"fillet"``).

    Returns
    -------
    list[ResidualStressProfile]
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
