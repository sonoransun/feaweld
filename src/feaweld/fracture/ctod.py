"""Crack-tip opening displacement (CTOD) estimators (Track F3).

Provides two MVP estimators:

* :func:`ctod_displacement_extrapolation` — pick one node on each crack
  flank a fixed distance behind the tip and return the magnitude of the
  opening displacement. Simple, noisy, but easy to sanity-check.
* :func:`ctod_90_degree` — Rice's 45-degree-intercept (a.k.a. 90-degree
  clip) method: find the point on the deformed crack flank where a line
  from the tip makes a 45-degree angle with the crack axis, and take the
  perpendicular distance from that point to the undeformed crack line as
  CTOD.

Both methods return a :class:`CTODResult` which carries the scalar plus
a method tag and a small metadata dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CTODResult:
    ctod: float
    method: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_xy(vec: NDArray) -> NDArray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] not in (2, 3):
        raise ValueError(f"expected (2,) or (3,) vector, got shape {arr.shape}")
    return np.array([arr[0], arr[1]], dtype=np.float64)


def _unit(v: NDArray) -> NDArray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length vector cannot be normalised")
    return v / n


# ---------------------------------------------------------------------------
# Displacement extrapolation method
# ---------------------------------------------------------------------------


def ctod_displacement_extrapolation(
    fea_results: FEAResults,
    crack_tip: NDArray,
    crack_axis: NDArray,
    offset_distance: float,
) -> CTODResult:
    """CTOD from nodal displacements on either flank a fixed offset behind the tip.

    Selects the two nodes closest to the probe point
    ``tip - offset_distance * crack_axis`` — one with a strictly positive
    perpendicular coordinate (upper flank) and one strictly negative
    (lower flank) — and returns ``||u_upper - u_lower||``.

    ``crack_axis`` points *from* the crack mouth *toward* the tip; the
    probe is placed behind the tip by negating this direction.
    """
    tip = _as_xy(crack_tip)
    axis = _unit(_as_xy(crack_axis))
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)

    mesh = fea_results.mesh
    if fea_results.displacement is None:
        raise ValueError("FEAResults.displacement is required")

    nodes_xy = mesh.nodes[:, :2]
    u_xy = fea_results.displacement[:, :2]

    probe = tip - offset_distance * axis

    # Local coordinates relative to the tip, projected onto (axis, normal).
    rel = nodes_xy - tip[None, :]
    s = rel @ axis   # along-crack coordinate (negative = behind tip)
    t = rel @ normal  # perpendicular coordinate (sign tells flank)

    probe_rel = probe - tip
    s_probe = float(probe_rel @ axis)
    t_probe = float(probe_rel @ normal)

    # Distance in (s, t) from each node to probe point.
    ds = s - s_probe
    dt = t - t_probe
    dist = np.sqrt(ds**2 + dt**2)

    upper_mask = t > 1e-12
    lower_mask = t < -1e-12

    if not np.any(upper_mask) or not np.any(lower_mask):
        raise ValueError(
            "ctod_displacement_extrapolation: need nodes on both flanks"
        )

    upper_idx = int(np.where(upper_mask)[0][np.argmin(dist[upper_mask])])
    lower_idx = int(np.where(lower_mask)[0][np.argmin(dist[lower_mask])])

    du = u_xy[upper_idx] - u_xy[lower_idx]
    # Opening displacement is the projection onto the normal direction.
    opening = float(du @ normal)
    # Return absolute opening; compressive (overlapping) flanks clamped to 0.
    ctod = max(opening, 0.0)

    return CTODResult(
        ctod=ctod,
        method="displacement_extrapolation",
        metadata={
            "upper_node": upper_idx,
            "lower_node": lower_idx,
            "offset_distance": offset_distance,
            "raw_opening": opening,
        },
    )


# ---------------------------------------------------------------------------
# 90-degree (45-degree intercept) method
# ---------------------------------------------------------------------------


def ctod_90_degree(
    fea_results: FEAResults,
    crack_tip: NDArray,
    crack_axis: NDArray,
) -> CTODResult:
    """Rice's 45-degree intercept CTOD.

    Sample the upper and lower crack flanks at a ladder of offsets behind
    the tip, compute the deformed flank positions, and find the point on
    each flank where the ray from the tip makes a 45-degree angle with
    the crack axis. The CTOD is the total opening at that point, i.e.
    twice the perpendicular displacement of the 45-degree intercept above
    the undeformed crack line (equivalently, the distance between the
    upper and lower intercepts).
    """
    tip = _as_xy(crack_tip)
    axis = _unit(_as_xy(crack_axis))
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)

    mesh = fea_results.mesh
    if fea_results.displacement is None:
        raise ValueError("FEAResults.displacement is required")
    nodes_xy = mesh.nodes[:, :2]
    u_xy = fea_results.displacement[:, :2]

    rel = nodes_xy - tip[None, :]
    s = rel @ axis
    t = rel @ normal

    # Only look at mesh nodes strictly behind the tip.
    behind = s < -1e-12
    upper = behind & (t > 1e-12)
    lower = behind & (t < -1e-12)

    if not np.any(upper) or not np.any(lower):
        raise ValueError("ctod_90_degree: need nodes on both flanks behind the tip")

    def _flank_intercept(mask: NDArray, sign: float) -> tuple[float, float]:
        idx = np.where(mask)[0]
        s_flank = s[idx]
        t_flank = t[idx]
        # Rank by |s| (distance behind tip), take the 10 closest.
        order = np.argsort(np.abs(s_flank))
        take = idx[order[: min(10, idx.size)]]
        # Sort them by s ascending.
        take = take[np.argsort(s[take])]

        # Deformed nodal positions in (axis, normal) coords.
        rel_def = (nodes_xy[take] + u_xy[take]) - tip[None, :]
        s_def = rel_def @ axis
        t_def = rel_def @ normal

        # We want (t_def - 0) / (-s_def) = 1 for a 45-degree ray, i.e.
        # f(s) = sign * t_def(s) + s_def(s)  hits zero (because s_def < 0).
        f = sign * t_def + s_def
        # Find sign change; fall back to linear interp from the closest two.
        i_hit = None
        for k in range(f.size - 1):
            if f[k] * f[k + 1] <= 0.0 and abs(f[k] - f[k + 1]) > 1e-20:
                i_hit = k
                break
        if i_hit is None:
            # No bracket — use the last two points and extrapolate.
            i_hit = max(f.size - 2, 0)
        k = i_hit
        if f.size < 2:
            return float(s_def[0]), float(t_def[0])
        alpha = f[k] / (f[k] - f[k + 1])
        alpha = float(np.clip(alpha, 0.0, 1.0))
        s_star = float(s_def[k] + alpha * (s_def[k + 1] - s_def[k]))
        t_star = float(t_def[k] + alpha * (t_def[k + 1] - t_def[k]))
        return s_star, t_star

    s_up, t_up = _flank_intercept(upper, sign=+1.0)
    s_dn, t_dn = _flank_intercept(lower, sign=-1.0)

    opening = float(t_up - t_dn)
    ctod = max(opening, 0.0)

    return CTODResult(
        ctod=ctod,
        method="90_degree",
        metadata={
            "upper_intercept": (s_up, t_up),
            "lower_intercept": (s_dn, t_dn),
            "raw_opening": opening,
        },
    )
