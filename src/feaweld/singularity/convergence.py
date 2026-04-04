"""Automated mesh convergence studies using Richardson extrapolation.

This module implements the Grid Convergence Index (GCI) methodology of
Roache (1994) which provides a consistent, quantitative way to assess
mesh-independence of FEA results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    """Outcome of a mesh convergence study at a single point/quantity."""

    mesh_sizes: list[float]
    stress_values: list[float]
    extrapolated_value: float
    convergence_order: float
    gci: float  # Grid Convergence Index (fraction, e.g. 0.03 = 3 %)
    is_converged: bool


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def richardson_extrapolation(
    values: list[float],
    sizes: list[float],
) -> tuple[float, float]:
    """Richardson extrapolation from three mesh refinement levels.

    Parameters
    ----------
    values:
        Quantity of interest at three levels, ordered **finest → coarsest**
        (f1, f2, f3).
    sizes:
        Corresponding characteristic mesh sizes, ordered
        **finest → coarsest** (h1, h2, h3).

    Returns
    -------
    (extrapolated_value, convergence_order)
        The zero-mesh-size extrapolated value and the observed order of
        convergence *p*.

    Raises
    ------
    ValueError
        If fewer than three levels are provided or the data does not allow a
        valid extrapolation (e.g. oscillatory behaviour).
    """
    if len(values) < 3 or len(sizes) < 3:
        raise ValueError("Richardson extrapolation requires at least 3 levels.")

    # Unpack finest → coarsest
    f1, f2, f3 = float(values[0]), float(values[1]), float(values[2])
    h1, h2, h3 = float(sizes[0]), float(sizes[1]), float(sizes[2])

    # Refinement ratio (assumed constant; use h2/h1)
    r = h2 / h1
    if r <= 1.0:
        raise ValueError(
            f"Mesh sizes must increase from finest to coarsest (got r={r:.4f})."
        )

    denom = f2 - f1
    numer = f3 - f2
    if denom == 0.0:
        # f1 == f2: already converged between the two finest meshes.
        return f1, float("inf")

    ratio = numer / denom
    if ratio <= 0.0:
        # Oscillatory — fall back to a safe estimate.
        # Use the absolute values to get a rough order.
        if abs(numer) == 0.0:
            p = float("inf")
        else:
            p = abs(math.log(abs(ratio)) / math.log(r))
        extrapolated = f1  # best available
        return extrapolated, p

    p = math.log(ratio) / math.log(r)

    rp = r ** p
    extrapolated = f1 + (f1 - f2) / (rp - 1.0)

    return extrapolated, p


def grid_convergence_index(
    f_fine: float,
    f_coarse: float,
    r: float,
    p: float,
    safety_factor: float = 1.25,
) -> float:
    """Compute the Grid Convergence Index (GCI).

    GCI = *safety_factor* * |epsilon| / (r^p - 1)

    where epsilon = (f_coarse - f_fine) / f_fine.

    Parameters
    ----------
    f_fine:
        Value on the finer mesh.
    f_coarse:
        Value on the coarser mesh.
    r:
        Mesh refinement ratio (h_coarse / h_fine).
    p:
        Observed convergence order.
    safety_factor:
        Safety factor (1.25 for >=3 meshes, 3.0 for 2 meshes).

    Returns
    -------
    float
        GCI as a fraction (e.g. 0.03 = 3 %).
    """
    if f_fine == 0.0:
        if f_coarse == 0.0:
            return 0.0
        return float("inf")

    epsilon = (f_coarse - f_fine) / f_fine
    rp = r ** p
    if rp <= 1.0:
        return float("inf")
    return safety_factor * abs(epsilon) / (rp - 1.0)


def convergence_study(
    stress_at_point: list[float],
    mesh_sizes: list[float],
) -> ConvergenceResult:
    """Run a full mesh convergence analysis for a scalar quantity.

    Parameters
    ----------
    stress_at_point:
        The quantity of interest at multiple refinement levels, ordered
        **finest → coarsest**.
    mesh_sizes:
        Corresponding characteristic element sizes, ordered
        **finest → coarsest**.

    Returns
    -------
    ConvergenceResult
        Complete convergence metrics.  ``is_converged`` is ``True`` when
        GCI < 0.05 (5 %).
    """
    if len(stress_at_point) < 3 or len(mesh_sizes) < 3:
        raise ValueError("Convergence study requires at least 3 mesh levels.")
    if len(stress_at_point) != len(mesh_sizes):
        raise ValueError("stress_at_point and mesh_sizes must have the same length.")

    # Ensure ordering is finest → coarsest.
    order = np.argsort(mesh_sizes)
    sizes_sorted = [mesh_sizes[i] for i in order]
    values_sorted = [stress_at_point[i] for i in order]

    f1 = values_sorted[0]
    f2 = values_sorted[1]

    extrapolated, p = richardson_extrapolation(values_sorted, sizes_sorted)

    r = sizes_sorted[1] / sizes_sorted[0]
    gci = grid_convergence_index(f1, f2, r, p)

    return ConvergenceResult(
        mesh_sizes=sizes_sorted,
        stress_values=values_sorted,
        extrapolated_value=extrapolated,
        convergence_order=p,
        gci=gci,
        is_converged=gci < 0.05,
    )
