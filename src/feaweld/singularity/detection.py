"""Automatic singular point detection in FEA results.

Singularities arise at re-entrant corners, point loads, and sharp notches
where the theoretical stress is unbounded.  This module compares results on
two (or more) mesh refinement levels and flags nodes whose stress grows
without converging — the hallmark of a mesh-dependent singularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults, FEMesh, StressField


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SingularityInfo:
    """Information about a detected (or candidate) singular point."""

    node_id: int
    stress_value: float
    convergence_rate: float
    is_singular: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_singularities(
    results_coarse: FEAResults,
    results_fine: FEAResults,
    threshold: float = 0.20,
) -> list[SingularityInfo]:
    """Compare stress between coarse and fine mesh results and flag singularities.

    For each node in the fine mesh the nearest node in the coarse mesh is
    located.  If the von-Mises stress increased by more than *threshold*
    (default 20 %) the node is flagged as *potentially* singular.  A true
    singularity is one where the stress increases **without converging**,
    i.e. the rate of increase suggests unbounded growth.

    Parameters
    ----------
    results_coarse:
        FEA results on the coarser mesh.
    results_fine:
        FEA results on the finer mesh.
    threshold:
        Fractional stress increase considered suspicious (0.20 = 20 %).

    Returns
    -------
    list[SingularityInfo]
        One entry per fine-mesh node that exceeds the threshold, with a
        boolean ``is_singular`` flag set when the convergence rate
        indicates divergent behaviour.
    """
    if results_coarse.stress is None or results_fine.stress is None:
        raise ValueError("Both results must contain a stress field.")

    coarse_nodes = results_coarse.mesh.nodes  # (n_c, 3)
    fine_nodes = results_fine.mesh.nodes  # (n_f, 3)

    vm_coarse = results_coarse.stress.von_mises  # (n_c,)
    vm_fine = results_fine.stress.von_mises  # (n_f,)

    # For each fine node, find nearest coarse node (brute-force but correct).
    # Using broadcasting: dist[i, j] = ||fine[i] - coarse[j]||
    # For large meshes this could be replaced with a KD-tree, but the
    # squared-distance matrix approach is clear and fine for typical sizes.
    nearest_coarse_idx = _nearest_nodes(fine_nodes, coarse_nodes)

    # Characteristic mesh sizes (average edge length proxy: cube-root of
    # bounding-box volume / n_nodes).
    h_coarse = _characteristic_mesh_size(results_coarse.mesh)
    h_fine = _characteristic_mesh_size(results_fine.mesh)

    detections: list[SingularityInfo] = []
    for fine_idx in range(len(vm_fine)):
        coarse_idx = nearest_coarse_idx[fine_idx]
        s_fine = float(vm_fine[fine_idx])
        s_coarse = float(vm_coarse[coarse_idx])

        if s_coarse == 0.0:
            # Cannot compute relative increase; skip.
            continue

        relative_increase = (s_fine - s_coarse) / abs(s_coarse)

        if relative_increase <= threshold:
            continue

        # Estimate a pseudo-convergence rate from these two levels.
        # A converging quantity has rate > 0; a singular one ~ 0 or < 0.
        rate = estimate_convergence_rate(
            stress_values=[s_fine, s_coarse],
            mesh_sizes=[h_fine, h_coarse],
        )

        # Heuristic: if the convergence rate is <= 0.5 (well below the
        # theoretical p=1 or p=2 for FE), treat as singular.
        is_singular = rate < 0.5

        detections.append(
            SingularityInfo(
                node_id=fine_idx,
                stress_value=s_fine,
                convergence_rate=rate,
                is_singular=is_singular,
            )
        )

    return detections


def estimate_convergence_rate(
    stress_values: list[float],
    mesh_sizes: list[float],
) -> float:
    """Estimate the convergence rate of a stress quantity.

    Given stress values at a node for **two or more** mesh refinement
    levels, the convergence rate *p* is estimated.  For two levels, a
    simple log-ratio is used.  For three or more levels, Richardson
    extrapolation provides a more robust estimate.

    A **positive** rate means the quantity is converging as the mesh is
    refined (mesh size decreases).  A **zero or negative** rate suggests
    that the quantity diverges — the signature of a singularity.

    Parameters
    ----------
    stress_values:
        Stress at the point of interest, ordered from **finest** mesh to
        **coarsest** mesh.
    mesh_sizes:
        Corresponding characteristic mesh sizes (finest first).

    Returns
    -------
    float
        Estimated convergence order *p*.
    """
    if len(stress_values) < 2 or len(stress_values) != len(mesh_sizes):
        raise ValueError(
            "Need at least two stress / mesh-size pairs of equal length."
        )

    # Sort so that h increases (coarsest last).
    order = np.argsort(mesh_sizes)
    h = np.array(mesh_sizes, dtype=np.float64)[order]
    s = np.array(stress_values, dtype=np.float64)[order]

    if len(s) == 2:
        # Two levels: p = ln(|Δs|) / ln(r), where r = h[1]/h[0].
        r = h[1] / h[0]
        if r <= 1.0:
            raise ValueError("Mesh sizes must be distinct.")
        ds = s[1] - s[0]
        if ds == 0.0:
            # Stress unchanged → perfectly converged.
            return float("inf")
        # Rate of change of stress with mesh refinement.
        # If stress *increases* as h *decreases*, the sign flips.
        # We define convergence rate as the exponent in |error| ~ h^p.
        # |f_exact - f_h| ~ C * h^p  →  log-ratio gives p.
        # With only two values we approximate error as |s_coarse - s_fine|.
        error_abs = abs(ds)
        # Relate to mesh-size ratio
        p = np.log(error_abs / abs(s[0] - s[1])) / np.log(r) if False else 0.0
        # Actually, with two points the best we can do is:
        #   error_coarse / error_fine ≈ (h_coarse / h_fine)^p
        # but we don't know the exact value.  Use the *fractional change*
        # as a proxy: if stress is converging, |s2-s1|/|s1| shrinks with h.
        # A simple indicator: p = ln(|s1/s2|) / ln(h2/h1)
        # For a converging field, s1 ≈ s2 and s1/s2 → 1 → p → ∞ (good).
        # For a divergent field, |s1| << |s2| (stress grows) → p < 0.
        if s[0] == 0.0 or s[1] == 0.0:
            return 0.0
        p = float(np.log(abs(s[1]) / abs(s[0])) / np.log(r))
        return p

    # Three or more levels: use Richardson-style estimate.
    # We use the three finest levels.
    f1, f2, f3 = s[0], s[1], s[2]
    h1, h2, h3 = h[0], h[1], h[2]
    r21 = h2 / h1
    r32 = h3 / h2

    denom = f2 - f1
    numer = f3 - f2
    if denom == 0.0:
        return float("inf")  # perfectly converged between finest levels
    ratio = numer / denom
    if ratio <= 0.0:
        # Oscillatory or non-monotone behaviour.  Return 0 to signal
        # non-convergence / possible singularity.
        return 0.0

    p = float(np.log(abs(ratio)) / np.log(r21))
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_nodes(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Return the index of the nearest *reference* node for each *query* node.

    Uses a chunked approach to avoid allocating a full (n_q, n_r) distance
    matrix when either set is large.
    """
    n_q = query.shape[0]
    chunk = max(1, min(1000, n_q))
    result = np.empty(n_q, dtype=np.int64)
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        # (chunk, 1, 3) - (1, n_r, 3) → (chunk, n_r)
        diff = query[start:end, np.newaxis, :] - reference[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        result[start:end] = np.argmin(dist_sq, axis=1)
    return result


def _characteristic_mesh_size(mesh: FEMesh) -> float:
    """Estimate a characteristic element size from the bounding box and node count."""
    bbox_min = mesh.nodes.min(axis=0)
    bbox_max = mesh.nodes.max(axis=0)
    volume = float(np.prod(bbox_max - bbox_min + 1e-30))
    return (volume / max(mesh.n_nodes, 1)) ** (1.0 / mesh.ndim)
