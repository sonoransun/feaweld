"""Volumetric Strain Energy Density via Monte Carlo over arbitrary control volumes.

This module generalises the Lazzarin spherical SED helper in
:mod:`feaweld.postprocess.sed`. Instead of hard-coding a sphere we
evaluate the averaged SED over any control volume defined by a
user-supplied predicate plus a bounding box, using a Monte Carlo sampler
driven by :func:`numpy.random.default_rng`.

The SED at each accepted sample is evaluated by nearest-neighbour
interpolation on the FE nodal stresses (and, when available, strains).
For elastic isotropic analyses the fallback is the familiar
``σ_vm² / (2 E)`` von Mises form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from feaweld.core.types import FEAResults


@dataclass
class VolumetricSEDResult:
    """Monte Carlo averaged SED result over an arbitrary control volume."""

    sed_average: float
    sed_std: float
    n_samples: int
    volume: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Predicate factories
# ---------------------------------------------------------------------------


def cylindrical_control_volume(
    axis_origin: NDArray,
    axis_direction: NDArray,
    radius: float,
    height: float,
) -> Callable[[NDArray], NDArray]:
    """Return a predicate testing membership in a finite cylinder.

    Parameters
    ----------
    axis_origin:
        3-vector: one endpoint of the cylinder axis (the "base" center).
    axis_direction:
        3-vector: direction of the axis (will be normalised). The cylinder
        extends from ``axis_origin`` to ``axis_origin + height * axis_hat``.
    radius:
        Cylinder radius.
    height:
        Length of the cylinder along the axis.
    """
    origin = np.asarray(axis_origin, dtype=np.float64).reshape(3)
    direction = np.asarray(axis_direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(direction))
    if n < 1e-15:
        raise ValueError("axis_direction must be non-zero")
    axis_hat = direction / n
    r_sq = float(radius) ** 2
    h = float(height)

    def predicate(points: NDArray) -> NDArray:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        rel = p - origin[None, :]
        along = rel @ axis_hat  # projection onto axis, shape (n,)
        perp = rel - np.outer(along, axis_hat)
        perp_r_sq = np.sum(perp * perp, axis=1)
        mask = (along >= 0.0) & (along <= h) & (perp_r_sq <= r_sq)
        if points.ndim == 1:
            return mask[0:1]
        return mask

    return predicate


def ellipsoidal_control_volume(
    center: NDArray,
    semi_axes: NDArray,
) -> Callable[[NDArray], NDArray]:
    """Return a predicate testing membership in an axis-aligned ellipsoid.

    An axis-aligned ellipsoid centred at ``center`` with semi-axes
    ``(a, b, c)`` along (x, y, z):

        ((x - cx) / a)^2 + ((y - cy) / b)^2 + ((z - cz) / c)^2 <= 1.
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    ax = np.asarray(semi_axes, dtype=np.float64).reshape(3)
    if np.any(ax <= 0.0):
        raise ValueError("semi_axes must all be positive")

    def predicate(points: NDArray) -> NDArray:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        rel = (p - c[None, :]) / ax[None, :]
        val = np.sum(rel * rel, axis=1)
        mask = val <= 1.0
        if points.ndim == 1:
            return mask[0:1]
        return mask

    return predicate


def spherical_control_volume(
    center: NDArray,
    radius: float,
) -> Callable[[NDArray], NDArray]:
    """Return a predicate for a sphere (the degenerate ellipsoid case).

    Provided so the volumetric helper can be validated against the legacy
    Lazzarin spherical SED implementation.
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r_sq = float(radius) ** 2

    def predicate(points: NDArray) -> NDArray:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        rel = p - c[None, :]
        d_sq = np.sum(rel * rel, axis=1)
        mask = d_sq <= r_sq
        if points.ndim == 1:
            return mask[0:1]
        return mask

    return predicate


# ---------------------------------------------------------------------------
# Defect-wrapping helper
# ---------------------------------------------------------------------------


def defect_wrapping_volume(
    defect, padding: float = 1.0
) -> tuple[Callable[[NDArray], NDArray], tuple[NDArray, NDArray]]:
    """Return ``(predicate, bounding_box)`` for a volume that wraps a defect.

    The returned control volume is padded by ``padding`` mm on all sides
    around the defect's bounding geometry.

    Dispatch is via duck typing on the ``defect_type`` attribute:

    * ``"pore"`` -> sphere of radius ``d/2 + padding`` about the center.
    * ``"cluster_porosity"`` -> sphere of radius ``cluster.radius + padding``.
    * ``"slag"`` -> axis-aligned ellipsoid with padded semi-axes.
    * ``"surface_crack"``/``"undercut"`` -> cylinder capped around the
      linear segment with radius ``depth + padding``.
    * Anything else -> spherical fallback with ``radius = padding``
      around the defect's ``center`` (or best-effort position).
    """
    dtype = getattr(defect, "defect_type", None)

    def _center_xyz(point) -> NDArray:
        # Accept both Point3D and ndarray-like.
        if hasattr(point, "x"):
            return np.array([point.x, point.y, point.z], dtype=np.float64)
        return np.asarray(point, dtype=np.float64).reshape(3)

    if dtype == "pore":
        center = _center_xyz(defect.center)
        r = 0.5 * float(defect.diameter) + float(padding)
        predicate = spherical_control_volume(center, r)
        bbox = (center - r, center + r)
        return predicate, bbox

    if dtype == "cluster_porosity":
        center = _center_xyz(defect.center)
        r = float(defect.radius) + float(padding)
        predicate = spherical_control_volume(center, r)
        bbox = (center - r, center + r)
        return predicate, bbox

    if dtype == "slag":
        center = _center_xyz(defect.center)
        axes = np.asarray(defect.semi_axes, dtype=np.float64) + float(padding)
        predicate = ellipsoidal_control_volume(center, axes)
        bbox = (center - axes, center + axes)
        return predicate, bbox

    if dtype in ("surface_crack", "undercut", "root_gap"):
        start = _center_xyz(defect.start)
        end = _center_xyz(defect.end)
        axis_vec = end - start
        length = float(np.linalg.norm(axis_vec))
        if length < 1e-12:
            # Degenerate line — fall back to a sphere of radius padding.
            center = start
            predicate = spherical_control_volume(center, float(padding))
            bbox = (center - padding, center + padding)
            return predicate, bbox
        axis_hat = axis_vec / length
        depth = float(getattr(defect, "depth", padding))
        radius = depth + float(padding)
        # Extend height on both ends by padding.
        base = start - padding * axis_hat
        height = length + 2.0 * float(padding)
        predicate = cylindrical_control_volume(base, axis_hat, radius, height)

        # Cylinder bbox: conservative axis-aligned expansion.
        end_pt = base + height * axis_hat
        pad_vec = np.full(3, radius + float(padding))
        bbox_min = np.minimum(base, end_pt) - pad_vec
        bbox_max = np.maximum(base, end_pt) + pad_vec
        return predicate, (bbox_min, bbox_max)

    # Fallback: use best-effort center with a spherical padding.
    center_attr = getattr(defect, "center", None) or getattr(defect, "plane_origin", None)
    if center_attr is None:
        raise TypeError(
            f"defect_wrapping_volume cannot infer a bounding volume for "
            f"defect_type={dtype!r}"
        )
    center = _center_xyz(center_attr)
    predicate = spherical_control_volume(center, float(padding))
    bbox = (center - padding, center + padding)
    return predicate, bbox


# ---------------------------------------------------------------------------
# Monte Carlo averaged SED
# ---------------------------------------------------------------------------


def averaged_sed_over_volume(
    fea_results: FEAResults,
    volume_predicate: Callable[[NDArray], NDArray],
    bounding_box: tuple[NDArray, NDArray],
    n_samples: int = 500,
    seed: int = 0,
    E: float = 210000.0,
    nu: float = 0.3,
) -> VolumetricSEDResult:
    """Monte Carlo average of strain-energy density over an arbitrary volume.

    Samples ``n_samples`` points uniformly inside the bounding box, keeps
    those that pass the ``volume_predicate``, interpolates the stress (and
    strain, when available) at each accepted point by nearest-neighbour
    lookup on the FE mesh nodes, and averages

        SED = 0.5 * σ : ε

    falling back to ``σ_vm² / (2 E)`` if the nodal strain array is not
    populated.

    The reported ``volume`` is the Monte Carlo estimate of the control
    volume,

        V ≈ V_bbox * (n_inside / n_samples),

    and ``sed_std`` is the standard error of the mean (``std / sqrt(n)``).
    """
    if fea_results.stress is None:
        raise ValueError("FEAResults.stress is required for averaged_sed_over_volume")
    mesh = fea_results.mesh
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    if nodes.shape[1] == 2:
        nodes = np.column_stack([nodes, np.zeros(nodes.shape[0])])

    bbox_min = np.asarray(bounding_box[0], dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bounding_box[1], dtype=np.float64).reshape(3)
    if np.any(bbox_max <= bbox_min):
        raise ValueError("bounding_box max must be strictly greater than min on all axes")
    bbox_extents = bbox_max - bbox_min
    V_bbox = float(np.prod(bbox_extents))

    rng = np.random.default_rng(seed)
    samples = rng.uniform(size=(int(n_samples), 3))
    points = bbox_min[None, :] + samples * bbox_extents[None, :]

    inside = np.asarray(volume_predicate(points), dtype=bool)
    n_inside = int(inside.sum())

    volume_estimate = V_bbox * (n_inside / max(n_samples, 1))

    if n_inside == 0:
        return VolumetricSEDResult(
            sed_average=0.0,
            sed_std=0.0,
            n_samples=0,
            volume=0.0,
            metadata={
                "bbox_volume": V_bbox,
                "n_requested": int(n_samples),
                "n_inside": 0,
                "fallback": "von_mises",
            },
        )

    tree = cKDTree(nodes)
    accepted = points[inside]
    _, nearest_idx = tree.query(accepted, k=1)
    nearest_idx = np.atleast_1d(nearest_idx).astype(np.int64)

    sigma = fea_results.stress.values  # (n_nodes, 6)
    strain = fea_results.strain  # (n_nodes, 6) or None

    if strain is not None:
        s = sigma[nearest_idx]
        e = strain[nearest_idx]
        # Full Voigt contraction σ:ε = Σ σ_i ε_i with engineering shear
        # factors (τ_xy * γ_xy = 2 τ_xy * ε_xy, etc.).
        sed_per_sample = 0.5 * (
            s[:, 0] * e[:, 0]
            + s[:, 1] * e[:, 1]
            + s[:, 2] * e[:, 2]
            + s[:, 3] * e[:, 3]
            + s[:, 4] * e[:, 4]
            + s[:, 5] * e[:, 5]
        )
        fallback = "full_voigt"
    else:
        vm = fea_results.stress.von_mises[nearest_idx]
        sed_per_sample = (vm**2) / (2.0 * float(E))
        fallback = "von_mises"

    mean_sed = float(np.mean(sed_per_sample))
    if n_inside > 1:
        sample_std = float(np.std(sed_per_sample, ddof=1))
    else:
        sample_std = 0.0
    mc_std = sample_std / np.sqrt(n_inside) if n_inside > 0 else 0.0

    return VolumetricSEDResult(
        sed_average=mean_sed,
        sed_std=float(mc_std),
        n_samples=n_inside,
        volume=float(volume_estimate),
        metadata={
            "bbox_volume": V_bbox,
            "n_requested": int(n_samples),
            "n_inside": n_inside,
            "fallback": fallback,
            "sample_std": sample_std,
        },
    )
