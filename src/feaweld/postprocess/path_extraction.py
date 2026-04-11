"""Arc-length stress extraction along a :class:`WeldPath`.

This module provides RBF-interpolated sampling of nodal stress fields
along a 3D weld path defined by :class:`feaweld.geometry.weld_path.WeldPath`.
It complements :mod:`feaweld.postprocess.hotspot` for curved weld toes
where the discrete-node tangent estimator cannot produce meaningful
extrapolation directions.

The RBF kernel (``thin_plate_spline``) matches the one already used by
:mod:`feaweld.singularity.submodeling` for boundary displacement transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults, StressField
from feaweld.geometry.weld_path import WeldPath


FieldName = Literal["von_mises", "max_principal", "tresca", "hydrostatic"]
InterpMethod = Literal["rbf", "nearest"]


@dataclass
class PathExtractionResult:
    """Nodal field values sampled along arc-length positions on a path."""

    s: NDArray                      # (n_samples,) arc-length coordinates
    values: NDArray                 # (n_samples,) or (n_samples, k) field values
    points: NDArray                 # (n_samples, 3) sample coordinates
    frames: NDArray | None = None   # (n_samples, 3, 3) Frenet (T, N, B) stacked
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_along_path(
    fea_results: FEAResults,
    path: WeldPath,
    n_samples: int = 50,
    field: FieldName = "von_mises",
    method: InterpMethod = "rbf",
) -> PathExtractionResult:
    """Sample a nodal field along arc-length positions on a :class:`WeldPath`.

    Parameters
    ----------
    fea_results:
        FEA results containing a stress field.
    path:
        Weld path to sample along.
    n_samples:
        Number of arc-length samples (>= 2).
    field:
        Which derived stress quantity to interpolate.  One of
        ``"von_mises"``, ``"max_principal"``, ``"tresca"``, or
        ``"hydrostatic"``.
    method:
        ``"rbf"`` uses :class:`scipy.interpolate.RBFInterpolator` with a
        thin-plate-spline kernel (the same kernel used by
        :mod:`feaweld.singularity.submodeling`).  ``"nearest"`` uses a
        KD-tree nearest-neighbour lookup.  When ``"rbf"`` is requested
        and the solve fails (e.g. collinear / ill-conditioned nodes),
        the function falls back to ``"nearest"`` and records the
        fallback in :attr:`PathExtractionResult.metadata`.

    Returns
    -------
    PathExtractionResult
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    if fea_results.stress is None:
        raise ValueError("fea_results must contain a stress field")

    stress = fea_results.stress
    nodes = fea_results.mesh.nodes

    field_values = _field_values(stress, field)

    # Arc-length sample positions and physical coordinates
    total = float(path.arc_length())
    s_vals = np.linspace(0.0, total, n_samples)
    pts = np.asarray(path.evaluate_s(s_vals), dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)

    metadata: dict = {"field": field, "method_requested": method}
    used_method: InterpMethod = method

    if method == "rbf":
        try:
            values = _rbf_interpolate(nodes, field_values, pts)
        except Exception as exc:  # noqa: BLE001 - any scipy solver failure
            metadata["rbf_error"] = repr(exc)
            metadata["fallback"] = "nearest"
            used_method = "nearest"
            values = _nearest_interpolate(nodes, field_values, pts)
    elif method == "nearest":
        values = _nearest_interpolate(nodes, field_values, pts)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown method: {method!r}")

    metadata["method_used"] = used_method

    return PathExtractionResult(
        s=s_vals,
        values=values,
        points=pts,
        frames=None,
        metadata=metadata,
    )


def extract_tangent_normal_frame(
    path: WeldPath, s: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Return (T, N, B) at arc-length coordinate ``s`` as orthonormal unit vectors."""
    total = float(path.arc_length())
    if total <= 0.0:
        raise ValueError("path has zero arc length")
    s_clip = max(0.0, min(float(s), total))
    u = path._u_for_arc_length(s_clip, total)  # internal but deterministic
    t, n, b = path.frenet_frame(u)
    return (
        np.asarray(t, dtype=np.float64),
        np.asarray(n, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
    )


def sample_offset_points(
    path: WeldPath,
    n_samples: int,
    offset_distances: list[float],
    offset_direction: Literal["normal", "binormal"] = "normal",
) -> NDArray:
    """Sample a grid of points offset from a path along its Frenet frame.

    Parameters
    ----------
    path:
        Weld path to sample along.
    n_samples:
        Number of arc-length positions.
    offset_distances:
        Signed offsets (in model units) to apply at each arc-length
        position.  A positive offset moves along the chosen direction,
        negative moves against it.
    offset_direction:
        ``"normal"`` uses the Frenet principal normal ``N``;
        ``"binormal"`` uses the binormal ``B``.

    Returns
    -------
    NDArray
        Points of shape ``(n_samples, n_offsets, 3)``.  ``[:, 0, :]``
        does *not* automatically match ``path.sample`` — each row is
        the path point *plus* the offset at the corresponding offset
        distance.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    if len(offset_distances) == 0:
        raise ValueError("offset_distances must contain at least one value")
    if offset_direction not in ("normal", "binormal"):
        raise ValueError(
            f"offset_direction must be 'normal' or 'binormal', got {offset_direction!r}"
        )

    total = float(path.arc_length())
    s_vals = np.linspace(0.0, total, n_samples)

    n_offsets = len(offset_distances)
    out = np.empty((n_samples, n_offsets, 3), dtype=np.float64)

    for i, s in enumerate(s_vals):
        t, n_vec, b = extract_tangent_normal_frame(path, float(s))
        base = np.asarray(path.evaluate_s(float(s)), dtype=np.float64).reshape(3)
        direction = n_vec if offset_direction == "normal" else b
        for j, d in enumerate(offset_distances):
            out[i, j, :] = base + float(d) * direction

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _field_values(stress: StressField, field: FieldName) -> NDArray:
    if field == "von_mises":
        return stress.von_mises
    if field == "tresca":
        return stress.tresca
    if field == "hydrostatic":
        return stress.hydrostatic
    if field == "max_principal":
        return stress.principal[:, -1]
    raise ValueError(f"unknown field: {field!r}")


def _rbf_interpolate(
    source_coords: NDArray,
    source_values: NDArray,
    target_coords: NDArray,
) -> NDArray:
    """Thin-plate-spline RBF interpolation.

    Mirrors the kernel choice used by :mod:`feaweld.singularity.submodeling`.
    Any solver failure is propagated to the caller so it can fall back.
    """
    from scipy.interpolate import RBFInterpolator  # type: ignore[import-untyped]

    # RBFInterpolator supports scalar (1-D) and vector-valued targets; normalise
    # shape handling here so callers can always pass a 1-D array.
    was_1d = source_values.ndim == 1
    if was_1d:
        vals = source_values.reshape(-1, 1)
    else:
        vals = source_values

    rbf = RBFInterpolator(
        np.asarray(source_coords, dtype=np.float64),
        np.asarray(vals, dtype=np.float64),
        kernel="thin_plate_spline",
    )
    out = np.asarray(rbf(np.asarray(target_coords, dtype=np.float64)), dtype=np.float64)
    if was_1d:
        out = out.reshape(-1)
    # A degenerate point cloud (e.g. all collinear) can yield non-finite
    # results without raising; treat that as a solve failure.
    if not np.all(np.isfinite(out)):
        raise RuntimeError("RBF solve produced non-finite values")
    return out


def _nearest_interpolate(
    source_coords: NDArray,
    source_values: NDArray,
    target_coords: NDArray,
) -> NDArray:
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(source_coords, dtype=np.float64))
    _, idx = tree.query(np.asarray(target_coords, dtype=np.float64))
    return np.asarray(source_values, dtype=np.float64)[idx]
