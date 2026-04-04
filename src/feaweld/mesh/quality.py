"""Mesh quality metrics for finite-element meshes.

Provides element-level quality measures (aspect ratio, scaled Jacobian)
and a summary report that flags poorly shaped elements.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import ElementType, FEMesh


# ---------------------------------------------------------------------------
# Aspect ratio
# ---------------------------------------------------------------------------

def aspect_ratio(mesh: FEMesh) -> NDArray[np.float64]:
    """Compute the aspect ratio for every element in *mesh*.

    For triangular elements the aspect ratio is defined as the longest
    edge divided by the shortest altitude (normalised so that an
    equilateral triangle has AR = 1).  For quadrilateral elements the
    ratio of the longest to shortest edge is used as an approximation.

    Higher values indicate worse quality.
    """
    nodes = mesh.nodes
    conn = mesh.elements
    n_elems = mesh.n_elements
    ar = np.empty(n_elems, dtype=np.float64)

    is_tri = mesh.element_type in (ElementType.TRI3, ElementType.TRI6)
    is_quad = mesh.element_type in (ElementType.QUAD4, ElementType.QUAD8)

    for i in range(n_elems):
        if is_tri:
            # Use first 3 nodes (corners) even for TRI6
            idx = conn[i, :3]
            p0, p1, p2 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]
            edges = np.array([
                np.linalg.norm(p1 - p0),
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p0 - p2),
            ])
            longest = edges.max()
            # Area via cross product
            area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            if area < 1e-30:
                ar[i] = 1e10
            else:
                shortest_altitude = 2.0 * area / longest
                # Normalise: equilateral triangle has AR = 2/sqrt(3) ~ 1.155
                ar[i] = longest / shortest_altitude / (2.0 / np.sqrt(3.0))
        elif is_quad:
            idx = conn[i, :4]
            pts = nodes[idx]
            edges = np.array([
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[3] - pts[2]),
                np.linalg.norm(pts[0] - pts[3]),
            ])
            e_max = edges.max()
            e_min = edges.min()
            ar[i] = e_max / e_min if e_min > 1e-30 else 1e10
        else:
            # 3-D elements: use edge-length ratio as a simple proxy
            n_corner = {
                ElementType.TET4: 4, ElementType.TET10: 4,
                ElementType.HEX8: 8, ElementType.HEX20: 8,
            }.get(mesh.element_type, conn.shape[1])
            idx = conn[i, :n_corner]
            pts = nodes[idx]
            edge_lengths: list[float] = []
            for a in range(n_corner):
                for b in range(a + 1, n_corner):
                    edge_lengths.append(float(np.linalg.norm(pts[a] - pts[b])))
            el = np.array(edge_lengths)
            ar[i] = el.max() / el.min() if el.min() > 1e-30 else 1e10

    return ar


# ---------------------------------------------------------------------------
# Scaled Jacobian
# ---------------------------------------------------------------------------

def jacobian_quality(mesh: FEMesh) -> NDArray[np.float64]:
    """Compute the scaled Jacobian quality metric for each element.

    For triangles: J_scaled = 2 * area / (max_edge^2 * sin(60 deg)),
    normalised so an equilateral triangle gives 1.0.

    For quads: the minimum ratio of the cross-product of edge vectors
    at each corner to the product of their lengths, normalised so a
    square gives 1.0.

    For 3-D elements a simplified ratio of actual volume to the volume
    of a regular reference element is computed.

    Values close to 1.0 are ideal; values <= 0 indicate inverted elements.
    """
    nodes = mesh.nodes
    conn = mesh.elements
    n_elems = mesh.n_elements
    jq = np.empty(n_elems, dtype=np.float64)

    is_tri = mesh.element_type in (ElementType.TRI3, ElementType.TRI6)
    is_quad = mesh.element_type in (ElementType.QUAD4, ElementType.QUAD8)
    is_tet = mesh.element_type in (ElementType.TET4, ElementType.TET10)

    for i in range(n_elems):
        if is_tri:
            idx = conn[i, :3]
            p0, p1, p2 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]
            area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            edges = np.array([
                np.linalg.norm(p1 - p0),
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p0 - p2),
            ])
            max_edge = edges.max()
            # Ideal equilateral: area = sqrt(3)/4 * L^2
            if max_edge < 1e-30:
                jq[i] = 0.0
            else:
                jq[i] = 2.0 * area / (max_edge ** 2 * np.sin(np.pi / 3.0))

        elif is_quad:
            idx = conn[i, :4]
            pts = nodes[idx]
            # Evaluate scaled Jacobian at each corner
            sj_vals: list[float] = []
            for c in range(4):
                v1 = pts[(c + 1) % 4] - pts[c]
                v2 = pts[(c + 3) % 4] - pts[c]
                cross = np.cross(v1, v2)
                cross_mag = np.linalg.norm(cross)
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                if denom < 1e-30:
                    sj_vals.append(0.0)
                else:
                    sj_vals.append(cross_mag / denom)
            jq[i] = min(sj_vals)

        elif is_tet:
            idx = conn[i, :4]
            p0, p1, p2, p3 = (nodes[idx[0]], nodes[idx[1]],
                               nodes[idx[2]], nodes[idx[3]])
            v1 = p1 - p0
            v2 = p2 - p0
            v3 = p3 - p0
            vol = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
            edges = [
                np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p0),
                np.linalg.norm(p3 - p0), np.linalg.norm(p2 - p1),
                np.linalg.norm(p3 - p1), np.linalg.norm(p3 - p2),
            ]
            max_edge = max(edges)
            # Regular tet: vol = L^3 / (6*sqrt(2))
            if max_edge < 1e-30:
                jq[i] = 0.0
            else:
                vol_ideal = max_edge ** 3 / (6.0 * np.sqrt(2.0))
                jq[i] = vol / vol_ideal

        else:
            # Hex or unknown: fall back to a simple volume-based metric
            n_corner = min(8, conn.shape[1])
            idx = conn[i, :n_corner]
            pts = nodes[idx]
            centroid = pts.mean(axis=0)
            dists = np.linalg.norm(pts - centroid, axis=1)
            jq[i] = dists.min() / dists.max() if dists.max() > 1e-30 else 0.0

    return jq


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def mesh_quality_report(mesh: FEMesh) -> dict:
    """Produce a summary of mesh quality for *mesh*.

    Returns a dictionary with keys:

    - ``aspect_ratio``: dict with ``min``, ``max``, ``mean``
    - ``jacobian``: dict with ``min``, ``max``, ``mean``
    - ``n_poor_elements``: count of elements with AR > 5 or Jacobian < 0.3
    - ``poor_element_indices``: 1-D integer array of flagged element indices
    """
    ar = aspect_ratio(mesh)
    jq = jacobian_quality(mesh)

    poor_mask = (ar > 5.0) | (jq < 0.3)

    return {
        "aspect_ratio": {
            "min": float(ar.min()),
            "max": float(ar.max()),
            "mean": float(ar.mean()),
        },
        "jacobian": {
            "min": float(jq.min()),
            "max": float(jq.max()),
            "mean": float(jq.mean()),
        },
        "n_poor_elements": int(poor_mask.sum()),
        "poor_element_indices": np.where(poor_mask)[0],
    }
