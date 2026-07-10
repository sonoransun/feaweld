"""Hot-spot stress extrapolation methods per IIW recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from feaweld.core.types import FEAResults, FEMesh, StressField, WeldLineDefinition


class HotSpotType(str, Enum):
    TYPE_A = "type_a"  # Weld toe on plate surface — linear extrapolation
    TYPE_B = "type_b"  # Weld toe at plate edge — quadratic extrapolation


@dataclass
class HotSpotResult:
    """Result of hot-spot stress extrapolation."""
    hot_spot_stress: float            # σ_hs (MPa)
    reference_stresses: list[float]   # stresses at reference points
    reference_distances: list[float]  # distances of reference points from weld toe (mm)
    extrapolation_type: HotSpotType
    weld_toe_location: NDArray[np.float64]
    well_resolved: bool = True        # False when the mesh is too coarse to
                                      # place distinct reference points


def hotspot_stress_linear(
    results: FEAResults,
    weld_line: WeldLineDefinition,
    hot_spot_type: HotSpotType = HotSpotType.TYPE_A,
) -> list[HotSpotResult]:
    """Compute hot-spot stress using linear surface extrapolation.

    For Type A (surface): σ_hs = 1.67·σ(0.4t) - 0.67·σ(1.0t)
    For Type B (edge): σ_hs = 3·σ(4mm) - 3·σ(8mm) + σ(12mm) [quadratic]

    Parameters
    ----------
    results : FEAResults
        FEA results with stress field
    weld_line : WeldLineDefinition
        Definition of the weld toe line
    hot_spot_type : HotSpotType
        Type A or Type B

    Returns
    -------
    list[HotSpotResult]
        List of HotSpotResult, one per weld toe node.
    """
    if results.stress is None:
        raise ValueError("No stress data in results")

    mesh = results.mesh
    tree = cKDTree(mesh.nodes)
    weld_centroid = _weld_region_centroid(mesh)

    results_list = []
    t = weld_line.plate_thickness
    normal = weld_line.normal_direction

    for node_id in weld_line.node_ids:
        toe_pos = mesh.nodes[node_id]

        if hot_spot_type == HotSpotType.TYPE_A:
            # Reference points at 0.4t and 1.0t from weld toe along plate surface
            d1, d2 = 0.4 * t, 1.0 * t
            ref_distances = [d1, d2]
        else:
            # Type B: fixed distances 4, 8, 12 mm
            ref_distances = [4.0, 8.0, 12.0]

        # Find reference points perpendicular to weld toe along plate surface
        # Direction: away from weld, along plate surface (perpendicular to normal and weld line)
        weld_tangent = _estimate_weld_tangent(mesh, weld_line, node_id)
        surface_dir = np.cross(normal, weld_tangent)
        _sd_norm = np.linalg.norm(surface_dir)
        if _sd_norm < 1e-10:
            raise ValueError(
                f"Plate normal is parallel to weld tangent at node {node_id} — "
                "cannot determine surface direction for hot-spot extrapolation."
            )
        surface_dir = surface_dir / _sd_norm
        surface_dir = _orient_away_from_weld(surface_dir, toe_pos, weld_centroid)

        ref_stresses, ref_nodes = [], []
        well_resolved = True
        for d in ref_distances:
            ref_point = toe_pos + d * surface_dir
            dist, nearest = tree.query(ref_point)
            if dist > 0.5 * d:
                well_resolved = False
            ref_nodes.append(int(nearest))
            stress_vm = results.stress.von_mises[nearest]
            ref_stresses.append(float(stress_vm))
        if len(set(ref_nodes)) < len(ref_nodes):
            well_resolved = False

        # Extrapolation
        if hot_spot_type == HotSpotType.TYPE_A:
            # Linear: σ_hs = 1.67·σ(0.4t) - 0.67·σ(1.0t)
            sigma_hs = 1.67 * ref_stresses[0] - 0.67 * ref_stresses[1]
        else:
            # Quadratic: σ_hs = 3·σ(4mm) - 3·σ(8mm) + σ(12mm)
            sigma_hs = 3.0 * ref_stresses[0] - 3.0 * ref_stresses[1] + ref_stresses[2]

        results_list.append(HotSpotResult(
            hot_spot_stress=sigma_hs,
            reference_stresses=ref_stresses,
            reference_distances=ref_distances,
            extrapolation_type=hot_spot_type,
            weld_toe_location=toe_pos.copy(),
            well_resolved=well_resolved,
        ))

    return results_list


def hotspot_stress_quadratic(
    results: FEAResults,
    weld_line: WeldLineDefinition,
) -> list[HotSpotResult]:
    """Compute hot-spot stress using quadratic surface extrapolation (Type B).

    σ_hs = 3·σ(0.4t) - 3·σ(0.9t) + σ(1.4t)

    This variant uses thickness-relative distances (IIW alternative).
    """
    if results.stress is None:
        raise ValueError("No stress data in results")

    mesh = results.mesh
    tree = cKDTree(mesh.nodes)
    weld_centroid = _weld_region_centroid(mesh)
    t = weld_line.plate_thickness
    normal = weld_line.normal_direction

    results_list = []
    ref_distances = [0.4 * t, 0.9 * t, 1.4 * t]

    for node_id in weld_line.node_ids:
        toe_pos = mesh.nodes[node_id]

        weld_tangent = _estimate_weld_tangent(mesh, weld_line, node_id)
        surface_dir = np.cross(normal, weld_tangent)
        _sd_norm = np.linalg.norm(surface_dir)
        if _sd_norm < 1e-10:
            raise ValueError(
                f"Plate normal is parallel to weld tangent at node {node_id} — "
                "cannot determine surface direction for hot-spot extrapolation."
            )
        surface_dir = surface_dir / _sd_norm
        surface_dir = _orient_away_from_weld(surface_dir, toe_pos, weld_centroid)

        ref_stresses, ref_nodes = [], []
        well_resolved = True
        for d in ref_distances:
            ref_point = toe_pos + d * surface_dir
            dist, nearest = tree.query(ref_point)
            if dist > 0.5 * d:
                well_resolved = False
            ref_nodes.append(int(nearest))
            ref_stresses.append(float(results.stress.von_mises[nearest]))
        if len(set(ref_nodes)) < len(ref_nodes):
            well_resolved = False

        # Quadratic extrapolation
        sigma_hs = 3.0 * ref_stresses[0] - 3.0 * ref_stresses[1] + ref_stresses[2]

        results_list.append(HotSpotResult(
            hot_spot_stress=sigma_hs,
            reference_stresses=ref_stresses,
            reference_distances=ref_distances,
            extrapolation_type=HotSpotType.TYPE_B,
            weld_toe_location=toe_pos.copy(),
            well_resolved=well_resolved,
        ))

    return results_list


def max_hotspot_stress(results_list: list[HotSpotResult]) -> HotSpotResult:
    """Return the hot-spot result with maximum stress."""
    return max(results_list, key=lambda r: r.hot_spot_stress)


def order_weld_line_nodes(
    mesh: FEMesh,
    node_ids: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Order weld-toe nodes spatially along the dominant toe direction.

    Node sets recovered from mesh physical groups carry no spatial
    ordering, but tangent estimation treats neighbouring array entries as
    neighbouring points.  The node coordinates are centred and projected
    onto their first principal axis (via `numpy.linalg.svd`), and the ids
    are sorted by that projection.

    Parameters
    ----------
    mesh : FEMesh
        Mesh providing the node coordinates.
    node_ids : NDArray[np.int64]
        Weld-toe node indices in arbitrary order.

    Returns
    -------
    NDArray[np.int64]
        The same ids sorted along the weld line.  Which end comes first
        is arbitrary (the principal-axis sign is not defined).
    """
    ids = np.asarray(node_ids, dtype=np.int64)
    if ids.size <= 1:
        return ids.copy()

    coords = mesh.nodes[ids]
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[0]
    return ids[np.argsort(projection, kind="stable")]


def _weld_region_centroid(mesh: FEMesh) -> NDArray[np.float64] | None:
    """Mean node coordinate of "weld"-named element groups, or None."""
    group_elems = [
        elems
        for name, elems in mesh.physical_groups.items()
        if "weld" in name.lower()
    ]
    if not group_elems:
        return None

    elem_ids = np.unique(np.concatenate(group_elems))
    nodes = np.unique(mesh.elements[elem_ids].ravel())
    return mesh.nodes[nodes].mean(axis=0)


def _orient_away_from_weld(
    surface_dir: NDArray[np.float64],
    toe_pos: NDArray[np.float64],
    weld_centroid: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Flip *surface_dir* so extrapolation points away from the weld region.

    ``cross(normal, tangent)`` has arbitrary sign; sampling into the weld
    metal instead of the loaded plate silently corrupts the extrapolation.
    Without a "weld" element group the direction is kept as computed.
    """
    if weld_centroid is None:
        return surface_dir
    if np.dot(surface_dir, weld_centroid - toe_pos) > 0.0:
        return -surface_dir
    return surface_dir


def _estimate_weld_tangent(
    mesh: FEMesh,
    weld_line: WeldLineDefinition,
    node_id: int,
) -> NDArray[np.float64]:
    """Estimate the tangent direction of the weld line at a given node.

    ``weld_line.node_ids`` is taken as spatially ordered (see
    [order_weld_line_nodes][feaweld.postprocess.hotspot.order_weld_line_nodes]);
    the tangent is the chord between the node's ordered neighbours.
    """
    ids = np.asarray(weld_line.node_ids)
    matches = np.nonzero(ids == node_id)[0]
    idx = int(matches[0]) if matches.size > 0 else 0

    if idx == 0 and len(ids) > 1:
        p0 = mesh.nodes[ids[0]]
        p1 = mesh.nodes[ids[1]]
    elif idx == len(ids) - 1 and len(ids) > 1:
        p0 = mesh.nodes[ids[-2]]
        p1 = mesh.nodes[ids[-1]]
    elif len(ids) > 2:
        p0 = mesh.nodes[ids[idx - 1]]
        p1 = mesh.nodes[ids[idx + 1]]
    else:
        return np.array([1.0, 0.0, 0.0])

    tangent = p1 - p0
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return tangent / norm
