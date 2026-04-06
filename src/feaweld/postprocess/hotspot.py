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


def hotspot_stress_linear(
    results: FEAResults,
    weld_line: WeldLineDefinition,
    hot_spot_type: HotSpotType = HotSpotType.TYPE_A,
) -> list[HotSpotResult]:
    """Compute hot-spot stress using linear surface extrapolation.

    For Type A (surface): σ_hs = 1.67·σ(0.4t) - 0.67·σ(1.0t)
    For Type B (edge): σ_hs = 3·σ(4mm) - 3·σ(8mm) + σ(12mm) [quadratic]

    Args:
        results: FEA results with stress field
        weld_line: Definition of the weld toe line
        hot_spot_type: Type A or Type B

    Returns:
        List of HotSpotResult, one per weld toe node.
    """
    if results.stress is None:
        raise ValueError("No stress data in results")

    mesh = results.mesh
    tree = cKDTree(mesh.nodes)

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

        ref_stresses = []
        for d in ref_distances:
            ref_point = toe_pos + d * surface_dir
            _, nearest = tree.query(ref_point)
            stress_vm = results.stress.von_mises[nearest]
            ref_stresses.append(float(stress_vm))

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

        ref_stresses = []
        for d in ref_distances:
            ref_point = toe_pos + d * surface_dir
            _, nearest = tree.query(ref_point)
            ref_stresses.append(float(results.stress.von_mises[nearest]))

        # Quadratic extrapolation
        sigma_hs = 3.0 * ref_stresses[0] - 3.0 * ref_stresses[1] + ref_stresses[2]

        results_list.append(HotSpotResult(
            hot_spot_stress=sigma_hs,
            reference_stresses=ref_stresses,
            reference_distances=ref_distances,
            extrapolation_type=HotSpotType.TYPE_B,
            weld_toe_location=toe_pos.copy(),
        ))

    return results_list


def max_hotspot_stress(results_list: list[HotSpotResult]) -> HotSpotResult:
    """Return the hot-spot result with maximum stress."""
    return max(results_list, key=lambda r: r.hot_spot_stress)


def _estimate_weld_tangent(
    mesh: FEMesh,
    weld_line: WeldLineDefinition,
    node_id: int,
) -> NDArray[np.float64]:
    """Estimate the tangent direction of the weld line at a given node."""
    ids = weld_line.node_ids
    idx = np.searchsorted(ids, node_id)
    idx = min(max(idx, 0), len(ids) - 1)

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
