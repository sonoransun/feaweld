"""Macro-scale analysis for multi-scale weld modeling.

Handles global structural model, stress field extraction at submodel
boundaries, and boundary condition interpolation for meso-scale handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from feaweld.core.types import FEAResults, FEMesh, StressField, LoadCase


@dataclass
class SubregionSpec:
    """Specification for a subregion to extract from the macro model."""
    center: NDArray[np.float64]   # (3,) center of subregion
    radius: float                  # extraction radius (mm)
    name: str = "subregion"


@dataclass
class MacroToMesoTransfer:
    """Data transfer package from macro to meso scale."""
    boundary_nodes: NDArray[np.int64]           # node IDs on subregion boundary
    boundary_positions: NDArray[np.float64]     # (n_boundary, 3) positions
    boundary_displacements: NDArray[np.float64] # (n_boundary, 3) displacement BCs
    boundary_temperatures: NDArray[np.float64] | None = None  # (n_boundary,) if thermal
    interior_nodes: NDArray[np.int64] = field(default_factory=lambda: np.array([], dtype=np.int64))
    stress_field_interior: StressField | None = None


def extract_subregion(
    results: FEAResults,
    spec: SubregionSpec,
) -> MacroToMesoTransfer:
    """Extract subregion data from macro model results.

    Identifies nodes within the subregion, separates boundary from interior,
    and packages displacement BCs for the meso model.

    Args:
        results: Macro-scale FEA results
        spec: Subregion specification (center and radius)

    Returns:
        MacroToMesoTransfer with boundary conditions and interior data.
    """
    mesh = results.mesh
    nodes = mesh.nodes

    # Find nodes in subregion
    distances = np.linalg.norm(nodes - spec.center, axis=1)
    in_region = distances <= spec.radius
    on_boundary = (distances > spec.radius * 0.9) & (distances <= spec.radius)

    region_ids = np.where(in_region)[0]
    boundary_ids = np.where(on_boundary)[0]
    interior_ids = np.where(in_region & ~on_boundary)[0]

    # Extract boundary displacements
    if results.displacement is not None:
        boundary_disp = results.displacement[boundary_ids]
    else:
        boundary_disp = np.zeros((len(boundary_ids), 3))

    # Extract boundary temperatures if available
    boundary_temp = None
    if results.temperature is not None:
        if results.temperature.ndim == 1:
            boundary_temp = results.temperature[boundary_ids]
        else:
            boundary_temp = results.temperature[-1, boundary_ids]  # last timestep

    # Extract interior stress field
    interior_stress = None
    if results.stress is not None:
        interior_values = results.stress.values[interior_ids]
        interior_stress = StressField(values=interior_values)

    return MacroToMesoTransfer(
        boundary_nodes=boundary_ids,
        boundary_positions=nodes[boundary_ids],
        boundary_displacements=boundary_disp,
        boundary_temperatures=boundary_temp,
        interior_nodes=interior_ids,
        stress_field_interior=interior_stress,
    )


def interpolate_boundary_conditions(
    transfer: MacroToMesoTransfer,
    target_positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate macro displacement BCs onto meso-scale boundary nodes.

    Uses radial basis function (RBF) interpolation for smooth transfer.

    Args:
        transfer: MacroToMesoTransfer data
        target_positions: (n_target, 3) positions of meso boundary nodes

    Returns:
        (n_target, 3) interpolated displacements
    """
    if len(transfer.boundary_positions) < 4:
        # Too few points for RBF, use nearest-neighbor
        tree = cKDTree(transfer.boundary_positions)
        _, indices = tree.query(target_positions)
        return transfer.boundary_displacements[indices]

    result = np.empty((len(target_positions), 3))
    for comp in range(3):
        rbf = RBFInterpolator(
            transfer.boundary_positions,
            transfer.boundary_displacements[:, comp],
            kernel="thin_plate_spline",
            smoothing=0.0,
        )
        result[:, comp] = rbf(target_positions)

    return result


def check_equilibrium(
    transfer: MacroToMesoTransfer,
    meso_boundary_forces: NDArray[np.float64],
    tolerance: float = 0.01,
) -> dict:
    """Verify force equilibrium between macro and meso models.

    Args:
        transfer: Transfer data from macro model
        meso_boundary_forces: (n_boundary, 3) forces from meso model at boundary
        tolerance: Relative force balance tolerance

    Returns:
        Dict with force sums, imbalance, and pass/fail.
    """
    total_force = np.sum(meso_boundary_forces, axis=0)
    force_magnitude = np.linalg.norm(total_force)
    max_nodal_force = np.max(np.linalg.norm(meso_boundary_forces, axis=1))

    relative_imbalance = force_magnitude / (max_nodal_force + 1e-12)

    return {
        "total_force": total_force,
        "force_magnitude": force_magnitude,
        "max_nodal_force": max_nodal_force,
        "relative_imbalance": relative_imbalance,
        "passes": relative_imbalance < tolerance,
    }
