"""Through-thickness stress linearization per ASME methodology.

Decomposes actual stress distribution through a section into membrane,
bending, and peak (nonlinear) components. Used for fatigue assessment
and singularity resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from feaweld.core.types import FEAResults, FEMesh, StressField


@dataclass
class LinearizationResult:
    """Result of through-thickness stress linearization."""
    membrane: NDArray[np.float64]   # (6,) membrane stress tensor components
    bending: NDArray[np.float64]    # (6,) bending stress tensor components
    peak: NDArray[np.float64]       # (6,) peak (nonlinear) stress tensor components
    z_coords: NDArray[np.float64]   # through-thickness coordinates
    total_stress: NDArray[np.float64]  # (n_points, 6) actual stress distribution
    membrane_scalar: float          # von Mises of membrane
    bending_scalar: float           # von Mises of bending
    peak_scalar: float              # von Mises of peak
    membrane_plus_bending_scalar: float  # von Mises of membrane+bending

    @property
    def linearized_stress(self) -> NDArray[np.float64]:
        """Linearized (membrane + bending) stress distribution."""
        n = len(self.z_coords)
        t = self.z_coords[-1] - self.z_coords[0]
        z_mid = (self.z_coords[0] + self.z_coords[-1]) / 2.0
        result = np.empty((n, 6))
        for i in range(n):
            z_rel = (self.z_coords[i] - z_mid) / (t / 2.0)  # normalized: -1 to +1
            result[i] = self.membrane + self.bending * z_rel
        return result


def linearize_through_thickness(
    results: FEAResults,
    start_point: NDArray[np.float64],
    end_point: NDArray[np.float64],
    n_points: int = 20,
) -> LinearizationResult:
    """Linearize stress through a section defined by two points.

    Separates the stress into:
    - Membrane: σ_m = (1/t) ∫₀ᵗ σ dz
    - Bending: σ_b = (6/t²) ∫₀ᵗ σ·(z - t/2) dz
    - Peak: σ_F = σ_total - σ_linearized

    Args:
        results: FEA results with stress field
        start_point: Start of section line (inner surface)
        end_point: End of section line (outer surface)
        n_points: Number of sampling points through thickness

    Returns:
        LinearizationResult with all stress components.
    """
    if results.stress is None:
        raise ValueError("No stress data in results")

    mesh = results.mesh
    tree = cKDTree(mesh.nodes)

    # Section line
    direction = end_point - start_point
    thickness = float(np.linalg.norm(direction))
    direction_unit = direction / thickness

    # Sample points along section
    z_coords = np.linspace(0, thickness, n_points)
    sample_points = np.array([start_point + z * direction_unit for z in z_coords])

    # Find nearest nodes and extract stress tensors
    _, nearest_ids = tree.query(sample_points)
    total_stress = results.stress.values[nearest_ids]  # (n_points, 6)

    # Membrane: average through thickness
    # σ_m,ij = (1/t) ∫₀ᵗ σ_ij(z) dz
    membrane = np.zeros(6)
    for comp in range(6):
        membrane[comp] = np.trapezoid(total_stress[:, comp], z_coords) / thickness

    # Bending: linear moment component
    # σ_b,ij = (6/t²) ∫₀ᵗ σ_ij(z)·(z - t/2) dz
    z_centered = z_coords - thickness / 2.0
    bending = np.zeros(6)
    for comp in range(6):
        bending[comp] = 6.0 * np.trapezoid(
            total_stress[:, comp] * z_centered, z_coords
        ) / (thickness ** 2)

    # Peak: remainder at the point of maximum total stress
    # Construct linearized distribution and subtract from total
    z_mid = thickness / 2.0
    linearized = np.empty_like(total_stress)
    for i in range(n_points):
        z_rel = (z_coords[i] - z_mid) / (thickness / 2.0)
        linearized[i] = membrane + bending * z_rel

    peak_field = total_stress - linearized
    # Peak at the point of maximum von Mises
    vm_total = _von_mises_6(total_stress)
    max_idx = np.argmax(vm_total)
    peak = peak_field[max_idx]

    # Scalar values (von Mises)
    membrane_scalar = float(_von_mises_single(membrane))
    bending_scalar = float(_von_mises_single(bending))
    peak_scalar = float(_von_mises_single(peak))
    mb = membrane + bending  # at outer surface (z_rel = 1)
    mb_scalar = float(_von_mises_single(mb))

    return LinearizationResult(
        membrane=membrane,
        bending=bending,
        peak=peak,
        z_coords=z_coords,
        total_stress=total_stress,
        membrane_scalar=membrane_scalar,
        bending_scalar=bending_scalar,
        peak_scalar=peak_scalar,
        membrane_plus_bending_scalar=mb_scalar,
    )


def linearize_at_weld_toe(
    results: FEAResults,
    weld_toe_node: int,
    plate_thickness: float,
    surface_normal: NDArray[np.float64],
    n_points: int = 20,
) -> LinearizationResult:
    """Convenience function to linearize through thickness at a weld toe.

    Args:
        results: FEA results
        weld_toe_node: Node ID at the weld toe (outer surface)
        plate_thickness: Plate thickness t (mm)
        surface_normal: Outward normal to the plate surface
        n_points: Number of sampling points

    Returns:
        LinearizationResult
    """
    mesh = results.mesh
    outer_point = mesh.nodes[weld_toe_node]
    inner_point = outer_point - plate_thickness * surface_normal

    return linearize_through_thickness(results, inner_point, outer_point, n_points)


def _von_mises_single(s: NDArray[np.float64]) -> float:
    """Von Mises stress from a single (6,) Voigt stress vector."""
    return float(np.sqrt(
        0.5 * (
            (s[0] - s[1]) ** 2
            + (s[1] - s[2]) ** 2
            + (s[2] - s[0]) ** 2
            + 6.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
        )
    ))


def _von_mises_6(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Von Mises stress from (n, 6) Voigt stress array."""
    return np.sqrt(
        0.5 * (
            (s[:, 0] - s[:, 1]) ** 2
            + (s[:, 1] - s[:, 2]) ** 2
            + (s[:, 2] - s[:, 0]) ** 2
            + 6.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
        )
    )
