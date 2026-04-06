"""ASME VIII Division 2 nominal stress categorization and assessment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults, StressField, WeldLineDefinition


class StressCategory(str, Enum):
    """ASME VIII Div 2 stress categories."""
    PM = "primary_membrane"          # General primary membrane
    PL = "primary_local_membrane"    # Local primary membrane
    PB = "primary_bending"           # Primary bending
    Q = "secondary"                  # Secondary (membrane + bending)
    F = "peak"                       # Peak stress


@dataclass
class StressCategorization:
    """Result of ASME VIII stress categorization at a section."""
    membrane: float          # σ_m (MPa) — linearized membrane
    bending: float           # σ_b (MPa) — linearized bending
    peak: float              # σ_F (MPa) — peak = total - membrane - bending
    total: float             # σ_total (MPa) — actual max stress
    stress_intensity: float  # Tresca stress intensity (MPa)

    @property
    def primary_membrane(self) -> float:
        return self.membrane

    @property
    def primary_plus_bending(self) -> float:
        return self.membrane + self.bending

    @property
    def primary_plus_secondary(self) -> float:
        return self.membrane + self.bending + self.peak


def categorize_stress_section(
    stress_through_thickness: NDArray[np.float64],
    thickness: float,
    z_coords: NDArray[np.float64],
) -> StressCategorization:
    """Categorize stress through a section per ASME VIII Div 2.

    Args:
        stress_through_thickness: (n_points,) stress values through thickness
        thickness: plate/shell thickness (mm)
        z_coords: (n_points,) z-coordinates through thickness (0 to t)

    Returns:
        StressCategorization with membrane, bending, peak components.
    """
    t = thickness

    # Membrane: average through thickness
    # σ_m = (1/t) ∫₀ᵗ σ dz
    sigma_m = float(np.trapezoid(stress_through_thickness, z_coords) / t)

    # Bending: linear component
    # σ_b = (6/t²) ∫₀ᵗ σ·(z - t/2) dz
    z_centered = z_coords - t / 2.0
    sigma_b_integral = np.trapezoid(stress_through_thickness * z_centered, z_coords)
    sigma_b = float(6.0 * sigma_b_integral / (t ** 2))

    # Peak: remainder
    sigma_total = float(np.max(np.abs(stress_through_thickness)))
    sigma_peak = sigma_total - abs(sigma_m) - abs(sigma_b)

    # Stress intensity (Tresca = max principal difference)
    stress_intensity = sigma_total  # simplified for 1D; full 3D uses principal stresses

    return StressCategorization(
        membrane=sigma_m,
        bending=sigma_b,
        peak=max(0.0, sigma_peak),
        total=sigma_total,
        stress_intensity=stress_intensity,
    )


def asme_allowable_check(
    categorization: StressCategorization,
    S_m: float,
    S_y: float,
) -> dict[str, dict]:
    """Check stress limits per ASME VIII Division 2 Part 5.

    Args:
        categorization: StressCategorization result
        S_m: allowable stress intensity (MPa) = min(σ_u/2.4, σ_y/1.5)
        S_y: yield strength (MPa)

    Returns:
        Dict with category checks: {category: {value, limit, ratio, passes}}
    """
    checks = {}

    # Primary membrane: P_m ≤ S_m
    checks["Pm"] = {
        "value": abs(categorization.membrane),
        "limit": S_m,
        "ratio": abs(categorization.membrane) / S_m if S_m > 0 else float("inf"),
        "passes": abs(categorization.membrane) <= S_m,
    }

    # Primary local membrane: P_L ≤ 1.5 * S_m
    checks["PL"] = {
        "value": abs(categorization.membrane),
        "limit": 1.5 * S_m,
        "ratio": abs(categorization.membrane) / (1.5 * S_m) if S_m > 0 else float("inf"),
        "passes": abs(categorization.membrane) <= 1.5 * S_m,
    }

    # Primary membrane + bending: P_m + P_b ≤ 1.5 * S_m
    pm_pb = abs(categorization.membrane) + abs(categorization.bending)
    checks["Pm+Pb"] = {
        "value": pm_pb,
        "limit": 1.5 * S_m,
        "ratio": pm_pb / (1.5 * S_m) if S_m > 0 else float("inf"),
        "passes": pm_pb <= 1.5 * S_m,
    }

    # Primary + secondary: P_L + P_b + Q ≤ S_PS = max(3*S_m, 2*S_y)
    S_PS = max(3.0 * S_m, 2.0 * S_y)
    total_pq = abs(categorization.membrane) + abs(categorization.bending) + categorization.peak
    checks["PL+Pb+Q"] = {
        "value": total_pq,
        "limit": S_PS,
        "ratio": total_pq / S_PS if S_PS > 0 else float("inf"),
        "passes": total_pq <= S_PS,
    }

    return checks


def extract_stress_along_path(
    results: FEAResults,
    start_node: int,
    end_node: int,
    n_points: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract stress values along a path through the mesh.

    Args:
        results: FEA results with stress field
        start_node: node ID at path start (e.g., inner surface)
        end_node: node ID at path end (e.g., outer surface)
        n_points: number of interpolation points

    Returns:
        (distances, stress_values) along the path
    """
    if results.stress is None:
        raise ValueError("No stress data in results")

    mesh = results.mesh
    p0 = mesh.nodes[start_node]
    p1 = mesh.nodes[end_node]

    path_length = float(np.linalg.norm(p1 - p0))
    direction = (p1 - p0) / path_length

    # Sample points along path
    distances = np.linspace(0, path_length, n_points)
    points = np.array([p0 + d * direction for d in distances])

    # Find nearest mesh nodes to each sample point
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.nodes)
    _, nearest_ids = tree.query(points)

    # Extract von Mises stress at nearest nodes
    stress_values = results.stress.von_mises[nearest_ids]

    return distances, stress_values
