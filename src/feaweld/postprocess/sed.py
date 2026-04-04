"""Strain Energy Density (SED) method for fatigue assessment.

Implements the averaged SED approach by Lazzarin and colleagues. The method
is inherently mesh-insensitive because it uses energy averaged over a control
volume rather than peak stress values.

References:
    Lazzarin, P. and Zambardi, R. (2001). "A finite-volume-energy based approach
    to predict the static and fatigue behavior of components with sharp V-shaped
    notches." Int. J. Fracture, 112, 275-298.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults, FEMesh, StressField


# Critical SED values for common steels (from literature)
# W_cr = σ_u² / (2E) approximately
DEFAULT_CONTROL_RADIUS = {
    "steel_low_carbon": 0.28,     # mm (R₀ for structural steel)
    "steel_high_strength": 0.15,  # mm
    "aluminum": 0.12,             # mm
}


@dataclass
class SEDResult:
    """Result of Strain Energy Density analysis."""
    averaged_sed: float              # W̄ (MJ/m³)
    control_radius: float            # R₀ (mm)
    control_volume: float            # V_c (mm³)
    critical_sed: float | None       # W_cr for fatigue assessment
    fatigue_life: float | None       # N cycles if critical SED is known
    sed_field: NDArray[np.float64] | None = None  # SED at each element


def compute_sed_field(
    results: FEAResults,
    elastic_modulus: float,
) -> NDArray[np.float64]:
    """Compute strain energy density at each node.

    W = σ²/(2E) for uniaxial, or W = (1/2E)[σ₁² + σ₂² + σ₃² - 2ν(σ₁σ₂ + σ₂σ₃ + σ₁σ₃)]

    Simplified to W = σ_vm² / (2E) for von Mises-based estimate.

    Args:
        results: FEA results with stress field
        elastic_modulus: E (MPa)

    Returns:
        SED values at each node (MJ/m³ = MPa)
    """
    if results.stress is None:
        raise ValueError("No stress data in results")

    vm = results.stress.von_mises
    return vm ** 2 / (2.0 * elastic_modulus)


def averaged_sed(
    results: FEAResults,
    center_point: NDArray[np.float64],
    control_radius: float,
    elastic_modulus: float,
    poisson_ratio: float = 0.3,
) -> SEDResult:
    """Compute averaged SED over a cylindrical/spherical control volume.

    W̄ = (1/V_c) ∫_Vc W dV

    The control volume is centered at the notch tip (weld toe) with radius R₀.

    Args:
        results: FEA results with stress field
        center_point: Center of control volume (weld toe/root location)
        control_radius: R₀ (mm)
        elastic_modulus: E (MPa)
        poisson_ratio: ν

    Returns:
        SEDResult with averaged SED and volume.
    """
    if results.stress is None:
        raise ValueError("No stress data in results")

    mesh = results.mesh
    nodes = mesh.nodes

    # Find nodes within control volume (sphere of radius R₀)
    distances = np.linalg.norm(nodes - center_point, axis=1)
    mask = distances <= control_radius

    if not np.any(mask):
        # No nodes in control volume — use nearest node
        nearest = np.argmin(distances)
        mask[nearest] = True

    # Compute SED at nodes within control volume
    sed_values = compute_sed_field(results, elastic_modulus)
    sed_in_volume = sed_values[mask]

    # Average SED (volume-weighted average using Voronoi-like approximation)
    # For simplicity, use arithmetic mean (accurate for uniform mesh)
    W_bar = float(np.mean(sed_in_volume))

    # Estimate control volume (sphere)
    n_nodes_in = np.sum(mask)
    ndim = mesh.ndim
    if ndim == 3:
        V_c = (4.0 / 3.0) * np.pi * control_radius ** 3
    else:
        V_c = np.pi * control_radius ** 2  # 2D: circular area

    return SEDResult(
        averaged_sed=W_bar,
        control_radius=control_radius,
        control_volume=V_c,
        critical_sed=None,
        fatigue_life=None,
        sed_field=sed_values,
    )


def sed_fatigue_life(
    sed_result: SEDResult,
    W_ref: float,
    N_ref: float = 2e6,
    slope: float = 1.5,
) -> SEDResult:
    """Estimate fatigue life from averaged SED using power law.

    N = N_ref · (W_ref / W̄)^(1/slope_factor)

    where W_ref is the SED at reference life N_ref.

    Args:
        sed_result: SEDResult from averaged_sed
        W_ref: Reference SED at N_ref cycles (MJ/m³)
        N_ref: Reference fatigue life (default 2e6 cycles)
        slope: SED-life curve slope (typically ~1.5 for steel)

    Returns:
        Updated SEDResult with fatigue_life.
    """
    if sed_result.averaged_sed > 0:
        life = N_ref * (W_ref / sed_result.averaged_sed) ** (1.0 / slope)
    else:
        life = float("inf")

    return SEDResult(
        averaged_sed=sed_result.averaged_sed,
        control_radius=sed_result.control_radius,
        control_volume=sed_result.control_volume,
        critical_sed=W_ref,
        fatigue_life=life,
        sed_field=sed_result.sed_field,
    )


def estimate_control_radius(
    fracture_toughness: float,
    ultimate_strength: float,
    elastic_modulus: float,
) -> float:
    """Estimate control radius R₀ from material properties.

    R₀ ≈ (K_Ic / σ_u)² / (4π)  (plane strain, mode I)

    where K_Ic is fracture toughness.

    Args:
        fracture_toughness: K_Ic (MPa·√mm)
        ultimate_strength: σ_u (MPa)
        elastic_modulus: E (MPa)

    Returns:
        Estimated R₀ (mm)
    """
    return (fracture_toughness / ultimate_strength) ** 2 / (4.0 * np.pi)
