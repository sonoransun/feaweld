"""Effective notch stress method per IIW recommendations.

Uses fictitious notch rounding (typically 1mm radius for steel) to compute
stress at weld toes and roots, then assesses fatigue life using FAT225 S-N curve.

References:
    IIW-2006-09: "Guideline for the Fatigue Assessment by Notch Stress Analysis
    for Welded Structures"
    Fricke, W. (2012). "IIW Recommendations for the Fatigue Assessment of Welded
    Structures by Notch Stress Analysis."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults, FEMesh, SNCurve, SNSegment, SNStandard


# FAT225 two-slope S-N curve for the effective notch stress method (IIW).
# Detail class 225 MPa at N = 2e6 cycles.  The slope steepens from m = 3 to
# m = 5 at the knee (N = 1e7), and the variable-amplitude endurance cutoff is
# at N = 1e9.  SNCurve.life() scans segments in order and uses the first whose
# stress_threshold is met, so the m = 3 segment (threshold = knee stress)
# governs above the knee and the m = 5 segment (threshold = 0) governs below.
FAT225_KNEE_STRESS = 225.0 * (2e6 / 1e7) ** (1.0 / 3.0)  # ~131.6 MPa at N = 1e7
FAT225_CURVE = SNCurve(
    name="FAT225",
    standard=SNStandard.IIW,
    segments=[
        SNSegment(m=3.0, C=225.0**3 * 2e6, stress_threshold=FAT225_KNEE_STRESS),
        SNSegment(m=5.0, C=FAT225_KNEE_STRESS**5 * 1e7, stress_threshold=0.0),
    ],
    cutoff_cycles=1e9,
)

# Default fictitious radius values per IIW
FICTITIOUS_RADIUS_STEEL = 1.0     # mm, for plate thickness t >= 5mm
FICTITIOUS_RADIUS_ALUMINUM = 1.0  # mm
FICTITIOUS_RADIUS_THIN = 0.05     # mm, for thin sheets (t < 5mm)


@dataclass
class NotchStressResult:
    """Result of effective notch stress analysis."""
    max_notch_stress: float                    # Maximum notch stress (MPa)
    notch_stress_range: float                  # Stress range for fatigue (MPa)
    stress_concentration_factor: float         # K_t = σ_notch / σ_nominal
    fatigue_life: float                        # N cycles from FAT225
    notch_stress_field: NDArray[np.float64] | None = None  # Full field if available
    fictitious_radius: float = 1.0             # radius used (mm)


def effective_notch_stress(
    results: FEAResults,
    notch_node_ids: NDArray[np.int64],
    nominal_stress: float,
    fictitious_radius: float = FICTITIOUS_RADIUS_STEEL,
) -> NotchStressResult:
    """Compute effective notch stress from a model with fictitious rounding.

    The FEA model must already have the fictitious radius applied at weld
    toes/roots (see feaweld.geometry.notch). This function extracts the
    maximum stress at the notch root nodes.

    Parameters
    ----------
    results : FEAResults
        FEA results from model with fictitious notch rounding
    notch_node_ids : NDArray[np.int64]
        Node IDs at the notch root (bottom of rounded area)
    nominal_stress : float
        Nominal stress for SCF calculation (MPa)
    fictitious_radius : float
        Radius used in the model (mm)

    Returns
    -------
    NotchStressResult
        NotchStressResult with max stress, SCF, and fatigue life.
    """
    if results.stress is None:
        raise ValueError("No stress data in results")

    # Extract von Mises stress at notch nodes
    notch_stresses = results.stress.von_mises[notch_node_ids]
    max_notch = float(np.max(notch_stresses))
    notch_range = max_notch  # for R=0 loading; adjust for other R-ratios

    # Stress concentration factor
    scf = max_notch / nominal_stress if nominal_stress > 0 else float("inf")

    # Fatigue life from FAT225 curve
    life = FAT225_CURVE.life(notch_range)

    return NotchStressResult(
        max_notch_stress=max_notch,
        notch_stress_range=notch_range,
        stress_concentration_factor=scf,
        fatigue_life=life,
        notch_stress_field=notch_stresses,
        fictitious_radius=fictitious_radius,
    )


def notch_stress_scf_parametric(
    toe_radius: float,
    toe_angle: float,
    plate_thickness: float,
    weld_toe_type: str = "fillet",
    geometry: str | None = None,
) -> float:
    """Estimate SCF using parametric formulas for common weld geometries.

    Based on parametric studies from literature:
    K_t ≈ 1 + a · (t/ρ)^b · (θ/π)^c

    Parameters
    ----------
    toe_radius : float
        Weld toe radius ρ (mm)
    toe_angle : float
        Weld toe flank angle θ (degrees)
    plate_thickness : float
        Plate thickness t (mm)
    weld_toe_type : str
        "fillet" or "butt" (used if geometry is None)
    geometry : str | None
        Optional SCF database geometry key (e.g., "cruciform",
        "t_joint"). When provided, coefficients are loaded from the
        SCF parametric database instead of the inline defaults.

    Returns
    -------
    float
        Estimated stress concentration factor K_t.
    """
    if geometry is not None:
        from feaweld.data.scf import compute_scf
        return compute_scf(geometry, toe_radius, toe_angle, plate_thickness)

    rho = max(toe_radius, 0.01)  # avoid division by zero
    theta_rad = np.radians(toe_angle)

    if weld_toe_type == "fillet":
        # Parametric SCF for fillet welds (Anthes et al., 1993)
        a, b, c = 0.469, 0.572, 0.469
    else:
        # Butt welds
        a, b, c = 0.350, 0.550, 0.400

    K_t = 1.0 + a * (plate_thickness / rho) ** b * (theta_rad / np.pi) ** c
    return float(K_t)


def assess_notch_fatigue(
    notch_result: NotchStressResult,
    sn_curve: SNCurve | None = None,
) -> dict:
    """Full fatigue assessment using notch stress method.

    Parameters
    ----------
    notch_result : NotchStressResult
        NotchStressResult from effective_notch_stress
    sn_curve : SNCurve | None
        S-N curve to use (default: FAT225 for notch method)

    Returns
    -------
    dict
        Assessment dict with life, utilization ratio, and pass/fail.
    """
    curve = sn_curve or FAT225_CURVE
    life = curve.life(notch_result.notch_stress_range)

    return {
        "notch_stress_range": notch_result.notch_stress_range,
        "scf": notch_result.stress_concentration_factor,
        "sn_curve": curve.name,
        "fatigue_life_cycles": life,
        "fictitious_radius_mm": notch_result.fictitious_radius,
    }
