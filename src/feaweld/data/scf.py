"""Stress concentration factor (SCF) parametric lookup and computation.

Provides SCF coefficients for common weld geometries and computes the
stress concentration factor using the Anthes-type parametric formula:

    K_t = 1 + a * (t / rho)^b * (theta / pi)^c

where *t* is plate thickness, *rho* is toe radius, and *theta* is toe angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feaweld.data.cache import get_cache


@dataclass
class SCFCoefficients:
    """Parametric SCF coefficients for one weld geometry."""

    geometry: str
    a: float
    b: float
    c: float
    description: str = ""
    reference: str = ""


def get_scf_coefficients(geometry: str) -> SCFCoefficients:
    """Look up SCF parametric coefficients by geometry name.

    Args:
        geometry: Geometry identifier (e.g. ``"fillet_toe"``, ``"butt_toe"``).

    Returns:
        SCFCoefficients dataclass.

    Raises:
        KeyError: If the geometry is not found in the dataset.
    """
    data = get_cache().get("scf/parametric_coefficients")
    for entry in data:
        if entry["geometry"] == geometry:
            return SCFCoefficients(
                geometry=entry["geometry"],
                a=entry["a"],
                b=entry["b"],
                c=entry["c"],
                description=entry.get("description", ""),
                reference=entry.get("reference", ""),
            )
    available = [e["geometry"] for e in data]
    raise KeyError(f"SCF geometry not found: {geometry!r}. Available: {available}")


def compute_scf(
    geometry: str,
    toe_radius: float,
    toe_angle: float,
    plate_thickness: float,
) -> float:
    """Compute the stress concentration factor for a weld toe.

    Uses the parametric formula:
        K_t = 1 + a * (t / rho)^b * (theta / pi)^c

    Args:
        geometry: Geometry identifier (e.g. ``"fillet_toe"``).
        toe_radius: Weld toe radius in mm.  Clamped to >= 0.01 mm.
        toe_angle: Weld toe angle in degrees.
        plate_thickness: Plate thickness in mm.

    Returns:
        Stress concentration factor K_t (dimensionless, >= 1.0).
    """
    coeff = get_scf_coefficients(geometry)
    rho = max(toe_radius, 0.01)
    theta_rad = np.radians(toe_angle)
    return float(
        1.0
        + coeff.a
        * (plate_thickness / rho) ** coeff.b
        * (theta_rad / np.pi) ** coeff.c
    )


def list_scf_geometries() -> list[str]:
    """Return a list of all available SCF geometry identifiers."""
    data = get_cache().get("scf/parametric_coefficients")
    return [entry["geometry"] for entry in data]
