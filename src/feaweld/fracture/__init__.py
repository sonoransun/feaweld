"""Fracture mechanics: phase-field (Track A4) + J-integral / CTOD (Track F3)."""

from __future__ import annotations

from feaweld.fracture.types import FractureResult
from feaweld.fracture.phase_field import (
    EnergyDecomposition,
    PhaseFieldConfig,
    solve_phase_field,
)
from feaweld.fracture.j_integral import (
    JResult,
    compute_k_from_j_elastic,
    interaction_integral,
    j_integral_2d,
)
from feaweld.fracture.ctod import (
    CTODResult,
    ctod_90_degree,
    ctod_displacement_extrapolation,
)

__all__ = [
    "EnergyDecomposition",
    "FractureResult",
    "PhaseFieldConfig",
    "solve_phase_field",
    "JResult",
    "j_integral_2d",
    "interaction_integral",
    "compute_k_from_j_elastic",
    "CTODResult",
    "ctod_displacement_extrapolation",
    "ctod_90_degree",
]
