"""Singularity detection, convergence studies, and submodeling."""

from feaweld.singularity.convergence import (
    ConvergenceResult,
    convergence_study,
    grid_convergence_index,
    richardson_extrapolation,
)
from feaweld.singularity.detection import (
    SingularityInfo,
    detect_singularities,
    estimate_convergence_rate,
)

__all__ = [
    "ConvergenceResult",
    "SingularityInfo",
    "convergence_study",
    "detect_singularities",
    "estimate_convergence_rate",
    "grid_convergence_index",
    "richardson_extrapolation",
]
