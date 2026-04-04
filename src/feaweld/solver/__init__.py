"""Solver subpackage: FEA backends, constitutive models, and thermal/mechanical utilities."""

from feaweld.solver.backend import SolverBackend, get_backend
from feaweld.solver.thermal import GoldakHeatSource, ElementBirthDeath
from feaweld.solver.mechanical import (
    linear_elastic_stress,
    j2_return_mapping,
    von_mises,
    deviatoric_stress,
)
from feaweld.solver.constitutive import (
    ConstitutiveModel,
    LinearElastic,
    J2Plastic,
    TemperatureDependent,
    MaterialState,
)
from feaweld.solver.creep import norton_bailey_rate, simulate_pwht
from feaweld.solver.thermomechanical import sequential_coupled_solve, compute_thermal_stress

__all__ = [
    "SolverBackend",
    "get_backend",
    "GoldakHeatSource",
    "ElementBirthDeath",
    "linear_elastic_stress",
    "j2_return_mapping",
    "von_mises",
    "deviatoric_stress",
    "ConstitutiveModel",
    "LinearElastic",
    "J2Plastic",
    "TemperatureDependent",
    "MaterialState",
    "norton_bailey_rate",
    "simulate_pwht",
    "sequential_coupled_solve",
    "compute_thermal_stress",
]
