"""Unified solver backend interface for FEA analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from feaweld.core.types import FEAResults, FEMesh, LoadCase, SolverType
from feaweld.core.materials import Material


class SolverBackend(ABC):
    """Unified interface for FEA solver backends.

    All concrete backends must implement the four core solve methods.
    Each method accepts solver-agnostic data structures and returns
    :class:`FEAResults`.
    """

    @abstractmethod
    def solve_static(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float = 20.0,
    ) -> FEAResults:
        """Linear or nonlinear static mechanical analysis.

        Parameters
        ----------
        mesh : FEMesh
            Finite element mesh.
        material : Material
            Temperature-dependent material.
        load_case : LoadCase
            Loads and boundary conditions.
        temperature : float
            Uniform temperature for property evaluation (C).

        Returns
        -------
        FEAResults
            Displacements, stresses, strains, and reaction forces.
        """

    @abstractmethod
    def solve_thermal_steady(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
    ) -> FEAResults:
        """Steady-state thermal analysis.

        Parameters
        ----------
        mesh : FEMesh
            Finite element mesh.
        material : Material
            Temperature-dependent material (uses conductivity).
        load_case : LoadCase
            Thermal loads and boundary conditions.

        Returns
        -------
        FEAResults
            Nodal temperature field.
        """

    @abstractmethod
    def solve_thermal_transient(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        time_steps: NDArray,
        heat_source: object | None = None,
        initial_temperature: NDArray | None = None,
    ) -> FEAResults:
        """Transient thermal analysis.

        Parameters
        ----------
        mesh : FEMesh
            Finite element mesh.
        material : Material
            Temperature-dependent material.
        load_case : LoadCase
            Thermal loads and boundary conditions.
        time_steps : NDArray
            Array of time values (s) at which to solve.
        heat_source : object or None
            Optional moving heat source (e.g. GoldakHeatSource).
        initial_temperature : NDArray or None
            Optional ``(n_nodes,)`` initial temperature field.  If *None*,
            a uniform 20 C ambient is assumed.

        Returns
        -------
        FEAResults
            Temperature history at all nodes across time steps.
        """

    @abstractmethod
    def solve_coupled(
        self,
        mesh: FEMesh,
        material: Material,
        mechanical_lc: LoadCase,
        thermal_lc: LoadCase,
        time_steps: NDArray,
    ) -> FEAResults:
        """Coupled thermomechanical analysis (sequential coupling).

        Parameters
        ----------
        mesh : FEMesh
            Finite element mesh.
        material : Material
            Temperature-dependent material.
        mechanical_lc : LoadCase
            Mechanical loads and constraints.
        thermal_lc : LoadCase
            Thermal loads and boundary conditions.
        time_steps : NDArray
            Array of time values (s).

        Returns
        -------
        FEAResults
            Combined thermal and mechanical results.
        """


def get_backend(preference: str = "auto") -> SolverBackend:
    """Get a solver backend instance.

    Parameters
    ----------
    preference : str
        Backend name: ``"fenics"``, ``"calculix"``, ``"jax"``, ``"neural"``,
        or ``"auto"``. ``"auto"`` tries FEniCSx first, then CalculiX, then JAX.

    Returns
    -------
    SolverBackend
        A concrete solver backend.

    Raises
    ------
    ImportError
        If no suitable backend is available.
    """
    if preference == "fenics":
        from feaweld.solver.fenics_backend import FEniCSBackend
        return FEniCSBackend()

    if preference == "calculix":
        from feaweld.solver.calculix_backend import CalculiXBackend
        return CalculiXBackend()

    if preference == "jax":
        from feaweld.solver.jax_backend import JAXBackend
        return JAXBackend()

    if preference == "neural":
        from feaweld.solver.neural_backend import NeuralBackend
        return NeuralBackend()

    # auto: try FEniCSx first
    try:
        from feaweld.solver.fenics_backend import FEniCSBackend
        # Verify dolfinx is importable
        import dolfinx  # noqa: F401
        return FEniCSBackend()
    except ImportError:
        pass

    try:
        from feaweld.solver.calculix_backend import CalculiXBackend
        return CalculiXBackend()
    except ImportError:
        pass

    try:
        from feaweld.solver.jax_backend import JAXBackend
        return JAXBackend()
    except ImportError:
        pass

    raise ImportError(
        "No FEA solver backend available. Install fenics-dolfinx, CalculiX (ccx), "
        "or JAX. See https://fenicsproject.org, https://www.calculix.de, or "
        "https://jax.readthedocs.io for installation."
    )
