"""Coupled thermomechanical solver using sequential (staggered) coupling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from feaweld.core.materials import Material
from feaweld.core.types import (
    BoundaryCondition,
    FEAResults,
    FEMesh,
    LoadCase,
    LoadType,
    StressField,
)
from feaweld.solver.backend import SolverBackend
from feaweld.solver.mechanical import linear_elastic_stress


def sequential_coupled_solve(
    backend: SolverBackend,
    mesh: FEMesh,
    material: Material,
    thermal_lc: LoadCase,
    mechanical_lc: LoadCase,
    time_steps: NDArray,
    heat_source: object | None = None,
) -> FEAResults:
    """Sequentially-coupled thermomechanical analysis.

    The algorithm proceeds as follows:

    1.  Solve the transient thermal problem to obtain temperature
        history *T(x, t)* at all nodes and time steps.
    2.  For each time step (or the final state), compute thermal
        strains from the coefficient of thermal expansion and add
        the corresponding body forces to the mechanical problem.
    3.  Solve the static mechanical problem for each time step.
    4.  Combine thermal and mechanical results into a single
        :class:`FEAResults` object.

    Parameters
    ----------
    backend : SolverBackend
        FEA solver backend to use.
    mesh : FEMesh
        Finite element mesh.
    material : Material
        Temperature-dependent material.
    thermal_lc : LoadCase
        Thermal loads and boundary conditions.
    mechanical_lc : LoadCase
        Mechanical loads and constraints.
    time_steps : NDArray
        Time values (s) for the transient analysis.
    heat_source : object or None
        Optional moving heat source (e.g. :class:`GoldakHeatSource`).

    Returns
    -------
    FEAResults
        Combined results with displacement, stress, strain, and
        temperature fields.
    """
    time_steps = np.asarray(time_steps, dtype=np.float64)
    T_ref = 20.0  # reference (stress-free) temperature in C

    # --- Step 1: Solve transient thermal problem ---
    thermal_results = backend.solve_thermal_transient(
        mesh=mesh,
        material=material,
        load_case=thermal_lc,
        time_steps=time_steps,
        heat_source=heat_source,
    )

    n_nodes = mesh.n_nodes
    n_steps = len(time_steps)

    # Extract temperature history
    # Expect shape (n_steps, n_nodes) or (n_nodes,) for single step
    if thermal_results.temperature is not None:
        temp_field = thermal_results.temperature
        if temp_field.ndim == 1:
            # Single time step: replicate for all steps
            temp_history = np.tile(temp_field, (n_steps, 1))
        else:
            temp_history = temp_field
    else:
        # No thermal result available; assume uniform reference temperature
        temp_history = np.full((n_steps, n_nodes), T_ref)

    # --- Step 2 & 3: Mechanical solve at each time step ---
    displacement_history: list[NDArray] = []
    stress_history: list[NDArray] = []
    strain_history: list[NDArray] = []

    for step_idx in range(n_steps):
        T_nodes = temp_history[step_idx]
        T_avg = float(np.mean(T_nodes))

        # Compute thermal strain at each node
        # eps_th = alpha(T) * (T - T_ref) for each node
        thermal_strains = np.zeros((n_nodes, 6))
        for node_i in range(n_nodes):
            T_node = float(T_nodes[node_i])
            alpha_val = material.alpha(T_node)
            eps_th = alpha_val * (T_node - T_ref)
            thermal_strains[node_i, 0] = eps_th
            thermal_strains[node_i, 1] = eps_th
            thermal_strains[node_i, 2] = eps_th
            # Shear components are zero for isotropic thermal expansion

        # Create a modified load case that includes thermal body forces
        # Add temperature distribution as a load
        thermal_body_bc = BoundaryCondition(
            node_set="all",
            bc_type=LoadType.TEMPERATURE,
            values=T_nodes,
        )

        combined_loads = list(mechanical_lc.loads) + [thermal_body_bc]
        combined_lc = LoadCase(
            name=f"{mechanical_lc.name}_step_{step_idx}",
            loads=combined_loads,
            constraints=list(mechanical_lc.constraints),
        )

        # Solve static mechanical problem at this temperature
        mech_result = backend.solve_static(
            mesh=mesh,
            material=material,
            load_case=combined_lc,
            temperature=T_avg,
        )

        # Store results
        if mech_result.displacement is not None:
            displacement_history.append(mech_result.displacement)
        if mech_result.stress is not None:
            stress_history.append(mech_result.stress.values)
        if mech_result.strain is not None:
            strain_history.append(mech_result.strain)

    # --- Step 4: Assemble combined results ---
    # Use the final time step for the primary fields
    final_displacement = displacement_history[-1] if displacement_history else None
    final_stress = (
        StressField(values=stress_history[-1]) if stress_history else None
    )
    final_strain = strain_history[-1] if strain_history else None

    # Build time history dict
    time_hist: dict[str, list[NDArray]] = {}
    if displacement_history:
        time_hist["displacement"] = displacement_history
    if stress_history:
        time_hist["stress"] = stress_history
    if strain_history:
        time_hist["strain"] = strain_history

    combined = FEAResults(
        mesh=mesh,
        displacement=final_displacement,
        stress=final_stress,
        strain=final_strain,
        temperature=temp_history[-1] if temp_history is not None else None,
        time_steps=time_steps,
        time_history=time_hist if time_hist else None,
        metadata={
            "analysis_type": "thermomechanical_sequential",
            "n_time_steps": n_steps,
            "thermal_reference_temperature": T_ref,
        },
    )
    return combined


def compute_thermal_stress(
    material: Material,
    temperature: NDArray,
    T_ref: float = 20.0,
) -> NDArray:
    """Compute thermal stress at nodes assuming full constraint.

    For a fully constrained body, thermal stress is:
        sigma_th = -C : eps_th

    This is useful for estimating residual stress without a full FEA solve.

    Parameters
    ----------
    material : Material
        Material with temperature-dependent properties.
    temperature : NDArray
        Nodal temperatures (C), shape ``(n_nodes,)``.
    T_ref : float
        Reference (stress-free) temperature.

    Returns
    -------
    NDArray
        Thermal stress in Voigt notation, shape ``(n_nodes, 6)``.
    """
    temperature = np.asarray(temperature, dtype=np.float64)
    n_nodes = temperature.shape[0]
    stress = np.zeros((n_nodes, 6))

    for i in range(n_nodes):
        T = float(temperature[i])
        alpha_val = material.alpha(T)
        dT = T - T_ref
        eps_th = np.array([alpha_val * dT] * 3 + [0.0, 0.0, 0.0])
        C = material.elasticity_tensor_3d(T)
        stress[i] = -(C @ eps_th)

    return stress
