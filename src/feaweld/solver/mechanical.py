"""Mechanical solver utilities: constitutive model helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from feaweld.core.materials import Material

if TYPE_CHECKING:
    from feaweld.core.types import FEAResults, FEMesh, LoadCase
    from feaweld.solver.backend import SolverBackend


def linear_elastic_stress(strain: NDArray, C: NDArray) -> NDArray:
    """Compute stress from strain using the linear elastic constitutive law.

    Parameters
    ----------
    strain : NDArray
        Strain in Voigt notation.  Shape ``(6,)`` for a single point or
        ``(n, 6)`` for multiple points.  Components:
        [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz].
    C : NDArray
        6x6 elasticity matrix in Voigt notation.

    Returns
    -------
    NDArray
        Stress in Voigt notation, same shape as *strain*.
        [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz].
    """
    strain = np.asarray(strain, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    if strain.ndim == 1:
        return C @ strain
    # (n, 6)
    return (C @ strain.T).T


def deviatoric_stress(stress: NDArray) -> NDArray:
    """Extract the deviatoric part of a stress tensor in Voigt notation.

    Parameters
    ----------
    stress : NDArray
        Stress vector(s) in Voigt notation, shape ``(6,)`` or ``(n, 6)``.

    Returns
    -------
    NDArray
        Deviatoric stress, same shape as *stress*.
    """
    stress = np.asarray(stress, dtype=np.float64)
    single = stress.ndim == 1
    if single:
        stress = stress[np.newaxis, :]

    hydro = (stress[:, 0] + stress[:, 1] + stress[:, 2]) / 3.0
    s = stress.copy()
    s[:, 0] -= hydro
    s[:, 1] -= hydro
    s[:, 2] -= hydro
    # Shear components are unchanged

    if single:
        return s[0]
    return s


def von_mises(stress: NDArray) -> NDArray | float:
    """Compute von Mises equivalent stress from Voigt notation.

    Parameters
    ----------
    stress : NDArray
        Shape ``(6,)`` or ``(n, 6)``.

    Returns
    -------
    float or NDArray
        Von Mises stress.
    """
    stress = np.asarray(stress, dtype=np.float64)
    single = stress.ndim == 1
    if single:
        stress = stress[np.newaxis, :]

    s = deviatoric_stress(stress)
    # von Mises = sqrt(3/2 * s:s) with proper treatment of shear in Voigt
    # s:s = s_xx^2 + s_yy^2 + s_zz^2 + 2*(s_xy^2 + s_yz^2 + s_xz^2)
    s_contract = (
        s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
        + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    )
    vm = np.sqrt(1.5 * s_contract)

    if single:
        return float(vm[0])
    return vm


def j2_return_mapping(
    strain_trial: NDArray,
    strain_plastic_n: NDArray,
    material: Material,
    T: float,
    equiv_plastic_strain: float = 0.0,
) -> tuple[NDArray, NDArray, float]:
    """Radial return mapping for J2 (von Mises) plasticity.

    Implements the classical implicit (backward-Euler) return-mapping
    algorithm with linear isotropic hardening.  For linear hardening under
    proportional loading the algorithm is exact, so provided the accumulated
    hardening is threaded through *equiv_plastic_strain* the final state is
    independent of how the load is subdivided into increments.

    Parameters
    ----------
    strain_trial : NDArray
        Total trial strain at the current increment, shape ``(6,)``.
    strain_plastic_n : NDArray
        Accumulated plastic strain from the previous increment, shape ``(6,)``.
    material : Material
        Material with temperature-dependent properties and hardening
        parameters (``hardening_modulus``, ``hardening_exponent``).
    T : float
        Current temperature (C) for property evaluation.
    equiv_plastic_strain : float, optional
        Accumulated equivalent plastic strain from previous increments.  The
        yield surface is hardened to ``sigma_y + H * equiv_plastic_strain``
        before the yield check, so re-yielding starts from the current
        (hardened) yield stress rather than the virgin one.  Defaults to
        ``0.0`` (single-shot use), which reproduces virgin-yield behaviour.

    Returns
    -------
    stress : NDArray
        Updated Cauchy stress in Voigt notation, shape ``(6,)``.
    strain_plastic_new : NDArray
        Updated plastic strain in Voigt notation, shape ``(6,)``.
    delta_gamma : float
        Increment of equivalent plastic strain for this call.  Adding this to
        *equiv_plastic_strain* gives the updated accumulated value.
    """
    strain_trial = np.asarray(strain_trial, dtype=np.float64)
    strain_plastic_n = np.asarray(strain_plastic_n, dtype=np.float64)

    C = material.elasticity_tensor_3d(T)
    mu = material.lame_mu(T)
    sigma_y = material.sigma_y(T)
    H = material.hardening_modulus  # linear hardening modulus

    # Current (hardened) yield stress after prior plastic flow.  With
    # equiv_plastic_strain == 0 this is the virgin yield stress.
    sigma_y_eff = sigma_y + H * equiv_plastic_strain

    # Step 1: Trial elastic strain
    strain_elastic_trial = strain_trial - strain_plastic_n

    # Step 2: Trial stress
    sigma_trial = C @ strain_elastic_trial

    # Step 3: Deviatoric trial stress
    s_trial = deviatoric_stress(sigma_trial)

    # Norm of deviatoric stress (accounting for Voigt shear factor)
    # ||s|| = sqrt(s:s) where s:s = s_ij * s_ij
    # In Voigt: s:s = s_xx^2 + s_yy^2 + s_zz^2 + 2*(s_xy^2 + s_yz^2 + s_xz^2)
    s_norm_sq = (
        s_trial[0] ** 2 + s_trial[1] ** 2 + s_trial[2] ** 2
        + 2.0 * (s_trial[3] ** 2 + s_trial[4] ** 2 + s_trial[5] ** 2)
    )
    s_norm = np.sqrt(s_norm_sq)

    # Step 4: Von Mises stress and yield function (against hardened surface)
    sigma_vm = np.sqrt(1.5 * s_norm_sq)
    f_trial = sigma_vm - sigma_y_eff

    # Step 5: Check yield
    if f_trial <= 0.0:
        # Elastic step - no plastic correction
        return sigma_trial.copy(), strain_plastic_n.copy(), 0.0

    # Step 6: Plastic correction (radial return)
    # Consistency parameter: Delta_gamma = f / (3*mu + H)
    delta_gamma = f_trial / (3.0 * mu + H)

    # Normal to yield surface: n_hat = s_trial / ||s_trial||
    if s_norm < 1e-30:
        # Deviatoric stress is zero — elastic return (defensive guard)
        return sigma_trial.copy(), strain_plastic_n.copy(), 0.0
    n_hat = s_trial / s_norm

    # Update stress: subtract the plastic corrector from the trial stress.
    # The associative flow rule gives d_eps_p = delta_gamma * (3/2) * s/sigma_vm,
    # and the deviatoric stress relaxes by 2*mu*d_eps_p, so
    #   sigma = sigma_trial - 2*mu*delta_gamma * (3/2) * s_trial / sigma_vm
    #         = sigma_trial - 3*mu*delta_gamma * s_trial / sigma_vm.
    # With this correction vm(sigma) == sigma_y_eff + H*delta_gamma exactly
    # (linear-hardening consistency), since vm(sigma) = sigma_vm - 3*mu*delta_gamma.
    factor = 3.0 * mu * delta_gamma / sigma_vm
    sigma = sigma_trial.copy()
    sigma[0] -= factor * s_trial[0]
    sigma[1] -= factor * s_trial[1]
    sigma[2] -= factor * s_trial[2]
    sigma[3] -= factor * s_trial[3]
    sigma[4] -= factor * s_trial[4]
    sigma[5] -= factor * s_trial[5]

    # Update plastic strain
    # d_eps_p = delta_gamma * sqrt(3/2) * n_hat (in Voigt)
    # n_hat = s / ||s||, but we need the tensorial normal to Voigt:
    # For the normal components the factor is 1, for shear it's already
    # consistent since we stored shear strains as engineering.
    # d_eps_p_ij = delta_gamma * (3/2) * s_ij / sigma_vm
    strain_plastic_new = strain_plastic_n.copy()
    dep = 1.5 * delta_gamma / sigma_vm
    strain_plastic_new[0] += dep * s_trial[0]
    strain_plastic_new[1] += dep * s_trial[1]
    strain_plastic_new[2] += dep * s_trial[2]
    # For engineering shear in Voigt, the plastic shear strain increment
    # d_gamma_xy = 2 * d_eps_xy = 2 * dep * s_xy
    strain_plastic_new[3] += 2.0 * dep * s_trial[3]
    strain_plastic_new[4] += 2.0 * dep * s_trial[4]
    strain_plastic_new[5] += 2.0 * dep * s_trial[5]

    return sigma, strain_plastic_new, float(delta_gamma)


def solve_elastoplastic(
    backend: SolverBackend,
    mesh: FEMesh,
    material: Material,
    load_case: LoadCase,
    temperature: float = 20.0,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
) -> FEAResults:
    """Elastoplastic solve via elastic solution + J2 stress correction.

    Runs a linear elastic solve on *backend*, then applies the radial
    return-mapping algorithm ([j2_return_mapping][feaweld.solver.mechanical.j2_return_mapping]) point-by-point,
    ramping the total strain over load increments.  This is a
    *post-correction* approximation: stresses are capped at the yield
    surface with linear isotropic hardening, but global equilibrium is
    not re-established (no stress redistribution).  For weld assessment
    this bounds the local plastic response; a natively nonlinear solve
    (e.g. CalculiX ``*PLASTIC``) is required for large-scale yielding.

    Parameters
    ----------
    backend : SolverBackend
        FEA solver backend used for the elastic predictor solve.
    mesh : FEMesh
        Finite element mesh.
    material : Material
        Material with yield strength and hardening parameters.
    load_case : LoadCase
        Loads and boundary conditions.
    temperature : float
        Uniform temperature for property evaluation (C).
    max_iterations : int
        Upper bound on strain increments (capped at 10).
    tolerance : float
        Convergence tolerance on the relative stress change between
        the final two increments.

    Returns
    -------
    FEAResults
        Results with the plastically corrected stress field and a
        ``metadata["plasticity"]`` summary.
    """
    elastic = backend.solve_static(
        mesh, material, load_case, temperature=temperature,
    )
    if elastic.stress is None:
        return elastic

    from feaweld.core.types import FEAResults as _FEAResults, StressField

    C = material.elasticity_tensor_3d(temperature)
    C_inv = np.linalg.inv(C)

    stress_vals = elastic.stress.values
    if elastic.strain is not None and elastic.strain.shape == stress_vals.shape:
        strain_total = elastic.strain
    else:
        # Recover total strain from the elastic stress: eps = C^-1 sigma
        strain_total = stress_vals @ C_inv.T

    n_pts = stress_vals.shape[0]
    corrected = np.empty_like(stress_vals)
    plastic_strain = np.zeros_like(stress_vals)
    equiv_plastic = np.zeros(n_pts)

    n_increments = int(max(1, min(max_iterations, 10)))
    max_rel_change = 0.0

    for pt in range(n_pts):
        eps_p = np.zeros(6)
        delta_gamma_total = 0.0
        sigma_prev = None
        sigma = stress_vals[pt].copy()
        for inc in range(1, n_increments + 1):
            eps_trial = strain_total[pt] * (inc / n_increments)
            # Harden the yield surface with the plastic strain accumulated so
            # far, so re-yielding in later increments does not restart from the
            # virgin yield stress.  This makes the final state independent of
            # n_increments for proportional loading.
            sigma, eps_p, dg = j2_return_mapping(
                eps_trial, eps_p, material, temperature,
                equiv_plastic_strain=delta_gamma_total,
            )
            delta_gamma_total += dg
            if sigma_prev is not None:
                denom = max(float(np.linalg.norm(sigma_prev)), 1e-12)
                max_rel_change = max(
                    max_rel_change,
                    float(np.linalg.norm(sigma - sigma_prev)) / denom,
                )
            sigma_prev = sigma
        corrected[pt] = sigma
        plastic_strain[pt] = eps_p
        equiv_plastic[pt] = delta_gamma_total

    yielded = equiv_plastic > 0.0
    metadata = {
        **elastic.metadata,
        "plasticity": {
            "model": "j2_radial_return_postcorrection",
            "n_increments": n_increments,
            "n_yielded_points": int(np.sum(yielded)),
            "max_equiv_plastic_strain": float(np.max(equiv_plastic)),
            "final_relative_stress_change": max_rel_change,
            "converged": bool(max_rel_change <= max(tolerance, 1e-12) or n_increments == 1),
            "yield_strength": material.sigma_y(temperature),
        },
    }

    return _FEAResults(
        mesh=elastic.mesh,
        displacement=elastic.displacement,
        stress=StressField(values=corrected, location=elastic.stress.location),
        strain=strain_total,
        temperature=elastic.temperature,
        nodal_forces=elastic.nodal_forces,
        time_steps=elastic.time_steps,
        time_history=elastic.time_history,
        metadata=metadata,
    )
