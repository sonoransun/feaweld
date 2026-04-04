"""Mechanical solver utilities: constitutive model helper functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from feaweld.core.materials import Material


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
) -> tuple[NDArray, NDArray, float]:
    """Radial return mapping for J2 (von Mises) plasticity.

    Implements the classical implicit return-mapping algorithm with
    linear isotropic hardening.

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

    Returns
    -------
    stress : NDArray
        Updated Cauchy stress in Voigt notation, shape ``(6,)``.
    strain_plastic_new : NDArray
        Updated plastic strain in Voigt notation, shape ``(6,)``.
    equiv_plastic_strain : float
        Increment of equivalent plastic strain Delta-gamma.
    """
    strain_trial = np.asarray(strain_trial, dtype=np.float64)
    strain_plastic_n = np.asarray(strain_plastic_n, dtype=np.float64)

    C = material.elasticity_tensor_3d(T)
    mu = material.lame_mu(T)
    sigma_y = material.sigma_y(T)
    H = material.hardening_modulus  # linear hardening modulus

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

    # Step 4: Von Mises stress and yield function
    sigma_vm = np.sqrt(1.5 * s_norm_sq)
    f_trial = sigma_vm - sigma_y

    # Step 5: Check yield
    if f_trial <= 0.0:
        # Elastic step - no plastic correction
        return sigma_trial.copy(), strain_plastic_n.copy(), 0.0

    # Step 6: Plastic correction (radial return)
    # Consistency parameter: Delta_gamma = f / (3*mu + H)
    delta_gamma = f_trial / (3.0 * mu + H)

    # Normal to yield surface: n_hat = s_trial / ||s_trial||
    if s_norm < 1e-30:
        n_hat = np.zeros(6)
    else:
        n_hat = s_trial / s_norm

    # Update stress: subtract plastic corrector from trial stress
    # sigma = sigma_trial - 2*mu*Delta_gamma * sqrt(3/2) * n_hat
    # Because sigma_vm = sqrt(3/2)*||s||, and the flow rule involves
    # the normalised deviatoric direction, the correction is:
    # sigma = sigma_trial - 2*mu*delta_gamma * (3/2) * s_trial / sigma_vm
    # which simplifies to:
    corrector = 2.0 * mu * delta_gamma
    # The corrector in Voigt space applied to the deviatoric stress
    # n in tensorial = s/||s||, in Voigt the flow direction for sigma_vm:
    # d_sigma = 2*mu*delta_gamma * sqrt(3/2) * n_hat
    # where n_hat = s/||s|| and ||s|| = sigma_vm / sqrt(3/2)
    # So d_sigma = 2*mu*delta_gamma * (3/2) * s / sigma_vm
    factor = 3.0 * mu * delta_gamma / sigma_vm
    sigma = sigma_trial.copy()
    sigma[0] -= 2.0 * factor * s_trial[0]
    sigma[1] -= 2.0 * factor * s_trial[1]
    sigma[2] -= 2.0 * factor * s_trial[2]
    sigma[3] -= 2.0 * factor * s_trial[3]
    sigma[4] -= 2.0 * factor * s_trial[4]
    sigma[5] -= 2.0 * factor * s_trial[5]

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
