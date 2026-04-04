"""Constitutive model abstractions for FEA integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from feaweld.core.materials import Material
from feaweld.solver.mechanical import (
    j2_return_mapping,
    linear_elastic_stress,
)


@dataclass
class MaterialState:
    """Internal state variables carried between increments.

    Each integration point maintains its own state instance.
    """

    plastic_strain: NDArray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64)
    )
    equiv_plastic_strain: float = 0.0
    temperature: float = 20.0
    extra: dict[str, Any] = field(default_factory=dict)


class ConstitutiveModel(ABC):
    """Base class for constitutive (stress-strain) models.

    A constitutive model computes stress from strain and updates
    internal state variables.
    """

    @abstractmethod
    def stress(
        self, strain: NDArray, state: MaterialState
    ) -> tuple[NDArray, MaterialState]:
        """Compute stress and update state.

        Parameters
        ----------
        strain : NDArray
            Total strain in Voigt notation, shape ``(6,)``.
        state : MaterialState
            Current internal state (plastic strain, etc.).

        Returns
        -------
        stress : NDArray
            Cauchy stress in Voigt notation, shape ``(6,)``.
        new_state : MaterialState
            Updated state after this increment.
        """

    @abstractmethod
    def tangent(self, strain: NDArray, state: MaterialState) -> NDArray:
        """Algorithmic tangent modulus (6x6) for Newton-Raphson iterations.

        Parameters
        ----------
        strain : NDArray
            Current strain, Voigt notation ``(6,)``.
        state : MaterialState
            Current internal state.

        Returns
        -------
        NDArray
            6x6 consistent tangent matrix.
        """


class LinearElastic(ConstitutiveModel):
    """Isotropic linear elastic constitutive model.

    Parameters
    ----------
    material : Material
        Material with temperature-dependent elastic properties.
    temperature : float
        Temperature (C) at which to evaluate properties.
    """

    def __init__(self, material: Material, temperature: float = 20.0) -> None:
        self.material = material
        self.temperature = temperature
        self._C = material.elasticity_tensor_3d(temperature)

    def stress(
        self, strain: NDArray, state: MaterialState
    ) -> tuple[NDArray, MaterialState]:
        sigma = linear_elastic_stress(strain, self._C)
        # Linear elastic: state is unchanged (no plasticity)
        new_state = MaterialState(
            plastic_strain=state.plastic_strain.copy(),
            equiv_plastic_strain=state.equiv_plastic_strain,
            temperature=self.temperature,
            extra=state.extra.copy(),
        )
        return sigma, new_state

    def tangent(self, strain: NDArray, state: MaterialState) -> NDArray:
        return self._C.copy()

    def update_temperature(self, T: float) -> None:
        """Recompute the elasticity tensor for a new temperature."""
        self.temperature = T
        self._C = self.material.elasticity_tensor_3d(T)


class J2Plastic(ConstitutiveModel):
    """J2 (von Mises) plasticity with isotropic linear hardening.

    Parameters
    ----------
    material : Material
        Material with yield strength and hardening parameters.
    temperature : float
        Temperature (C) for property evaluation.
    """

    def __init__(self, material: Material, temperature: float = 20.0) -> None:
        self.material = material
        self.temperature = temperature

    def stress(
        self, strain: NDArray, state: MaterialState
    ) -> tuple[NDArray, MaterialState]:
        sigma, eps_p_new, d_gamma = j2_return_mapping(
            strain_trial=strain,
            strain_plastic_n=state.plastic_strain,
            material=self.material,
            T=self.temperature,
        )
        new_state = MaterialState(
            plastic_strain=eps_p_new,
            equiv_plastic_strain=state.equiv_plastic_strain + d_gamma,
            temperature=self.temperature,
            extra=state.extra.copy(),
        )
        return sigma, new_state

    def tangent(self, strain: NDArray, state: MaterialState) -> NDArray:
        """Consistent algorithmic tangent for J2 plasticity.

        For elastic loading or unloading, returns the elastic tangent.
        For active plastic loading, returns the elasto-plastic tangent
        derived from the radial return mapping.
        """
        C = self.material.elasticity_tensor_3d(self.temperature)
        mu = self.material.lame_mu(self.temperature)
        sigma_y = self.material.sigma_y(self.temperature)
        H = self.material.hardening_modulus

        # Check if currently yielding by doing a trial
        strain_e_trial = strain - state.plastic_strain
        sigma_trial = C @ strain_e_trial
        # Deviatoric
        hydro = (sigma_trial[0] + sigma_trial[1] + sigma_trial[2]) / 3.0
        s = sigma_trial.copy()
        s[0] -= hydro
        s[1] -= hydro
        s[2] -= hydro
        s_norm_sq = (
            s[0] ** 2 + s[1] ** 2 + s[2] ** 2
            + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
        )
        sigma_vm = np.sqrt(1.5 * s_norm_sq)
        f_trial = sigma_vm - sigma_y

        if f_trial <= 0.0:
            return C.copy()

        # Plastic tangent (consistent tangent for radial return)
        s_norm = np.sqrt(s_norm_sq)
        if s_norm < 1e-30:
            return C.copy()

        delta_gamma = f_trial / (3.0 * mu + H)
        # Normal in Voigt (tensor product needs care)
        n = s / s_norm  # (6,)

        # Outer product n x n with Voigt weight
        # For Voigt: N_ij = n_i * n_j * w  where w accounts for shear
        # But for the tangent we work in the Mandel-corrected form and then
        # convert.  For simplicity, use the standard formula:
        # C_ep = C - (2*mu)^2 * delta_gamma / s_norm * P_dev
        #        + (2*mu)^2 * (delta_gamma / s_norm - 1/(3*mu + H)) * n x n
        # where P_dev is the deviatoric projector.
        # Simplified consistent tangent:
        theta = 1.0 - 2.0 * mu * delta_gamma / s_norm
        theta_bar = 1.0 / (1.0 + H / (3.0 * mu)) - (1.0 - theta)

        # Identity and deviatoric projector in Voigt
        I2 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        # Deviatoric projector P = I4_sym - 1/3 * I2 x I2
        I4_sym = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        P_dev = I4_sym - np.outer(I2, I2) / 3.0

        # n outer n with proper Voigt handling
        # For Voigt representation, shear terms need factor 2 in the outer product
        # for the off-diagonal coupling.  We use Mandel-like weighting.
        n_mandel = n.copy()
        n_mandel[3:] *= np.sqrt(2.0)
        nn = np.outer(n_mandel, n_mandel)
        # Convert back from Mandel to Voigt
        nn_voigt = nn.copy()
        nn_voigt[3:, :] /= np.sqrt(2.0)
        nn_voigt[:, 3:] /= np.sqrt(2.0)

        lam = self.material.lame_lambda(self.temperature)
        # Volumetric part
        K = lam + 2.0 * mu / 3.0  # bulk modulus
        C_ep = (
            K * np.outer(I2, I2)
            + 2.0 * mu * theta * P_dev
            - 2.0 * mu * theta_bar * nn_voigt
        )
        return C_ep

    def update_temperature(self, T: float) -> None:
        """Update temperature for property evaluation."""
        self.temperature = T


class TemperatureDependent(ConstitutiveModel):
    """Wrapper that updates a base constitutive model's temperature.

    At each call, the wrapped model's properties are re-evaluated at the
    current temperature stored in the state.

    Parameters
    ----------
    base_model : ConstitutiveModel
        A constitutive model that has an ``update_temperature`` method.
    material : Material
        The material (used for reference).
    """

    def __init__(
        self, base_model: ConstitutiveModel, material: Material
    ) -> None:
        self.base_model = base_model
        self.material = material

    def stress(
        self, strain: NDArray, state: MaterialState
    ) -> tuple[NDArray, MaterialState]:
        T = state.temperature
        if hasattr(self.base_model, "update_temperature"):
            self.base_model.update_temperature(T)
        return self.base_model.stress(strain, state)

    def tangent(self, strain: NDArray, state: MaterialState) -> NDArray:
        T = state.temperature
        if hasattr(self.base_model, "update_temperature"):
            self.base_model.update_temperature(T)
        return self.base_model.tangent(strain, state)
