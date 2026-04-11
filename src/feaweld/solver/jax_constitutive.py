"""Batched JAX constitutive models for the differentiable backend.

The existing `feaweld.solver.constitutive.ConstitutiveModel` ABC operates
on scalar Voigt-6 strain arrays per integration point, which forces a
host/device sync per call when used from a JAX solver. This module
defines a parallel protocol that takes **batched** `(n_elements, 6)`
strain arrays and returns stress + tangent for the whole batch in one
pass, compatible with `jax.vmap` / `lax.scan`.

Temperature-dependent material properties are materialized to plain
floats at construction time, matching how `LinearElastic.__init__`
freezes the elasticity tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jax = None  # type: ignore
    jnp = None  # type: ignore

from feaweld.core.materials import Material


class JAXConstitutiveModel(Protocol):
    """Protocol for batched JAX constitutive models.

    Implementations must be pytree-compatible so they can be closed over
    by `jax.jit`-compiled solver code without leaking Python state.
    """

    def stress(self, strain: "jnp.ndarray") -> "jnp.ndarray":
        """Compute stress for a batch of strain tensors.

        Parameters
        ----------
        strain : jnp.ndarray, shape (..., 6)
            Voigt strain tensors [ε_xx, ε_yy, ε_zz, 2ε_xy, 2ε_yz, 2ε_xz].

        Returns
        -------
        jnp.ndarray, shape (..., 6)
            Voigt Cauchy stress tensors.
        """

    def tangent(self, strain: "jnp.ndarray") -> "jnp.ndarray":
        """Consistent tangent modulus (..., 6, 6) for the same batch."""


def _require_jax() -> None:
    if not _HAS_JAX:
        raise ImportError(
            "JAX is required for feaweld.solver.jax_constitutive. "
            "Install with: pip install 'feaweld[jax]'"
        )


def _elasticity_tensor_voigt(E: float, nu: float) -> np.ndarray:
    """Build the 6x6 Voigt elasticity tensor for an isotropic material."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


@dataclass(frozen=True)
class JAXLinearElastic:
    """Batched isotropic linear elastic model.

    The elasticity tensor is frozen at construction from the material's
    temperature-dependent properties, mirroring `LinearElastic`.
    """

    C: "jnp.ndarray"  # (6, 6)

    @classmethod
    def from_material(
        cls, material: Material, temperature: float = 20.0
    ) -> "JAXLinearElastic":
        _require_jax()
        C_np = _elasticity_tensor_voigt(
            material.E(temperature), material.nu(temperature)
        )
        return cls(C=jnp.asarray(C_np, dtype=jnp.float64))

    def stress(self, strain: "jnp.ndarray") -> "jnp.ndarray":
        return strain @ self.C.T

    def tangent(self, strain: "jnp.ndarray") -> "jnp.ndarray":
        batch_shape = strain.shape[:-1]
        return jnp.broadcast_to(self.C, (*batch_shape, 6, 6))


@dataclass(frozen=True)
class JAXJ2Plasticity:
    """Batched J2 (von Mises) plasticity with isotropic linear hardening.

    Radial return mapping implemented in pure JAX so gradients flow
    through the plastic strain update via implicit differentiation of
    the consistency condition. Hardening is linear: σ_y(εp) = σ_y0 + H εp.

    This model is *stateless* in the JAX sense: the accumulated plastic
    strain must be carried externally by the solver's `lax.scan` state.
    For the initial A1 milestone (linear-elastic solve) it is unused;
    A2 wires it into the Newton loop.
    """

    C: "jnp.ndarray"  # (6, 6)
    mu: float
    sigma_y0: float
    H: float  # isotropic hardening modulus

    @classmethod
    def from_material(
        cls, material: Material, temperature: float = 20.0
    ) -> "JAXJ2Plasticity":
        _require_jax()
        C_np = _elasticity_tensor_voigt(
            material.E(temperature), material.nu(temperature)
        )
        return cls(
            C=jnp.asarray(C_np, dtype=jnp.float64),
            mu=float(material.lame_mu(temperature)),
            sigma_y0=float(material.sigma_y(temperature)),
            H=float(material.hardening_modulus),
        )

    def _trial_stress(self, strain_elastic: "jnp.ndarray") -> "jnp.ndarray":
        return strain_elastic @ self.C.T

    def _radial_return(
        self,
        strain: "jnp.ndarray",
        plastic_strain_prev: "jnp.ndarray",
        eqps_prev: "jnp.ndarray",
    ) -> tuple["jnp.ndarray", "jnp.ndarray", "jnp.ndarray"]:
        """One-step radial return for a single batch element.

        Returns (stress, new_plastic_strain, new_eqps).
        """
        strain_el_trial = strain - plastic_strain_prev
        sigma_trial = self._trial_stress(strain_el_trial)

        hydro = (sigma_trial[0] + sigma_trial[1] + sigma_trial[2]) / 3.0
        s = sigma_trial.at[0].add(-hydro).at[1].add(-hydro).at[2].add(-hydro)
        s_norm_sq = (
            s[0] ** 2 + s[1] ** 2 + s[2] ** 2
            + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
        )
        sigma_vm = jnp.sqrt(1.5 * s_norm_sq + 1e-30)
        sigma_y_curr = self.sigma_y0 + self.H * eqps_prev
        f_trial = sigma_vm - sigma_y_curr

        # Radial return increment (zero if elastic)
        dgamma = jnp.where(
            f_trial > 0.0, f_trial / (3.0 * self.mu + self.H), 0.0
        )

        # Unit deviatoric direction (safe division)
        s_norm = jnp.sqrt(s_norm_sq + 1e-30)
        n = s / s_norm

        # Plastic strain increment (Voigt form; engineering shear doubled)
        dep = dgamma * jnp.sqrt(1.5) * n
        dep = dep.at[3:].multiply(2.0)

        plastic_strain_new = plastic_strain_prev + dep
        eqps_new = eqps_prev + dgamma * jnp.sqrt(2.0 / 3.0)

        strain_el_new = strain - plastic_strain_new
        stress = self._trial_stress(strain_el_new)
        return stress, plastic_strain_new, eqps_new

    def stress_stateful(
        self,
        strain_batch: "jnp.ndarray",
        plastic_strain_batch: "jnp.ndarray",
        eqps_batch: "jnp.ndarray",
    ) -> tuple["jnp.ndarray", "jnp.ndarray", "jnp.ndarray"]:
        """Batched stress update: returns (stress, new_eps_p, new_eqps)."""
        return jax.vmap(self._radial_return)(
            strain_batch, plastic_strain_batch, eqps_batch
        )

    def stress(self, strain: "jnp.ndarray") -> "jnp.ndarray":
        """Stateless stress (treats previous plastic strain as zero).

        Needed to satisfy the JAXConstitutiveModel protocol for simple
        one-shot calls. For incremental loading use `stress_stateful`.
        """
        zeros = jnp.zeros_like(strain)
        zeros_eqps = jnp.zeros(strain.shape[:-1])
        stress, _, _ = self.stress_stateful(strain, zeros, zeros_eqps)
        return stress

    def tangent(self, strain: "jnp.ndarray") -> "jnp.ndarray":
        """Consistent tangent via jax.vjp on the stateless stress.

        This is the elegant payoff of routing through JAX: the
        algorithmic tangent falls out of reverse-mode AD automatically,
        no manual derivation of the J2 tangent needed.
        """
        def single_stress(e):
            return self.stress(e[None, :])[0]

        jac_fn = jax.vmap(jax.jacfwd(single_stress))
        return jac_fn(strain)
