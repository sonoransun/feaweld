"""Meric-Cailletaud rate-dependent crystal plasticity for FCC metals.

Single-crystal plasticity with 12 {111}<110> slip systems, implemented
as a `JAXConstitutiveModel`-compatible dataclass so it plugs directly
into `JAXBackend.solve_static_incremental` without modifying the
backend.

The flow rule is a classical viscoplastic power law:

    gamma_dot_alpha = gamma_dot_0 * (|tau_alpha| / tau_c_alpha) ** n
                      * sign(tau_alpha)

with linear isotropic self-hardening on the critical resolved shear
stress (no cross-hardening for this MVP):

    tau_c_alpha = tau0 + H * eqps

The signature of `stress_stateful(strain, plastic_strain, eqps)` exactly
matches `JAXJ2Plasticity` so the backend's operator-split Newton loop
can treat it interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jax = None  # type: ignore
    jnp = None  # type: ignore

from feaweld.core.materials import Material
from feaweld.solver.jax_constitutive import _elasticity_tensor_voigt


# ---------------------------------------------------------------------------
# FCC slip system geometry ({111}<110>)
# ---------------------------------------------------------------------------


def _build_fcc_slip_systems() -> tuple[np.ndarray, np.ndarray]:
    """Return normalized (12, 3) normals and directions for FCC slip."""
    planes = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
        ]
    )
    directions_per_plane = [
        # (1, 1, 1)
        [[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]],
        # (-1, 1, 1)
        [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, -1.0, 1.0]],
        # (1, -1, 1)
        [[1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        # (1, 1, -1)
        [[-1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    ]
    normals = np.zeros((12, 3))
    directions = np.zeros((12, 3))
    idx = 0
    for p_idx, plane in enumerate(planes):
        n_hat = plane / np.linalg.norm(plane)
        for d in directions_per_plane[p_idx]:
            d_arr = np.asarray(d, dtype=float)
            d_hat = d_arr / np.linalg.norm(d_arr)
            normals[idx] = n_hat
            directions[idx] = d_hat
            idx += 1
    return normals, directions


FCC_SLIP_NORMALS, FCC_SLIP_DIRECTIONS = _build_fcc_slip_systems()


def schmid_tensors(normals: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Symmetric Schmid tensors in Voigt-6 form.

    Returns an (N, 6) array where each row encodes the symmetric part
    of ``s (x) n`` as
    ``[P_xx, P_yy, P_zz, 2*P_xy, 2*P_yz, 2*P_xz]``. The engineering-shear
    doubling means a simple inner product with a Voigt stress vector
    recovers the resolved shear stress on that slip system.
    """
    n = normals.shape[0]
    out = np.zeros((n, 6))
    for a in range(n):
        s = directions[a]
        m = normals[a]
        P = 0.5 * (np.outer(s, m) + np.outer(m, s))
        out[a, 0] = P[0, 0]
        out[a, 1] = P[1, 1]
        out[a, 2] = P[2, 2]
        out[a, 3] = 2.0 * P[0, 1]
        out[a, 4] = 2.0 * P[1, 2]
        out[a, 5] = 2.0 * P[0, 2]
    return out


def _require_jax() -> None:
    if not _HAS_JAX:
        raise ImportError(
            "JAX is required for feaweld.solver.jax_crystal_plasticity. "
            "Install with: pip install 'feaweld[jax]'"
        )


# ---------------------------------------------------------------------------
# Meric-Cailletaud crystal plasticity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JAXCrystalPlasticity:
    """Rate-dependent single-crystal plasticity for FCC metals.

    Implements the `JAXConstitutiveModel` protocol with the same
    `stress_stateful` signature as `JAXJ2Plasticity`, so it drops into
    `JAXBackend.solve_static_incremental` unchanged.
    """

    C: "jnp.ndarray"  # (6, 6) elastic stiffness in Voigt form
    schmid: "jnp.ndarray"  # (n_sys, 6) Schmid tensors with engineering shear
    tau0: float  # initial critical resolved shear stress
    H: float  # linear self-hardening modulus on tau_c
    gamma_dot_0: float = 1.0e-3  # reference slip rate
    n_power: int = 20  # rate exponent (high -> near rate-independent)
    dt: float = 1.0  # pseudo-time step for the viscoplastic update

    @classmethod
    def from_material(
        cls,
        material: Material,
        temperature: float = 20.0,
        tau0: float | None = None,
        H: float | None = None,
        gamma_dot_0: float = 1.0e-3,
        n_power: int = 20,
        dt: float = 1.0,
    ) -> "JAXCrystalPlasticity":
        _require_jax()
        C_np = _elasticity_tensor_voigt(
            material.E(temperature), material.nu(temperature)
        )
        schmid_np = schmid_tensors(FCC_SLIP_NORMALS, FCC_SLIP_DIRECTIONS)
        if tau0 is None:
            tau0 = material.sigma_y(temperature) / sqrt(3.0)
        if H is None:
            H = float(material.hardening_modulus)
        return cls(
            C=jnp.asarray(C_np, dtype=jnp.float64),
            schmid=jnp.asarray(schmid_np, dtype=jnp.float64),
            tau0=float(tau0),
            H=float(H),
            gamma_dot_0=float(gamma_dot_0),
            n_power=int(n_power),
            dt=float(dt),
        )

    # -- Core single-point update -------------------------------------------

    def _single_step(
        self,
        strain: "jnp.ndarray",
        plastic_strain_prev: "jnp.ndarray",
        eqps_prev: "jnp.ndarray",
    ) -> tuple["jnp.ndarray", "jnp.ndarray", "jnp.ndarray"]:
        """Viscoplastic update for a single integration point.

        Uses an elastic predictor (trial stress from the previous
        plastic strain) and a one-step explicit evaluation of the
        power-law slip rates at the trial state. Good enough for
        Newton-driven incremental loading at moderate dt and high n.
        """
        strain_el_trial = strain - plastic_strain_prev
        sigma_trial = self.C @ strain_el_trial

        # Resolved shear stresses on each slip system: tau_a = P_a : sigma.
        # Our Schmid rows carry the engineering-shear factor of 2 on the
        # off-diagonal components, so a plain dot product gives the
        # correct resolved shear against a Voigt stress vector.
        tau = self.schmid @ sigma_trial  # (n_sys,)

        tau_c = self.tau0 + self.H * eqps_prev  # scalar (same for all)
        tau_c_safe = jnp.maximum(tau_c, 1.0e-12)

        abs_tau = jnp.abs(tau)
        ratio = abs_tau / tau_c_safe
        gamma_dot = (
            self.gamma_dot_0 * ratio ** self.n_power * jnp.sign(tau)
        )

        # Plastic strain increment: sum_a (dt * gamma_dot_a) * P_a (Voigt)
        dep = self.dt * (gamma_dot @ self.schmid)  # (6,)

        plastic_strain_new = plastic_strain_prev + dep

        # Approximate equivalent plastic strain update: accumulated
        # total slip scaled by sqrt(2/3). Fine for linear self-hardening.
        eqps_new = (
            eqps_prev
            + self.dt * jnp.sum(jnp.abs(gamma_dot)) * jnp.sqrt(2.0 / 3.0)
        )

        strain_el_new = strain - plastic_strain_new
        stress = self.C @ strain_el_new
        return stress, plastic_strain_new, eqps_new

    # -- Protocol surface ---------------------------------------------------

    def stress_stateful(
        self,
        strain_batch: "jnp.ndarray",
        plastic_strain_batch: "jnp.ndarray",
        eqps_batch: "jnp.ndarray",
    ) -> tuple["jnp.ndarray", "jnp.ndarray", "jnp.ndarray"]:
        """Batched viscoplastic update: (stress, new_eps_p, new_eqps)."""
        return jax.vmap(self._single_step)(
            strain_batch, plastic_strain_batch, eqps_batch
        )

    def stress(self, strain: "jnp.ndarray") -> "jnp.ndarray":
        """Stateless stress (treats prior plastic state as zero)."""
        zeros = jnp.zeros_like(strain)
        zeros_eqps = jnp.zeros(strain.shape[:-1])
        stress, _, _ = self.stress_stateful(strain, zeros, zeros_eqps)
        return stress

    def tangent(self, strain: "jnp.ndarray") -> "jnp.ndarray":
        """Algorithmic tangent via forward-mode AD on stateless stress."""
        def single_stress(e: "jnp.ndarray") -> "jnp.ndarray":
            return self.stress(e[None, :])[0]

        jac_fn = jax.vmap(jax.jacfwd(single_stress))
        return jac_fn(strain)
