"""Tests for the Meric-Cailletaud FCC crystal plasticity model (A3)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from feaweld.core.materials import Material
from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEMesh,
    LoadCase,
    LoadType,
)
from feaweld.solver.jax_backend import JAXBackend
from feaweld.solver.jax_crystal_plasticity import (
    FCC_SLIP_DIRECTIONS,
    FCC_SLIP_NORMALS,
    JAXCrystalPlasticity,
    schmid_tensors,
)


pytestmark = pytest.mark.requires_jax


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def steel():
    return Material(
        name="A36",
        density=7850.0,
        elastic_modulus={20.0: 200_000.0},
        poisson_ratio={20.0: 0.3},
        yield_strength={20.0: 250.0},
        ultimate_strength={20.0: 400.0},
        thermal_conductivity={20.0: 51.9},
        specific_heat={20.0: 440.0},
        thermal_expansion={20.0: 11.7e-6},
        hardening_modulus=500.0,
    )


@pytest.fixture
def tri_plate():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    )
    elements = np.array([[0, 1, 2], [0, 2, 3]])
    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
        node_sets={
            "bottom": np.array([0, 1]),
            "top": np.array([2, 3]),
        },
    )


@pytest.fixture
def axial_load_case():
    return LoadCase(
        name="tension",
        loads=[
            BoundaryCondition(
                node_set="top",
                bc_type=LoadType.FORCE,
                values=np.array([0.0, 5000.0, 0.0]),
            )
        ],
        constraints=[
            BoundaryCondition(
                node_set="bottom",
                bc_type=LoadType.DISPLACEMENT,
                values=np.array([0.0, 0.0, 0.0]),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Slip-system geometry
# ---------------------------------------------------------------------------


def test_schmid_tensors_orthogonality():
    # Slip direction must be perpendicular to slip normal on every FCC system.
    assert FCC_SLIP_NORMALS.shape == (12, 3)
    assert FCC_SLIP_DIRECTIONS.shape == (12, 3)
    dots = np.sum(FCC_SLIP_NORMALS * FCC_SLIP_DIRECTIONS, axis=1)
    assert np.allclose(dots, 0.0, atol=1e-12)

    norms_n = np.linalg.norm(FCC_SLIP_NORMALS, axis=1)
    norms_d = np.linalg.norm(FCC_SLIP_DIRECTIONS, axis=1)
    assert np.allclose(norms_n, 1.0, atol=1e-12)
    assert np.allclose(norms_d, 1.0, atol=1e-12)

    P = schmid_tensors(FCC_SLIP_NORMALS, FCC_SLIP_DIRECTIONS)
    assert P.shape == (12, 6)


# ---------------------------------------------------------------------------
# Constitutive behaviour
# ---------------------------------------------------------------------------


def test_crystal_plasticity_elastic_at_zero_strain(steel):
    model = JAXCrystalPlasticity.from_material(steel)
    strain = jnp.zeros((1, 6))
    ep = jnp.zeros((1, 6))
    eqps = jnp.zeros((1,))
    stress, ep_new, eqps_new = model.stress_stateful(strain, ep, eqps)
    assert np.allclose(np.asarray(stress), 0.0, atol=1e-10)
    assert np.allclose(np.asarray(ep_new), 0.0, atol=1e-10)
    assert np.allclose(np.asarray(eqps_new), 0.0, atol=1e-10)


def test_crystal_plasticity_activates_above_crss(steel):
    # Apply an elastic uniaxial strain well above the CRSS threshold.
    # sigma_yy = E * eps_yy; choose eps_yy so sigma_yy >> tau0*sqrt(3).
    model = JAXCrystalPlasticity.from_material(steel, n_power=10)
    eps = 0.01  # 1% axial strain -> ~2000 MPa trial stress (way above CRSS)
    strain = jnp.asarray([[0.0, eps, 0.0, 0.0, 0.0, 0.0]])
    ep0 = jnp.zeros((1, 6))
    eqps0 = jnp.zeros((1,))
    _stress, ep_new, eqps_new = model.stress_stateful(strain, ep0, eqps0)
    assert float(eqps_new[0]) > 0.0
    assert float(jnp.sum(jnp.abs(ep_new))) > 0.0


def test_crystal_plasticity_hardening_increases_yield(steel):
    # Stress-relaxation style probe: hold the total strain fixed and
    # take two viscoplastic sub-steps. Linear self-hardening must make
    # the second sub-step's plastic increment strictly smaller than the
    # first (CRSS has grown, ratio |tau|/tau_c has shrunk).
    # A very small dt keeps the explicit power-law update stable.
    model = JAXCrystalPlasticity.from_material(
        steel, n_power=10, H=1.0e5, dt=1.0e-6
    )
    eps = 3.0e-3  # above yield but moderate, so we stay in the stable regime
    strain = jnp.asarray([[0.0, eps, 0.0, 0.0, 0.0, 0.0]])
    ep0 = jnp.zeros((1, 6))
    eqps0 = jnp.zeros((1,))

    _s1, ep1, eqps1 = model.stress_stateful(strain, ep0, eqps0)
    _s2, ep2, eqps2 = model.stress_stateful(strain, ep1, eqps1)

    deqps1 = float(eqps1[0] - eqps0[0])
    deqps2 = float(eqps2[0] - eqps1[0])

    assert deqps1 > 0.0
    assert deqps2 > 0.0
    assert deqps2 < deqps1


# ---------------------------------------------------------------------------
# Backend integration
# ---------------------------------------------------------------------------


def test_crystal_plasticity_integrates_with_jax_backend(
    tri_plate, steel, axial_load_case
):
    model = JAXCrystalPlasticity.from_material(steel, n_power=10)
    backend = JAXBackend(constitutive=model)
    result = backend.solve_static_incremental(
        tri_plate,
        steel,
        axial_load_case,
        constitutive=model,
        n_steps=3,
    )
    assert result is not None
    assert result.displacement is not None
    assert np.all(np.isfinite(result.displacement))
    assert result.stress is not None
    assert np.all(np.isfinite(result.stress.values))
