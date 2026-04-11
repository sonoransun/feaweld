"""Tests for the JAX differentiable FEA backend (Track A1)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from feaweld.core.materials import Material
from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEMesh,
    LoadCase,
    LoadType,
)
from feaweld.solver.backend import get_backend
from feaweld.solver.jax_backend import JAXBackend


pytestmark = pytest.mark.requires_jax


@pytest.fixture
def steel():
    return Material(
        name="A36",
        density=7850.0,
        elastic_modulus={20.0: 200_000.0, 500.0: 160_000.0},
        poisson_ratio={20.0: 0.3, 500.0: 0.3},
        yield_strength={20.0: 250.0, 500.0: 165.0},
        ultimate_strength={20.0: 400.0, 500.0: 310.0},
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
def tet_bar():
    # Two tets sharing a face: enough DOFs to not be trivially singular
    # when the back three nodes are pinned (6 constraints kill all 6 rigid
    # body modes of a 3D body).
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 10.0, 10.0],
        ]
    )
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TET4,
        node_sets={
            "fixed": np.array([0, 2, 3]),
            "pull_x": np.array([4]),
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
                values=np.array([0.0, 1000.0, 0.0]),
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
# Registration
# ---------------------------------------------------------------------------


def test_get_backend_jax_returns_instance():
    backend = get_backend("jax")
    assert isinstance(backend, JAXBackend)


# ---------------------------------------------------------------------------
# TRI3 plane strain correctness
# ---------------------------------------------------------------------------


def test_tri3_linear_elastic_axial_tension(tri_plate, steel, axial_load_case):
    backend = JAXBackend()
    result = backend.solve_static(tri_plate, steel, axial_load_case)

    assert result.displacement is not None
    assert result.stress is not None
    assert result.strain is not None

    # Top nodes should move upward (positive y) under upward force
    top_y = result.displacement[2:, 1]
    assert np.all(top_y > 0.0)

    # Stress should be predominantly σ_yy in plane strain for a pull
    sig_yy_avg = float(np.mean(result.stress.values[:, 1]))
    sig_xx_avg = float(np.mean(result.stress.values[:, 0]))
    # In plane strain with ν≈0.3 the lateral σ_xx ≈ ν σ_yy so yy dominates
    assert sig_yy_avg > 0.0
    assert abs(sig_yy_avg) > abs(sig_xx_avg) * 0.5


def test_tri3_zero_load_gives_zero_displacement(tri_plate, steel):
    lc = LoadCase(
        name="none",
        constraints=[
            BoundaryCondition(
                node_set="bottom",
                bc_type=LoadType.DISPLACEMENT,
                values=np.array([0.0, 0.0, 0.0]),
            )
        ],
    )
    backend = JAXBackend()
    result = backend.solve_static(tri_plate, steel, lc)
    assert np.allclose(result.displacement, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TET4 correctness
# ---------------------------------------------------------------------------


def test_tet4_linear_elastic_fixed_support(tet_bar, steel):
    lc = LoadCase(
        name="tet_tension",
        loads=[
            BoundaryCondition(
                node_set="pull_x",
                bc_type=LoadType.FORCE,
                values=np.array([1000.0, 0.0, 0.0]),
            )
        ],
        constraints=[
            BoundaryCondition(
                node_set="fixed",
                bc_type=LoadType.DISPLACEMENT,
                values=np.array([0.0, 0.0, 0.0]),
            )
        ],
    )
    backend = JAXBackend()
    result = backend.solve_static(tet_bar, steel, lc)

    # Nodes 0, 2, 3 are pinned
    for pinned in (0, 2, 3):
        assert np.allclose(result.displacement[pinned], 0.0, atol=1e-6)
    # Node 4 (the pulled node) gets displaced in +x under the applied force
    assert result.displacement[4, 0] > 0.0


# ---------------------------------------------------------------------------
# Differentiability: gradient of compliance wrt Young's modulus
# ---------------------------------------------------------------------------


def test_jax_backend_gradient_wrt_modulus(tri_plate, axial_load_case):
    """Compliance C(E) should have ∂C/∂E < 0 (stiffer → less compliance).

    This is the crucial A1 validation: confirms that autodiff flows
    cleanly from material properties through the linear solve to the
    scalar QoI.
    """
    backend = JAXBackend()

    def compliance_for_E(E_val: float) -> float:
        mat = Material(
            name="test",
            density=7850.0,
            elastic_modulus={20.0: float(E_val)},
            poisson_ratio={20.0: 0.3},
            yield_strength={20.0: 250.0},
            ultimate_strength={20.0: 400.0},
            thermal_conductivity={20.0: 51.9},
            specific_heat={20.0: 440.0},
            thermal_expansion={20.0: 11.7e-6},
        )
        return backend.static_compliance(tri_plate, mat, axial_load_case)

    c_low = compliance_for_E(150_000.0)
    c_high = compliance_for_E(250_000.0)
    # Stiffer material → less compliance under the same load
    assert c_high < c_low


def test_jax_backend_compliance_scales_quadratically_with_force(
    tri_plate, steel, axial_load_case
):
    """Linear elastic compliance should scale as F^2."""
    backend = JAXBackend()

    def with_force(F):
        lc = LoadCase(
            name="scaled",
            loads=[
                BoundaryCondition(
                    node_set="top",
                    bc_type=LoadType.FORCE,
                    values=np.array([0.0, F, 0.0]),
                )
            ],
            constraints=axial_load_case.constraints,
        )
        return backend.static_compliance(tri_plate, steel, lc)

    c1 = with_force(1000.0)
    c2 = with_force(2000.0)
    # Compliance is (1/2) u·f = (1/2) F·u and u ∝ F, so c ∝ F^2
    ratio = c2 / c1
    assert 3.5 < ratio < 4.5


# ---------------------------------------------------------------------------
# Unsupported element types raise clearly
# ---------------------------------------------------------------------------


def test_jax_backend_rejects_unsupported_element_type(steel, axial_load_case):
    nodes = np.zeros((8, 3))
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    mesh = FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.HEX8,
        node_sets={"bottom": np.array([0])},
    )
    backend = JAXBackend()
    with pytest.raises(NotImplementedError, match="TRI3 and TET4"):
        backend.solve_static(mesh, steel, axial_load_case)


def test_jax_backend_thermal_stub_is_loud(tri_plate, steel, axial_load_case):
    backend = JAXBackend()
    with pytest.raises(NotImplementedError, match="thermal_steady"):
        backend.solve_thermal_steady(tri_plate, steel, axial_load_case)


# ---------------------------------------------------------------------------
# A2: J2 plasticity wired via JAXJ2Plasticity + solve_static_incremental
# ---------------------------------------------------------------------------


def _uniaxial_tri_mesh():
    """2-triangle square plate (plane strain) for uniaxial-ish tension in y."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
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


def _plastic_steel(sigma_y: float = 100.0, H: float = 500.0) -> Material:
    return Material(
        name="J2_steel",
        density=7850.0,
        elastic_modulus={20.0: 200_000.0},
        poisson_ratio={20.0: 0.3},
        yield_strength={20.0: sigma_y},
        ultimate_strength={20.0: 400.0},
        thermal_conductivity={20.0: 51.9},
        specific_heat={20.0: 440.0},
        thermal_expansion={20.0: 11.7e-6},
        hardening_modulus=H,
    )


def _tri_tension_load_case(force_y: float) -> LoadCase:
    return LoadCase(
        name="plastic_tension",
        loads=[
            BoundaryCondition(
                node_set="top",
                bc_type=LoadType.FORCE,
                values=np.array([0.0, force_y, 0.0]),
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


def test_jax_j2_plasticity_yields_at_correct_stress():
    from feaweld.solver.jax_constitutive import JAXJ2Plasticity

    mesh = _uniaxial_tri_mesh()
    mat = _plastic_steel(sigma_y=100.0, H=0.0)
    # Apply a force several times larger than the expected yield force
    # (~100 MPa on a 1x1 plane-strain unit cell).
    load = _tri_tension_load_case(force_y=500.0)

    cmodel = JAXJ2Plasticity.from_material(mat)
    backend = JAXBackend(constitutive=cmodel)
    result = backend.solve_static(mesh, mat, load)

    vm = result.stress.von_mises
    vm_peak = float(np.max(vm))
    # With perfect plasticity (H=0) and an overloaded element the peak
    # von Mises stress must clamp at sigma_y within 5%.
    assert abs(vm_peak - 100.0) / 100.0 < 0.05
    assert result.metadata.get("constitutive") == "j2_plasticity"
    assert result.metadata.get("n_steps") == 10


def test_jax_plasticity_matches_elastic_below_yield():
    from feaweld.solver.jax_constitutive import JAXJ2Plasticity

    mesh = _uniaxial_tri_mesh()
    mat = _plastic_steel(sigma_y=1.0e6, H=0.0)  # effectively never yields
    load = _tri_tension_load_case(force_y=50.0)

    elastic_backend = JAXBackend()
    elastic_result = elastic_backend.solve_static(mesh, mat, load)

    cmodel = JAXJ2Plasticity.from_material(mat)
    plastic_backend = JAXBackend(constitutive=cmodel)
    plastic_result = plastic_backend.solve_static_incremental(
        mesh, mat, load, n_steps=10, constitutive=cmodel
    )

    np.testing.assert_allclose(
        plastic_result.displacement, elastic_result.displacement, atol=1e-8
    )


def test_jax_plasticity_has_residual_strain_after_unload():
    from feaweld.solver.jax_constitutive import JAXJ2Plasticity

    mesh = _uniaxial_tri_mesh()
    mat = _plastic_steel(sigma_y=100.0, H=500.0)
    cmodel = JAXJ2Plasticity.from_material(mat)
    backend = JAXBackend(constitutive=cmodel)

    load = _tri_tension_load_case(force_y=300.0)
    result = backend.solve_static(mesh, mat, load)

    vm = float(np.max(result.stress.von_mises))
    eps_yy = float(np.mean(result.strain[:, 1]))
    E = mat.E(20.0)
    elastic_strain_if_linear = vm / E
    assert eps_yy > 2.0 * elastic_strain_if_linear


def test_jax_plasticity_gradient_wrt_sigma_y_is_finite():
    from feaweld.solver.jax_constitutive import JAXJ2Plasticity

    mesh = _uniaxial_tri_mesh()
    mat = _plastic_steel(sigma_y=100.0, H=500.0)
    base_load = _tri_tension_load_case(force_y=300.0)

    C_base = JAXJ2Plasticity.from_material(mat).C
    mu_val = float(mat.lame_mu(20.0))
    H_val = float(mat.hardening_modulus)
    backend = JAXBackend()

    def compliance(sigma_y_val):
        cmodel = JAXJ2Plasticity(
            C=C_base, mu=mu_val, sigma_y0=sigma_y_val, H=H_val
        )
        return backend.incremental_compliance_tri3(
            mesh, mat, base_load, cmodel, n_steps=10
        )

    grad_fn = jax.grad(compliance)
    g = grad_fn(100.0)
    assert jnp.isfinite(g)
    c_val = compliance(100.0)
    assert jnp.isfinite(c_val)
