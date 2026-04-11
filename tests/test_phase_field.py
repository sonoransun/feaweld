"""Tests for variational phase-field fracture (Track A4)."""

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
from feaweld.fracture import FractureResult, PhaseFieldConfig, solve_phase_field


pytestmark = pytest.mark.requires_jax


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _structured_tri_plate(
    lx: float,
    ly: float,
    nx: int,
    ny: int,
) -> FEMesh:
    """Structured triangular plate: nx x ny rectangles, each split into 2 tris.

    Node sets exposed: ``bottom``, ``top``, ``left``, ``right``, ``left_edge_bottom``.
    """
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    nodes = np.array(
        [[x, y, 0.0] for y in ys for x in xs], dtype=np.float64
    )
    elements: list[list[int]] = []

    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    for j in range(ny):
        for i in range(nx):
            n0 = nid(i, j)
            n1 = nid(i + 1, j)
            n2 = nid(i + 1, j + 1)
            n3 = nid(i, j + 1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

    node_sets = {
        "bottom": np.array([nid(i, 0) for i in range(nx + 1)], dtype=np.int64),
        "top": np.array([nid(i, ny) for i in range(nx + 1)], dtype=np.int64),
        "left": np.array([nid(0, j) for j in range(ny + 1)], dtype=np.int64),
        "right": np.array([nid(nx, j) for j in range(ny + 1)], dtype=np.int64),
        "left_edge_bottom": np.array(
            [nid(0, j) for j in range(ny // 2 + 1)], dtype=np.int64
        ),
    }
    return FEMesh(
        nodes=nodes,
        elements=np.array(elements, dtype=np.int64),
        element_type=ElementType.TRI3,
        node_sets=node_sets,
    )


@pytest.fixture
def steel() -> Material:
    return Material(
        name="TestSteel",
        density=7850.0,
        elastic_modulus={20.0: 210_000.0},
        poisson_ratio={20.0: 0.3},
        yield_strength={20.0: 250.0},
        ultimate_strength={20.0: 400.0},
        thermal_conductivity={20.0: 50.0},
        specific_heat={20.0: 450.0},
        thermal_expansion={20.0: 12.0e-6},
    )


@pytest.fixture
def fine_plate() -> FEMesh:
    # 10x5 rectangles -> avg edge about 0.22, well under l0/4 = 0.125 only
    # if l0=0.5 — we override l0 in tests accordingly.
    return _structured_tri_plate(lx=1.0, ly=0.5, nx=10, ny=5)


@pytest.fixture
def coarse_plate() -> FEMesh:
    return _structured_tri_plate(lx=1.0, ly=0.5, nx=2, ny=2)


def _tension_load_case(force_y: float = 10.0) -> LoadCase:
    return LoadCase(
        name="tension",
        loads=[
            BoundaryCondition(
                node_set="top",
                bc_type=LoadType.FORCE,
                values=np.array([0.0, force_y]),
            ),
        ],
        constraints=[
            BoundaryCondition(
                node_set="bottom",
                bc_type=LoadType.DISPLACEMENT,
                values=np.array([0.0, 0.0]),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Mesh density guard
# ---------------------------------------------------------------------------


def test_mesh_density_guard(coarse_plate: FEMesh, steel: Material) -> None:
    """A coarse mesh with tiny l0 must be rejected at pre-check."""
    cfg = PhaseFieldConfig(l0=0.01, min_mesh_ratio=4.0, n_load_steps=1)
    with pytest.raises(ValueError, match="Mesh too coarse"):
        solve_phase_field(coarse_plate, steel, _tension_load_case(), cfg)


# ---------------------------------------------------------------------------
# Zero load -> zero damage
# ---------------------------------------------------------------------------


def test_zero_load_gives_zero_damage(fine_plate: FEMesh, steel: Material) -> None:
    cfg = PhaseFieldConfig(
        l0=0.5,
        Gc=2.7,
        n_load_steps=2,
        max_load=0.0,  # zero total load
        max_staggered_iters=5,
    )
    lc = _tension_load_case(force_y=0.0)
    result = solve_phase_field(fine_plate, steel, lc, cfg)
    assert isinstance(result, FractureResult)
    assert result.damage.shape == (fine_plate.n_nodes,)
    assert np.allclose(result.damage, 0.0, atol=1e-10)
    # Reaction history should also be zero.
    assert np.allclose(result.reaction_history, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Monotone damage
# ---------------------------------------------------------------------------


def test_damage_monotonic(fine_plate: FEMesh, steel: Material) -> None:
    cfg = PhaseFieldConfig(
        l0=0.5,
        Gc=0.05,  # small Gc so damage develops quickly
        n_load_steps=5,
        max_load=1.0,
        staggered_tol=1e-3,
        max_staggered_iters=20,
    )
    lc = _tension_load_case(force_y=200.0)
    result = solve_phase_field(fine_plate, steel, lc, cfg)

    hist = np.stack(result.damage_history, axis=0)  # (n_steps, n_nodes)
    # Non-decreasing per node across steps.
    diffs = np.diff(hist, axis=0)
    assert np.all(diffs >= -1e-10), (
        f"Damage is not monotone: min diff {diffs.min()}"
    )


# ---------------------------------------------------------------------------
# Bourdin softening signature
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dcb_softening_appears(steel: Material) -> None:
    """Pre-notched plate pulled under displacement control shows softening.

    The qualitative Bourdin signature: as damage grows under a prescribed
    displacement on the top edge, the reaction force (K u evaluated at the
    loaded DOFs) passes through a maximum and then decreases.  We drive
    the top edge in y-displacement (proportional to the load step
    multiplier) while fixing the bottom, and track the row-sum of (K u)
    at the top nodes.
    """
    nx, ny = 16, 8
    mesh = _structured_tri_plate(lx=2.0, ly=1.0, nx=nx, ny=ny)

    # Pre-existing edge notch: damage = 1 at a few left-edge nodes near mid.
    initial_damage = np.zeros(mesh.n_nodes, dtype=np.float64)
    mid_y_start = (ny // 2) - 1
    mid_y_end = (ny // 2) + 1
    for j in range(mid_y_start, mid_y_end + 1):
        initial_damage[j * (nx + 1) + 0] = 1.0
        initial_damage[j * (nx + 1) + 1] = 1.0

    cfg = PhaseFieldConfig(
        l0=0.25,
        Gc=0.005,
        n_load_steps=20,
        max_load=0.01,  # max top-edge displacement (mm)
        staggered_tol=1e-3,
        max_staggered_iters=15,
        min_mesh_ratio=1.5,  # plate has h ~ 0.125, l0/h ~ 2
    )

    # Displacement control: prescribe u_y on the top edge.  We also need
    # a tiny force BC so the solver has a loaded DOF set to track — use a
    # minuscule force at the top to mark it and then rely on the imposed
    # displacement to actually drive the problem.
    lc = LoadCase(
        name="dcb_displacement",
        loads=[
            BoundaryCondition(
                node_set="top",
                bc_type=LoadType.FORCE,
                values=np.array([0.0, 1e-12]),
            ),
        ],
        constraints=[
            BoundaryCondition(
                node_set="bottom",
                bc_type=LoadType.DISPLACEMENT,
                values=np.array([0.0, 0.0]),
            ),
            BoundaryCondition(
                node_set="top",
                bc_type=LoadType.DISPLACEMENT,
                values=np.array([0.0, 1.0]),  # will be scaled per step
            ),
        ],
    )

    result = solve_phase_field(
        mesh, steel, lc, cfg, initial_damage=initial_damage
    )

    reactions = np.abs(result.reaction_history)
    assert reactions.size == 20
    assert reactions.max() > reactions[-1], (
        f"Expected post-peak softening: max {reactions.max():.3e}, "
        f"last {reactions[-1]:.3e}\nhistory={reactions}"
    )
    # Damage should have grown beyond the initial notch somewhere.
    assert result.damage.max() >= initial_damage.max() - 1e-12
    assert result.damage.sum() > initial_damage.sum()
