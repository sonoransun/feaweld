"""Tests for the FEniCSx (DOLFINx) solver backend.

The module is written so it imports cleanly with DOLFINx absent; the
unmarked tests below verify that contract and the element-type coverage of
the cell map.  The heavier solve tests are gated behind
``@pytest.mark.requires_fenics`` and a per-test ``importorskip("dolfinx")``
so they are collected (not module-skipped) but only execute where DOLFINx
is installed.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from feaweld.core.materials import Material
from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    LoadCase,
    LoadType,
)
from feaweld.solver import fenics_backend
from feaweld.solver.fenics_backend import (
    FEniCSBackend,
    _femesh_to_dolfinx,
    _require_dolfinx,
)


def _material() -> Material:
    return Material(
        name="FenicsSteel",
        density=7850.0,
        elastic_modulus={20.0: 200000.0},
        poisson_ratio={20.0: 0.3},
        yield_strength={20.0: 250.0},
        ultimate_strength={20.0: 400.0},
        thermal_conductivity={20.0: 50.0},
        specific_heat={20.0: 500.0},
        thermal_expansion={20.0: 12e-6},
    )


# ---------------------------------------------------------------------------
# Unmarked: importable and correct without DOLFINx installed
# ---------------------------------------------------------------------------

def test_module_imports_without_dolfinx():
    """The backend module and class import even when DOLFINx is missing."""
    assert fenics_backend.FEniCSBackend is not None
    assert issubclass(FEniCSBackend, fenics_backend.SolverBackend)


def test_require_dolfinx_error_message():
    """When DOLFINx is absent, the guard raises with an install hint."""
    try:
        import dolfinx  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"feaweld\[fenics\]"):
            _require_dolfinx()
    else:
        pytest.skip("dolfinx is installed; nothing to assert about its absence")


def test_cell_map_covers_every_element_type():
    """Every ElementType member is mapped in ``_femesh_to_dolfinx``."""
    source = inspect.getsource(_femesh_to_dolfinx)
    missing = [m.name for m in ElementType if f"ElementType.{m.name}" not in source]
    assert missing == [], f"cell_map is missing element types: {missing}"


# ---------------------------------------------------------------------------
# Marked: real DOLFINx solves (skipped where dolfinx is unavailable)
# ---------------------------------------------------------------------------

def _assert_solved(results, mesh):
    # Displacement is returned in INPUT node order: one row per input node,
    # and input node 0 (in the fully fixed 'bottom' set) sits at row 0 with
    # ~zero displacement — proving the reorder undoes DOLFINx's vertex
    # permutation rather than returning dof-order data.
    assert results.displacement is not None
    assert results.displacement.shape == (mesh.n_nodes, 3)
    np.testing.assert_allclose(results.displacement[0], 0.0, atol=1e-9)
    assert np.any(np.abs(results.displacement) > 0.0)
    # Stress is nodal (input node order), (n_nodes, 6), per the solver-agnostic
    # FEAResults contract shared with the CalculiX backend.
    assert results.stress is not None
    assert results.stress.values.shape == (mesh.n_nodes, 6)
    assert results.stress.location == "nodes"


@pytest.mark.requires_fenics
def test_solve_static_traction(simple_plate_mesh):
    pytest.importorskip("dolfinx")
    lc = LoadCase(
        name="traction",
        loads=[BoundaryCondition(
            "top", LoadType.FORCE, np.array([100.0]),
            direction=np.array([0.0, 1.0, 0.0]),
        )],
        constraints=[BoundaryCondition(
            "bottom", LoadType.DISPLACEMENT, np.array([0.0, 0.0, 0.0]),
        )],
    )
    _assert_solved(FEniCSBackend().solve_static(simple_plate_mesh, _material(), lc), simple_plate_mesh)


@pytest.mark.requires_fenics
def test_solve_static_pressure(simple_plate_mesh):
    pytest.importorskip("dolfinx")
    lc = LoadCase(
        name="pressure",
        loads=[BoundaryCondition("top", LoadType.PRESSURE, np.array([5.0]))],
        constraints=[BoundaryCondition(
            "bottom", LoadType.DISPLACEMENT, np.array([0.0, 0.0, 0.0]),
        )],
    )
    _assert_solved(FEniCSBackend().solve_static(simple_plate_mesh, _material(), lc), simple_plate_mesh)


@pytest.mark.requires_fenics
def test_solve_static_temperature_scalar(simple_plate_mesh):
    pytest.importorskip("dolfinx")
    lc = LoadCase(
        name="thermal",
        loads=[BoundaryCondition("all", LoadType.TEMPERATURE, np.array([120.0]))],
        constraints=[BoundaryCondition(
            "bottom", LoadType.DISPLACEMENT, np.array([0.0, 0.0, 0.0]),
        )],
    )
    _assert_solved(FEniCSBackend().solve_static(simple_plate_mesh, _material(), lc), simple_plate_mesh)


@pytest.mark.requires_fenics
def test_solve_static_per_node_forces(simple_plate_mesh):
    pytest.importorskip("dolfinx")
    top = simple_plate_mesh.node_sets["top"]
    forces = np.tile([0.0, 50.0, 0.0], (len(top), 1))
    lc = LoadCase(
        name="nodal_forces",
        loads=[BoundaryCondition("top", LoadType.FORCE, forces)],
        constraints=[BoundaryCondition(
            "bottom", LoadType.DISPLACEMENT, np.array([0.0, 0.0, 0.0]),
        )],
    )
    _assert_solved(FEniCSBackend().solve_static(simple_plate_mesh, _material(), lc), simple_plate_mesh)


@pytest.mark.requires_fenics
def test_solve_static_per_node_displacement_bc(simple_plate_mesh):
    pytest.importorskip("dolfinx")
    top = simple_plate_mesh.node_sets["top"]
    imposed = np.tile([0.0, 0.02, 0.0], (len(top), 1))
    lc = LoadCase(
        name="disp_bc",
        constraints=[
            BoundaryCondition(
                "bottom", LoadType.DISPLACEMENT, np.array([0.0, 0.0, 0.0]),
            ),
            BoundaryCondition("top", LoadType.DISPLACEMENT, imposed),
        ],
    )
    _assert_solved(FEniCSBackend().solve_static(simple_plate_mesh, _material(), lc), simple_plate_mesh)
