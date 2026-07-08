"""Tests for global-local submodeling and its CalculiX boundary transfer.

Covers the interpolation fallback, a real (fake-backend) cut-boundary solve,
the ``backend="auto"`` path on a machine with no FE solver, region
extraction, uniform refinement, and the per-node CalculiX ``*BOUNDARY`` /
``*CLOAD`` / ``*TEMPERATURE`` writers.  No FEniCSx / CalculiX binary required.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from feaweld.core.materials import Material
from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEAResults,
    FEMesh,
    LoadCase,
    LoadType,
    StressField,
)
from feaweld.singularity.submodeling import (
    SubmodelSolver,
    _refine_mesh,
    create_submodel_region,
    solve_submodel,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _material() -> Material:
    return Material(
        name="SubmodelSteel",
        density=7850.0,
        elastic_modulus={20.0: 200000.0},
        poisson_ratio={20.0: 0.3},
        yield_strength={20.0: 250.0},
        ultimate_strength={20.0: 400.0},
        thermal_conductivity={20.0: 50.0},
        specific_heat={20.0: 500.0},
        thermal_expansion={20.0: 12e-6},
    )


@pytest.fixture
def parent_results():
    """A small solved tet mesh (displacement + stress) to submodel from."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [5.0, 10.0, 0.0],
        [5.0, 5.0, 10.0],
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 10.0],
    ])
    elements = np.array([[0, 1, 2, 3], [0, 1, 3, 4]], dtype=np.int64)
    mesh = FEMesh(nodes=nodes, elements=elements, element_type=ElementType.TET4)

    disp = np.zeros((mesh.n_nodes, 3))
    disp[:, 1] = np.linspace(0.0, 0.01, mesh.n_nodes)
    stress = np.zeros((mesh.n_nodes, 6))
    stress[:, 1] = np.linspace(50.0, 200.0, mesh.n_nodes)

    return FEAResults(
        mesh=mesh,
        displacement=disp,
        stress=StressField(values=stress),
    )


def _center_radius(parent_results):
    center = parent_results.mesh.nodes.mean(axis=0)
    return center, 8.0


# ---------------------------------------------------------------------------
# solve_submodel: fallback and real-solve paths
# ---------------------------------------------------------------------------

def test_submodel_interpolation_fallback(parent_results):
    """With no backend the parent field is interpolated, with a warning."""
    center, radius = _center_radius(parent_results)
    _, region_nodes = create_submodel_region(parent_results.mesh, center, radius)

    with pytest.warns(RuntimeWarning):
        result = solve_submodel(
            parent_results, center, radius, material=None, backend=None,
        )

    assert result.metadata["submodel_solve"] == "interpolated"
    # The refined submodel is denser than the extracted parent region.
    assert result.mesh.n_nodes > len(region_nodes)


def test_submodel_backend_solve_records_cut_boundary(parent_results, fake_backend):
    """An injected backend gets a per-node displacement BC on 'boundary'."""
    center, radius = _center_radius(parent_results)

    result = solve_submodel(
        parent_results, center, radius, material=_material(), backend=fake_backend,
    )

    static_calls = fake_backend.calls_to("solve_static")
    assert len(static_calls) == 1
    lc = static_calls[0]["load_case"]
    assert lc.constraints[0].node_set == "boundary"
    assert np.asarray(lc.constraints[0].values).ndim == 2

    assert result.metadata["submodel_solve"] == "FakeBackend"
    assert np.isfinite(result.metadata["boundary_bc_residual"])


def test_submodel_auto_backend_without_solvers(parent_results):
    """``backend='auto'`` degrades gracefully when no FE solver runs."""
    center, radius = _center_radius(parent_results)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_submodel(
            parent_results, center, radius, material=_material(), backend="auto",
        )

    # Don't assert which path won — only that it returned a labelled result.
    assert isinstance(result.metadata["submodel_solve"], str)
    assert result.mesh.n_nodes > 0


# ---------------------------------------------------------------------------
# Region extraction and refinement
# ---------------------------------------------------------------------------

def test_create_submodel_region_invariants(parent_results):
    center, radius = _center_radius(parent_results)
    elem_ids, node_ids = create_submodel_region(parent_results.mesh, center, radius)

    # Every returned node lies within the extraction sphere.
    coords = parent_results.mesh.nodes[node_ids]
    assert np.all(np.linalg.norm(coords - center, axis=1) <= radius + 1e-9)

    # Every returned element touches at least one region node.
    region = set(node_ids.tolist())
    for eid in elem_ids:
        conn = parent_results.mesh.elements[eid]
        assert any(int(n) in region for n in conn)


@pytest.mark.parametrize("factor,ratio", [(2, 4), (4, 16)])
def test_refine_mesh_tri_bisection(factor, ratio):
    """Each bisection quadruples triangle count (4^n_bisections total)."""
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
    ])
    elements = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)

    refined_nodes, refined_elems = _refine_mesh(nodes, elements, factor)
    assert len(refined_elems) == len(elements) * ratio
    assert len(refined_nodes) > len(nodes)


# ---------------------------------------------------------------------------
# CalculiX per-node writers (no ccx binary needed)
# ---------------------------------------------------------------------------

def _ccx_mesh() -> FEMesh:
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 1.0, 1.0],
    ])
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    return FEMesh(
        nodes=nodes, elements=elements, element_type=ElementType.TET4,
        node_sets={"boundary": np.array([0, 1])},
    )


def test_ccx_per_node_displacement_boundary(tmp_path):
    from feaweld.solver.calculix_backend import generate_inp

    mesh = _ccx_mesh()
    values = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    lc = LoadCase(
        name="disp",
        constraints=[BoundaryCondition("boundary", LoadType.DISPLACEMENT, values)],
    )

    content = generate_inp(
        mesh, _material(), lc, tmp_path / "disp.inp", analysis="static",
    ).read_text()

    assert "*BOUNDARY" in content
    # Node 1 (1-based), dof 1, value 0.1 ; node 2, dof 3, value 0.6.
    assert "1, 1, 1, 0.1" in content
    assert "2, 3, 3, 0.6" in content


def test_ccx_per_node_cload(tmp_path):
    from feaweld.solver.calculix_backend import generate_inp

    mesh = _ccx_mesh()
    values = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    lc = LoadCase(
        name="force",
        loads=[BoundaryCondition("boundary", LoadType.FORCE, values)],
    )

    content = generate_inp(
        mesh, _material(), lc, tmp_path / "force.inp", analysis="static",
    ).read_text()

    assert "*CLOAD" in content
    assert "1, 1, 10" in content
    assert "2, 2, 50" in content


def test_ccx_static_temperature_and_initial_conditions(tmp_path):
    from feaweld.solver.calculix_backend import generate_inp

    mesh = _ccx_mesh()
    lc = LoadCase(
        name="thermal_load",
        loads=[BoundaryCondition("all", LoadType.TEMPERATURE, np.array([120.0]))],
    )

    content = generate_inp(
        mesh, _material(), lc, tmp_path / "temp.inp", analysis="static",
    ).read_text()

    assert "*TEMPERATURE" in content
    assert "*INITIAL CONDITIONS, TYPE=TEMPERATURE" in content
