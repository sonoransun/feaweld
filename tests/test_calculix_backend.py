"""Tests for the CalculiX (ccx) solver backend.

The module imports cleanly with CalculiX absent; the unmarked tests below
exercise the pure-Python deck writer (element-type coverage and the Gmsh ->
CalculiX quadratic-tet node permutation) without invoking ``ccx``.  The two
end-to-end tests are gated behind ``@pytest.mark.requires_calculix`` and a
``skipif`` on ``shutil.which("ccx")`` so they are collected (not
module-skipped) but only run where the ``ccx`` executable is on ``PATH``.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEMesh,
    LoadCase,
    LoadType,
    StressField,
)
from feaweld.solver import calculix_backend
from feaweld.solver.backend import SolverBackend
from feaweld.solver.calculix_backend import (
    _ELEMENT_TYPE_MAP,
    _GMSH_TO_CCX_PERMUTATION,
    CalculiXBackend,
    generate_inp,
    parse_frd,
)

# ``ccx`` is not installed in the default dev environment, so the gated tests
# skip there; where it is present they run for real.
requires_ccx = pytest.mark.skipif(
    shutil.which("ccx") is None,
    reason="ccx executable not found on PATH",
)


def _one_tet10_mesh() -> FEMesh:
    """A single straight-edged C3D10 tetrahedron in Gmsh node order.

    Corner nodes 0..3 span a unit-scale tetrahedron; mid-edge nodes 4..9 sit
    at the exact edge midpoints in Gmsh's TET10 edge order
    ``(0,1),(1,2),(2,0),(0,3),(2,3),(1,3)``.  The ``bottom`` node set is the
    fully clamped ``z = 0`` face (three corners plus their three mid-edge
    nodes) and ``apex`` is the single free corner carrying the load.

    Returns
    -------
    FEMesh
        A one-element quadratic-tet mesh.
    """
    nodes = np.array([
        [0.0, 0.0, 0.0],    # 0  corner
        [10.0, 0.0, 0.0],   # 1  corner
        [0.0, 10.0, 0.0],   # 2  corner
        [0.0, 0.0, 10.0],   # 3  corner (apex)
        [5.0, 0.0, 0.0],    # 4  mid-edge (0,1)
        [5.0, 5.0, 0.0],    # 5  mid-edge (1,2)
        [0.0, 5.0, 0.0],    # 6  mid-edge (2,0)
        [0.0, 0.0, 5.0],    # 7  mid-edge (0,3)
        [0.0, 5.0, 5.0],    # 8  mid-edge (2,3)
        [5.0, 0.0, 5.0],    # 9  mid-edge (1,3)
    ])
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TET10,
        node_sets={
            "bottom": np.array([0, 1, 2, 4, 5, 6]),
            "apex": np.array([3]),
        },
    )


# ---------------------------------------------------------------------------
# Unmarked: importable and deck writer correct without CalculiX installed
# ---------------------------------------------------------------------------

def test_module_imports_without_calculix():
    """The backend module and class import even when ccx is missing."""
    assert calculix_backend.CalculiXBackend is not None
    assert issubclass(CalculiXBackend, SolverBackend)


def test_element_type_map_covers_every_element_type():
    """Every ElementType member maps to a CalculiX/Abaqus element keyword."""
    missing = [m.name for m in ElementType if m not in _ELEMENT_TYPE_MAP]
    assert missing == [], f"_ELEMENT_TYPE_MAP is missing element types: {missing}"


def test_tet10_node_permutation_written_to_deck(tmp_path, steel_material):
    """The generated C3D10 element line reorders mid-edge nodes 8 and 9.

    Gmsh and CalculiX disagree on the last two mid-edge nodes of a quadratic
    tetrahedron; ``_GMSH_TO_CCX_PERMUTATION[TET10]`` swaps them.  With a mesh
    stored in Gmsh order (connectivity ``0..9``), the written 1-based line must
    therefore read ``1,2,3,4,5,6,7,8,10,9`` — positions 8 and 9 exchanged.
    """
    assert _GMSH_TO_CCX_PERMUTATION[ElementType.TET10] == [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]

    mesh = _one_tet10_mesh()
    lc = LoadCase(
        name="clamp",
        constraints=[BoundaryCondition(
            "bottom", LoadType.DISPLACEMENT, np.array([0.0, 0.0, 0.0]),
        )],
    )
    inp = generate_inp(mesh, steel_material, lc, tmp_path / "tet10.inp", analysis="static")

    element_line = None
    lines = inp.read_text().splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("*ELEMENT"):
            element_line = lines[idx + 1]
            break
    assert element_line is not None, "no *ELEMENT data line written"

    connectivity = [int(tok) for tok in element_line.split(",")]
    # First token is the 1-based element id; the rest is the node connectivity.
    assert connectivity == [1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9]


# ---------------------------------------------------------------------------
# Marked: real ccx solves (skipped where the executable is unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.requires_calculix
@requires_ccx
def test_solve_static_tet4(simple_3d_mesh, steel_material):
    """A tiny TET4 mesh solves end-to-end and parses finite nodal stress."""
    lc = LoadCase(
        name="axial",
        loads=[BoundaryCondition(
            "top", LoadType.FORCE, np.array([1000.0]),
            direction=np.array([0.0, 0.0, 1.0]),
        )],
        constraints=[BoundaryCondition(
            "bottom", LoadType.DISPLACEMENT, np.array([0.0, 0.0, 0.0]),
        )],
    )
    results = CalculiXBackend().solve_static(simple_3d_mesh, steel_material, lc)

    assert isinstance(results.stress, StressField)
    assert results.stress.values.shape[1] == 6
    assert results.stress.values.shape[0] > 0
    assert np.all(np.isfinite(results.stress.values))
    assert results.displacement is not None
    assert np.all(np.isfinite(results.displacement))
    assert np.any(np.abs(results.displacement) > 0.0)


@pytest.mark.requires_calculix
@requires_ccx
def test_tet10_deck_accepted_by_ccx(tmp_path, steel_material):
    """ccx accepts the permuted C3D10 deck and returns finite stress.

    An incorrect mid-edge node order leaves two mid-side nodes off the edges
    ccx expects, giving a non-affine map with a non-positive Jacobian that ccx
    rejects; a clean ``returncode == 0`` and parseable stress thus validate the
    Gmsh -> CalculiX permutation numerically.
    """
    mesh = _one_tet10_mesh()
    lc = LoadCase(
        name="apex_load",
        loads=[BoundaryCondition(
            "apex", LoadType.FORCE, np.array([500.0]),
            direction=np.array([1.0, 0.0, 0.0]),
        )],
        constraints=[BoundaryCondition(
            "bottom", LoadType.DISPLACEMENT, np.array([0.0, 0.0, 0.0]),
        )],
    )
    inp = generate_inp(mesh, steel_material, lc, tmp_path / "model.inp", analysis="static")

    proc = subprocess.run(
        ["ccx", "-i", inp.stem],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (proc.stdout[-2000:] + proc.stderr[-2000:])

    frd = tmp_path / f"{inp.stem}.frd"
    assert frd.exists()
    stress = parse_frd(frd).get("stress")
    assert stress is not None
    assert stress.shape[1] == 6
    assert np.all(np.isfinite(stress))
