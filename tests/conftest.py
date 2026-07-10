"""Shared test fixtures for feaweld test suite."""

import numpy as np
import pytest

from feaweld.core.types import (
    FEMesh, FEAResults, StressField, ElementType,
    WeldLineDefinition, LoadCase, BoundaryCondition, LoadType,
    SNCurve, SNSegment, SNStandard,
)
from feaweld.core.materials import Material
from feaweld.solver.backend import SolverBackend


@pytest.fixture
def simple_plate_mesh():
    """A simple 2D plate mesh (2 triangles forming a square)."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    elements = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
        node_sets={
            "bottom": np.array([0, 1]),
            "top": np.array([2, 3]),
            "weld_toe": np.array([1, 2]),
        },
    )


@pytest.fixture
def simple_3d_mesh():
    """A simple 3D mesh (2 tetrahedra forming a prism)."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [5.0, 10.0, 0.0],
        [5.0, 5.0, 10.0],
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 10.0],
    ])
    elements = np.array([
        [0, 1, 2, 3],
        [0, 1, 3, 4],
    ])
    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TET4,
        node_sets={
            "bottom": np.array([0, 1, 2]),
            "top": np.array([3, 4, 5]),
        },
    )


@pytest.fixture
def uniform_stress_results(simple_plate_mesh):
    """FEA results with uniform uniaxial stress (σ_yy = 100 MPa)."""
    n = simple_plate_mesh.n_nodes
    stress_vals = np.zeros((n, 6))
    stress_vals[:, 1] = 100.0  # σ_yy = 100 MPa

    displacement = np.zeros((n, 3))
    displacement[:, 1] = np.linspace(0, 0.005, n)  # small y-displacement

    return FEAResults(
        mesh=simple_plate_mesh,
        displacement=displacement,
        stress=StressField(values=stress_vals),
    )


@pytest.fixture
def gradient_stress_results(simple_plate_mesh):
    """FEA results with stress gradient (σ varies linearly through thickness)."""
    n = simple_plate_mesh.n_nodes
    stress_vals = np.zeros((n, 6))
    # Stress varies linearly with y-coordinate: 0 at bottom, 200 at top
    for i in range(n):
        y = simple_plate_mesh.nodes[i, 1]
        stress_vals[i, 1] = 200.0 * y / 10.0  # σ_yy varies 0 to 200 MPa

    return FEAResults(
        mesh=simple_plate_mesh,
        stress=StressField(values=stress_vals),
    )


@pytest.fixture
def sample_weld_line(simple_plate_mesh):
    """A simple weld line definition for testing."""
    return WeldLineDefinition(
        name="test_weld_toe",
        node_ids=np.array([1, 2]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )


@pytest.fixture
def steel_material():
    """A simple A36-like steel material for testing (no YAML dependency)."""
    return Material(
        name="TestSteel",
        density=7850.0,
        elastic_modulus={20.0: 200000.0, 500.0: 160000.0, 800.0: 70000.0},
        poisson_ratio={20.0: 0.26, 500.0: 0.30},
        yield_strength={20.0: 250.0, 500.0: 165.0, 800.0: 25.0},
        ultimate_strength={20.0: 400.0, 500.0: 310.0},
        thermal_conductivity={20.0: 51.9, 500.0: 37.2},
        specific_heat={20.0: 440.0, 500.0: 650.0},
        thermal_expansion={20.0: 11.7e-6, 500.0: 14.0e-6},
    )


@pytest.fixture
def simple_sn_curve():
    """A simple two-slope S-N curve (IIW FAT90-like)."""
    return SNCurve(
        name="TestFAT90",
        standard=SNStandard.IIW,
        segments=[
            SNSegment(m=3.0, C=90.0**3 * 2e6, stress_threshold=0.0),
        ],
        cutoff_cycles=1e7,
    )


@pytest.fixture
def simple_load_case():
    """A simple load case with fixed bottom and force on top."""
    return LoadCase(
        name="test_static",
        loads=[
            BoundaryCondition(
                node_set="top",
                bc_type=LoadType.FORCE,
                values=np.array([0.0, 1000.0, 0.0]),
            ),
        ],
        constraints=[
            BoundaryCondition(
                node_set="bottom",
                bc_type=LoadType.DISPLACEMENT,
                values=np.array([0.0, 0.0, 0.0]),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Fake solver backend for exercising solver-agnostic wiring without FEniCSx /
# CalculiX.  Every solve method returns a canned, gradient-stress FEAResults
# built from the mesh it is handed and records the call so tests can assert
# which method ran and inspect the LoadCase it received.
# ---------------------------------------------------------------------------

class FakeBackend(SolverBackend):
    """A SolverBackend that fabricates deterministic results and logs calls.

    The canned stress field is a through-thickness gradient (σ_yy rising with
    the y-coordinate plus small σ_xx / τ_xy terms) so that post-processing
    methods have a non-trivial field to chew on.  Displacements are small and
    monotone in y; thermal solves return a plausible temperature ramp.

    Each ``solve_*`` call appends a dict to :attr:`calls` with the method name
    and its arguments (``load_case`` for mechanical, ``load_case`` +
    ``time_steps`` for transient, etc.).  Use :meth:`methods_called` and
    :meth:`calls_to` to inspect the recorded history.
    """

    def __init__(self, stress_scale: float = 1.0, temp_peak: float = 120.0):
        self.calls: list[dict] = []
        self.stress_scale = float(stress_scale)
        self.temp_peak = float(temp_peak)

    # -- canned field builders ------------------------------------------------

    @staticmethod
    def _yfrac(mesh: FEMesh) -> np.ndarray:
        y = np.asarray(mesh.nodes, dtype=np.float64)[:, 1]
        span = float(y.max() - y.min())
        if span < 1e-12:
            return np.zeros(mesh.n_nodes)
        return (y - y.min()) / span

    def canned_stress(self, mesh: FEMesh) -> np.ndarray:
        """Gradient stress field (n, 6) in Voigt notation (MPa)."""
        f = self._yfrac(mesh)
        s = np.zeros((mesh.n_nodes, 6))
        s[:, 1] = (50.0 + 150.0 * f) * self.stress_scale   # σ_yy: 50 → 200
        s[:, 0] = 20.0 * f * self.stress_scale             # small σ_xx
        s[:, 3] = 10.0 * f * self.stress_scale             # small τ_xy
        return s

    def canned_displacement(self, mesh: FEMesh) -> np.ndarray:
        """Small monotone displacement field (n, 3)."""
        f = self._yfrac(mesh)
        d = np.zeros((mesh.n_nodes, 3))
        d[:, 1] = 0.01 * f
        return d

    def canned_temperature(self, mesh: FEMesh) -> np.ndarray:
        """Plausible nodal temperature ramp (n,) in C."""
        f = self._yfrac(mesh)
        return 20.0 + (self.temp_peak - 20.0) * f

    # -- SolverBackend interface ---------------------------------------------

    def solve_static(self, mesh, material, load_case, temperature=20.0):
        self.calls.append({
            "method": "solve_static", "mesh": mesh, "material": material,
            "load_case": load_case, "temperature": temperature,
        })
        return FEAResults(
            mesh=mesh,
            displacement=self.canned_displacement(mesh),
            stress=StressField(values=self.canned_stress(mesh)),
            metadata={"solver": "fake", "temperature": temperature},
        )

    def solve_thermal_steady(self, mesh, material, load_case):
        self.calls.append({
            "method": "solve_thermal_steady", "mesh": mesh,
            "material": material, "load_case": load_case,
        })
        return FEAResults(
            mesh=mesh,
            temperature=self.canned_temperature(mesh),
            metadata={"solver": "fake", "analysis": "thermal_steady"},
        )

    def solve_thermal_transient(self, mesh, material, load_case, time_steps,
                                heat_source=None):
        ts = np.asarray(time_steps, dtype=np.float64)
        self.calls.append({
            "method": "solve_thermal_transient", "mesh": mesh,
            "material": material, "load_case": load_case,
            "time_steps": ts, "heat_source": heat_source,
        })
        temp = np.tile(self.canned_temperature(mesh), (len(ts), 1))
        return FEAResults(
            mesh=mesh,
            temperature=temp,
            time_steps=ts,
            metadata={"solver": "fake", "analysis": "thermal_transient"},
        )

    def solve_coupled(self, mesh, material, mechanical_lc, thermal_lc, time_steps):
        ts = np.asarray(time_steps, dtype=np.float64)
        self.calls.append({
            "method": "solve_coupled", "mesh": mesh, "material": material,
            "mechanical_lc": mechanical_lc, "thermal_lc": thermal_lc,
            "time_steps": ts,
        })
        return FEAResults(
            mesh=mesh,
            displacement=self.canned_displacement(mesh),
            stress=StressField(values=self.canned_stress(mesh)),
            temperature=self.canned_temperature(mesh),
            time_steps=ts,
            metadata={"solver": "fake", "analysis": "coupled"},
        )

    # -- inspection helpers ---------------------------------------------------

    def methods_called(self) -> list[str]:
        return [c["method"] for c in self.calls]

    def calls_to(self, method: str) -> list[dict]:
        return [c for c in self.calls if c["method"] == method]


@pytest.fixture
def fake_backend():
    """A fresh :class:`FakeBackend` instance (records nothing yet)."""
    return FakeBackend()


@pytest.fixture
def fake_backend_cls():
    """The :class:`FakeBackend` class, for tests that build their own."""
    return FakeBackend


@pytest.fixture
def grid_plate_mesh():
    """A structured 2D plate mesh with bottom/top/weld_toe node sets.

    The plate spans x in [0, 40] (width) and y in [0, 20] (through-thickness),
    a 5x5 node grid triangulated into 32 TRI3 elements.  ``bottom`` is the
    y = 0 edge, ``top`` the y = 20 edge, and ``weld_toe`` three ordered nodes
    on the top surface — enough structure for surface extrapolation and
    through-thickness linearization to run against a real gradient.
    """
    nx, ny = 5, 5
    dx, dy = 10.0, 5.0
    nodes = np.array(
        [[i * dx, j * dy, 0.0] for j in range(ny) for i in range(nx)],
        dtype=np.float64,
    )

    def nid(i, j):
        return j * nx + i

    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            elements.append([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1)])
            elements.append([nid(i, j), nid(i + 1, j + 1), nid(i, j + 1)])
    elements = np.array(elements, dtype=np.int64)

    bottom = np.array([nid(i, 0) for i in range(nx)], dtype=np.int64)
    top = np.array([nid(i, ny - 1) for i in range(nx)], dtype=np.int64)
    weld_toe = np.array([nid(1, ny - 1), nid(2, ny - 1), nid(3, ny - 1)],
                        dtype=np.int64)

    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
        node_sets={"bottom": bottom, "top": top, "weld_toe": weld_toe},
    )


@pytest.fixture
def grid_solid_mesh():
    """A structured 3D solid mesh with bottom/top/weld_toe node sets.

    The block spans x in [0, 40] (width), y in [0, 20] (through-thickness),
    and z in [0, 40] (weld direction): a 5x5x5 node grid whose 64 cells are
    each split into 6 TET4s (Kuhn subdivision, conforming across cells).
    ``bottom`` is the y = 0 face, ``top`` the y = 20 face, and
    ``weld_toe`` / ``weld_toe_0`` the 5 nodes along the toe line
    (x = 20, y = 20, z = 0..40).  A small ``"weld"`` element group on the
    x < 20 side of the toe exercises hot-spot surface-direction sign
    selection.
    """
    nx = ny = nz = 5
    dx, dy, dz = 10.0, 5.0, 10.0
    nodes = np.array(
        [[i * dx, j * dy, k * dz]
         for k in range(nz) for j in range(ny) for i in range(nx)],
        dtype=np.float64,
    )

    def nid(i, j, k):
        return (k * ny + j) * nx + i

    # Kuhn subdivision: 6 tets per cell, one per permutation of the axis
    # steps from the cell's low corner to its high corner.
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    elements = []
    weld_elems = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                corner = {
                    (di, dj, dk): nid(i + di, j + dj, k + dk)
                    for di in (0, 1) for dj in (0, 1) for dk in (0, 1)
                }
                for perm in perms:
                    step = [0, 0, 0]
                    tet = [corner[(0, 0, 0)]]
                    for axis in perm:
                        step = list(step)
                        step[axis] = 1
                        tet.append(corner[tuple(step)])
                    if i == 1 and j == ny - 2:
                        weld_elems.append(len(elements))
                    elements.append(tet)
    elements = np.array(elements, dtype=np.int64)

    bottom = np.array(sorted(nid(i, 0, k) for i in range(nx)
                             for k in range(nz)), dtype=np.int64)
    top = np.array(sorted(nid(i, ny - 1, k) for i in range(nx)
                          for k in range(nz)), dtype=np.int64)
    toe = np.array([nid(2, ny - 1, k) for k in range(nz)], dtype=np.int64)

    return FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TET4,
        physical_groups={"weld": np.array(weld_elems, dtype=np.int64)},
        node_sets={
            "bottom": bottom,
            "top": top,
            "weld_toe": toe.copy(),
            "weld_toe_0": toe,
        },
    )
