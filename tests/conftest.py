"""Shared test fixtures for feaweld test suite."""

import numpy as np
import pytest

from feaweld.core.types import (
    FEMesh, FEAResults, StressField, ElementType,
    WeldLineDefinition, LoadCase, BoundaryCondition, LoadType,
    SNCurve, SNSegment, SNStandard,
)
from feaweld.core.materials import Material


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
