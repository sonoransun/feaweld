"""Tests for Battelle/Dong structural stress method."""

import numpy as np
import pytest

from feaweld.core.types import FEMesh, FEAResults, StressField, ElementType, WeldLineDefinition
from feaweld.postprocess.dong import (
    dong_structural_stress,
    dong_fatigue_life,
    MASTER_SN_C,
    MASTER_SN_H,
)


@pytest.fixture
def plate_with_nodal_forces():
    """Create a simple plate mesh with nodal forces for Dong method testing."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 0.0],
        [20.0, 10.0, 0.0],
        [0.0, 20.0, 0.0],
        [10.0, 20.0, 0.0],
        [20.0, 20.0, 0.0],
    ])
    elements = np.array([
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6],
        [4, 5, 8], [4, 8, 7],
    ])

    mesh = FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
        node_sets={"weld_toe": np.array([0, 1, 2])},
    )

    # Uniform stress σ_yy = 100 MPa
    stress_vals = np.zeros((9, 6))
    stress_vals[:, 1] = 100.0

    # Nodal forces consistent with σ_yy = 100 MPa on bottom edge
    # Force per node ≈ σ * tributary_length * thickness
    nodal_forces = np.zeros((9, 3))
    nodal_forces[0, 1] = 500.0   # corner: half tributary
    nodal_forces[1, 1] = 1000.0  # mid: full tributary
    nodal_forces[2, 1] = 500.0   # corner: half tributary

    results = FEAResults(
        mesh=mesh,
        stress=StressField(values=stress_vals),
        nodal_forces=nodal_forces,
    )
    return results, mesh


def test_dong_structural_stress_returns_result(plate_with_nodal_forces):
    """Test that structural stress computation returns valid result."""
    results, mesh = plate_with_nodal_forces

    weld_line = WeldLineDefinition(
        name="bottom_toe",
        node_ids=np.array([0, 1, 2]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )

    dong = dong_structural_stress(results, weld_line)
    assert dong.membrane_stress is not None
    assert len(dong.membrane_stress) == 3
    assert dong.structural_stress is not None
    assert np.all(np.isfinite(dong.structural_stress))


def test_dong_bending_ratio_bounded(plate_with_nodal_forces):
    """Test that bending ratio r is between 0 and 1."""
    results, mesh = plate_with_nodal_forces

    weld_line = WeldLineDefinition(
        name="bottom_toe",
        node_ids=np.array([0, 1, 2]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )

    dong = dong_structural_stress(results, weld_line)
    assert np.all(dong.bending_ratio >= 0)
    assert np.all(dong.bending_ratio <= 1.0 + 1e-10)


def test_dong_fatigue_life_positive(plate_with_nodal_forces):
    """Test that fatigue life is computed and positive."""
    results, mesh = plate_with_nodal_forces

    weld_line = WeldLineDefinition(
        name="bottom_toe",
        node_ids=np.array([0, 1, 2]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )

    dong = dong_structural_stress(results, weld_line)
    dong_with_life = dong_fatigue_life(dong, plate_thickness=10.0)

    assert dong_with_life.fatigue_life is not None
    assert np.all(dong_with_life.fatigue_life > 0)
    assert dong_with_life.equivalent_stress_range is not None


def test_dong_master_sn_constants():
    """Test that master S-N curve constants are reasonable."""
    assert MASTER_SN_C > 0
    assert 2.0 < MASTER_SN_H < 5.0  # typical range


def test_dong_no_data_raises():
    """Test that missing force and stress data raises error."""
    mesh = FEMesh(
        nodes=np.zeros((3, 3)),
        elements=np.array([[0, 1, 2]]),
        element_type=ElementType.TRI3,
    )
    results = FEAResults(mesh=mesh)

    weld_line = WeldLineDefinition(
        name="test",
        node_ids=np.array([0]),
        plate_thickness=10.0,
        normal_direction=np.array([0.0, 1.0, 0.0]),
    )

    with pytest.raises(ValueError):
        dong_structural_stress(results, weld_line)
