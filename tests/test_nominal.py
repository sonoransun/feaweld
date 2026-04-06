"""Tests for ASME VIII nominal stress categorization and assessment."""

import numpy as np
import pytest

from feaweld.core.types import FEMesh, FEAResults, StressField, ElementType
from feaweld.postprocess.nominal import (
    categorize_stress_section,
    asme_allowable_check,
    extract_stress_along_path,
    StressCategorization,
)


# ---------------------------------------------------------------------------
# categorize_stress_section — uniform stress (pure membrane)
# ---------------------------------------------------------------------------


def test_categorize_uniform_stress_membrane():
    """Uniform stress through thickness should yield pure membrane, bending ~ 0."""
    n_points = 21
    thickness = 10.0
    z = np.linspace(0, thickness, n_points)
    stress = np.full(n_points, 100.0)  # constant 100 MPa

    result = categorize_stress_section(stress, thickness, z)

    assert result.membrane == pytest.approx(100.0, abs=1e-6)
    assert result.bending == pytest.approx(0.0, abs=1e-6)
    assert result.peak == pytest.approx(0.0, abs=0.1)
    assert result.total == pytest.approx(100.0, abs=1e-6)


def test_categorize_uniform_stress_properties():
    """StressCategorization properties for uniform stress."""
    n_points = 21
    thickness = 10.0
    z = np.linspace(0, thickness, n_points)
    stress = np.full(n_points, 80.0)

    result = categorize_stress_section(stress, thickness, z)

    assert result.primary_membrane == pytest.approx(80.0, abs=1e-6)
    assert result.primary_plus_bending == pytest.approx(80.0, abs=0.5)
    assert result.stress_intensity == pytest.approx(80.0, abs=1e-6)


# ---------------------------------------------------------------------------
# categorize_stress_section — linearly varying stress (has bending)
# ---------------------------------------------------------------------------


def test_categorize_linear_stress_has_bending():
    """Linear stress through thickness should have nonzero bending component."""
    n_points = 101
    thickness = 10.0
    z = np.linspace(0, thickness, n_points)
    # Stress varies linearly: 0 at z=0, 200 at z=t
    stress = np.linspace(0.0, 200.0, n_points)

    result = categorize_stress_section(stress, thickness, z)

    # Membrane = average = (0 + 200) / 2 = 100
    assert result.membrane == pytest.approx(100.0, rel=1e-3)

    # Bending should be nonzero for linearly varying stress
    assert abs(result.bending) > 1.0

    # Total = max(|stress|) = 200
    assert result.total == pytest.approx(200.0, abs=1e-6)


def test_categorize_linear_stress_bending_value():
    """Verify bending value for a symmetric linear profile.

    For stress = sigma_0 * (z/t - 0.5) (antisymmetric about midplane):
      membrane = 0 (integral of odd function about midplane)
      bending = sigma_0 / 2 (from the linearization formula)
    """
    n_points = 201
    thickness = 10.0
    z = np.linspace(0, thickness, n_points)
    sigma_0 = 300.0
    # Pure bending: stress = sigma_0 * (z/t - 0.5)
    # At z=0 stress = -150, at z=t stress = +150
    stress = sigma_0 * (z / thickness - 0.5)

    result = categorize_stress_section(stress, thickness, z)

    assert result.membrane == pytest.approx(0.0, abs=0.5)
    assert result.bending == pytest.approx(sigma_0 / 2.0, rel=1e-2)
    assert result.total == pytest.approx(150.0, abs=0.5)


def test_categorize_combined_membrane_and_bending():
    """Combined membrane + bending: constant offset plus linear variation."""
    n_points = 201
    thickness = 20.0
    z = np.linspace(0, thickness, n_points)
    # stress = 50 (membrane) + 60*(z/t - 0.5) (bending)
    stress = 50.0 + 60.0 * (z / thickness - 0.5)

    result = categorize_stress_section(stress, thickness, z)

    assert result.membrane == pytest.approx(50.0, abs=0.5)
    assert result.bending == pytest.approx(30.0, rel=1e-2)


# ---------------------------------------------------------------------------
# asme_allowable_check — passes when below limits
# ---------------------------------------------------------------------------


def test_asme_check_all_pass():
    """When stresses are well below limits, all checks should pass."""
    cat = StressCategorization(
        membrane=50.0,
        bending=20.0,
        peak=5.0,
        total=75.0,
        stress_intensity=75.0,
    )
    S_m = 150.0
    S_y = 250.0

    checks = asme_allowable_check(cat, S_m, S_y)

    assert checks["Pm"]["passes"] is True
    assert checks["PL"]["passes"] is True
    assert checks["Pm+Pb"]["passes"] is True
    assert checks["PL+Pb+Q"]["passes"] is True

    # Ratios should all be below 1.0
    for key in checks:
        assert checks[key]["ratio"] < 1.0


def test_asme_check_values_and_limits():
    """Verify the numeric values and limits in the check result."""
    cat = StressCategorization(
        membrane=100.0,
        bending=30.0,
        peak=10.0,
        total=140.0,
        stress_intensity=140.0,
    )
    S_m = 160.0
    S_y = 250.0

    checks = asme_allowable_check(cat, S_m, S_y)

    # Pm check
    assert checks["Pm"]["value"] == pytest.approx(100.0)
    assert checks["Pm"]["limit"] == pytest.approx(160.0)
    assert checks["Pm"]["ratio"] == pytest.approx(100.0 / 160.0)

    # Pm+Pb check
    assert checks["Pm+Pb"]["value"] == pytest.approx(130.0)
    assert checks["Pm+Pb"]["limit"] == pytest.approx(1.5 * 160.0)

    # PL+Pb+Q check
    S_PS = max(3.0 * S_m, 2.0 * S_y)
    assert checks["PL+Pb+Q"]["limit"] == pytest.approx(S_PS)


# ---------------------------------------------------------------------------
# asme_allowable_check — fails on Pm exceeding S_m
# ---------------------------------------------------------------------------


def test_asme_check_fails_pm():
    """Primary membrane exceeding S_m should cause Pm check to fail."""
    cat = StressCategorization(
        membrane=200.0,
        bending=10.0,
        peak=5.0,
        total=215.0,
        stress_intensity=215.0,
    )
    S_m = 150.0
    S_y = 250.0

    checks = asme_allowable_check(cat, S_m, S_y)

    assert checks["Pm"]["passes"] is False
    assert checks["Pm"]["ratio"] > 1.0
    assert checks["Pm"]["value"] == pytest.approx(200.0)


def test_asme_check_negative_membrane_exceeds():
    """Negative membrane stress with |Pm| > S_m should also fail."""
    cat = StressCategorization(
        membrane=-180.0,
        bending=10.0,
        peak=5.0,
        total=195.0,
        stress_intensity=195.0,
    )
    S_m = 150.0
    S_y = 250.0

    checks = asme_allowable_check(cat, S_m, S_y)

    assert checks["Pm"]["passes"] is False
    assert checks["Pm"]["value"] == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# asme_allowable_check — fails on Pm+Pb exceeding 1.5*S_m
# ---------------------------------------------------------------------------


def test_asme_check_fails_pm_plus_pb():
    """Pm+Pb exceeding 1.5*S_m should cause that check to fail."""
    cat = StressCategorization(
        membrane=100.0,
        bending=130.0,
        peak=5.0,
        total=235.0,
        stress_intensity=235.0,
    )
    S_m = 150.0  # limit = 1.5 * 150 = 225
    S_y = 250.0

    checks = asme_allowable_check(cat, S_m, S_y)

    # Pm alone is OK
    assert checks["Pm"]["passes"] is True

    # But Pm+Pb = 230 > 225 should fail
    assert checks["Pm+Pb"]["passes"] is False
    assert checks["Pm+Pb"]["value"] == pytest.approx(230.0)
    assert checks["Pm+Pb"]["ratio"] > 1.0


def test_asme_check_pm_plus_pb_boundary():
    """At the exact boundary (Pm+Pb == 1.5*S_m) the check should pass."""
    cat = StressCategorization(
        membrane=100.0,
        bending=125.0,
        peak=0.0,
        total=225.0,
        stress_intensity=225.0,
    )
    S_m = 150.0  # limit = 225

    checks = asme_allowable_check(cat, S_m, S_y=250.0)

    assert checks["Pm+Pb"]["passes"] is True
    assert checks["Pm+Pb"]["ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# extract_stress_along_path — with known stress values
# ---------------------------------------------------------------------------


def _make_path_mesh():
    """Create a mesh with nodes along z-axis and known von Mises stress."""
    n = 11
    nodes = np.zeros((n, 3))
    nodes[:, 2] = np.linspace(0, 10.0, n)  # z from 0 to 10 mm

    elements = np.array([[i, i + 1, i + 1] for i in range(n - 1)])

    stress_vals = np.zeros((n, 6))
    # σ_yy = 100 + 20*z  -> von Mises = σ_yy for uniaxial
    stress_vals[:, 1] = 100.0 + 20.0 * nodes[:, 2]

    mesh = FEMesh(
        nodes=nodes,
        elements=elements,
        element_type=ElementType.TRI3,
    )
    results = FEAResults(
        mesh=mesh,
        stress=StressField(values=stress_vals),
    )
    return results


def test_extract_stress_along_path_shape():
    """extract_stress_along_path should return arrays of correct length."""
    results = _make_path_mesh()

    distances, stresses = extract_stress_along_path(
        results, start_node=0, end_node=10, n_points=15,
    )

    assert distances.shape == (15,)
    assert stresses.shape == (15,)


def test_extract_stress_along_path_endpoints():
    """Stress at start and end nodes should match the known values."""
    results = _make_path_mesh()

    distances, stresses = extract_stress_along_path(
        results, start_node=0, end_node=10, n_points=11,
    )

    # Distance should span 0 to 10
    assert distances[0] == pytest.approx(0.0)
    assert distances[-1] == pytest.approx(10.0)

    # Von Mises at start node (z=0): σ_yy=100 -> VM=100
    assert stresses[0] == pytest.approx(100.0, rel=0.1)
    # Von Mises at end node (z=10): σ_yy=300 -> VM=300
    assert stresses[-1] == pytest.approx(300.0, rel=0.1)


def test_extract_stress_along_path_monotonic():
    """With monotonically increasing stress, extracted values should increase."""
    results = _make_path_mesh()

    distances, stresses = extract_stress_along_path(
        results, start_node=0, end_node=10, n_points=11,
    )

    # Stress should be non-decreasing along the path
    assert np.all(np.diff(stresses) >= -1e-6)


def test_extract_stress_no_data_raises():
    """Missing stress data should raise ValueError."""
    mesh = FEMesh(
        nodes=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [1.0, 0.0, 5.0]]),
        elements=np.array([[0, 1, 2]]),
        element_type=ElementType.TRI3,
    )
    results = FEAResults(mesh=mesh)

    with pytest.raises(ValueError, match="No stress data"):
        extract_stress_along_path(results, start_node=0, end_node=1)
