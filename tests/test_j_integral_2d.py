"""Tests for 2D domain J-integral and interaction integral (Track F3)."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import ElementType, FEAResults, FEMesh, StressField
from feaweld.fracture import (
    JResult,
    compute_k_from_j_elastic,
    interaction_integral,
    j_integral_2d,
)
from feaweld.fracture.j_integral import _q_function, _williams_auxiliary_fields


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def _structured_tri_plate(
    lx: float,
    ly: float,
    nx: int,
    ny: int,
    origin: tuple[float, float] = (0.0, 0.0),
) -> FEMesh:
    """nx x ny rectangles (each split into 2 TRI3s) spanning the plate."""
    x0, y0 = origin
    xs = np.linspace(x0, x0 + lx, nx + 1)
    ys = np.linspace(y0, y0 + ly, ny + 1)
    nodes = np.array(
        [[x, y, 0.0] for y in ys for x in xs], dtype=np.float64
    )

    def nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    elements: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            n0 = nid(i, j)
            n1 = nid(i + 1, j)
            n2 = nid(i + 1, j + 1)
            n3 = nid(i, j + 1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

    return FEMesh(
        nodes=nodes,
        elements=np.array(elements, dtype=np.int64),
        element_type=ElementType.TRI3,
    )


def _voigt_from_plane_stress(sxx, syy, sxy):
    """Expand (σ_xx, σ_yy, τ_xy) into a 6-component Voigt array with zeros elsewhere."""
    n = sxx.shape[0]
    out = np.zeros((n, 6))
    out[:, 0] = sxx
    out[:, 1] = syy
    out[:, 3] = sxy
    return out


def _impose_williams_field(
    mesh: FEMesh,
    crack_tip: np.ndarray,
    K_I: float,
    K_II: float,
    E: float,
    nu: float,
    plane_strain: bool = True,
) -> FEAResults:
    """Analytically evaluate the Williams K field at each node and wrap as FEAResults."""
    nodes_xy = mesh.nodes[:, :2]
    rel = nodes_xy - crack_tip[None, :2]
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])

    disp = np.zeros((mesh.n_nodes, 3))
    sxx = np.zeros(mesh.n_nodes)
    syy = np.zeros(mesh.n_nodes)
    sxy = np.zeros(mesh.n_nodes)
    eps_xx = np.zeros(mesh.n_nodes)
    eps_yy = np.zeros(mesh.n_nodes)
    gxy = np.zeros(mesh.n_nodes)

    for amp, mode in ((K_I, "I"), (K_II, "II")):
        if amp == 0.0:
            continue
        u, s, e = _williams_auxiliary_fields(
            r, theta, mode, E, nu, plane_strain=plane_strain
        )
        disp[:, 0] += amp * u[:, 0]
        disp[:, 1] += amp * u[:, 1]
        sxx += amp * s[:, 0]
        syy += amp * s[:, 1]
        sxy += amp * s[:, 2]
        eps_xx += amp * e[:, 0]
        eps_yy += amp * e[:, 1]
        gxy += amp * e[:, 2]

    stress_values = _voigt_from_plane_stress(sxx, syy, sxy)
    strain_values = np.zeros((mesh.n_nodes, 6))
    strain_values[:, 0] = eps_xx
    strain_values[:, 1] = eps_yy
    strain_values[:, 3] = gxy

    return FEAResults(
        mesh=mesh,
        displacement=disp,
        stress=StressField(values=stress_values),
        strain=strain_values,
    )


# ---------------------------------------------------------------------------
# q-function helper
# ---------------------------------------------------------------------------


def test_q_function_shape():
    r = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    q = _q_function(r, inner_r=1.0, outer_r=2.0)
    # Plateau
    assert q[0] == pytest.approx(1.0)
    assert q[1] == pytest.approx(1.0)
    assert q[2] == pytest.approx(1.0)
    # Linear decay halfway through annulus
    assert q[3] == pytest.approx(0.5, abs=1e-12)
    # Outer boundary
    assert q[4] == pytest.approx(0.0, abs=1e-12)
    # Strictly outside
    assert q[5] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# E/J relationship
# ---------------------------------------------------------------------------


def test_compute_k_from_j_plane_strain_vs_plane_stress():
    J = 0.5
    E = 210_000.0
    nu = 0.3
    K_ps = compute_k_from_j_elastic(J, E, nu, plane_strain=False)
    K_pe = compute_k_from_j_elastic(J, E, nu, plane_strain=True)
    assert K_ps == pytest.approx(np.sqrt(J * E), rel=1e-12)
    assert K_pe == pytest.approx(np.sqrt(J * E / (1.0 - nu**2)), rel=1e-12)
    # Plane strain always stiffer than plane stress.
    assert K_pe > K_ps


# ---------------------------------------------------------------------------
# Uniform field — no crack — J should be (approximately) zero
# ---------------------------------------------------------------------------


def test_j_integral_on_uniform_stress_field_is_zero():
    mesh = _structured_tri_plate(lx=10.0, ly=10.0, nx=20, ny=20)
    E = 210_000.0
    nu = 0.3
    sig_yy = 100.0

    # Plane-strain uniaxial pull in y:
    # eps_yy = (1 - nu^2) / E * σ_yy,  eps_xx = -nu(1+nu)/E σ_yy
    eps_yy = (1.0 - nu**2) / E * sig_yy
    eps_xx = -nu * (1.0 + nu) / E * sig_yy

    nodes_xy = mesh.nodes[:, :2]
    disp = np.zeros((mesh.n_nodes, 3))
    disp[:, 0] = eps_xx * nodes_xy[:, 0]
    disp[:, 1] = eps_yy * nodes_xy[:, 1]

    stress_values = np.zeros((mesh.n_nodes, 6))
    stress_values[:, 1] = sig_yy  # σ_yy

    strain_values = np.zeros((mesh.n_nodes, 6))
    strain_values[:, 0] = eps_xx
    strain_values[:, 1] = eps_yy

    results = FEAResults(
        mesh=mesh,
        displacement=disp,
        stress=StressField(values=stress_values),
        strain=strain_values,
    )

    tip = np.array([5.0, 5.0])
    res = j_integral_2d(
        results,
        crack_tip=tip,
        q_function_radius=2.0,
        mode="elastic_plane_strain",
        E=E,
        nu=nu,
    )

    # Reference energy scale used to non-dimensionalise: W_ref * L_ref^2
    W_ref = 0.5 * sig_yy * eps_yy
    # Tolerate numerical noise from the CST projection; J should be
    # orders-of-magnitude smaller than the "dummy" scale.
    assert abs(res.J_value) < 1e-3 * W_ref * 4.0  # very loose, just sanity
    assert isinstance(res, JResult)


# ---------------------------------------------------------------------------
# Center-cracked plate — imposed K field
# ---------------------------------------------------------------------------


def test_j_integral_center_cracked_plate():
    # 20x20 plate centered at origin; impose a pure K_I = σ sqrt(π a) Williams
    # field with the crack tip at the origin.
    sigma = 100.0  # MPa
    a = 2.0        # mm half-crack
    E = 210_000.0
    nu = 0.3
    K_I_target = sigma * np.sqrt(np.pi * a)

    mesh = _structured_tri_plate(lx=20.0, ly=20.0, nx=20, ny=20, origin=(-10.0, -10.0))
    tip = np.array([0.0, 0.0])

    results = _impose_williams_field(mesh, tip, K_I=K_I_target, K_II=0.0, E=E, nu=nu)

    # Pick a q-function radius well inside the plate to keep the
    # integration domain away from the outer boundary.
    res = j_integral_2d(
        results,
        crack_tip=tip,
        q_function_radius=5.0,
        mode="elastic_plane_strain",
        E=E,
        nu=nu,
    )

    E_eff = E / (1.0 - nu**2)
    J_target = K_I_target**2 / E_eff
    # Tight-ish tolerance: centroid quadrature on TRI3 with analytic fields
    # should match within ~15%.
    assert res.J_value == pytest.approx(J_target, rel=0.15)
    assert res.K_I == pytest.approx(K_I_target, rel=0.10)


# ---------------------------------------------------------------------------
# Interaction integral — pure K_I
# ---------------------------------------------------------------------------


def test_interaction_integral_pure_mode_I():
    sigma = 100.0
    a = 2.0
    E = 210_000.0
    nu = 0.3
    K_I_target = sigma * np.sqrt(np.pi * a)

    mesh = _structured_tri_plate(lx=20.0, ly=20.0, nx=20, ny=20, origin=(-10.0, -10.0))
    tip = np.array([0.0, 0.0])
    results = _impose_williams_field(mesh, tip, K_I=K_I_target, K_II=0.0, E=E, nu=nu)

    res_I = interaction_integral(
        results, crack_tip=tip, q_function_radius=5.0,
        auxiliary_mode="I", E=E, nu=nu,
    )
    res_II = interaction_integral(
        results, crack_tip=tip, q_function_radius=5.0,
        auxiliary_mode="II", E=E, nu=nu,
    )

    assert res_I.K_I == pytest.approx(K_I_target, rel=0.20)
    # Mode II channel should be small for a pure mode-I field.
    assert abs(res_II.K_II) < 0.2 * K_I_target
