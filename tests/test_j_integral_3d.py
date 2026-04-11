"""Tests for 3D J-integral along a crack front (Track F5)."""

from __future__ import annotations

import numpy as np
import pytest

from feaweld.core.types import ElementType, FEAResults, FEMesh, Point3D, StressField
from feaweld.fracture.j_integral import j_integral_3d, _williams_auxiliary_fields
from feaweld.geometry.weld_path import WeldPath


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------


def _tet4_plate(
    lx: float,
    ly: float,
    lz: float,
    nx: int,
    ny: int,
    nz: int,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> FEMesh:
    """Build a simple TET4 mesh of a plate by splitting each hex into 6 tets."""
    x0, y0, z0 = origin
    xs = np.linspace(x0, x0 + lx, nx + 1)
    ys = np.linspace(y0, y0 + ly, ny + 1)
    zs = np.linspace(z0, z0 + lz, nz + 1)

    nodes = np.array(
        [[x, y, z] for z in zs for y in ys for x in xs],
        dtype=np.float64,
    )

    def nid(i: int, j: int, k: int) -> int:
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i

    # Standard hex -> 6 tets decomposition (Delaunay-consistent).
    tets: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = nid(i, j, k)
                n1 = nid(i + 1, j, k)
                n2 = nid(i + 1, j + 1, k)
                n3 = nid(i, j + 1, k)
                n4 = nid(i, j, k + 1)
                n5 = nid(i + 1, j, k + 1)
                n6 = nid(i + 1, j + 1, k + 1)
                n7 = nid(i, j + 1, k + 1)
                tets.append([n0, n1, n2, n6])
                tets.append([n0, n2, n3, n6])
                tets.append([n0, n3, n7, n6])
                tets.append([n0, n7, n4, n6])
                tets.append([n0, n4, n5, n6])
                tets.append([n0, n5, n1, n6])

    return FEMesh(
        nodes=nodes,
        elements=np.array(tets, dtype=np.int64),
        element_type=ElementType.TET4,
    )


def _impose_williams_3d(
    mesh: FEMesh,
    crack_tip_xy: tuple[float, float],
    K_I: float,
    E: float = 210000.0,
    nu: float = 0.3,
) -> FEAResults:
    """Impose a plane-strain Mode-I Williams field on a 3D TET4 mesh.

    The field is set up in the (x, y) plane; the z axis is along the
    crack front and is taken to be plane-strain.
    """
    nodes = mesh.nodes
    rel = nodes[:, :2] - np.asarray(crack_tip_xy, dtype=np.float64)[None, :]
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])

    u, s, eps = _williams_auxiliary_fields(r, theta, "I", E, nu, plane_strain=True)
    disp = np.zeros((mesh.n_nodes, 3))
    disp[:, 0] = K_I * u[:, 0]
    disp[:, 1] = K_I * u[:, 1]

    sxx = K_I * s[:, 0]
    syy = K_I * s[:, 1]
    txy = K_I * s[:, 2]
    # Plane strain: σ_zz = nu * (σ_xx + σ_yy)
    szz = nu * (sxx + syy)

    eps_xx = K_I * eps[:, 0]
    eps_yy = K_I * eps[:, 1]
    gxy = K_I * eps[:, 2]

    stress_values = np.zeros((mesh.n_nodes, 6))
    stress_values[:, 0] = sxx
    stress_values[:, 1] = syy
    stress_values[:, 2] = szz
    stress_values[:, 3] = txy  # τ_xy
    # τ_yz, τ_xz stay zero

    strain_values = np.zeros((mesh.n_nodes, 6))
    strain_values[:, 0] = eps_xx
    strain_values[:, 1] = eps_yy
    # eps_zz is zero (plane strain)
    strain_values[:, 3] = gxy

    return FEAResults(
        mesh=mesh,
        displacement=disp,
        stress=StressField(values=stress_values),
        strain=strain_values,
    )


def _impose_varying_williams_3d(
    mesh: FEMesh,
    crack_tip_xy: tuple[float, float],
    K_I_max: float,
    E: float = 210000.0,
    nu: float = 0.3,
) -> FEAResults:
    """Impose a plane-strain Williams field with K_I that peaks at mid-span.

    K_I(z) = K_I_max * sin(pi * z / lz), giving a "deepest-point" peak
    behaviour.
    """
    nodes = mesh.nodes
    z = nodes[:, 2]
    z_min, z_max = z.min(), z.max()
    lz = z_max - z_min
    amp = np.sin(np.pi * (z - z_min) / lz) * K_I_max

    rel = nodes[:, :2] - np.asarray(crack_tip_xy, dtype=np.float64)[None, :]
    r = np.linalg.norm(rel, axis=1)
    theta = np.arctan2(rel[:, 1], rel[:, 0])

    u, s, eps = _williams_auxiliary_fields(r, theta, "I", E, nu, plane_strain=True)

    disp = np.zeros((mesh.n_nodes, 3))
    disp[:, 0] = amp * u[:, 0]
    disp[:, 1] = amp * u[:, 1]

    sxx = amp * s[:, 0]
    syy = amp * s[:, 1]
    txy = amp * s[:, 2]
    szz = nu * (sxx + syy)

    eps_xx = amp * eps[:, 0]
    eps_yy = amp * eps[:, 1]
    gxy = amp * eps[:, 2]

    stress_values = np.zeros((mesh.n_nodes, 6))
    stress_values[:, 0] = sxx
    stress_values[:, 1] = syy
    stress_values[:, 2] = szz
    stress_values[:, 3] = txy

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
# Tests
# ---------------------------------------------------------------------------


def test_j_integral_3d_straight_front_symmetric():
    """Symmetric (uniform) K field along the front -> J at symmetric samples agree."""
    lx, ly, lz = 20.0, 20.0, 10.0
    mesh = _tet4_plate(
        lx=lx, ly=ly, lz=lz, nx=10, ny=10, nz=6, origin=(-10.0, -10.0, 0.0)
    )
    E, nu = 210_000.0, 0.3
    K_I_target = 100.0 * np.sqrt(np.pi * 2.0)  # arbitrary amplitude
    results = _impose_williams_3d(mesh, (0.0, 0.0), K_I=K_I_target, E=E, nu=nu)

    # Straight crack front along z from z=0 -> z=lz at x=0, y=0.
    front = WeldPath(
        control_points=[Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, lz)],
        mode="linear",
    )

    j_samples = j_integral_3d(
        results,
        crack_front=front,
        q_function_radius=4.0,
        n_front_samples=5,
        E=E,
        nu=nu,
    )

    assert len(j_samples) == 5
    for res in j_samples:
        assert res.J_value >= 0.0

    # Samples symmetric about the mid-point should agree within 5 %
    # (take interior samples 1 and 3, avoiding boundary sample 0 and 4).
    j_mid = j_samples[2].J_value
    j_left = j_samples[1].J_value
    j_right = j_samples[3].J_value
    assert j_mid > 0.0
    rel_err = abs(j_left - j_right) / max(j_mid, 1e-30)
    assert rel_err < 0.05, f"Symmetric samples differ by {rel_err:.3f}"


def test_j_integral_3d_varying_front_peak_at_center():
    """K field that peaks mid-span -> J at the center sample > J at the ends."""
    lx, ly, lz = 20.0, 20.0, 10.0
    mesh = _tet4_plate(
        lx=lx, ly=ly, lz=lz, nx=10, ny=10, nz=8, origin=(-10.0, -10.0, 0.0)
    )
    E, nu = 210_000.0, 0.3
    results = _impose_varying_williams_3d(
        mesh, (0.0, 0.0), K_I_max=100.0 * np.sqrt(np.pi * 2.0), E=E, nu=nu
    )

    front = WeldPath(
        control_points=[Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, lz)],
        mode="linear",
    )

    j_samples = j_integral_3d(
        results,
        crack_front=front,
        q_function_radius=4.0,
        n_front_samples=5,
        E=E,
        nu=nu,
    )

    assert len(j_samples) == 5
    for res in j_samples:
        assert res.J_value >= 0.0

    # Mid-span sample should dominate the end samples (sin peaks at s=lz/2).
    j_center = j_samples[2].J_value
    j_end_lo = j_samples[0].J_value
    j_end_hi = j_samples[-1].J_value
    # The ends may be essentially zero (boundary/no nodes) so just
    # enforce strict ordering with a small margin.
    assert j_center > j_end_lo
    assert j_center > j_end_hi


def test_j_integral_3d_wrong_element_type_raises():
    """Calling with a TRI3 mesh must raise NotImplementedError."""
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = FEMesh(nodes=nodes, elements=elements, element_type=ElementType.TRI3)
    results = FEAResults(
        mesh=mesh,
        displacement=np.zeros((3, 3)),
        stress=StressField(values=np.zeros((3, 6))),
    )
    front = WeldPath(
        control_points=[Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, 1.0)],
        mode="linear",
    )
    with pytest.raises(NotImplementedError, match="TET4"):
        j_integral_3d(
            results,
            crack_front=front,
            q_function_radius=0.5,
            n_front_samples=3,
        )
