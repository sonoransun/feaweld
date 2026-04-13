"""Bourdin/Francfort variational phase-field brittle fracture (Track A4).

Implements the AT2-style regularized fracture functional

    Psi(u, d) = int (1 - d)^2 psi_el(eps(u))
              + G_c * ( d^2 / (2 l0) + (l0/2) |grad d|^2 ) dV

on top of the A1 JAX differentiable backend for TRI3 / plane-strain meshes.
The solve is a classic staggered alternating minimization:

1. Apply the load in ``n_load_steps`` equal fractions.
2. At each load step, alternate

       u-step : solve K_u(d) u = f(lambda)     with degraded elasticity
       d-step : solve K_d(H) d = f_d(H)        with history H = max_t psi_el

   until ``||d_{k+1} - d_k|| < staggered_tol`` or ``max_staggered_iters``.
3. Track the monotone history field so damage can only increase.
4. Record the reaction force along the loaded node set at each step.

The displacement element stiffness reuses ``_tri3_stiffness`` from
``feaweld.solver.jax_backend`` — we multiply each element contribution by
the degradation ``(1 - <d_e>)^2 + k`` where ``<d_e>`` is the mean of the
three nodal damage values and ``k ~ 1e-6`` is the residual stiffness.

JAX is imported with a guard: the module imports cleanly without JAX,
but calling :func:`solve_phase_field` raises :class:`ImportError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:  # pragma: no cover - JAX optional
    _HAS_JAX = False
    jax = None  # type: ignore
    jnp = None  # type: ignore

from feaweld.core.materials import Material
from feaweld.core.types import (
    BoundaryCondition,
    ElementType,
    FEMesh,
    LoadCase,
    LoadType,
)
from feaweld.fracture.types import FractureResult


_PENALTY = 1.0e12
_RESIDUAL_STIFFNESS = 1.0e-6  # k in g(d) = (1-d)^2 + k


# ---------------------------------------------------------------------------
# Energy decomposition enum
# ---------------------------------------------------------------------------


class EnergyDecomposition(str, Enum):
    """Elastic energy split strategy for the phase-field driving force."""
    NONE = "none"
    SPECTRAL = "spectral"
    VOLUMETRIC_DEVIATORIC = "volumetric_deviatoric"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PhaseFieldConfig:
    """Configuration for :func:`solve_phase_field`.

    Attributes
    ----------
    l0:
        Regularization length (same units as mesh coordinates, typically mm).
        Controls the width of the diffuse crack; must be resolved by the mesh.
    Gc:
        Critical energy release rate (N/mm for 2D plane strain).
    n_load_steps:
        Number of equal-increment load steps from 0 to ``max_load``.
    max_load:
        Final load multiplier applied at step ``n_load_steps``.
    staggered_tol:
        Convergence tolerance on ``||d_{k+1} - d_k||`` (L2 norm) per load step.
    max_staggered_iters:
        Maximum alternating u/d iterations per load step.
    min_mesh_ratio:
        Minimum ``l0 / h_avg`` required; raises ``ValueError`` otherwise.
        The usual Bourdin rule of thumb is ``h <= l0 / 2`` to resolve the
        diffuse crack; we default to ``4`` for safety.
    energy_split:
        Elastic energy decomposition for the damage driving force.
        ``NONE`` uses the full energy (original AT2), ``SPECTRAL`` uses the
        Miehe spectral split, ``VOLUMETRIC_DEVIATORIC`` uses the Amor split.
    """

    l0: float = 0.1
    Gc: float = 2.7
    n_load_steps: int = 20
    max_load: float = 1.0
    staggered_tol: float = 1e-4
    max_staggered_iters: int = 50
    min_mesh_ratio: float = 4.0
    energy_split: EnergyDecomposition = EnergyDecomposition.NONE


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _require_jax() -> None:
    if not _HAS_JAX:
        raise ImportError(
            "JAX is required for solve_phase_field. "
            "Install with: pip install 'feaweld[jax]'"
        )
    # Match the A1 backend: enable float64 for FEA-grade accuracy.
    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------


def _tri_edge_lengths(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Edge lengths for a single TRI3 (coords is (3, 2) or (3, 3))."""
    e0 = np.linalg.norm(coords[1] - coords[0])
    e1 = np.linalg.norm(coords[2] - coords[1])
    e2 = np.linalg.norm(coords[0] - coords[2])
    return np.array([e0, e1, e2])


def _tet_edge_lengths(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Edge lengths for a single TET4 (coords is (4, 3))."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return np.array([np.linalg.norm(coords[j] - coords[i]) for i, j in pairs])


def _average_edge_length(mesh: FEMesh) -> float:
    coords = mesh.nodes
    acc = 0.0
    cnt = 0
    if mesh.element_type is ElementType.TET4:
        for conn in mesh.elements:
            tet = coords[conn]
            lens = _tet_edge_lengths(tet)
            acc += float(lens.sum())
            cnt += 6
    else:
        for conn in mesh.elements:
            tri = coords[conn]
            lens = _tri_edge_lengths(tri)
            acc += float(lens.sum())
            cnt += 3
    if cnt == 0:
        return 0.0
    return acc / cnt


# ---------------------------------------------------------------------------
# Element-level damage operators (pure numpy — cheap)
# ---------------------------------------------------------------------------


def _tri3_bd_area(coords_2d: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Return the scalar-gradient B matrix (2,3) and element area for a CST.

    For a linear triangle with nodes ``(x_i, y_i)``

        B_d = (1 / (2 A)) * [[y1 - y2, y2 - y0, y0 - y1],
                              [x2 - x1, x0 - x2, x1 - x0]]

    such that ``grad phi = B_d @ phi_nodes``.
    """
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    two_area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    area = 0.5 * abs(two_area)
    Bd = np.stack([b, c]) / two_area
    return Bd, area


# Consistent CST mass matrix: (A/12) * [[2,1,1],[1,2,1],[1,1,2]]
_M_CST = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0


# ---------------------------------------------------------------------------
# Assembly helpers (displacement path uses _tri3_stiffness from A1)
# ---------------------------------------------------------------------------


def _precompute_element_kinematics(
    mesh: FEMesh, C_ps: NDArray[np.float64], thickness: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Precompute per-element (B, K0, area) for a TRI3 mesh.

    ``K0`` is the undegraded element stiffness ``t * area * B^T C B``.
    ``B`` has shape ``(n_elem, 3, 6)`` and ``K0`` shape ``(n_elem, 6, 6)``.

    We use the A1 ``_tri3_stiffness`` for a single canonical element so the
    B-matrix layout stays identical; since CST kinematics are trivial the
    JAX call is a one-off *smoke check*, not a per-iteration cost.
    """
    # Delegate to the JAX backend for one element so we can assert our CST
    # B matrix matches.  The actual per-element loop below uses pure NumPy.
    from feaweld.solver.jax_backend import _tri3_stiffness  # local import

    n_elem = mesh.n_elements
    coords2d = np.asarray(mesh.nodes[:, :2], dtype=np.float64)

    B_all = np.zeros((n_elem, 3, 6), dtype=np.float64)
    K0_all = np.zeros((n_elem, 6, 6), dtype=np.float64)
    area_all = np.zeros(n_elem, dtype=np.float64)

    for e in range(n_elem):
        conn = mesh.elements[e]
        tri = coords2d[conn]
        x = tri[:, 0]
        y = tri[:, 1]
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
        two_area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        area = 0.5 * abs(two_area)

        B = np.zeros((3, 6), dtype=np.float64)
        for i in range(3):
            B[0, 2 * i] = b[i]
            B[1, 2 * i + 1] = c[i]
            B[2, 2 * i] = c[i]
            B[2, 2 * i + 1] = b[i]
        B = B / two_area

        ke0 = thickness * area * (B.T @ C_ps @ B)
        B_all[e] = B
        K0_all[e] = ke0
        area_all[e] = area

    # One-off sanity check against the JAX backend implementation.
    if n_elem > 0:
        tri0 = jnp.asarray(coords2d[mesh.elements[0]], dtype=jnp.float64)
        C_j = jnp.asarray(C_ps, dtype=jnp.float64)
        ke_ref, _ = _tri3_stiffness(tri0, C_j, thickness)
        ke_ref_np = np.asarray(ke_ref, dtype=np.float64)
        if not np.allclose(K0_all[0], ke_ref_np, rtol=1e-8, atol=1e-8):
            # This should never trigger — guard only.
            raise RuntimeError(
                "CST stiffness disagrees with jax_backend._tri3_stiffness"
            )

    return B_all, K0_all, area_all


def _element_dofs(mesh: FEMesh) -> NDArray[np.int64]:
    """Return (n_elem, 6) global DOF indices per element for TRI3."""
    elems = mesh.elements.astype(np.int64)
    n_elem = elems.shape[0]
    dofs = np.empty((n_elem, 6), dtype=np.int64)
    for i in range(3):
        dofs[:, 2 * i] = 2 * elems[:, i]
        dofs[:, 2 * i + 1] = 2 * elems[:, i] + 1
    return dofs


def _assemble_Ku_tri3(
    mesh: FEMesh,
    K0_all: NDArray[np.float64],
    dofs_all: NDArray[np.int64],
    damage: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble the degraded displacement stiffness on TRI3 / plane strain.

    Each element contribution is scaled by ``g(<d>) = (1 - <d_e>)^2 + k``.
    ``K0_all`` holds undegraded element stiffnesses (precomputed once).
    """
    n_dof = mesh.n_nodes * 2
    K = np.zeros((n_dof, n_dof), dtype=np.float64)

    elems = mesh.elements.astype(np.int64)
    d_nodal = damage[elems]              # (n_elem, 3)
    d_bar = d_nodal.mean(axis=1)         # (n_elem,)
    g = (1.0 - d_bar) ** 2 + _RESIDUAL_STIFFNESS  # (n_elem,)

    for e in range(mesh.n_elements):
        ke = g[e] * K0_all[e]
        dofs = dofs_all[e]
        # np.ix_ scatter
        K[np.ix_(dofs, dofs)] += ke

    return K


def _precompute_damage_kinematics(
    mesh: FEMesh, thickness: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Precompute per-element (Me, Be) for the damage subproblem.

    ``Me`` is the consistent mass matrix contribution ``(vol/12) * [[2,1,1],[1,2,1],[1,1,2]]``
    and ``Be = vol * B_d^T B_d`` is the Laplacian contribution for a CST.
    """
    n_elem = mesh.n_elements
    coords = mesh.nodes[:, :2]
    Me_all = np.zeros((n_elem, 3, 3), dtype=np.float64)
    Be_all = np.zeros((n_elem, 3, 3), dtype=np.float64)

    for e in range(n_elem):
        conn = mesh.elements[e].astype(np.int64)
        tri = coords[conn]
        Bd, area = _tri3_bd_area(tri)
        vol = area * thickness
        Me_all[e] = _M_CST * vol
        Be_all[e] = (Bd.T @ Bd) * vol
    return Me_all, Be_all


def _assemble_Kd_fd_tri3(
    mesh: FEMesh,
    history: NDArray[np.float64],
    Gc: float,
    l0: float,
    Me_all: NDArray[np.float64],
    Be_all: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Assemble the damage sub-problem stiffness and right-hand side.

    The AT2 functional yields the linear tangent

        K_d = int [ (Gc/l0 + 2 H) N^T N + Gc l0 (grad N)^T (grad N) ] dV
        f_d = int [ 2 H N ] dV

    with ``H`` the element-wise history field (mean of nodal values).
    """
    n = mesh.n_nodes
    Kd = np.zeros((n, n), dtype=np.float64)
    fd = np.zeros(n, dtype=np.float64)

    elems = mesh.elements.astype(np.int64)
    H_node = history[elems]           # (n_elem, 3)
    H_e = H_node.mean(axis=1)         # (n_elem,)

    # int N dV = vol / 3 * ones for a CST.  vol = sum(Me) / 1?  Actually
    # sum of row of the consistent mass matrix (unit density, unit field)
    # equals vol/3 exactly — use that to recover the volume without
    # re-deriving it.
    vol_all = Me_all.sum(axis=(1, 2))  # (n_elem,) — (4*vol/12)*3 = vol
    # Sanity: vol/3 * 3 nodes * mean = vol.  Me sums to vol.
    # We can double-check: (A/12)*( (2+1+1) + (1+2+1) + (1+1+2) ) = 12A/12 = A. ✓

    for e in range(mesh.n_elements):
        ke = (Gc / l0 + 2.0 * H_e[e]) * Me_all[e] + Gc * l0 * Be_all[e]
        # Load: 2 H * int N dV = 2 H * (vol / 3) * ones
        fe = (2.0 * H_e[e]) * (vol_all[e] / 3.0) * np.ones(3)

        conn = elems[e]
        Kd[np.ix_(conn, conn)] += ke
        fd[conn] += fe

    return Kd, fd


# ---------------------------------------------------------------------------
# TET4 / 3D element routines
# ---------------------------------------------------------------------------


# Consistent TET4 mass matrix: (V/20) * [[2,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2]]
_M_TET4 = np.array([
    [2.0, 1.0, 1.0, 1.0],
    [1.0, 2.0, 1.0, 1.0],
    [1.0, 1.0, 2.0, 1.0],
    [1.0, 1.0, 1.0, 2.0],
]) / 20.0


def _tet4_shape_derivs(
    coords: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Return shape-function derivatives (4,3) and volume for a TET4.

    Uses cofactors of the ``[1, x, y, z]`` matrix.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    M = np.ones((4, 4), dtype=np.float64)
    M[:, 1] = x
    M[:, 2] = y
    M[:, 3] = z
    det_M = np.linalg.det(M)
    vol = abs(det_M) / 6.0

    # Cofactors of columns 1, 2, 3 give dN/dx, dN/dy, dN/dz
    inv_M = np.linalg.inv(M)  # inv_M.T[j, i] = cofactor(i, j) / det
    dN = inv_M[1:, :].T  # (4, 3) — rows = nodes, cols = (dx, dy, dz)
    return dN, vol


def _precompute_element_kinematics_3d(
    mesh: FEMesh, C6: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Precompute per-element (B, K0, vol) for a TET4 mesh.

    ``B`` has shape ``(n_elem, 6, 12)`` and ``K0`` shape ``(n_elem, 12, 12)``.
    """
    n_elem = mesh.n_elements
    coords = np.asarray(mesh.nodes, dtype=np.float64)

    B_all = np.zeros((n_elem, 6, 12), dtype=np.float64)
    K0_all = np.zeros((n_elem, 12, 12), dtype=np.float64)
    vol_all = np.zeros(n_elem, dtype=np.float64)

    for e in range(n_elem):
        conn = mesh.elements[e]
        tet = coords[conn]
        dN, vol = _tet4_shape_derivs(tet)

        B = np.zeros((6, 12), dtype=np.float64)
        for i in range(4):
            c = 3 * i
            b_i, c_i, d_i = dN[i, 0], dN[i, 1], dN[i, 2]
            B[0, c] = b_i          # eps_xx
            B[1, c + 1] = c_i      # eps_yy
            B[2, c + 2] = d_i      # eps_zz
            B[3, c] = c_i          # gamma_xy
            B[3, c + 1] = b_i
            B[4, c + 1] = d_i      # gamma_yz
            B[4, c + 2] = c_i
            B[5, c] = d_i          # gamma_xz
            B[5, c + 2] = b_i

        K0 = vol * (B.T @ C6 @ B)
        B_all[e] = B
        K0_all[e] = K0
        vol_all[e] = vol

    return B_all, K0_all, vol_all


def _element_dofs_3d(mesh: FEMesh) -> NDArray[np.int64]:
    """Return (n_elem, 12) global DOF indices per element for TET4."""
    elems = mesh.elements.astype(np.int64)
    n_elem = elems.shape[0]
    dofs = np.empty((n_elem, 12), dtype=np.int64)
    for i in range(4):
        dofs[:, 3 * i] = 3 * elems[:, i]
        dofs[:, 3 * i + 1] = 3 * elems[:, i] + 1
        dofs[:, 3 * i + 2] = 3 * elems[:, i] + 2
    return dofs


def _assemble_Ku_tet4(
    mesh: FEMesh,
    K0_all: NDArray[np.float64],
    dofs_all: NDArray[np.int64],
    damage: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble the degraded displacement stiffness on TET4 / 3D."""
    n_dof = mesh.n_nodes * 3
    K = np.zeros((n_dof, n_dof), dtype=np.float64)

    elems = mesh.elements.astype(np.int64)
    d_nodal = damage[elems]
    d_bar = d_nodal.mean(axis=1)
    g = (1.0 - d_bar) ** 2 + _RESIDUAL_STIFFNESS

    for e in range(mesh.n_elements):
        ke = g[e] * K0_all[e]
        dofs = dofs_all[e]
        K[np.ix_(dofs, dofs)] += ke

    return K


def _precompute_damage_kinematics_3d(
    mesh: FEMesh,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Precompute per-element (Me, Be) for the damage subproblem on TET4."""
    n_elem = mesh.n_elements
    coords = np.asarray(mesh.nodes, dtype=np.float64)
    Me_all = np.zeros((n_elem, 4, 4), dtype=np.float64)
    Be_all = np.zeros((n_elem, 4, 4), dtype=np.float64)

    for e in range(n_elem):
        conn = mesh.elements[e].astype(np.int64)
        tet = coords[conn]
        dN, vol = _tet4_shape_derivs(tet)
        Me_all[e] = _M_TET4 * vol
        Be_all[e] = (dN @ dN.T) * vol  # (4,3)(3,4) = (4,4)
    return Me_all, Be_all


def _assemble_Kd_fd_tet4(
    mesh: FEMesh,
    history: NDArray[np.float64],
    Gc: float,
    l0: float,
    Me_all: NDArray[np.float64],
    Be_all: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Assemble the damage sub-problem stiffness and RHS for TET4."""
    n = mesh.n_nodes
    Kd = np.zeros((n, n), dtype=np.float64)
    fd = np.zeros(n, dtype=np.float64)

    elems = mesh.elements.astype(np.int64)
    H_node = history[elems]
    H_e = H_node.mean(axis=1)
    vol_all = Me_all.sum(axis=(1, 2))

    for e in range(mesh.n_elements):
        ke = (Gc / l0 + 2.0 * H_e[e]) * Me_all[e] + Gc * l0 * Be_all[e]
        fe = (2.0 * H_e[e]) * (vol_all[e] / 4.0) * np.ones(4)

        conn = elems[e]
        Kd[np.ix_(conn, conn)] += ke
        fd[conn] += fe

    return Kd, fd


# ---------------------------------------------------------------------------
# Energy split functions
# ---------------------------------------------------------------------------


def _spectral_split_2d(
    eps_batch: NDArray[np.float64],
    lam: float,
    mu: float,
) -> NDArray[np.float64]:
    """Miehe spectral split for 2D plane strain. Returns psi_plus per element.

    ``eps_batch`` has shape ``(n_elem, 3)`` in Voigt: [eps_xx, eps_yy, gamma_xy].
    For plane strain, eps_zz = 0 is the third eigenvalue.
    """
    n = eps_batch.shape[0]
    psi_plus = np.zeros(n, dtype=np.float64)

    for i in range(n):
        exx, eyy, gxy = eps_batch[i, 0], eps_batch[i, 1], eps_batch[i, 2]
        # 2x2 strain tensor
        eps_tensor = np.array([[exx, 0.5 * gxy], [0.5 * gxy, eyy]])
        eigvals = np.linalg.eigvalsh(eps_tensor)
        # Include eps_zz = 0 as third eigenvalue for plane strain
        all_eigvals = np.array([eigvals[0], eigvals[1], 0.0])
        tr_eps = all_eigvals.sum()
        tr_plus = max(tr_eps, 0.0)
        eigvals_plus = np.maximum(all_eigvals, 0.0)
        psi_plus[i] = 0.5 * lam * tr_plus ** 2 + mu * np.sum(eigvals_plus ** 2)

    return psi_plus


def _amor_split_2d(
    eps_batch: NDArray[np.float64],
    kappa: float,
    mu: float,
) -> NDArray[np.float64]:
    """Amor volumetric-deviatoric split for 2D plane strain. Returns psi_plus.

    ``kappa`` is the bulk modulus: lambda + 2*mu/ndim (ndim=2 for plane strain).
    """
    n = eps_batch.shape[0]
    psi_plus = np.zeros(n, dtype=np.float64)

    for i in range(n):
        exx, eyy, gxy = eps_batch[i, 0], eps_batch[i, 1], eps_batch[i, 2]
        tr_eps = exx + eyy  # eps_zz = 0 for plane strain
        tr_plus = max(tr_eps, 0.0)
        # Deviatoric strain in Voigt (plane strain: eps_zz = 0, include it)
        e_vol = tr_eps / 3.0
        dev_xx = exx - e_vol
        dev_yy = eyy - e_vol
        dev_zz = -e_vol  # 0 - tr/3
        dev_xy = 0.5 * gxy
        dev_sq = dev_xx ** 2 + dev_yy ** 2 + dev_zz ** 2 + 2.0 * dev_xy ** 2
        psi_plus[i] = 0.5 * kappa * tr_plus ** 2 + mu * dev_sq

    return psi_plus


def _spectral_split_3d(
    eps_batch: NDArray[np.float64],
    lam: float,
    mu: float,
) -> NDArray[np.float64]:
    """Miehe spectral split for 3D. Returns psi_plus per element.

    ``eps_batch`` has shape ``(n_elem, 6)`` in Voigt:
    [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz].
    """
    n = eps_batch.shape[0]
    psi_plus = np.zeros(n, dtype=np.float64)

    for i in range(n):
        e = eps_batch[i]
        eps_tensor = np.array([
            [e[0], 0.5 * e[3], 0.5 * e[5]],
            [0.5 * e[3], e[1], 0.5 * e[4]],
            [0.5 * e[5], 0.5 * e[4], e[2]],
        ])
        eigvals = np.linalg.eigvalsh(eps_tensor)
        tr_eps = eigvals.sum()
        tr_plus = max(tr_eps, 0.0)
        eigvals_plus = np.maximum(eigvals, 0.0)
        psi_plus[i] = 0.5 * lam * tr_plus ** 2 + mu * np.sum(eigvals_plus ** 2)

    return psi_plus


def _amor_split_3d(
    eps_batch: NDArray[np.float64],
    kappa: float,
    mu: float,
) -> NDArray[np.float64]:
    """Amor volumetric-deviatoric split for 3D. Returns psi_plus."""
    n = eps_batch.shape[0]
    psi_plus = np.zeros(n, dtype=np.float64)

    for i in range(n):
        e = eps_batch[i]
        tr_eps = e[0] + e[1] + e[2]
        tr_plus = max(tr_eps, 0.0)
        e_vol = tr_eps / 3.0
        dev = np.array([
            e[0] - e_vol, e[1] - e_vol, e[2] - e_vol,
            0.5 * e[3], 0.5 * e[4], 0.5 * e[5],
        ])
        dev_sq = dev[0] ** 2 + dev[1] ** 2 + dev[2] ** 2 + 2.0 * (
            dev[3] ** 2 + dev[4] ** 2 + dev[5] ** 2
        )
        psi_plus[i] = 0.5 * kappa * tr_plus ** 2 + mu * dev_sq

    return psi_plus


# ---------------------------------------------------------------------------
# Elastic energy density
# ---------------------------------------------------------------------------


def _element_psi_el(
    mesh: FEMesh,
    u: NDArray[np.float64],
    C: NDArray[np.float64],
    B_all: NDArray[np.float64],
    dofs_all: NDArray[np.int64],
    energy_split: EnergyDecomposition = EnergyDecomposition.NONE,
    lam: float = 0.0,
    mu: float = 0.0,
) -> NDArray[np.float64]:
    """Return per-element *undegraded* elastic energy density (or psi_plus).

    When ``energy_split`` is ``NONE``, returns the full energy density
    ``psi_el = 0.5 * eps^T C eps``.  Otherwise returns the tensile part only.
    """
    n_voigt = B_all.shape[1]  # 3 for TRI3, 6 for TET4
    dof_per_elem = B_all.shape[2]
    u_elem = u[dofs_all]                               # (n_elem, dof_per_elem)
    eps = np.einsum("eij,ej->ei", B_all, u_elem)       # (n_elem, n_voigt)

    if energy_split is EnergyDecomposition.NONE:
        sig = eps @ C.T
        psi = 0.5 * np.sum(eps * sig, axis=1)
        return psi

    is_3d = (n_voigt == 6)
    if energy_split is EnergyDecomposition.SPECTRAL:
        if is_3d:
            return _spectral_split_3d(eps, lam, mu)
        return _spectral_split_2d(eps, lam, mu)
    else:  # VOLUMETRIC_DEVIATORIC
        kappa = lam + 2.0 * mu / (3.0 if is_3d else 2.0)
        if is_3d:
            return _amor_split_3d(eps, kappa, mu)
        return _amor_split_2d(eps, kappa, mu)


def _project_element_to_nodes(
    mesh: FEMesh, elem_vals: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Simple area-weighted (actually count-weighted) element→node projection."""
    n = mesh.n_nodes
    acc = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    for e_idx in range(mesh.n_elements):
        conn = mesh.elements[e_idx]
        for node in conn:
            acc[int(node)] += elem_vals[e_idx]
            cnt[int(node)] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return acc / cnt


# ---------------------------------------------------------------------------
# Dirichlet / force BC gathering
# ---------------------------------------------------------------------------


def _gather_disp_bcs(
    mesh: FEMesh, load_case: LoadCase, scale: float,
    dof_per_node: int = 2,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], list[int]]:
    """Return (constrained_dofs, constrained_vals, f_global, loaded_dofs).

    ``scale`` multiplies both the prescribed displacements (if any) and the
    applied force magnitudes, so a single call produces the BCs at a given
    load fraction. ``loaded_dofs`` lists DOFs where a force BC was applied
    — these are used to evaluate the reaction at the end of each step.
    """
    n_dof = mesh.n_nodes * dof_per_node
    f = np.zeros(n_dof, dtype=np.float64)
    cons_dofs: list[int] = []
    cons_vals: list[float] = []
    loaded_dofs: list[int] = []

    for bc in load_case.constraints:
        if bc.bc_type != LoadType.DISPLACEMENT:
            continue
        if bc.node_set not in mesh.node_sets:
            continue
        nodes = mesh.node_sets[bc.node_set]
        vals = np.asarray(bc.values, dtype=np.float64).reshape(-1)
        for node in nodes:
            for k in range(dof_per_node):
                dof = int(node) * dof_per_node + k
                v = float(vals[k]) if k < len(vals) else 0.0
                cons_dofs.append(dof)
                cons_vals.append(v * scale)

    for bc in load_case.loads:
        if bc.bc_type != LoadType.FORCE:
            continue
        if bc.node_set not in mesh.node_sets:
            continue
        nodes = mesh.node_sets[bc.node_set]
        vals = np.asarray(bc.values, dtype=np.float64).reshape(-1)
        share = vals / max(len(nodes), 1)
        for node in nodes:
            for k in range(dof_per_node):
                dof = int(node) * dof_per_node + k
                comp = float(share[k]) if k < len(share) else 0.0
                f[dof] += comp * scale
                if abs(comp) > 0.0:
                    loaded_dofs.append(dof)

    return (
        np.asarray(cons_dofs, dtype=np.int64),
        np.asarray(cons_vals, dtype=np.float64),
        f,
        loaded_dofs,
    )


def _apply_penalty(
    K: NDArray[np.float64],
    f: NDArray[np.float64],
    cons_dofs: NDArray[np.int64],
    cons_vals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    K2 = K.copy()
    f2 = f.copy()
    for d, v in zip(cons_dofs, cons_vals):
        K2[int(d), int(d)] += _PENALTY
        f2[int(d)] += _PENALTY * float(v)
    return K2, f2


def _reaction_at_loaded(
    K_u: NDArray[np.float64],
    u_flat: NDArray[np.float64],
    loaded_dofs: list[int],
) -> float:
    """Reaction along the loaded DOFs as the row-sum (K u)_i over loaded i.

    Signed sum matches the direction of the applied force: for a tensile
    load this equals the externally reacted force in magnitude.
    """
    if not loaded_dofs:
        return 0.0
    ku = K_u @ u_flat
    return float(np.sum(ku[np.asarray(loaded_dofs, dtype=np.int64)]))


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def solve_phase_field(
    mesh: FEMesh,
    material: Material,
    load_case: LoadCase,
    config: PhaseFieldConfig = PhaseFieldConfig(),
    initial_damage: NDArray[np.float64] | None = None,
) -> FractureResult:
    """Staggered AT2 phase-field fracture solver for TRI3 and TET4 meshes.

    Parameters
    ----------
    mesh:
        FEMesh — must be :attr:`ElementType.TRI3` (2D plane strain) or
        :attr:`ElementType.TET4` (3D).
    material:
        Linear-elastic material; evaluated at ``20 C``.
    load_case:
        Force and/or displacement BCs; each step applies a scaled copy.
    config:
        :class:`PhaseFieldConfig` — regularization, load steps, tolerances.
    initial_damage:
        Optional ``(n_nodes,)`` initial damage field (e.g. a pre-existing
        notch represented by ``d=1`` at a few nodes).

    Returns
    -------
    FractureResult
    """
    _require_jax()

    is_3d = mesh.element_type is ElementType.TET4
    if mesh.element_type not in (ElementType.TRI3, ElementType.TET4):
        raise NotImplementedError(
            f"solve_phase_field supports TRI3 and TET4 only; "
            f"got {mesh.element_type.value}."
        )

    dpn = 3 if is_3d else 2  # DOFs per node
    n_per_elem = 4 if is_3d else 3

    # -- Mesh density guard --------------------------------------------------
    h_avg = _average_edge_length(mesh)
    h_max_allowed = config.l0 / config.min_mesh_ratio
    if h_avg > h_max_allowed:
        raise ValueError(
            f"Mesh too coarse for phase-field l0={config.l0:.4g}: "
            f"avg edge length h={h_avg:.4g} > l0 / min_mesh_ratio = {h_max_allowed:.4g}. "
            f"Refine the mesh or increase config.l0."
        )

    n = mesh.n_nodes
    thickness = 1.0

    if initial_damage is None:
        d = np.zeros(n, dtype=np.float64)
    else:
        d = np.asarray(initial_damage, dtype=np.float64).reshape(-1).copy()
        if d.shape[0] != n:
            raise ValueError(
                f"initial_damage has {d.shape[0]} entries but mesh has {n} nodes."
            )
        d = np.clip(d, 0.0, 1.0)

    history = np.zeros(n, dtype=np.float64)

    # -- Elasticity tensor and Lame parameters for energy split ---------------
    lam_val = material.lame_lambda(20.0)
    mu_val = material.lame_mu(20.0)

    if is_3d:
        C_mat = material.elasticity_tensor_3d(20.0)
    else:
        C_mat = material.elasticity_tensor_2d(20.0, plane="strain")

    # -- Precompute kinematics once (they only depend on geometry) ----------
    if is_3d:
        B_all, K0_all, _vol_all = _precompute_element_kinematics_3d(mesh, C_mat)
        dofs_all = _element_dofs_3d(mesh)
        Me_all, Be_all = _precompute_damage_kinematics_3d(mesh)
    else:
        B_all, K0_all, _area_all = _precompute_element_kinematics(
            mesh, C_mat, thickness
        )
        dofs_all = _element_dofs(mesh)
        Me_all, Be_all = _precompute_damage_kinematics(mesh, thickness)

    # Load fractions: n_load_steps >= 1 steps from > 0 to max_load.
    n_steps = int(config.n_load_steps)
    if n_steps < 1:
        raise ValueError("n_load_steps must be >= 1")
    load_fractions = np.linspace(
        config.max_load / n_steps, config.max_load, n_steps
    )

    reaction_history = np.zeros(n_steps, dtype=np.float64)
    damage_history: list[NDArray[np.float64]] = []
    all_converged = True

    u_flat = np.zeros(n * dpn, dtype=np.float64)
    esplit = config.energy_split

    # Select assembly functions based on element type
    _assemble_Ku = _assemble_Ku_tet4 if is_3d else _assemble_Ku_tri3
    _assemble_Kd_fd = _assemble_Kd_fd_tet4 if is_3d else _assemble_Kd_fd_tri3

    for step_idx, lam in enumerate(load_fractions):
        cons_dofs, cons_vals, f, loaded_dofs = _gather_disp_bcs(
            mesh, load_case, scale=float(lam), dof_per_node=dpn,
        )

        step_converged = False
        for it in range(config.max_staggered_iters):
            d_old = d.copy()

            # -- u-solve with current damage ---------------------------------
            K_u = _assemble_Ku(mesh, K0_all, dofs_all, d)
            K_u_p, f_p = _apply_penalty(K_u, f, cons_dofs, cons_vals)
            u_flat = np.linalg.solve(K_u_p, f_p)

            # -- update history (monotone) -----------------------------------
            psi_e = _element_psi_el(
                mesh, u_flat, C_mat, B_all, dofs_all,
                energy_split=esplit, lam=lam_val, mu=mu_val,
            )
            psi_nodal = _project_element_to_nodes(mesh, psi_e)
            history = np.maximum(history, psi_nodal)

            # -- d-solve -----------------------------------------------------
            K_d, f_d = _assemble_Kd_fd(
                mesh, history, config.Gc, config.l0, Me_all, Be_all
            )

            # Irreversibility: nodes already at/near d=1 are pinned.
            irreversible_dofs = np.nonzero(d_old >= 1.0 - 1e-12)[0]
            K_d_p = K_d.copy()
            f_d_p = f_d.copy()
            for dof in irreversible_dofs:
                K_d_p[int(dof), int(dof)] += _PENALTY
                f_d_p[int(dof)] += _PENALTY * 1.0

            d_new = np.linalg.solve(K_d_p, f_d_p)
            d_new = np.clip(d_new, 0.0, 1.0)
            d_new = np.maximum(d_new, d_old)

            delta = float(np.linalg.norm(d_new - d_old))
            d = d_new

            if delta < config.staggered_tol:
                step_converged = True
                break

        if not step_converged:
            all_converged = False

        K_u_final = _assemble_Ku(mesh, K0_all, dofs_all, d)
        reaction_history[step_idx] = _reaction_at_loaded(
            K_u_final, u_flat, loaded_dofs
        )
        damage_history.append(d.copy())

    u_out = u_flat.reshape(n, dpn).copy()

    return FractureResult(
        displacement=u_out,
        damage=d.copy(),
        reaction_force=float(reaction_history[-1]),
        load_steps=load_fractions.copy(),
        reaction_history=reaction_history,
        damage_history=damage_history,
        converged=all_converged,
        metadata={
            "element_type": mesh.element_type.value,
            "l0": config.l0,
            "Gc": config.Gc,
            "n_load_steps": n_steps,
            "h_avg": h_avg,
            "energy_split": config.energy_split.value,
            "backend": "jax",
        },
    )
