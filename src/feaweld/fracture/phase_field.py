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
    """

    l0: float = 0.1
    Gc: float = 2.7
    n_load_steps: int = 20
    max_load: float = 1.0
    staggered_tol: float = 1e-4
    max_staggered_iters: int = 50
    min_mesh_ratio: float = 4.0


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


def _average_edge_length(mesh: FEMesh) -> float:
    coords = mesh.nodes
    acc = 0.0
    cnt = 0
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
# Elastic energy density (tri3 plane strain)
# ---------------------------------------------------------------------------


def _element_psi_el(
    mesh: FEMesh,
    u: NDArray[np.float64],
    C_ps: NDArray[np.float64],
    B_all: NDArray[np.float64],
    dofs_all: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return per-element *undegraded* elastic energy density psi_el.

    psi_el = 0.5 * eps : sigma_0 = 0.5 * eps^T C eps

    The "+" split (Amor / Miehe) is a small follow-up — for tensile-dominated
    DCB-style loading the full energy is an acceptable MVP approximation.
    """
    # Vectorized over elements using precomputed B matrices.
    u_elem = u[dofs_all]                          # (n_elem, 6)
    eps = np.einsum("eij,ej->ei", B_all, u_elem)  # (n_elem, 3)
    sig = eps @ C_ps.T                            # (n_elem, 3)
    psi = 0.5 * np.sum(eps * sig, axis=1)         # (n_elem,)
    return psi


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
    mesh: FEMesh, load_case: LoadCase, scale: float
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], list[int]]:
    """Return (constrained_dofs, constrained_vals, f_global, loaded_dofs).

    ``scale`` multiplies both the prescribed displacements (if any) and the
    applied force magnitudes, so a single call produces the BCs at a given
    load fraction. ``loaded_dofs`` lists DOFs where a force BC was applied
    — these are used to evaluate the reaction at the end of each step.
    """
    dof_per_node = 2
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
    """Staggered AT2 phase-field fracture solver on TRI3 / plane strain.

    Parameters
    ----------
    mesh:
        FEMesh — must be :attr:`ElementType.TRI3`. 3D / tets are not yet
        supported (raises ``NotImplementedError``).
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

    if mesh.element_type is not ElementType.TRI3:
        raise NotImplementedError(
            "solve_phase_field currently supports TRI3 / plane strain only; "
            f"got {mesh.element_type.value}. TET4 / 3D is a follow-up."
        )

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

    C_ps = material.elasticity_tensor_2d(20.0, plane="strain")

    # -- Precompute kinematics once (they only depend on geometry) ----------
    B_all, K0_all, _area_all = _precompute_element_kinematics(
        mesh, C_ps, thickness
    )
    dofs_all = _element_dofs(mesh)
    Me_all, Be_all = _precompute_damage_kinematics(mesh, thickness)

    # Load fractions: n_load_steps ≥ 1 steps from > 0 to max_load.
    n_steps = int(config.n_load_steps)
    if n_steps < 1:
        raise ValueError("n_load_steps must be >= 1")
    load_fractions = np.linspace(
        config.max_load / n_steps, config.max_load, n_steps
    )

    reaction_history = np.zeros(n_steps, dtype=np.float64)
    damage_history: list[NDArray[np.float64]] = []
    all_converged = True

    u_flat = np.zeros(n * 2, dtype=np.float64)

    for step_idx, lam in enumerate(load_fractions):
        cons_dofs, cons_vals, f, loaded_dofs = _gather_disp_bcs(
            mesh, load_case, scale=float(lam)
        )

        step_converged = False
        for it in range(config.max_staggered_iters):
            d_old = d.copy()

            # -- u-solve with current damage ---------------------------------
            K_u = _assemble_Ku_tri3(mesh, K0_all, dofs_all, d)
            K_u_p, f_p = _apply_penalty(K_u, f, cons_dofs, cons_vals)
            u_flat = np.linalg.solve(K_u_p, f_p)

            # -- update history (monotone) -----------------------------------
            psi_e = _element_psi_el(mesh, u_flat, C_ps, B_all, dofs_all)
            psi_nodal = _project_element_to_nodes(mesh, psi_e)
            history = np.maximum(history, psi_nodal)

            # -- d-solve -----------------------------------------------------
            K_d, f_d = _assemble_Kd_fd_tri3(
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
            # Damage is bounded [0,1] and monotone non-decreasing.
            d_new = np.clip(d_new, 0.0, 1.0)
            d_new = np.maximum(d_new, d_old)

            delta = float(np.linalg.norm(d_new - d_old))
            d = d_new

            if delta < config.staggered_tol:
                step_converged = True
                break

        if not step_converged:
            all_converged = False

        # Reaction at the loaded DOFs (use the undegraded-in-BC stiffness).
        K_u_final = _assemble_Ku_tri3(mesh, K0_all, dofs_all, d)
        reaction_history[step_idx] = _reaction_at_loaded(
            K_u_final, u_flat, loaded_dofs
        )
        damage_history.append(d.copy())

    u_out = u_flat.reshape(n, 2).copy()

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
            "backend": "jax",
        },
    )
