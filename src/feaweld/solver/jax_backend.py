"""JAX-based differentiable FEA backend.

Implements `SolverBackend` with pure JAX numerics so the entire
assembly + linear solve + post-processing chain is differentiable via
`jax.grad`. Supports linear elastic static analysis on TRI3 (plane
strain) and TET4 meshes.

This backend is the differentiable substrate under everything in
Track A of the experimental extensions plan: it unlocks gradient-based
inverse problems, neural operator ground-truth generation, and the
variational phase-field fracture solver.

Design notes
------------
- `FEMesh` and `FEAResults` are pure NumPy dataclasses; we convert to
  `jnp.ndarray` at the solver boundary and back to NumPy on return so
  the rest of the pipeline sees no JAX types.
- Element assembly is batched via `jax.vmap` over elements.
- Linear elastic uses closed-form element stiffness (constant strain
  triangle + constant strain tetrahedron) — no Gauss integration
  needed for these element types.
- Boundary conditions are applied via penalty on the linear system,
  which keeps the assembled matrix symmetric and JIT-friendly.
- Plasticity, thermal, and coupled solves are stubbed in A1 and wired
  in subsequent phases (A2+).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jax = None  # type: ignore
    jnp = None  # type: ignore

from feaweld.core.materials import Material
from feaweld.core.types import (
    ElementType,
    FEAResults,
    FEMesh,
    LoadCase,
    LoadType,
    StressField,
)
from feaweld.solver.backend import SolverBackend

if TYPE_CHECKING:
    from feaweld.solver.jax_constitutive import (
        JAXConstitutiveModel,
        JAXJ2Plasticity,
    )


_PENALTY = 1.0e12


def _require_jax() -> None:
    if not _HAS_JAX:
        raise ImportError(
            "JAX is required for JAXBackend. "
            "Install with: pip install 'feaweld[jax]'"
        )


def _enable_x64() -> None:
    """Enable float64 precision for FEA-grade accuracy."""
    if _HAS_JAX:
        jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Element-level kinematics (closed-form for CST and linear tet)
# ---------------------------------------------------------------------------


def _tri3_stiffness(
    coords: "jnp.ndarray", C_voigt: "jnp.ndarray", thickness: float
) -> "jnp.ndarray":
    """Constant-strain triangle stiffness, plane strain.

    Parameters
    ----------
    coords : (3, 2) vertex coordinates
    C_voigt : (3, 3) plane-strain elasticity (σ_xx, σ_yy, τ_xy)
    thickness : out-of-plane thickness

    Returns
    -------
    (6, 6) element stiffness matrix (DOF order u1x, u1y, u2x, u2y, u3x, u3y)
    """
    x = coords[:, 0]
    y = coords[:, 1]
    b = jnp.stack([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = jnp.stack([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    two_area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    area = 0.5 * jnp.abs(two_area)

    # B matrix (3 x 6) mapping nodal displacements to engineering strain
    B = jnp.zeros((3, 6))
    for i in range(3):
        B = B.at[0, 2 * i].set(b[i])
        B = B.at[1, 2 * i + 1].set(c[i])
        B = B.at[2, 2 * i].set(c[i])
        B = B.at[2, 2 * i + 1].set(b[i])
    B = B / two_area

    ke = thickness * area * (B.T @ C_voigt @ B)
    return ke, B


def _tet4_stiffness(
    coords: "jnp.ndarray", C6: "jnp.ndarray"
) -> tuple["jnp.ndarray", "jnp.ndarray"]:
    """Linear tetrahedron stiffness in 3D.

    Parameters
    ----------
    coords : (4, 3)
    C6 : (6, 6) full 3D Voigt elasticity tensor

    Returns
    -------
    (12, 12) element stiffness and (6, 12) B matrix.
    """
    # Volume from the 4x4 determinant form
    ones = jnp.ones((4, 1))
    M = jnp.concatenate([ones, coords], axis=1)  # (4, 4)
    six_vol = jnp.linalg.det(M)
    vol = jnp.abs(six_vol) / 6.0

    # Shape function coefficients: N_i = (a_i + b_i x + c_i y + d_i z)/6V
    # b_i = -det(of cofactor excluding x column of row i), etc.
    # Instead use direct derivation via cofactors.
    def cofactor(i):
        rows = [r for r in range(4) if r != i]
        sub = M[jnp.array(rows)]  # (3, 4)
        # b_i = cofactor w.r.t. column 1 (x)
        # cofactor_{i,1} = (-1)^{i+1} * det of minor
        sign = (-1.0) ** (i + 1)
        minor_x = jnp.stack([sub[:, 0], sub[:, 2], sub[:, 3]], axis=1)
        minor_y = jnp.stack([sub[:, 0], sub[:, 1], sub[:, 3]], axis=1)
        minor_z = jnp.stack([sub[:, 0], sub[:, 1], sub[:, 2]], axis=1)
        b_i = -sign * jnp.linalg.det(minor_x)
        c_i = sign * jnp.linalg.det(minor_y)
        d_i = -sign * jnp.linalg.det(minor_z)
        return b_i, c_i, d_i

    bs, cs, ds = [], [], []
    for i in range(4):
        b_i, c_i, d_i = cofactor(i)
        bs.append(b_i)
        cs.append(c_i)
        ds.append(d_i)
    b = jnp.stack(bs) / six_vol
    c = jnp.stack(cs) / six_vol
    d = jnp.stack(ds) / six_vol

    # B matrix (6, 12): Voigt strain = B u
    # Order: ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz
    B = jnp.zeros((6, 12))
    for i in range(4):
        col = 3 * i
        B = B.at[0, col].set(b[i])
        B = B.at[1, col + 1].set(c[i])
        B = B.at[2, col + 2].set(d[i])
        B = B.at[3, col].set(c[i])
        B = B.at[3, col + 1].set(b[i])
        B = B.at[4, col + 1].set(d[i])
        B = B.at[4, col + 2].set(c[i])
        B = B.at[5, col].set(d[i])
        B = B.at[5, col + 2].set(b[i])

    ke = vol * (B.T @ C6 @ B)
    return ke, B


# ---------------------------------------------------------------------------
# Assembly + solve
# ---------------------------------------------------------------------------


def _assemble_global(
    ke_batch: "jnp.ndarray",
    elements: NDArray[np.int64],
    n_dof: int,
    dof_per_node: int,
) -> "jnp.ndarray":
    """Scatter-assemble element stiffnesses into a dense global matrix.

    Dense assembly is used here (not sparse) to keep the entire solve
    on-device and JIT-friendly. For the mesh sizes the MVP targets
    (a few thousand nodes) this is fine; a sparse version is a follow-up.
    """
    K = jnp.zeros((n_dof, n_dof))
    n_elem, ke_dim, _ = ke_batch.shape
    nodes_per_elem = ke_dim // dof_per_node

    def element_dofs(conn):
        # conn: (nodes_per_elem,)
        return jnp.concatenate(
            [conn[i] * dof_per_node + jnp.arange(dof_per_node)
             for i in range(nodes_per_elem)]
        )

    dofs_all = jnp.stack([element_dofs(jnp.asarray(elements[e]))
                          for e in range(n_elem)])

    def scatter_one(K_acc, args):
        ke, dofs = args
        K_acc = K_acc.at[jnp.ix_(dofs, dofs)].add(ke)
        return K_acc, None

    K_final, _ = jax.lax.scan(scatter_one, K, (ke_batch, dofs_all))
    return K_final


def _apply_penalty_bcs(
    K: "jnp.ndarray",
    f: "jnp.ndarray",
    constrained_dofs: "jnp.ndarray",
    constrained_vals: "jnp.ndarray",
) -> tuple["jnp.ndarray", "jnp.ndarray"]:
    """Apply Dirichlet BCs by adding a large penalty on the diagonal."""
    K_p = K.at[constrained_dofs, constrained_dofs].add(_PENALTY)
    f_p = f.at[constrained_dofs].add(_PENALTY * constrained_vals)
    return K_p, f_p


# ---------------------------------------------------------------------------
# Public backend class
# ---------------------------------------------------------------------------


class JAXBackend(SolverBackend):
    """Differentiable FEA backend built on JAX.

    A1 (this file): linear elastic static on TRI3 / TET4 with penalty BCs.
    A2: JAX J2 plasticity wired via `JAXConstitutiveModel`.
    A3: crystal plasticity.
    A4: phase-field fracture (coupled u-d).
    A5: neural operator ground-truth generation harness.

    The backend eagerly enables x64 precision on import and performs all
    math in float64. Inputs and outputs are NumPy; JAX is an internal
    detail.
    """

    def __init__(self, constitutive: "JAXConstitutiveModel | None" = None) -> None:
        _require_jax()
        _enable_x64()
        self._constitutive = constitutive

    # -- Static elastic solve ------------------------------------------------

    def solve_static(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float = 20.0,
    ) -> FEAResults:
        etype = mesh.element_type
        if etype not in (ElementType.TRI3, ElementType.TET4):
            raise NotImplementedError(
                f"JAXBackend currently supports TRI3 and TET4 only, got {etype.value}. "
                f"Use FEniCSBackend for higher-order elements."
            )
        if self._constitutive is not None and hasattr(
            self._constitutive, "stress_stateful"
        ):
            return self.solve_static_incremental(
                mesh,
                material,
                load_case,
                temperature,
                constitutive=self._constitutive,
            )
        if etype == ElementType.TRI3:
            return self._solve_static_tri3(mesh, material, load_case, temperature)
        return self._solve_static_tet4(mesh, material, load_case, temperature)

    def _gather_bcs(
        self,
        mesh: FEMesh,
        load_case: LoadCase,
        dof_per_node: int,
    ) -> tuple["jnp.ndarray", "jnp.ndarray", "jnp.ndarray"]:
        """Translate LoadCase -> (constrained_dofs, constrained_vals, rhs)."""
        n_dof = mesh.n_nodes * dof_per_node
        f = np.zeros(n_dof)
        cons_dofs: list[int] = []
        cons_vals: list[float] = []

        for bc in load_case.constraints:
            if bc.bc_type != LoadType.DISPLACEMENT:
                continue
            if bc.node_set not in mesh.node_sets:
                continue
            nodes = mesh.node_sets[bc.node_set]
            vals = np.asarray(bc.values, dtype=np.float64).reshape(-1)
            for n in nodes:
                for k in range(dof_per_node):
                    dof = int(n) * dof_per_node + k
                    v = float(vals[k]) if k < len(vals) else 0.0
                    cons_dofs.append(dof)
                    cons_vals.append(v)

        for bc in load_case.loads:
            if bc.bc_type != LoadType.FORCE:
                continue
            if bc.node_set not in mesh.node_sets:
                continue
            nodes = mesh.node_sets[bc.node_set]
            vals = np.asarray(bc.values, dtype=np.float64).reshape(-1)
            # Distribute total force evenly across nodes in the set
            share = vals / max(len(nodes), 1)
            for n in nodes:
                for k in range(dof_per_node):
                    dof = int(n) * dof_per_node + k
                    comp = float(share[k]) if k < len(share) else 0.0
                    f[dof] += comp

        return (
            jnp.asarray(cons_dofs, dtype=jnp.int32),
            jnp.asarray(cons_vals, dtype=jnp.float64),
            jnp.asarray(f, dtype=jnp.float64),
        )

    def _solve_static_tri3(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float,
    ) -> FEAResults:
        C_ps = material.elasticity_tensor_2d(temperature, plane="strain")
        C_ps_j = jnp.asarray(C_ps, dtype=jnp.float64)

        node_coords_2d = jnp.asarray(mesh.nodes[:, :2], dtype=jnp.float64)
        elements = mesh.elements.astype(np.int64)
        thickness = 1.0

        def element_k(conn):
            coords = node_coords_2d[conn]
            ke, _ = _tri3_stiffness(coords, C_ps_j, thickness)
            return ke

        ke_batch = jax.vmap(element_k)(jnp.asarray(elements))
        K = _assemble_global(ke_batch, elements, mesh.n_nodes * 2, dof_per_node=2)

        cons_dofs, cons_vals, f = self._gather_bcs(mesh, load_case, dof_per_node=2)
        K_p, f_p = _apply_penalty_bcs(K, f, cons_dofs, cons_vals)

        u = jnp.linalg.solve(K_p, f_p)
        u_np = np.asarray(u).reshape(mesh.n_nodes, 2)

        # Per-element strain and stress via the B matrix
        def elem_stress(conn):
            coords = node_coords_2d[conn]
            _, B = _tri3_stiffness(coords, C_ps_j, thickness)
            u_e = jnp.stack([u[2 * conn[i] + k] for i in range(3) for k in range(2)])
            eps = B @ u_e
            sig_ps = C_ps_j @ eps  # [σ_xx, σ_yy, τ_xy] plane strain
            return eps, sig_ps

        strains_ps, stresses_ps = jax.vmap(elem_stress)(jnp.asarray(elements))

        # Project element values back onto nodes by simple averaging
        strain_node = np.zeros((mesh.n_nodes, 6))
        stress_node = np.zeros((mesh.n_nodes, 6))
        count = np.zeros(mesh.n_nodes)
        nu = material.nu(temperature)
        E = material.E(temperature)
        for e_idx, conn in enumerate(elements):
            eps_e = np.asarray(strains_ps[e_idx])   # [εxx, εyy, γxy]
            sig_e = np.asarray(stresses_ps[e_idx])  # [σxx, σyy, τxy]
            # σ_zz from plane-strain closure: ε_zz = 0 -> σ_zz = nu*(σxx+σyy)
            sig_zz = nu * (sig_e[0] + sig_e[1])
            eps_full = np.array([eps_e[0], eps_e[1], 0.0, eps_e[2], 0.0, 0.0])
            sig_full = np.array(
                [sig_e[0], sig_e[1], sig_zz, sig_e[2], 0.0, 0.0]
            )
            for n in conn:
                strain_node[n] += eps_full
                stress_node[n] += sig_full
                count[n] += 1
        count = np.maximum(count, 1)
        strain_node /= count[:, None]
        stress_node /= count[:, None]

        u_full = np.zeros((mesh.n_nodes, 3))
        u_full[:, :2] = u_np

        return FEAResults(
            mesh=mesh,
            displacement=u_full,
            stress=StressField(values=stress_node),
            strain=strain_node,
            metadata={"backend": "jax", "element_type": "tri3", "E": E, "nu": nu},
        )

    def _solve_static_tet4(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float,
    ) -> FEAResults:
        C6 = material.elasticity_tensor_3d(temperature)
        C6_j = jnp.asarray(C6, dtype=jnp.float64)

        node_coords = jnp.asarray(mesh.nodes, dtype=jnp.float64)
        elements = mesh.elements.astype(np.int64)

        def element_k(conn):
            coords = node_coords[conn]
            ke, _ = _tet4_stiffness(coords, C6_j)
            return ke

        ke_batch = jax.vmap(element_k)(jnp.asarray(elements))
        K = _assemble_global(ke_batch, elements, mesh.n_nodes * 3, dof_per_node=3)

        cons_dofs, cons_vals, f = self._gather_bcs(mesh, load_case, dof_per_node=3)
        K_p, f_p = _apply_penalty_bcs(K, f, cons_dofs, cons_vals)

        u = jnp.linalg.solve(K_p, f_p)
        u_np = np.asarray(u).reshape(mesh.n_nodes, 3)

        def elem_stress(conn):
            coords = node_coords[conn]
            _, B = _tet4_stiffness(coords, C6_j)
            u_e = jnp.stack(
                [u[3 * conn[i] + k] for i in range(4) for k in range(3)]
            )
            eps = B @ u_e
            sig = C6_j @ eps
            return eps, sig

        strains, stresses = jax.vmap(elem_stress)(jnp.asarray(elements))

        strain_node = np.zeros((mesh.n_nodes, 6))
        stress_node = np.zeros((mesh.n_nodes, 6))
        count = np.zeros(mesh.n_nodes)
        for e_idx, conn in enumerate(elements):
            eps_e = np.asarray(strains[e_idx])
            sig_e = np.asarray(stresses[e_idx])
            for n in conn:
                strain_node[n] += eps_e
                stress_node[n] += sig_e
                count[n] += 1
        count = np.maximum(count, 1)
        strain_node /= count[:, None]
        stress_node /= count[:, None]

        return FEAResults(
            mesh=mesh,
            displacement=u_np,
            stress=StressField(values=stress_node),
            strain=strain_node,
            metadata={"backend": "jax", "element_type": "tet4"},
        )

    # -- Incremental plasticity solve ---------------------------------------

    def solve_static_incremental(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float = 20.0,
        n_steps: int = 10,
        constitutive: "JAXJ2Plasticity | None" = None,
    ) -> FEAResults:
        """Incremental static solve with J2 plasticity via radial return.

        Applies the external load in ``n_steps`` equal pseudo-time
        increments. Each increment uses an elastic predictor / plastic
        corrector operator split: assemble the elastic stiffness once,
        accumulate a plastic pseudo-force from previously stored
        plastic strains, solve for displacements, then run a per-element
        radial return to update plastic strain and stress. Carried state
        ``(plastic_strain, eqps, displacement)`` threads through
        ``jax.lax.scan`` so reverse-mode autodiff flows cleanly.
        """
        etype = mesh.element_type
        if etype not in (ElementType.TRI3, ElementType.TET4):
            raise NotImplementedError(
                f"solve_static_incremental supports TRI3 and TET4 only, got {etype.value}."
            )
        cmodel = constitutive if constitutive is not None else self._constitutive
        if cmodel is None or not hasattr(cmodel, "stress_stateful"):
            raise ValueError(
                "solve_static_incremental requires a stateful constitutive model "
                "(e.g. JAXJ2Plasticity). Pass one via the constitutive= argument."
            )
        if etype == ElementType.TRI3:
            return self._solve_incremental_tri3(
                mesh, material, load_case, temperature, n_steps, cmodel
            )
        return self._solve_incremental_tet4(
            mesh, material, load_case, temperature, n_steps, cmodel
        )

    def _run_incremental_tri3(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float,
        n_steps: int,
        cmodel: "JAXJ2Plasticity",
    ):
        """Pure-JAX incremental TRI3 driver (plane strain).

        Returns ``(u_final, ep_final, eqps_final, f_ext,
        eps_full_batch_final, stress_batch, elements_np)`` for
        gradient-friendly consumption.
        """
        C_ps = material.elasticity_tensor_2d(temperature, plane="strain")
        C_ps_j = jnp.asarray(C_ps, dtype=jnp.float64)
        C6_j = cmodel.C

        node_coords_2d = jnp.asarray(mesh.nodes[:, :2], dtype=jnp.float64)
        elements_np = mesh.elements.astype(np.int64)
        elements_j = jnp.asarray(elements_np)
        thickness = 1.0
        n_elem = elements_np.shape[0]

        def element_k_and_B(conn):
            coords = node_coords_2d[conn]
            ke, B = _tri3_stiffness(coords, C_ps_j, thickness)
            x = coords[:, 0]
            y = coords[:, 1]
            two_area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            area = 0.5 * jnp.abs(two_area)
            return ke, B, area

        ke_batch, B_batch, area_batch = jax.vmap(element_k_and_B)(elements_j)
        K = _assemble_global(ke_batch, elements_np, mesh.n_nodes * 2, dof_per_node=2)

        cons_dofs, cons_vals, f_ext = self._gather_bcs(mesh, load_case, dof_per_node=2)

        def element_dofs_row(conn):
            return jnp.concatenate(
                [conn[i] * 2 + jnp.arange(2) for i in range(3)]
            )
        dofs_batch = jax.vmap(element_dofs_row)(elements_j)

        def plane_strain_eps_full(eps_ps):
            return jnp.array(
                [eps_ps[0], eps_ps[1], 0.0, eps_ps[2], 0.0, 0.0]
            )

        def assemble_plastic_force(ep_batch_full):
            def per_elem(B, area, ep_full):
                ep_ps = jnp.stack([ep_full[0], ep_full[1], ep_full[3]])
                sig_p = C_ps_j @ ep_ps
                return thickness * area * (B.T @ sig_p)
            f_elems = jax.vmap(per_elem)(B_batch, area_batch, ep_batch_full)
            f_p = jnp.zeros_like(f_ext)
            def scatter(acc, args):
                fe, dofs = args
                acc = acc.at[dofs].add(fe)
                return acc, None
            f_p, _ = jax.lax.scan(scatter, f_p, (f_elems, dofs_batch))
            return f_p

        def compute_strains(u):
            def per_elem(conn, B):
                u_e = jnp.stack(
                    [u[2 * conn[i] + k] for i in range(3) for k in range(2)]
                )
                eps_ps = B @ u_e
                return plane_strain_eps_full(eps_ps)
            return jax.vmap(per_elem)(elements_j, B_batch)

        def step(state, t_scale):
            ep_batch, eqps_batch, _u_prev = state
            f_p = assemble_plastic_force(ep_batch)
            rhs = t_scale * f_ext + f_p
            K_p, rhs_p = _apply_penalty_bcs(
                K, rhs, cons_dofs, t_scale * cons_vals
            )
            u = jnp.linalg.solve(K_p, rhs_p)
            eps_full_batch = compute_strains(u)
            _stress, ep_new, eqps_new = cmodel.stress_stateful(
                eps_full_batch, ep_batch, eqps_batch
            )
            return (ep_new, eqps_new, u), None

        ep0 = jnp.zeros((n_elem, 6))
        eqps0 = jnp.zeros((n_elem,))
        u0 = jnp.zeros((mesh.n_nodes * 2,))
        t_scales = (jnp.arange(1, n_steps + 1, dtype=jnp.float64)) / float(n_steps)
        (ep_final, eqps_final, u_final), _ = jax.lax.scan(
            step, (ep0, eqps0, u0), t_scales
        )

        eps_full_batch = compute_strains(u_final)
        def elem_stress_from_ep(eps_full, ep_full):
            return C6_j @ (eps_full - ep_full)
        stress_batch = jax.vmap(elem_stress_from_ep)(eps_full_batch, ep_final)
        return (
            u_final,
            ep_final,
            eqps_final,
            f_ext,
            eps_full_batch,
            stress_batch,
            elements_np,
        )

    def _solve_incremental_tri3(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float,
        n_steps: int,
        cmodel: "JAXJ2Plasticity",
    ) -> FEAResults:
        (
            u_final,
            _ep_final,
            _eqps_final,
            _f_ext,
            eps_full_batch,
            stress_batch,
            elements_np,
        ) = self._run_incremental_tri3(
            mesh, material, load_case, temperature, n_steps, cmodel
        )
        nu = material.nu(temperature)
        E = material.E(temperature)

        u_np = np.asarray(u_final).reshape(mesh.n_nodes, 2)
        strain_node = np.zeros((mesh.n_nodes, 6))
        stress_node = np.zeros((mesh.n_nodes, 6))
        count = np.zeros(mesh.n_nodes)
        eps_full_np = np.asarray(eps_full_batch)
        stress_np = np.asarray(stress_batch)
        # σ_zz comes directly from the full 3D constitutive law and
        # already accounts for plastic strain; do not overwrite with the
        # elastic plane-strain closure (σ_zz ≠ ν(σ_xx+σ_yy) when εp ≠ 0).
        for e_idx, conn in enumerate(elements_np):
            eps_full = eps_full_np[e_idx]
            sig_full = stress_np[e_idx]
            for n in conn:
                strain_node[n] += eps_full
                stress_node[n] += sig_full
                count[n] += 1
        count = np.maximum(count, 1)
        strain_node /= count[:, None]
        stress_node /= count[:, None]

        u_full = np.zeros((mesh.n_nodes, 3))
        u_full[:, :2] = u_np

        return FEAResults(
            mesh=mesh,
            displacement=u_full,
            stress=StressField(values=stress_node),
            strain=strain_node,
            metadata={
                "backend": "jax",
                "element_type": "tri3",
                "constitutive": "j2_plasticity",
                "n_steps": int(n_steps),
                "E": E,
                "nu": nu,
            },
        )

    def incremental_compliance_tet4(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        cmodel: "JAXJ2Plasticity",
        temperature: float = 20.0,
        n_steps: int = 10,
    ) -> "jnp.ndarray":
        """Pure-JAX scalar compliance for the incremental TET4 solve.

        Returns a JAX scalar (0.5 uᵀ f_ext) suitable for ``jax.grad``
        with respect to any JAX leaf inside ``cmodel``.
        """
        u_final, _, _, f_ext, _, _, _ = self._run_incremental_tet4(
            mesh, material, load_case, temperature, n_steps, cmodel
        )
        return 0.5 * jnp.dot(u_final, f_ext)

    def incremental_compliance_tri3(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        cmodel: "JAXJ2Plasticity",
        temperature: float = 20.0,
        n_steps: int = 10,
    ) -> "jnp.ndarray":
        """Pure-JAX scalar compliance for the incremental TRI3 solve."""
        u_final, _, _, f_ext, _, _, _ = self._run_incremental_tri3(
            mesh, material, load_case, temperature, n_steps, cmodel
        )
        return 0.5 * jnp.dot(u_final, f_ext)

    def _run_incremental_tet4(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float,
        n_steps: int,
        cmodel: "JAXJ2Plasticity",
    ):
        """Pure-JAX incremental TET4 driver.

        Returns ``(u_final, ep_final, eqps_final, f_ext, eps_batch_final,
        stress_batch, elements_np)`` all as JAX / NumPy leaves, with
        ``u_final``, ``ep_final``, ``eqps_final``, ``f_ext``,
        ``eps_batch_final`` and ``stress_batch`` being JAX arrays so
        ``jax.grad`` can trace the whole path.
        """
        C6 = material.elasticity_tensor_3d(temperature)
        C6_j = jnp.asarray(C6, dtype=jnp.float64)

        node_coords = jnp.asarray(mesh.nodes, dtype=jnp.float64)
        elements_np = mesh.elements.astype(np.int64)
        elements_j = jnp.asarray(elements_np)
        n_elem = elements_np.shape[0]

        def element_k_and_B(conn):
            coords = node_coords[conn]
            ke, B = _tet4_stiffness(coords, C6_j)
            ones = jnp.ones((4, 1))
            M = jnp.concatenate([ones, coords], axis=1)
            vol = jnp.abs(jnp.linalg.det(M)) / 6.0
            return ke, B, vol

        ke_batch, B_batch, vol_batch = jax.vmap(element_k_and_B)(elements_j)
        K = _assemble_global(ke_batch, elements_np, mesh.n_nodes * 3, dof_per_node=3)

        cons_dofs, cons_vals, f_ext = self._gather_bcs(mesh, load_case, dof_per_node=3)

        def element_dofs_row(conn):
            return jnp.concatenate(
                [conn[i] * 3 + jnp.arange(3) for i in range(4)]
            )
        dofs_batch = jax.vmap(element_dofs_row)(elements_j)

        def assemble_plastic_force(ep_batch):
            def per_elem(B, vol, ep):
                sig_p = C6_j @ ep
                return vol * (B.T @ sig_p)
            f_elems = jax.vmap(per_elem)(B_batch, vol_batch, ep_batch)
            f_p = jnp.zeros_like(f_ext)
            def scatter(acc, args):
                fe, dofs = args
                acc = acc.at[dofs].add(fe)
                return acc, None
            f_p, _ = jax.lax.scan(scatter, f_p, (f_elems, dofs_batch))
            return f_p

        def compute_strains(u):
            def per_elem(conn, B):
                u_e = jnp.stack(
                    [u[3 * conn[i] + k] for i in range(4) for k in range(3)]
                )
                return B @ u_e
            return jax.vmap(per_elem)(elements_j, B_batch)

        def step(state, t_scale):
            ep_batch, eqps_batch, _u_prev = state
            f_p = assemble_plastic_force(ep_batch)
            rhs = t_scale * f_ext + f_p
            K_p, rhs_p = _apply_penalty_bcs(
                K, rhs, cons_dofs, t_scale * cons_vals
            )
            u = jnp.linalg.solve(K_p, rhs_p)
            eps_batch = compute_strains(u)
            _stress, ep_new, eqps_new = cmodel.stress_stateful(
                eps_batch, ep_batch, eqps_batch
            )
            return (ep_new, eqps_new, u), None

        ep0 = jnp.zeros((n_elem, 6))
        eqps0 = jnp.zeros((n_elem,))
        u0 = jnp.zeros((mesh.n_nodes * 3,))
        t_scales = (jnp.arange(1, n_steps + 1, dtype=jnp.float64)) / float(n_steps)
        (ep_final, eqps_final, u_final), _ = jax.lax.scan(
            step, (ep0, eqps0, u0), t_scales
        )

        eps_batch_final = compute_strains(u_final)
        def elem_stress_from_ep(eps, ep):
            return C6_j @ (eps - ep)
        stress_batch = jax.vmap(elem_stress_from_ep)(eps_batch_final, ep_final)
        return (
            u_final,
            ep_final,
            eqps_final,
            f_ext,
            eps_batch_final,
            stress_batch,
            elements_np,
        )

    def _solve_incremental_tet4(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float,
        n_steps: int,
        cmodel: "JAXJ2Plasticity",
    ) -> FEAResults:
        (
            u_final,
            ep_final,
            _eqps_final,
            _f_ext,
            eps_batch_final,
            stress_batch,
            elements_np,
        ) = self._run_incremental_tet4(
            mesh, material, load_case, temperature, n_steps, cmodel
        )

        u_np = np.asarray(u_final).reshape(mesh.n_nodes, 3)
        strain_node = np.zeros((mesh.n_nodes, 6))
        stress_node = np.zeros((mesh.n_nodes, 6))
        count = np.zeros(mesh.n_nodes)
        eps_np = np.asarray(eps_batch_final)
        stress_np = np.asarray(stress_batch)
        for e_idx, conn in enumerate(elements_np):
            for n in conn:
                strain_node[n] += eps_np[e_idx]
                stress_node[n] += stress_np[e_idx]
                count[n] += 1
        count = np.maximum(count, 1)
        strain_node /= count[:, None]
        stress_node /= count[:, None]

        return FEAResults(
            mesh=mesh,
            displacement=u_np,
            stress=StressField(values=stress_node),
            strain=strain_node,
            metadata={
                "backend": "jax",
                "element_type": "tet4",
                "constitutive": "j2_plasticity",
                "n_steps": int(n_steps),
            },
        )

    # -- Thermal / coupled stubs --------------------------------------------

    def solve_thermal_steady(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
    ) -> FEAResults:
        raise NotImplementedError(
            "JAXBackend.solve_thermal_steady is not implemented yet. "
            "Track A1 only ships linear-elastic static. Use FEniCSBackend for thermal."
        )

    def solve_thermal_transient(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        time_steps: NDArray,
        heat_source: object | None = None,
    ) -> FEAResults:
        raise NotImplementedError(
            "JAXBackend.solve_thermal_transient is not implemented yet. "
            "Track A1 only ships linear-elastic static."
        )

    def solve_coupled(
        self,
        mesh: FEMesh,
        material: Material,
        mechanical_lc: LoadCase,
        thermal_lc: LoadCase,
        time_steps: NDArray,
    ) -> FEAResults:
        raise NotImplementedError(
            "JAXBackend.solve_coupled is not implemented yet. "
            "Track A1 only ships linear-elastic static."
        )

    # -- Differentiable API exposed to Track A5 (neural operator) ----------

    def static_compliance(
        self,
        mesh: FEMesh,
        material: Material,
        load_case: LoadCase,
        temperature: float = 20.0,
    ) -> float:
        """Scalar compliance functional for gradient-based design.

        Useful as a QoI for `jax.grad` in inverse/optimization loops and
        as a regression target for neural operator training.
        """
        result = self.solve_static(mesh, material, load_case, temperature)
        u = result.displacement.reshape(-1)
        # Compliance = uᵀ f, but without re-running assembly we use
        # the strain energy U = 0.5 uᵀ K u via stress·strain integration.
        sig = result.stress.values
        eps = result.strain
        return float(0.5 * np.einsum("ij,ij->", sig, eps))
